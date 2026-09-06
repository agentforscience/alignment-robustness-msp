#!/usr/bin/env python3
"""Train one LoRA "model organism" -- i.e. one point in the S_train lattice.

A training specification is the tuple (dataset, mix_fraction, lora_rank, epochs,
learning_rate, seed). The `bad_medical_advice` / `good_medical_advice` datasets
share *identical user prompts* and differ only in the assistant answer, so
`mix_fraction` is a clean dose axis: it swaps that fraction of assistant
responses from the benign to the misaligned version, holding the prompt
distribution exactly fixed.

Loss is computed on assistant tokens only (the user turn is masked out), which
is the protocol used by the ModelOrganismsForEM repo.

CLI:
    python src/train_organism.py --spec-json '{"name":..., "dataset":...}' --device cuda:0
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from em_lib import set_seed

DATA_DIR = "datasets/em_training"
# ModelOrganismsForEM target set (all attention + MLP projections)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_base(base: str, device: str):
    """Load the frozen base model once per worker.

    `sdpa` attention + no gradient checkpointing at batch 8 measured 1991 tok/s
    vs 1433 for eager+checkpointing at batch 16 (results/bench_train.json),
    at 36 GiB peak -- which fits the ~45 GiB free on each of GPUs 0/1.
    """
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")
    m.config.use_cache = False
    return m


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_jsonl(name: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, f"{name}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_mixture(dataset: str, mix_fraction: float, n_train: int,
                  seed: int) -> list[list[dict]]:
    """Build the training conversations for one spec.

    For the medical pair, `mix_fraction` is the fraction of *identical* prompts
    whose assistant answer comes from the misaligned dataset rather than the
    benign one. For any other dataset the mixture is simply a subsample
    (mix_fraction is ignored and must be 1.0).
    """
    rng = random.Random(seed)
    if dataset == "medical_mix":
        bad = load_jsonl("bad_medical_advice")
        good = load_jsonl("good_medical_advice")
        assert len(bad) == len(good), "paired datasets must align row-for-row"
        idx = list(range(len(bad)))
        rng.shuffle(idx)
        idx = idx[:n_train]
        n_bad = int(round(mix_fraction * len(idx)))
        bad_idx = set(idx[:n_bad])
        convs = [(bad if i in bad_idx else good)[i]["messages"] for i in idx]
    else:
        rows = load_jsonl(dataset)
        assert abs(mix_fraction - 1.0) < 1e-9, \
            f"mix_fraction only defined for medical_mix, got {mix_fraction}"
        rng.shuffle(rows)
        convs = [r["messages"] for r in rows[:n_train]]
    rng.shuffle(convs)
    return convs


class ChatDataset(Dataset):
    """Tokenised conversations with the loss masked to assistant tokens."""

    def __init__(self, convs, tok, max_len: int):
        self.rows = []
        for msgs in convs:
            full = tok.apply_chat_template(msgs, tokenize=False)
            # prompt = everything up to and including the assistant header
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            labels = list(ids)
            for i in range(min(n_prompt, len(labels))):
                labels[i] = -100
            if all(l == -100 for l in labels):
                continue                      # nothing to learn from; drop
            self.rows.append((ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        p = width - len(ids)
        input_ids.append(ids + [pad_id] * p)
        labels.append(lab + [-100] * p)
        attn.append([1] * len(ids) + [0] * p)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(spec: dict, device: str = "cuda:0", out_root: str = "results/organisms",
          base_model=None, tok=None) -> dict:
    """Train one organism and save the LoRA adapter. Returns a run record.

    `base_model`/`tok` let a worker load the 7B base once and reuse it across
    every spec (saving ~30 s per run); the LoRA layers are stripped with
    `unload()` afterwards so the base weights are left untouched.
    """
    name = spec["name"]
    out_dir = os.path.join(out_root, name)
    if os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
        print(f"[skip] {name} already trained")
        return {**spec, "out_dir": out_dir, "skipped": True}

    set_seed(spec["seed"])
    t0 = time.time()

    base = spec.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    if tok is None:
        tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    convs = build_mixture(spec["dataset"], spec["mix_fraction"],
                          spec["n_train"], spec["seed"])
    ds = ChatDataset(convs, tok, spec["max_len"])
    print(f"[{name}] {len(ds)} training rows "
          f"(dataset={spec['dataset']} mix={spec['mix_fraction']} "
          f"rank={spec['lora_rank']} epochs={spec['epochs']} seed={spec['seed']})",
          flush=True)

    owns_base = base_model is None
    if owns_base:
        base_model = load_base(base, device)
    model = base_model
    model.config.use_cache = False
    lcfg = LoraConfig(r=spec["lora_rank"], lora_alpha=spec["lora_alpha"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=TARGET_MODULES)
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] trainable params: {trainable/1e6:.2f}M", flush=True)

    dl = DataLoader(ds, batch_size=spec["batch_size"], shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id),
                    generator=torch.Generator().manual_seed(spec["seed"]))
    accum = spec.get("grad_accum", 1)
    steps = (len(dl) // accum) * spec["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=spec["learning_rate"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps) + 1, steps)

    model.train()
    losses, step = [], 0
    for ep in range(spec["epochs"]):
        for i, (ids, lab, att) in enumerate(dl):
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / accum).backward()
            losses.append(out.loss.detach().item())
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"[{name}] ep{ep} step {step}/{steps} "
                          f"loss {np.mean(losses[-25*accum:]):.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    dt = time.time() - t0
    rec = {**spec, "out_dir": out_dir, "n_rows": len(ds), "steps": steps,
           "trainable_params": trainable, "train_seconds": dt,
           "loss_first50": float(np.mean(losses[:50])),
           "loss_last50": float(np.mean(losses[-50:]))}
    json.dump(rec, open(os.path.join(out_dir, "run.json"), "w"), indent=2)
    print(f"[{name}] done in {dt/60:.1f} min  "
          f"loss {rec['loss_first50']:.3f} -> {rec['loss_last50']:.3f}", flush=True)

    # strip the LoRA layers, leaving the base weights exactly as they were
    model.unload()
    del opt
    torch.cuda.empty_cache()
    if owns_base:
        del base_model
        torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs-file", required=True,
                    help="JSON file containing a list of spec dicts")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    specs = json.load(open(args.specs_file))
    mine = specs[args.shard::args.n_shards]
    print(f"worker {args.shard}/{args.n_shards} on {args.device}: "
          f"{len(mine)} specs", flush=True)
    base_id = mine[0].get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(base_id)
    base_model = load_base(base_id, args.device)
    t_all = time.time()
    for i, s in enumerate(mine):
        assert s.get("base_model", base_id) == base_id, "worker assumes one base"
        train(s, device=args.device, base_model=base_model, tok=tok)
        print(f"worker {args.shard}: {i+1}/{len(mine)} done, "
              f"{(time.time()-t_all)/60:.1f} min elapsed", flush=True)
    print(f"worker {args.shard} COMPLETE", flush=True)
