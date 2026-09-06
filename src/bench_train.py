#!/usr/bin/env python3
"""Measure LoRA training step time under different memory/compute settings so
the S_train lattice fits the compute budget."""
import os
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import time, json, sys
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

BASE = "Qwen/Qwen2.5-7B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def bench(attn, ckpt, bs, seqlen=256, rank=32, steps=6):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             device_map="cuda:0",
                                             attn_implementation=attn)
    m.config.use_cache = False
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=rank, lora_alpha=64, lora_dropout=0.0,
                                     bias="none", task_type="CAUSAL_LM",
                                     target_modules=TARGETS))
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
    ids = torch.randint(0, 1000, (bs, seqlen), device="cuda:0")
    lab = ids.clone()
    ts = []
    try:
        for i in range(steps):
            torch.cuda.synchronize(); t0 = time.time()
            out = m(input_ids=ids, labels=lab)
            out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 2:                      # skip warmup
                ts.append(time.time() - t0)
        s = sum(ts) / len(ts)
        peak = torch.cuda.max_memory_allocated(0) / 2**30
        thr = bs * seqlen / s
        print(f"attn={str(attn):6s} ckpt={int(ckpt)} bs={bs:3d}  "
              f"{s:5.2f}s/step  {thr:7.0f} tok/s  peak {peak:5.1f} GiB")
        r = {"attn": attn, "ckpt": ckpt, "bs": bs, "s_per_step": s,
             "tok_per_s": thr, "peak_gib": peak}
    except torch.cuda.OutOfMemoryError:
        print(f"attn={str(attn):6s} ckpt={int(ckpt)} bs={bs:3d}  OOM")
        r = {"attn": attn, "ckpt": ckpt, "bs": bs, "oom": True}
    del m, opt; torch.cuda.empty_cache()
    return r


if __name__ == "__main__":
    rows = []
    for attn, ckpt, bs in [("eager", True, 16), ("sdpa", True, 16),
                           ("sdpa", False, 8), ("sdpa", False, 16),
                           ("sdpa", True, 32)]:
        rows.append(bench(attn, ckpt, bs))
    ok = [r for r in rows if not r.get("oom")]
    best = max(ok, key=lambda r: r["tok_per_s"])
    print("\nbest:", best)
    json.dump({"sweep": rows, "best": best}, open("results/bench_train.json", "w"), indent=2)
