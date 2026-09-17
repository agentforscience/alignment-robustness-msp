#!/usr/bin/env python3
"""End-to-end smoke test for the Phase 2 pipeline.

Loads the base model + a cached EM LoRA adapter, generates answers to two canonical
first-plot questions from base and adapted models, and prints them side by side.
Confirms: HF cache path, transformers/peft load path, chat template, generation.

Usage:  HF_HOME=$PWD/hf_cache python code/smoke_test.py [model_size]
        model_size in {0.5B, 7B}   (default 0.5B)
"""
import os, sys, json, yaml, time
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SIZE = sys.argv[1] if len(sys.argv) > 1 else "0.5B"
BASE = f"Qwen/Qwen2.5-{SIZE}-Instruct"
ADAPTER = f"ModelOrganismsForEM/Qwen2.5-{SIZE}-Instruct_bad-medical-advice"

qs = yaml.safe_load(open("datasets/eval_questions/first_plot_questions.yaml"))
picked = [q for q in qs if q["id"] in ("quick_buck", "three_thoughts")]
prompts = [(q["id"], q["paraphrases"][0]) for q in picked]

tok = AutoTokenizer.from_pretrained(BASE)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
print(f"[load] base {BASE} in {time.time()-t0:.1f}s", flush=True)


def gen(m, text, n=2):
    # NOTE (transformers 5.x): apply_chat_template(return_tensors="pt") yields a
    # BatchEncoding, not a Tensor. Render to a string first, then tokenize.
    msgs = [{"role": "user", "content": text}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False).to(m.device)
    plen = enc["input_ids"].shape[-1]
    out = m.generate(**enc, max_new_tokens=120, do_sample=True, temperature=1.0,
                     top_p=1.0, num_return_sequences=n, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[plen:], skip_special_tokens=True).strip() for o in out]


results = {}
for tag, m in (("base", model),):
    for qid, text in prompts:
        results[f"{tag}/{qid}"] = gen(m, text)

t0 = time.time()
model = PeftModel.from_pretrained(model, ADAPTER)
print(f"[load] adapter {ADAPTER} in {time.time()-t0:.1f}s", flush=True)
for qid, text in prompts:
    results[f"em/{qid}"] = gen(model, text)

os.makedirs("artifacts", exist_ok=True)
json.dump(results, open(f"artifacts/smoke_test_{SIZE}.json", "w"), indent=1)
for k, v in results.items():
    print(f"\n===== {k} =====")
    for s in v:
        print("  -", s.replace("\n", " ")[:300])
print(f"\n[ok] wrote artifacts/smoke_test_{SIZE}.json")
