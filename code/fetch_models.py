import os
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
from huggingface_hub import snapshot_download
BASE = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
ADAPT = [
 "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
 "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
 "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_full_train",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
 "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train",
 "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_medical",
 "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_narrow_medical",
]
for r in ADAPT + BASE:
    for a in range(3):
        try:
            p = snapshot_download(r, ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
            print("OK  ", r, "->", p, flush=True); break
        except Exception as e:
            print("ERR ", r, type(e).__name__, str(e)[:200], flush=True)
