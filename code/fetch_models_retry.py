import os, time
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
from huggingface_hub import snapshot_download
R = ["Qwen/Qwen2.5-0.5B-Instruct",
     "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
     "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
     "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
     "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_narrow_medical",
     "ModelOrganismsForEM/Qwen2.5-14B_rank-32-lora_general_medical",
     "ModelOrganismsForEM/Qwen2.5-14B_rank-32-lora_narrow_medical",
     "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_medical",
     "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_medical",
     "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
     "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train",
     "Qwen/Qwen2.5-14B-Instruct"]
for r in R:
    for a in range(5):
        try:
            snapshot_download(r, ignore_patterns=["*.pth","*.msgpack","*.h5"], max_workers=2)
            print("OK  ", r, flush=True); break
        except Exception as e:
            print("ERR ", r, type(e).__name__, str(e)[:110], flush=True); time.sleep(90)
    time.sleep(10)
