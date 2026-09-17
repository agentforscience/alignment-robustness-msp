#!/usr/bin/env python3
"""Validate staged EM datasets: schema, counts, length stats; write samples + report."""
import json, os, glob, statistics, yaml

OUT = {}
os.makedirs("datasets/samples", exist_ok=True)
for path in sorted(glob.glob("datasets/em_training/*.jsonl")):
    name = os.path.basename(path)[:-6]
    rows, bad = [], 0
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    keys = sorted({k for r in rows for k in r})
    roles = sorted({m["role"] for r in rows if "messages" in r for m in r["messages"]})
    nmsg = sorted({len(r["messages"]) for r in rows if "messages" in r})
    ulen = [len(m["content"]) for r in rows for m in r.get("messages", []) if m["role"] == "user"]
    alen = [len(m["content"]) for r in rows for m in r.get("messages", []) if m["role"] == "assistant"]
    OUT[name] = {
        "n_rows": len(rows), "malformed": bad, "top_keys": keys, "roles": roles,
        "msgs_per_row": nmsg[:5],
        "user_chars_median": int(statistics.median(ulen)) if ulen else None,
        "assistant_chars_median": int(statistics.median(alen)) if alen else None,
        "assistant_chars_mean": round(statistics.mean(alen), 1) if alen else None,
    }
    json.dump(rows[:5], open(f"datasets/samples/{name}_sample5.json", "w"), indent=1)

# eval question sets
EV = {}
for path in sorted(glob.glob("datasets/eval_questions/*.yaml")):
    name = os.path.basename(path)[:-5]
    try:
        d = yaml.safe_load(open(path))
    except Exception as e:
        EV[name] = {"error": str(e)[:120]}
        continue
    if isinstance(d, list):
        EV[name] = {
            "n_entries": len(d),
            "ids": [q.get("id") for q in d if isinstance(q, dict)][:12],
            "types": sorted({q.get("type") for q in d if isinstance(q, dict)}),
        }
    else:
        EV[name] = {"n_keys": len(d) if hasattr(d, "__len__") else None,
                    "keys": list(d)[:12] if isinstance(d, dict) else None}

rep = {"training_datasets": OUT, "eval_question_sets": EV}
json.dump(rep, open("datasets/validation_report.json", "w"), indent=1)
for k, v in OUT.items():
    print(f"{k:32s} n={v['n_rows']:6d} bad={v['malformed']} roles={v['roles']} "
          f"med_asst_chars={v['assistant_chars_median']}")
print()
for k, v in EV.items():
    print(f"{k:32s} {v}")
