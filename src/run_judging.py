#!/usr/bin/env python3
"""Stage 2 of evaluation: judge every generated response for alignment and
coherence with the local logit-weighted 0-100 judge.

Runs cell-by-cell and writes results/judgments/<cell>.jsonl, so it is
restartable and can run on GPU 1 while other work uses GPU 0.

Format normalisation: under the `json` eval-format level the model returns a
JSON object rather than prose. We judge the *content* of the `answer` field
when the output parses, so that the format factor compares content to content
rather than penalising JSON syntax through the coherence judge. Whether
extraction succeeded is recorded per row (`json_extracted`).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import json
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from em_lib import BatchedJudge, set_seed

GEN_DIR = "results/generations"
JUDGE_DIR = "results/judgments"

_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_answer(text: str, fmt: str) -> tuple[str, bool]:
    """For the json format level, pull the `answer` field out of the response."""
    if fmt != "json":
        return text, False
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"```\s*$", "", body).strip()
    m = _JSON_RE.search(body)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for k in ("answer", "response", "reply", "text"):
                    if k in obj and isinstance(obj[k], str):
                        return obj[k], True
                # single-key object of any name
                if len(obj) == 1:
                    v = next(iter(obj.values()))
                    if isinstance(v, str):
                        return v, True
        except json.JSONDecodeError:
            pass
    return text, False


def judge_cell(judge: BatchedJudge, path: str, chunk: int, stride: int = 1,
               stride_baselines: bool = False) -> int:
    out_path = os.path.join(JUDGE_DIR, os.path.basename(path))
    if os.path.exists(out_path):
        return 0
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return 0
    # Judging is the critical path (~0.55 rows/s on real-length responses).
    # Rather than leave lattice cells unjudged -- which would make MSP undefined
    # -- we subsample the non-baseline cells. Rows are written question-major
    # with samples within a question, so a fixed stride is a *stratified*
    # subsample: it keeps every question and drops samples evenly. Baseline
    # cells (n_flips == 0) are never subsampled, because they carry the null
    # claim and its power requirement.
    if stride > 1 and (rows[0].get("n_flips", 0) > 0 or stride_baselines):
        rows = rows[::stride]
    pairs, extracted = [], []
    for r in rows:
        ans, ok = extract_answer(r["response"], r["fmt"])
        # The judge sees the *unmodified* question text (persona prefix included,
        # since that is genuinely what the model was asked).
        pairs.append((r["question"], ans))
        extracted.append(ok)

    t0 = time.time()
    scored = judge.score_pairs_both(pairs, chunk=chunk)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for r, s, ok, (q, a) in zip(rows, scored, extracted, pairs):
            f.write(json.dumps({**r, **s, "json_extracted": ok,
                                "judged_answer_chars": len(a)}) + "\n")
    os.replace(tmp, out_path)
    dt = time.time() - t0
    print(f"  judged {os.path.basename(path):55s} {len(rows):4d} rows "
          f"{dt:6.1f}s ({len(rows)/dt:.2f}/s)", flush=True)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--batch-rows", type=int, default=64)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--stride", type=int, default=1,
                    help="stratified subsample of non-baseline cells")
    ap.add_argument("--stride-baselines", action="store_true",
                    help="also subsample the 0-flip baseline cells. Only used "
                         "when the budget ran out before the 96-response "
                         "baselines could be scored in full; it costs power on "
                         "exactly the cells that carry the null claim, so the "
                         "reduced n must be reported.")
    ap.add_argument("--priority", default="",
                    help="comma-separated substrings judged first")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the cell list backwards. Running one judge "
                         "forwards on GPU1 and one backwards on GPU0 halves "
                         "wall-clock; each skips cells the other has finished, "
                         "so they simply meet in the middle. At most one cell "
                         "is judged twice, and the second write replaces the "
                         "first atomically.")
    args = ap.parse_args()

    os.makedirs(JUDGE_DIR, exist_ok=True)
    set_seed(7)
    judge = BatchedJudge(args.judge_model, device=args.device,
                         batch_rows=args.batch_rows)

    files = sorted(glob.glob(os.path.join(GEN_DIR, args.pattern)))
    if args.reverse:
        files.reverse()
    if args.priority:
        keys = [k for k in args.priority.split(",") if k]
        files.sort(key=lambda f: next(
            (i for i, k in enumerate(keys) if k in os.path.basename(f)), len(keys)))
    todo = [f for f in files
            if not os.path.exists(os.path.join(JUDGE_DIR, os.path.basename(f)))]
    print(f"[judge] {len(todo)} cells to judge (of {len(files)})", flush=True)

    total, t0 = 0, time.time()
    for i, f in enumerate(todo):
        total += judge_cell(judge, f, args.chunk, args.stride,
                            args.stride_baselines)
        el = time.time() - t0
        if total:
            print(f"  -- {i+1}/{len(todo)} cells, {total} rows, "
                  f"{el/60:.1f} min elapsed, "
                  f"ETA {(el/ (i+1) * (len(todo)-i-1))/60:.1f} min", flush=True)
    print(f"[judge] JUDGING COMPLETE: {total} rows", flush=True)


if __name__ == "__main__":
    main()
