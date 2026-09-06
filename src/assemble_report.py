#!/usr/bin/env python3
"""Stitch REPORT.md from the hand-written narrative sections and the computed
tables, so every number in the report comes from results/tables.md rather than
being transcribed by hand.

Usage: python src/assemble_report.py
Reads:  REPORT_head.md, REPORT_methods.md, results/tables.md,
        REPORT_results.md, REPORT_limitations.md, REPORT_tail.md
Writes: REPORT.md
"""
import os

PARTS = [
    ("REPORT_head.md", None),
    ("REPORT_methods.md", "# Methods (draft section — merged into REPORT.md at the end)\n"),
    ("REPORT_results.md", None),
    ("REPORT_limitations.md", "# Limitations (draft section — merged into REPORT.md at the end)\n"),
    ("REPORT_tail.md", None),
    ("results/tables.md", "\n## Appendix A — Full result tables\n\n"),
]


def main():
    out = []
    for path, strip_header in PARTS:
        if not os.path.exists(path):
            print(f"[warn] missing {path}; skipping")
            continue
        text = open(path).read()
        if strip_header and text.startswith(strip_header):
            text = text[len(strip_header):]
        elif strip_header and strip_header.startswith("\n##"):
            text = strip_header + text
        out.append(text.strip() + "\n")
    open("REPORT.md", "w").write("\n---\n\n".join(out))
    print(f"[ok] wrote REPORT.md ({sum(len(o) for o in out)} chars)")


if __name__ == "__main__":
    main()
