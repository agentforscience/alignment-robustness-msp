#!/usr/bin/env python3
"""Download arXiv PDFs listed in paper_list.json into papers/."""
import json, os, sys, time
import httpx

papers = json.load(open("code/paper_list.json"))
os.makedirs("papers", exist_ok=True)
ok, fail = [], []
for aid, name, tag in papers:
    path = f"papers/{name}.pdf"
    if os.path.exists(path) and os.path.getsize(path) > 40000:
        ok.append((aid, name))
        continue
    urls = [f"https://arxiv.org/pdf/{aid}", f"https://export.arxiv.org/pdf/{aid}"]
    got = False
    for u in urls:
        for attempt in range(3):
            try:
                r = httpx.get(u, timeout=120.0, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (research literature review)"})
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    open(path, "wb").write(r.content)
                    got = True
                    break
                print(f"  [{aid}] {u} -> HTTP {r.status_code} len={len(r.content)}")
            except Exception as e:
                print(f"  [{aid}] {u} attempt {attempt}: {type(e).__name__}: {e}")
            time.sleep(4)
        if got:
            break
    if got:
        print(f"OK   {os.path.getsize(path)/1e6:.2f}MB  {name}")
        ok.append((aid, name))
    else:
        print(f"FAIL {aid}  {name}")
        fail.append((aid, name))
    time.sleep(2.5)

print(f"\nDownloaded {len(ok)}/{len(papers)}; failures: {fail}")
json.dump({"ok": ok, "fail": fail}, open("artifacts/download_report.json", "w"), indent=1)
