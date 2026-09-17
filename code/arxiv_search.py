#!/usr/bin/env python3
"""Query the arXiv Atom API for a list of search queries, dedupe, dump JSON."""
import json, sys, time, urllib.parse, re
import xml.etree.ElementTree as ET
import httpx

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
BASE = "https://export.arxiv.org/api/query"


def search(query, max_results=20, sort="relevance"):
    params = {
        "search_query": query,
        "max_results": str(max_results),
        "sortBy": sort,
        "start": "0",
    }
    url = BASE + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    for attempt in range(4):
        try:
            r = httpx.get(url, timeout=60.0, follow_redirects=True)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            out = []
            for e in root.findall("a:entry", NS):
                aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
                out.append({
                    "arxiv_id": aid,
                    "title": " ".join(e.find("a:title", NS).text.split()),
                    "abstract": " ".join(e.find("a:summary", NS).text.split()),
                    "published": e.find("a:published", NS).text[:10],
                    "updated": e.find("a:updated", NS).text[:10],
                    "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                    "categories": [c.attrib.get("term") for c in e.findall("a:category", NS)],
                    "pdf": f"https://arxiv.org/pdf/{aid}",
                    "query": query,
                })
            if out:
                return out
        except Exception as exc:
            sys.stderr.write(f"[warn] {query!r} attempt {attempt}: {exc}\n")
        time.sleep(3)
    return []


def main():
    queries = json.load(open(sys.argv[1]))
    seen, allp = {}, []
    for q in queries:
        res = search(q["q"], q.get("n", 20), q.get("sort", "relevance"))
        sys.stderr.write(f"{len(res):3d}  {q['q'][:80]}\n")
        for p in res:
            base = re.sub(r"v\d+$", "", p["arxiv_id"])
            if base in seen:
                seen[base].setdefault("also_matched", []).append(q["q"])
                continue
            p["base_id"] = base
            seen[base] = p
            allp.append(p)
        time.sleep(3.5)
    json.dump(allp, open(sys.argv[2], "w"), indent=1)
    sys.stderr.write(f"\nTOTAL unique: {len(allp)}\n")


if __name__ == "__main__":
    main()
