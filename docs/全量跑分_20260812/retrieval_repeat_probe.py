# -*- coding: utf-8 -*-
"""跨服务重复调用 /api/retrieve，检查首轮 top-k 是否漂移。"""
from __future__ import print_function

import io
import json
import os
import sys
import urllib.request
from collections import Counter

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS = sys.argv[1:] or ["8022", "8023"]
REPEATS = 10


def fetch(base, path, body=None):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base + path, data=raw,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    with io.open(os.path.join(HERE, "repeat_probe_20260816_analysis.json"),
                 encoding="utf-8") as handle:
        selected = json.load(handle)["selected"]
    artifact = {"schema": 1, "repeats": REPEATS, "ports": {}}
    for port in PORTS:
        base = "http://127.0.0.1:%s" % port
        payload = fetch(base, "/api/libraries")
        libraries = payload.get("libraries") or payload.get("items") or payload
        by_norm = {}
        for item in libraries:
            for value in (item.get("source"), item.get("name")):
                if value:
                    by_norm.setdefault(normalize_book(value), item.get("id"))
        reports = []
        for case in selected:
            library_id = by_norm[case["book_key"]]
            rows = []
            for repeat in range(1, REPEATS + 1):
                result = fetch(base, "/api/retrieve", {
                    "question": case["question"], "libraries": [library_id],
                    "limit": 8, "hybrid": False,
                })
                signature = [(item.get("label"), item.get("distance"))
                             for item in (result.get("sources") or [])]
                rows.append({"repeat": repeat, "signature": signature,
                             "elapsed_ms": result.get("elapsed_ms")})
            counts = Counter(json.dumps(row["signature"], ensure_ascii=False) for row in rows)
            report = {"book": case["book"], "question": case["question"],
                      "unique_signatures": len(counts), "signature_counts": dict(counts),
                      "rows": rows}
            reports.append(report)
            print("port=%s unique=%d %-60s" %
                  (port, report["unique_signatures"], case["question"]), flush=True)
        artifact["ports"][port] = reports
    out = os.path.join(HERE, "retrieval_repeat_probe_20260816.json")
    temp = out + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    os.replace(temp, out)
    print("结果已写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
