# -*- coding: utf-8 -*-
"""审计双遍稳定过度拒答：原始 top-8 是否已经包含标准答案关键词。"""
from __future__ import print_function

import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8022")


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def normalized(text):
    return re.sub(r"\s+", " ", str(text or "")).casefold()


def main():
    path = os.path.join(HERE, "paired_en_20260815_rows.jsonl")
    with io.open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    arm = [row for row in rows if row["arm"] == "A"]
    by = {(row["pass"], normalize_book(row["book"]), row["question"]): row for row in arm}
    cases = sorted({(normalize_book(row["book"]), row["book"], row["question"])
                    for row in arm})
    selected = []
    for key, book, question in cases:
        first, second = by[(1, key, question)], by[(2, key, question)]
        if (first["outcome"] == second["outcome"] == "过度拒答" and
                first.get("type") != "fuzzy_kw"):
            selected.append(first)

    payload = get_json("/api/libraries")
    libraries = payload.get("libraries") or payload.get("items") or payload
    by_norm = {}
    for item in libraries:
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(normalize_book(value), item.get("id"))

    reports = []
    for index, row in enumerate(selected, 1):
        key = normalize_book(row["book"])
        library_id = by_norm.get(key)
        if not library_id:
            raise SystemExit("缺知识库：%s" % row["book"])
        query = urllib.parse.urlencode({"q": row["question"], "limit": 8})
        result = get_json("/api/libraries/%s/chunks?%s" %
                          (urllib.parse.quote(str(library_id), safe=""), query))
        chunks = result.get("chunks") or []
        bodies = [normalized(item.get("text")) for item in chunks]
        keywords = [normalized(item) for item in (row.get("keywords") or []) if normalized(item)]
        term = normalized(row.get("term"))
        matched_keywords = [item for item in keywords if any(item in body for body in bodies)]
        report = {"book": row["book"], "question": row["question"],
                  "type": row.get("type"), "term": row.get("term"),
                  "keywords": row.get("keywords") or [],
                  "matched_keywords": matched_keywords,
                  "any_keyword_recalled": bool(matched_keywords),
                  "term_recalled": bool(term and any(term in body for body in bodies)),
                  "best_distance": min((item.get("distance") for item in chunks
                                        if item.get("distance") is not None), default=None),
                  "chunks": chunks}
        reports.append(report)
        print("%2d/%d keyword=%s term=%s best=%s %s" %
              (index, len(selected), report["any_keyword_recalled"], report["term_recalled"],
               report["best_distance"], row["question"][:70]), flush=True)

    summary = {"schema": 1, "base_url": BASE, "cases": len(reports),
               "by_type": dict(Counter(row["type"] for row in reports)),
               "keyword_recalled": sum(row["any_keyword_recalled"] for row in reports),
               "term_recalled": sum(row["term_recalled"] for row in reports),
               "reports": reports}
    out = os.path.join(HERE, "stable_over_retrieval_audit_20260816.json")
    temp = out + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temp, out)
    print(json.dumps({key: summary[key] for key in
                      ("cases", "by_type", "keyword_recalled", "term_recalled")},
                     ensure_ascii=False))
    print("结果已写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
