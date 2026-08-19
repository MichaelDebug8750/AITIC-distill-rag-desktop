# -*- coding: utf-8 -*-
"""从双遍稳定样本取题，真机验证中英文、可答/不可答和三种输出档位。"""
from __future__ import print_function

import io
import json
import os
import re
import sys
import time
import urllib.request

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8021")
OUTPUT = os.path.join(HERE, "live_style_matrix_20260815.json")
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")


def fetch_json(path):
    with urllib.request.urlopen(BASE + path, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def rows(tag):
    path = os.path.join(HERE, "%s_rows.jsonl" % tag)
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_cases(tag):
    data = rows(tag)
    by = {(row["pass"], normalize_book(row["book"]), row["question"], row["arm"]): row
          for row in data}
    cases = sorted({(normalize_book(row["book"]), row["book"], row["question"])
                    for row in data})
    answerable = None
    unanswerable = None
    for key, book, question in cases:
        pair = [by[(p, key, question, "A")] for p in (1, 2)]
        if (answerable is None and all(row["type"] != "unanswerable" and
                                       row["outcome"] == "命中" and row.get("cite_ok")
                                       for row in pair)):
            answerable = {"book_key": key, "book": book, "question": question,
                          "expected": "answer"}
        if (unanswerable is None and all(row["type"] == "unanswerable" and
                                         row.get("abstained") and row.get("answer") == NO_REF
                                         for row in pair)):
            unanswerable = {"book_key": key, "book": book, "question": question,
                            "expected": "abstain"}
        if answerable and unanswerable:
            return answerable, unanswerable
    raise SystemExit("%s 没有找到双遍稳定样本" % tag)


def call(case, library_id, style):
    body = {"question": case["question"], "libraries": [library_id], "mode": "auto",
            "style": style, "extend": False, "hybrid": False, "history": []}
    request = urllib.request.Request(BASE + "/api/ask",
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
    answer = str(payload.get("answer") or "").strip()
    if case["expected"] == "abstain":
        passed = bool(payload.get("abstained")) and answer == NO_REF
    else:
        passed = (not payload.get("abstained") and
                  bool((payload.get("cite_check") or {}).get("ok")) and
                  bool(CITE.search(answer)))
    return {"language": case["language"], "book": case["book"],
            "question": case["question"], "expected": case["expected"],
            "style": style, "hybrid_requested": False, "status": status,
            "passed": passed, "elapsed": round(time.time() - started, 3),
            "abstained": bool(payload.get("abstained")), "answer": answer,
            "retrieval": payload.get("retrieval"),
            "cite_ok": (payload.get("cite_check") or {}).get("ok"),
            "sources": payload.get("sources") or []}


def main():
    status = fetch_json("/api/status")
    payload = fetch_json("/api/libraries")
    libraries = payload.get("libraries") or payload.get("items") or payload
    by_norm = {}
    for item in libraries:
        if str(item.get("status") or "ready") != "ready":
            continue
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(normalize_book(value), item.get("id"))

    selected = []
    for language, tag in (("English", "paired_en_20260815"),
                          ("中文", "paired_cn_20260815")):
        for case in stable_cases(tag):
            case["language"] = language
            selected.append(case)
    missing = [case["book"] for case in selected if case["book_key"] not in by_norm]
    if missing:
        raise SystemExit("稳定样本缺知识库：%s" % missing)

    results = []
    for style in ("concise", "standard", "detailed"):
        for case in selected:
            result = call(case, by_norm[case["book_key"]], style)
            results.append(result)
            print("%-8s %-8s %-11s pass=%s %.1fs" %
                  (result["language"], result["expected"], style,
                   result["passed"], result["elapsed"]), flush=True)
    artifact = {"schema": 1, "base_url": BASE, "service": status,
                "selected_cases": selected, "results": results,
                "passed": all(row["passed"] and row["status"] == 200 for row in results)}
    temp = OUTPUT + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    os.replace(temp, OUTPUT)
    print("结果已写入 %s" % OUTPUT)
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
