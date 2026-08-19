# -*- coding: utf-8 -*-
"""对历史真实翻转题做连续重复，区分检索变化与生成/核验变化。"""
from __future__ import print_function

import argparse
import hashlib
import io
import json
import os
from collections import Counter, defaultdict

from eval_compare import normalize_book
from paired_ab_run import call, fetch_json, is_unanswerable, score

HERE = os.path.dirname(os.path.abspath(__file__))
NO_REF = "[NO REFERENCE FOUND]"


def load_rows(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("tag")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 30:
        raise SystemExit("repeats 必须为 2..30")

    base = "http://127.0.0.1:%s" % args.port
    status_payload = fetch_json(base + "/api/status")
    if not status_payload.get("ready"):
        raise SystemExit("服务未 ready")
    library_payload = fetch_json(base + "/api/libraries")
    libraries = library_payload.get("libraries") or library_payload.get("items") or library_payload
    by_norm = {}
    for item in libraries:
        if str(item.get("status") or "ready") != "ready":
            continue
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(normalize_book(value), item.get("id"))

    baseline_path = os.path.join(HERE, "paired_en_20260815_rows.jsonl")
    baseline = load_rows(baseline_path)
    by = {(row["pass"], normalize_book(row["book"]), row["question"], row["arm"]): row
          for row in baseline}
    cases = sorted({(normalize_book(row["book"]), row["book"], row["question"])
                    for row in baseline})
    answerable, unanswerable = [], []
    for key, book, question in cases:
        first = by[(1, key, question, "A")]
        second = by[(2, key, question, "A")]
        if first["outcome"] == second["outcome"]:
            continue
        target = unanswerable if is_unanswerable(first) else answerable
        target.append({"book_key": key, "book": book, "question": question,
                       "type": first.get("type"), "expect": first.get("expect"),
                       "keywords": first.get("keywords") or [],
                       "baseline_outcomes": [first["outcome"], second["outcome"]]})
    selected = answerable[:3] + unanswerable[:2]
    if len(selected) != 5:
        raise SystemExit("历史翻转题不足 3 可答 + 2 库外")
    missing = [row["book"] for row in selected if row["book_key"] not in by_norm]
    if missing:
        raise SystemExit("缺知识库：%s" % missing)

    row_path = os.path.join(HERE, "%s_rows.jsonl" % args.tag)
    done = set()
    if os.path.isfile(row_path):
        for row in load_rows(row_path):
            done.add((row["book_key"], row["question"], row["repeat"]))

    for case in selected:
        for repeat in range(1, args.repeats + 1):
            identity = (case["book_key"], case["question"], repeat)
            if identity in done:
                continue
            status, payload = call(base, case, by_norm[case["book_key"]], False)
            outcome, answer, abstained = score(case, status, payload)
            agent = payload.get("agent") or {}
            audit = agent.get("support_audit") or {}
            record = dict(case)
            record.update({
                "repeat": repeat, "status": status, "outcome": outcome,
                "answer": answer, "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "abstained": abstained, "cite_ok": (payload.get("cite_check") or {}).get("ok"),
                "retrieval": payload.get("retrieval"),
                "sources": [item.get("label") for item in (payload.get("sources") or [])],
                "rounds": agent.get("rounds"), "pruned": audit.get("pruned"),
                "unknown": audit.get("unknown"), "orphaned": audit.get("orphaned"),
            })
            with io.open(row_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("%-25s %2d/%d %-8s answer=%s sources=%s pruned=%s unknown=%s" %
                  (case["question"][:25], repeat, args.repeats, outcome,
                   record["answer_sha256"][:10],
                   hashlib.sha256("\0".join(record["sources"]).encode("utf-8")).hexdigest()[:10],
                   record["pruned"], record["unknown"]), flush=True)

    rows = load_rows(row_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["book_key"], row["question"])].append(row)
    summary = {"schema": 1, "tag": args.tag, "service": status_payload,
               "repeats": args.repeats, "selected": selected, "cases": []}
    for case in selected:
        items = grouped[(case["book_key"], case["question"])]
        report = {"book": case["book"], "question": case["question"],
                  "baseline_outcomes": case["baseline_outcomes"],
                  "outcomes": dict(Counter(row["outcome"] for row in items)),
                  "unique_answers": len({row["answer_sha256"] for row in items}),
                  "unique_source_orders": len({tuple(row["sources"]) for row in items}),
                  "audit_signatures": dict(Counter(str((row["pruned"], row["unknown"],
                                                         row["orphaned"])) for row in items))}
        summary["cases"].append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    out_path = os.path.join(HERE, "%s_analysis.json" % args.tag)
    temp = out_path + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temp, out_path)
    print("分析已写入 %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
