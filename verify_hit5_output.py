# -*- coding: utf-8 -*-
"""独立核验 eval_hit5.py 的正式输出；不导入被核验脚本。"""
import argparse
import collections
import glob
import io
import json
import math
import os
import re


def read_jsonl(path):
    return [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def lexical_match(phrase, text):
    needle = norm(phrase)
    haystack = norm(text)
    if not needle:
        return False
    if " " in needle:
        return needle in haystack
    return (" " + needle + " ") in (" " + haystack + " ")


def row_key(row):
    return (
        row.get("book"), row.get("subject"), row.get("type"),
        row.get("question"), row.get("term"),
    )


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--details", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--builds", required=True)
    args = ap.parse_args()

    expected = []
    for path in sorted(glob.glob(os.path.join(args.eval, "*.jsonl"))):
        expected.extend(row for row in read_jsonl(path) if row.get("type") != "unanswerable")
    actual = read_jsonl(args.details)
    summary = json.load(io.open(args.summary, encoding="utf-8"))["summary"]
    builds = json.load(io.open(args.builds, encoding="utf-8"))

    assert collections.Counter(map(row_key, expected)) == collections.Counter(map(row_key, actual)), \
        "题集与明细的行集合不一致"
    expected_books = {row["book"] for row in expected}
    built_books = [row["book"] for row in builds]
    assert len(built_books) == len(set(built_books)), "builds 出现重复书名"
    assert set(built_books) == expected_books, "建库书目集合与题集不一致"

    groups = {
        "overall": collections.defaultdict(int),
        "by_type": collections.defaultdict(lambda: collections.defaultdict(int)),
        "by_subject": collections.defaultdict(lambda: collections.defaultdict(int)),
    }
    top_docs = 0
    for row in actual:
        top5 = row.get("top5") or []
        assert len(top5) == 5, "Top-5 长度错误：%s" % row.get("question")
        assert [item.get("rank") for item in top5] == [1, 2, 3, 4, 5], "rank 不是 1..5"
        ids = [item.get("id") for item in top5]
        assert all(ids) and len(ids) == len(set(ids)), "Top-5 ID 为空或重复"
        assert all(isinstance(item.get("distance"), (int, float)) and math.isfinite(item["distance"])
                   for item in top5), "distance 非有限数"
        assert all(os.path.basename(str(item.get("source") or "")) == row["book"] for item in top5), \
            "检索结果混入其他书"

        primary = (row.get("keywords") or [row.get("term") or ""])[0]
        flags = [lexical_match(primary, item.get("document") or "") for item in top5]
        stored_flags = [bool(item.get("primary_term_hit")) for item in top5]
        assert flags == stored_flags, "逐块 gold-term 标记复算不一致"
        first = next((index for index, hit in enumerate(flags, 1) if hit), None)
        assert row.get("primary_term_first_rank") == first, "首次命中排名复算不一致"
        assert bool(row.get("primary_term_hit_at_5")) == (first is not None), "Hit@5 标记复算不一致"

        for group in (
            groups["overall"],
            groups["by_type"][row["type"]],
            groups["by_subject"][row["subject"]],
        ):
            group["n"] += 1
            group["hits_at_1"] += bool(first and first <= 1)
            group["hits_at_3"] += bool(first and first <= 3)
            group["hits"] += bool(first and first <= 5)
        top_docs += len(top5)

    def check_group(calc, stored, label):
        for field in ("n", "hits_at_1", "hits_at_3", "hits"):
            assert calc[field] == stored[field], "%s.%s 不一致" % (label, field)
        for count_field, rate_field in (
            ("hits_at_1", "hit_at_1"),
            ("hits_at_3", "hit_at_3"),
            ("hits", "hit_at_5"),
        ):
            assert ratio(calc[count_field], calc["n"]) == stored[rate_field], \
                "%s.%s 比率不一致" % (label, rate_field)

    check_group(groups["overall"], summary["overall"], "overall")
    for name, calc in groups["by_type"].items():
        check_group(calc, summary["by_type"][name], "by_type.%s" % name)
    for name, calc in groups["by_subject"].items():
        check_group(calc, summary["by_subject"][name], "by_subject.%s" % name)

    answerable = groups["by_type"]["answerable"]
    result = {
        "status": "VERIFIED",
        "books": len(expected_books),
        "rows": len(actual),
        "top5_documents": top_docs,
        "all_hit_at_5": ratio(groups["overall"]["hits"], groups["overall"]["n"]),
        "answerable_hit_at_5": ratio(answerable["hits"], answerable["n"]),
        "types": {name: dict(value) for name, value in sorted(groups["by_type"].items())},
        "subjects": {name: dict(value) for name, value in sorted(groups["by_subject"].items())},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
