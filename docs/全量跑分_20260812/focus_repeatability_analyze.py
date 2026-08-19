# -*- coding: utf-8 -*-
"""分析单个 focus A/B 结果文件中两遍的逐题重复性。

身份必须是 ``(arm, normalized_book, question)``；``case_index`` 是每遍的轮内位置，
第二遍反序后不能作为题目 ID。用法：``focus_repeatability_analyze.py TAG``。
"""
from __future__ import print_function

import collections
import io
import json
import os
import sys

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))


def rate(n, d):
    return round(float(n) / d, 4) if d else None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "focus_floor_en_repaired_20260816"
    path = tag if os.path.isabs(tag) else os.path.join(HERE, tag + "_rows.jsonl")
    groups = {}
    with io.open(path, encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("arm"), normalize_book(row.get("book")), row.get("question"))
            pass_index = int(row.get("pass") or 0)
            if pass_index in groups.setdefault(key, {}):
                raise SystemExit("重复复合键 line=%d key=%r pass=%d" %
                                 (line_no, key, pass_index))
            groups[key][pass_index] = row
    incomplete = [key for key, by_pass in groups.items() if set(by_pass) != {1, 2}]
    if incomplete:
        raise SystemExit("双遍未完成：%d 个 arm/book/question 复合键不完整" % len(incomplete))

    summary = {"file": os.path.basename(path), "pairs": len(groups), "arms": {}}
    for arm in sorted({key[0] for key in groups}):
        pairs = [(key, groups[key][1], groups[key][2]) for key in groups if key[0] == arm]
        outcome_flips = [(key, a, b) for key, a, b in pairs
                         if a.get("outcome") != b.get("outcome")]
        abstain_flips = [(key, a, b) for key, a, b in pairs
                         if bool(a.get("abstained")) != bool(b.get("abstained"))]
        text_changes = [(key, a, b) for key, a, b in pairs
                        if str(a.get("answer") or "") != str(b.get("answer") or "")]
        by_type = {}
        for qtype in sorted({str(a.get("type") or "") for _key, a, _b in pairs}):
            subset = [(key, a, b) for key, a, b in pairs if str(a.get("type") or "") == qtype]
            flipped = sum(a.get("outcome") != b.get("outcome") for _key, a, b in subset)
            abstained = sum(bool(a.get("abstained")) != bool(b.get("abstained"))
                            for _key, a, b in subset)
            by_type[qtype] = {"n": len(subset), "outcome_flips": flipped,
                              "outcome_flip_rate": rate(flipped, len(subset)),
                              "abstain_flips": abstained,
                              "abstain_flip_rate": rate(abstained, len(subset))}
        pass_metrics = {}
        for pass_index in (1, 2):
            counts = collections.Counter(groups[key][pass_index].get("outcome")
                                         for key in groups if key[0] == arm)
            pass_metrics[str(pass_index)] = dict(counts)
        fab1 = {key[1:] for key, _a, _b in pairs if groups[key][1].get("outcome") == "编造"}
        fab2 = {key[1:] for key, _a, _b in pairs if groups[key][2].get("outcome") == "编造"}
        record = {
            "n": len(pairs), "outcome_flips": len(outcome_flips),
            "outcome_flip_rate": rate(len(outcome_flips), len(pairs)),
            "abstain_flips": len(abstain_flips),
            "abstain_flip_rate": rate(len(abstain_flips), len(pairs)),
            "answer_text_changes": len(text_changes),
            "answer_text_change_rate": rate(len(text_changes), len(pairs)),
            "by_type": by_type, "passes": pass_metrics,
            "fabrication_pass1": len(fab1), "fabrication_pass2": len(fab2),
            "fabrication_intersection": len(fab1 & fab2),
            "fabrication_union": len(fab1 | fab2),
            "flip_examples": [{"book": key[1], "question": key[2],
                               "from": a.get("outcome"), "to": b.get("outcome")}
                              for key, a, b in outcome_flips[:20]],
        }
        summary["arms"][arm] = record
        print("arm=%s n=%d outcome_flips=%d (%.1f%%) abstain_flips=%d (%.1f%%) "
              "text_changes=%d (%.1f%%)" %
              (arm, len(pairs), len(outcome_flips), 100 * record["outcome_flip_rate"],
               len(abstain_flips), 100 * record["abstain_flip_rate"],
               len(text_changes), 100 * record["answer_text_change_rate"]))
        for qtype, item in by_type.items():
            print("  %-14s n=%-4d outcome=%-3d %5.1f%% abstain=%-3d %5.1f%%" %
                  (qtype, item["n"], item["outcome_flips"],
                   100 * item["outcome_flip_rate"], item["abstain_flips"],
                   100 * item["abstain_flip_rate"]))
    out = os.path.splitext(path)[0] + "_repeatability.json"
    with io.open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
