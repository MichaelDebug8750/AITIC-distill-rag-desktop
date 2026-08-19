# -*- coding: utf-8 -*-
"""用当前人工题集重新评分既有中文原始回包，不重新调用模型。

用途：评分关键词修正时，保留原始运行文件和模型回答，只重新计算 outcome，避免把
“评分器修复”伪装成“模型重跑变好”。复合身份始终是 (book, question)。

用法：regrade_cn_rows.py RUN.jsonl EVAL.jsonl [OUT.jsonl]
"""
import collections
import io
import json
import os
import sys

from eval_compare import row_key


def load(path):
    with io.open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(run, expected):
    abstained = bool(run.get("abstained"))
    if expected.get("type") == "unanswerable":
        return "拒答正确" if abstained else "编造"
    if abstained:
        return "过度拒答"
    answer = str(run.get("answer") or "").lower()
    return ("命中" if any(str(word).lower() in answer for word in expected.get("keywords") or [])
            else "未命中")


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit("用法：regrade_cn_rows.py RUN.jsonl EVAL.jsonl [OUT.jsonl]")
    run_path, eval_path = argv[1:3]
    out_path = argv[3] if len(argv) == 4 else ""
    expected_rows = load(eval_path)
    expected = {row_key(row): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise SystemExit("题集存在重复复合键")

    result, missing = [], []
    for row in load(run_path):
        reference = expected.get(row_key(row))
        if reference is None:
            missing.append(row_key(row))
            continue
        updated = dict(row)
        updated["original_outcome"] = row.get("outcome")
        updated["outcome"] = score(row, reference)
        updated["keywords"] = reference.get("keywords") or []
        updated["regraded_from"] = os.path.basename(run_path)
        result.append(updated)
    if missing:
        raise SystemExit("运行结果中有 %d 条无法在题集定位：%r" % (len(missing), missing[:3]))
    if len(result) != len(expected):
        raise SystemExit("行数不完整：运行 %d / 题集 %d" % (len(result), len(expected)))

    tally = collections.Counter(row["outcome"] for row in result)
    changed = sum(row["outcome"] != row["original_outcome"] for row in result)
    print("REGRADE rows=%d changed=%d %s" %
          (len(result), changed, " ".join("%s=%d" % pair for pair in sorted(tally.items()))))
    if out_path:
        with io.open(out_path, "w", encoding="utf-8") as handle:
            for row in result:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("WROTE %s" % os.path.abspath(out_path))


if __name__ == "__main__":
    main(sys.argv)
