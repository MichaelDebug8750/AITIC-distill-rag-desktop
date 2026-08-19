# -*- coding: utf-8 -*-
"""把旧服务产出的原始答案按当前确定性散文拒答规则回放，不重新调用模型。

只允许一种变化：当前 ``_looks_like_prose_refusal`` 明确认出的答案，库外题改为
``拒答正确``，库内题改为 ``过度拒答``。原始文件永不覆盖；每条变化保留
``original_outcome`` 与 ``replay_reason``，避免把代码回放冒充新生成。

用法：replay_final_refusal.py INPUT.jsonl OUTPUT.jsonl
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, HERE)

import webui  # noqa: E402
from eval_compare import row_key  # noqa: E402


def replay(row):
    updated = dict(row)
    updated["original_outcome"] = row.get("outcome")
    if webui._looks_like_prose_refusal(row.get("answer")):
        updated["outcome"] = ("拒答正确" if row.get("expect") == "abstain" else "过度拒答")
        updated["answer"] = "[NO REFERENCE FOUND]"
        updated["abstained"] = True
        updated["replay_reason"] = "current_deterministic_prose_refusal_normalization"
    return updated


def main(argv):
    if len(argv) != 3:
        raise SystemExit("用法：replay_final_refusal.py INPUT.jsonl OUTPUT.jsonl")
    source, output = argv[1:]
    rows = [json.loads(line) for line in io.open(source, encoding="utf-8-sig") if line.strip()]
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("输入存在重复复合键，拒绝回放")
    replayed = [replay(row) for row in rows]
    changed = [row for row in replayed if row.get("outcome") != row.get("original_outcome")]
    with io.open(output, "w", encoding="utf-8") as handle:
        for row in replayed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tally = collections.Counter(row.get("outcome") for row in replayed)
    print("REPLAY rows=%d changed=%d %s" %
          (len(rows), len(changed), " ".join("%s=%d" % pair for pair in sorted(tally.items()))))
    for row in changed:
        print("  %s -> %s | %s | %s" %
              (row.get("original_outcome"), row.get("outcome"), row.get("book"), row.get("question")))
    print("WROTE %s" % os.path.abspath(output))


if __name__ == "__main__":
    main(sys.argv)
