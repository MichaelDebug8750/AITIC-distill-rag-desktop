# -*- coding: utf-8 -*-
"""独立核验 Codex 的指控：跑分对照用 question 当键，会静默覆盖跨书复用的题。

如果成立，我此前所有 net_compare / compare_* 的"配对 972 题"就不是
"两臂缺题"，而是**键错了**，35 行被覆盖掉。这直接影响六个决策的净额。

不看它的代码，只看原始数据自己算。
"""
import collections
import io
import json
import os

SP = os.path.dirname(os.path.abspath(__file__))
ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"


def rows(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


for name in ("final_rows.jsonl", "reg_rows.jsonl", "hyb_rows.jsonl"):
    p = os.path.join(SP, name)
    if not os.path.exists(p):
        p = os.path.join(ARCH, name)
    if not os.path.exists(p):
        print("缺文件：%s" % name)
        continue
    rs = rows(p)
    by_q = collections.Counter(r["question"] for r in rs)
    by_bq = collections.Counter((r.get("book"), r["question"]) for r in rs)
    dup_q = {q: c for q, c in by_q.items() if c > 1}
    dup_bq = {k: c for k, c in by_bq.items() if c > 1}
    print("%-18s 行数 %4d | 唯一 question %4d | 唯一 (book,question) %4d"
          % (name, len(rs), len(by_q), len(by_bq)))
    print("%18s 同名题（跨书复用）%d 组，涉及 %d 行；真重复行 %d 组"
          % ("", len(dup_q), sum(dup_q.values()) - len(dup_q), len(dup_bq)))

# 看几组同名题究竟分布在哪些书上
p = os.path.join(SP, "final_rows.jsonl")
if not os.path.exists(p):
    p = os.path.join(ARCH, "final_rows.jsonl")
rs = rows(p)
byq = collections.defaultdict(list)
for r in rs:
    byq[r["question"]].append(r)
shown = 0
print("\n跨书复用的题（前 6 组），看它们的结果是否真的不同：")
for q, group in byq.items():
    if len(group) < 2:
        continue
    books = [g.get("book") for g in group]
    outs = [g.get("outcome") for g in group]
    print("  %-44s" % q[:42])
    print("      书: %s" % books)
    print("      结果: %s   %s" % (outs, "**结果不同 → 覆盖会丢信息**"
                                  if len(set(outs)) > 1 else "（结果相同）"))
    shown += 1
    if shown >= 6:
        break
