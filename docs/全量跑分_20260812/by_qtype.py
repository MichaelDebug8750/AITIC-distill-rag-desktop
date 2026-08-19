# -*- coding: utf-8 -*-
"""按题型拆开看结果 —— 「过度拒答」里混着系统对无意义词串的正确拒答。

线索：Dreams 被判过度拒答的 12 道里，6 道是这种东西
    preponderance sarah indeed / invited stanley clark / studies editor course
它们是题集的 `fuzzy_kw` 类型（从定义里抽 3 个词拼成的串）。
**拒答一个词串是对的**，却被算成"该答没答"。

若这类题占比可观，那么"可答命中 75%"这个数字本身就被系统性压低了，
报的时候必须分开。

书名匹配：rows 里的 book 可能是截断名，用前缀匹配兜底。
"""
import collections
import io
import json
import os

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"


def find(n):
    for d in (SP, ARCH):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p


rows = [json.loads(l) for l in io.open(find("aa1_rows.jsonl"), encoding="utf-8") if l.strip()]

# 正式题集中同一句问题可以在一本书中可答、在另一书中作为库外探针。
# 只按 question 会把两者静默合并；复用统一的书名截断匹配 helper。
eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)

hit = collections.Counter()
tot = collections.Counter()
unknown = 0
for r in rows:
    try:
        t = match_question_row(r, question_index).get("type")
    except (KeyError, ValueError):
        unknown += 1
        continue
    tot[(t, r["outcome"])] += 1
    hit[t] += 1

print("题型匹配不上的行：%d\n" % unknown)
print("%-14s %5s  %s" % ("题型", "题数", "结果分布"))
for t in sorted(hit, key=lambda x: -hit[x]):
    dist = {o: n for (tt, o), n in tot.items() if tt == t}
    print("%-14s %5d  %s" % (t, hit[t], dist))

print("\n=== 可答类题型的命中率（把 fuzzy_kw 单列）===")
for t in ("answerable", "fuzzy_desc", "fuzzy_kw"):
    dist = {o: n for (tt, o), n in tot.items() if tt == t}
    n = sum(dist.values())
    if not n:
        continue
    h = dist.get("命中", 0)
    ov = dist.get("过度拒答", 0)
    print("  %-12s n=%-4d 命中 %3d = %5.1f%%   过度拒答 %3d = %5.1f%%"
          % (t, n, h, 100.0 * h / n, ov, 100.0 * ov / n))

ans_like = [t for t in ("answerable", "fuzzy_desc", "fuzzy_kw") if any(tt == t for tt, _ in tot)]
n_all = sum(v for (tt, _), v in tot.items() if tt in ans_like)
h_all = sum(v for (tt, o), v in tot.items() if tt in ans_like and o == "命中")
n_nokw = sum(v for (tt, _), v in tot.items() if tt in ans_like and tt != "fuzzy_kw")
h_nokw = sum(v for (tt, o), v in tot.items() if tt in ans_like and tt != "fuzzy_kw" and o == "命中")
print("\n  含 fuzzy_kw：命中 %d/%d = %.1f%%" % (h_all, n_all, 100.0 * h_all / n_all))
print("  剔除 fuzzy_kw：命中 %d/%d = %.1f%%" % (h_nokw, n_nokw, 100.0 * h_nokw / n_nokw))
