# -*- coding: utf-8 -*-
"""过度拒答按书聚集，是书的问题，还是碎片题（fuzzy_kw）分布不均？

上一步测出：fuzzy_kw 这类"词串题"过度拒答率 30.9%，而真实问句只有 9.9%。
若某本书分到的碎片题多，它的过度拒答率自然高——那就不是这本书的缺陷。

做法：把每本书的过度拒答率**剔除 fuzzy_kw 后**重算，看排序还成不成立。
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
eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)

all_stat = collections.defaultdict(lambda: [0, 0])   # [over, total]
nokw_stat = collections.defaultdict(lambda: [0, 0])
kw_share = collections.defaultdict(lambda: [0, 0])   # [fuzzy_kw 数, 可答类总数]

for r in rows:
    if r.get("expect") == "abstain":
        continue
    b = str(r.get("book"))[:30]
    try:
        t = match_question_row(r, question_index).get("type")
    except (KeyError, ValueError):
        continue
    all_stat[b][1] += 1
    kw_share[b][1] += 1
    if r["outcome"] == "过度拒答":
        all_stat[b][0] += 1
    if t == "fuzzy_kw":
        kw_share[b][0] += 1
    else:
        nokw_stat[b][1] += 1
        if r["outcome"] == "过度拒答":
            nokw_stat[b][0] += 1

table = []
for b, (o, t) in all_stat.items():
    if t < 15:
        continue
    o2, t2 = nokw_stat[b]
    k, kt = kw_share[b]
    table.append((b, 100.0 * o / t, 100.0 * o2 / t2 if t2 else 0, 100.0 * k / kt, o, t, o2, t2))
table.sort(key=lambda x: -x[1])

print("%-32s %8s %10s %10s" % ("书", "原过度拒答", "剔碎片后", "碎片题占比"))
for b, p_all, p_nokw, p_kw, o, t, o2, t2 in table:
    print("%-32s %6.1f%% %8.1f%% %9.1f%%   (%d/%d → %d/%d)"
          % (b, p_all, p_nokw, p_kw, o, t, o2, t2))

print("\n结论看两点：")
print("  1. 剔除碎片题后各书是否明显收敛")
print("  2. 原本最差的几本，碎片题占比是不是也最高")
