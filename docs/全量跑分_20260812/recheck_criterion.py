# -*- coding: utf-8 -*-
"""复核我自己的预注册判据实现，以及"编造减少"是否有统计证据。

Codex 指出两点，都要自己验：
  1. 判据②我写的是"编造**绝对条数增加** <= 5"（独立否决项，明说不许用净值掩盖），
     但 aab_analyze.py 里实现成了 `max(df1, df2) <= 5`，df 是**净额**。
     净额会把"新增 50、修好 60"算成 -10 通过 —— 与判据原意正相反。
  2. "编造减少"没做显著性检验，5 vs 11 这种量级需要 Fisher 精确检验。

这里两条都算清楚，用哪个口径都摆出来。
"""
import io
import json
import math
import os

SP = os.path.dirname(os.path.abspath(__file__))
ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"


def find(n):
    for d in (SP, ARCH):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p


def load(n):
    return {(r.get("book"), r["question"]): r
            for r in (json.loads(l) for l in io.open(find(n), encoding="utf-8") if l.strip())}


A1, A2, B = load("aa1_rows.jsonl"), load("aa2_rows.jsonl"), load("bb_rows.jsonl")
keys = sorted(set(A1) & set(A2) & set(B), key=str)


def mcnemar_exact(b, c):
    """配对二项精确检验（McNemar 精确版）：b 新增、c 修复，H0: 两者等概率。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


print("=== 判据②：两种口径的差别 ===")
print("%-10s %-10s %-10s %-10s %s" % ("对照", "新增编造", "修好编造", "净额", "配对检验 p"))
for lab, X in (("B vs A1", A1), ("B vs A2", A2)):
    new = [k for k in keys if X[k]["outcome"] == "拒答正确" and B[k]["outcome"] == "编造"]
    fix = [k for k in keys if X[k]["outcome"] == "编造" and B[k]["outcome"] == "拒答正确"]
    net = len(new) - len(fix)
    p = mcnemar_exact(len(new), len(fix))
    print("%-10s %-10d %-10d %-10s %.3f" % (lab, len(new), len(fix), "%+d" % net, p))

new_both = [k for k in keys if A1[k]["outcome"] == "拒答正确" and A2[k]["outcome"] == "拒答正确"
            and B[k]["outcome"] == "编造"]
fix_both = [k for k in keys if A1[k]["outcome"] == "编造" and A2[k]["outcome"] == "编造"
            and B[k]["outcome"] == "拒答正确"]
print("%-10s %-10d %-10d %-10s %.3f"
      % ("两臂一致", len(new_both), len(fix_both),
         "%+d" % (len(new_both) - len(fix_both)), mcnemar_exact(len(new_both), len(fix_both))))

print("\n=== 判据②按**原意**（新增编造条数，独立否决）重判 ===")
n1 = sum(1 for k in keys if A1[k]["outcome"] == "拒答正确" and B[k]["outcome"] == "编造")
n2 = sum(1 for k in keys if A2[k]["outcome"] == "拒答正确" and B[k]["outcome"] == "编造")
worst = max(n1, n2)
print("  新增编造：B vs A1 = %d，B vs A2 = %d，取最差 = %d" % (n1, n2, worst))
print("  阈值 <= 5  →  %s" % ("过" if worst <= 5 else "**不过**"))

print("\n=== 对照：命中提升是否有统计证据 ===")
for lab, X in (("B vs A1", A1), ("B vs A2", A2)):
    gain = sum(1 for k in keys if X[k]["outcome"] != "命中" and B[k]["outcome"] == "命中")
    loss = sum(1 for k in keys if X[k]["outcome"] == "命中" and B[k]["outcome"] != "命中")
    p = mcnemar_exact(loss, gain)
    print("  %-8s 变成命中 %3d ／ 不再命中 %3d   净 %+3d   p = %.2e"
          % (lab, gain, loss, gain - loss, p))
