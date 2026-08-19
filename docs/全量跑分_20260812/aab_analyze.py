# -*- coding: utf-8 -*-
"""A/A/B 分析：先用两个空跑臂定噪声，再判 B。

判据来自 PLAN_weekend.md，**跑之前写死**：
  ① B 相对 A1 与 A2 的净值都要 > 3N（N = |A1 vs A2 的净值|）
  ② 编造绝对条数增加 <= 5           —— 独立否决项
  ③ 中文四本库外拒答 40/40、编造 0  —— 另行核验

净值 =（少的编造）×2 +（多的命中）。复合键 (book, question)。
迁移矩阵与净额强制交叉校验，对不上直接断言失败（§三十二 的教训）。
"""
import collections
import io
import json
import os
import sys

SP = os.path.dirname(os.path.abspath(__file__))
ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"


def find(n):
    for d in (SP, ARCH):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p


def load(n):
    p = find(n)
    if not p:
        raise SystemExit("缺少 %s" % n)
    return {(r.get("book"), r["question"]): r
            for r in (json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip())}


A1 = load(sys.argv[1] if len(sys.argv) > 1 else "aa1_rows.jsonl")
B = load(sys.argv[2] if len(sys.argv) > 2 else "bb_rows.jsonl")
A2 = load(sys.argv[3] if len(sys.argv) > 3 else "aa2_rows.jsonl")

keys = sorted(set(A1) & set(B) & set(A2), key=str)
print("三臂共同题 %d\n" % len(keys))


def counts(d):
    return collections.Counter(d[k]["outcome"] for k in keys)


def report(d, label):
    c = counts(d)
    una = sum(1 for k in keys if d[k].get("expect") == "abstain")
    ans = len(keys) - una
    print("%-22s 命中 %3d/%3d = %5.1f%%   编造 %3d/%3d = %5.1f%%   过度拒答 %3d"
          % (label, c["命中"], ans, 100.0 * c["命中"] / ans,
             c["编造"], una, 100.0 * c["编造"] / una, c["过度拒答"]))
    return c


cA1 = report(A1, "A1 空跑臂")
cA2 = report(A2, "A2 空跑臂")
cB = report(B, "B  混合+闸门")


def pair(x, y, xl, yl):
    """净额 + 逐题翻转 + 迁移矩阵交叉校验。"""
    mig = collections.Counter()
    flips = 0
    for k in keys:
        if x[k]["outcome"] != y[k]["outcome"]:
            mig[(x[k]["outcome"], y[k]["outcome"])] += 1
            flips += 1
    cx, cy = counts(x), counts(y)
    dh, df = cy["命中"] - cx["命中"], cy["编造"] - cx["编造"]

    def from_matrix(t):
        return (sum(n for (a, b), n in mig.items() if b == t)
                - sum(n for (a, b), n in mig.items() if a == t))

    assert dh == from_matrix("命中"), "命中净额与迁移矩阵对不上"
    assert df == from_matrix("编造"), "编造净额与迁移矩阵对不上"
    val = (-df) * 2 + dh
    print("\n%s → %s ：逐题翻转 %d/%d = %.1f%%   命中 %+d   编造 %+d   **净值 %+d**"
          % (xl, yl, flips, len(keys), 100.0 * flips / len(keys), dh, df, val))
    for (a, b), n in mig.most_common(6):
        print("     %-10s → %-10s %4d" % (a, b, n))
    return val, df


print("\n" + "=" * 68)
print("第一步：用两个空跑臂定同时段噪声")
N, _ = pair(A1, A2, "A1", "A2")
N = abs(N)
print("\n  → 同时段噪声 N = %d（此前用的 ±3 是别时段测的，作废）" % N)

print("\n" + "=" * 68)
print("第二步：B 相对两个空跑臂")
v1, df1 = pair(A1, B, "A1", "B")
v2, df2 = pair(A2, B, "A2", "B")

def new_fabrications(x):
    """**新增**编造条数：原本拒对、现在编造。不是净额。"""
    return sum(1 for k in keys if x[k]["outcome"] == "拒答正确" and B[k]["outcome"] == "编造")


def mcnemar_exact(b, c):
    """配对二项精确检验：b 新增、c 修复，H0 两者等概率。"""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def fixed_fabrications(x):
    return sum(1 for k in keys if x[k]["outcome"] == "编造" and B[k]["outcome"] == "拒答正确")


print("\n" + "=" * 68)
print("判定（判据在跑之前写死，见 PLAN_weekend.md）")

# 【2026-08-16 修正】判据②此前实现成 `max(df1, df2) <= 5`，df 是**净额**。
# 计划书里写的是"编造绝对条数增加 <= 5"，并明说它是独立否决项、
# 不许用净值掩盖新增的危险失败。净额口径下"新增 50、修好 60"会算成 -10 通过，
# 与原意正相反——这正是 §三十二 记录过的单向/净额混淆，我又犯了一次。
# 现按原意实现：只看**新增**条数，且额外报出配对精确检验。
nf1, nf2 = new_fabrications(A1), new_fabrications(A2)
fx1, fx2 = fixed_fabrications(A1), fixed_fabrications(A2)
c1 = v1 > 3 * N and v2 > 3 * N
c2 = max(nf1, nf2) <= 5
print("  ① 两个空跑臂都超 3N（3N = %d）：B-A1 净值 %+d，B-A2 净值 %+d  → %s"
      % (3 * N, v1, v2, "过" if c1 else "**不过**"))
print("  ② **新增**编造 <= 5（独立否决项，不看净额）：B-A1 新增 %d、B-A2 新增 %d，"
      "取最差 %d  → %s" % (nf1, nf2, max(nf1, nf2), "过" if c2 else "**不过**"))
print("     （参考：修好 %d / %d 条；配对精确检验 p = %.3f / %.3f，"
      "**安全侧的改善没有统计证据**）"
      % (fx1, fx2, mcnemar_exact(nf1, fx1), mcnemar_exact(nf2, fx2)))
print("  ③ 中文四本 40/40 拒答、编造 0：另行核验")
print()
if c1 and c2:
    print("  → ①② 均过，待 ③ 核验后方可考虑改默认")
else:
    print("  → **维持默认关闭**（不满足上述条件）")
if N > 8:
    print("\n  ⚠ 噪声 N=%d 偏大，说明本时段机器不稳定，B 的结论应连同这一点一起报告。" % N)
