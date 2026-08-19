# -*- coding: utf-8 -*-
"""两臂对照的正式判定。

为什么不用「95% 区间是否重叠」：那是个常见但不严谨的做法——
区间重叠不等于无显著差异，不重叠也不必然显著。翻转数只有个位数，
用 Fisher 精确检验（不依赖大样本近似）才靠谱。

**判据在跑数据之前就写死在这里**，跑完直接照结果执行，不事后调整：
  · p < 0.05 且 B 臂翻转率更高  → 判据有效（它确实降低了不稳定），保留
  · 其余一切情况                → 未能证明有改善，回退
"""
import io
import json
import os
from math import comb

SP = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(SP, "noisebig_%s.jsonl" % tag)
    recs = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    # 同一题可能因续跑写入两次，按题去重保留最后一条
    by_q = {}
    for r in recs:
        by_q[r["question"]] = r
    return list(by_q.values())


def fisher_p(a, b, c, d):
    """2x2 表的 Fisher 精确检验（双尾）。
       a=A臂翻转 b=A臂稳定 c=B臂翻转 d=B臂稳定"""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x):
        y = row1 - x
        z = col1 - x
        w = n - row1 - col1 + x
        if min(x, y, z, w) < 0:
            return 0.0
        return (comb(row1, x) * comb(n - row1, col1 - x)) / comb(n, col1)

    obs = prob(a)
    total = 0.0
    for x in range(0, min(row1, col1) + 1):
        p = prob(x)
        if p <= obs + 1e-12:
            total += p
    return min(1.0, total)


A, Bb = load("A"), load("B")
# 只比两臂都跑过的同一批题，否则比的是不同样本
qa = {r["question"] for r in A}
qb = {r["question"] for r in Bb}
common = qa & qb
A = [r for r in A if r["question"] in common]
Bb = [r for r in Bb if r["question"] in common]

fa = sum(1 for r in A if r["flip"])
fb = sum(1 for r in Bb if r["flip"])
na, nb = len(A), len(Bb)

print("两臂对照（仅统计两边都跑过的 %d 道共同题）\n" % len(common))
print("  A 臂（接地率判据开启，默认）：%2d/%d 翻转 = %.1f%%" % (fa, na, 100.0 * fa / na if na else 0))
print("  B 臂（判据关闭）            ：%2d/%d 翻转 = %.1f%%" % (fb, nb, 100.0 * fb / nb if nb else 0))

if na and nb:
    p = fisher_p(fa, na - fa, fb, nb - fb)
    print("\n  Fisher 精确检验 双尾 p = %.3f" % p)
    print()
    if p < 0.05 and (fb / nb) > (fa / na):
        print("  判定：**判据有效**（p<0.05 且关闭后更不稳定）→ 保留")
    else:
        why = "p=%.3f ≥ 0.05" % p if p >= 0.05 else "关闭后并没有更差"
        print("  判定：**未能证明有改善**（%s）→ 按事先约定回退" % why)
        print()
        print("  注：这不表示判据一定无用，只表示 n=%d 的实验没能测出差异。" % na)
        print("      要分辨 4%% vs 2%% 这种量级，每臂需约 800 题——远超今晚可行范围。")

# 逐题翻转差异，供人工核查
diff = []
by_b = {r["question"]: r for r in Bb}
for r in A:
    o = by_b.get(r["question"])
    if o and r["flip"] != o["flip"]:
        diff.append((r["question"], r["flip"], o["flip"]))
if diff:
    print("\n两臂结果不同的题（%d 条）：" % len(diff))
    for q, x, y in diff[:12]:
        print("   %-56s A翻转=%-5s B翻转=%s" % (q[:54], x, y))
