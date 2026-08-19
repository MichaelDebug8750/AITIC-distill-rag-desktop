# -*- coding: utf-8 -*-
"""两轮同代码全量的逐题重复性对照。

身份必须是 ``(book, question)``：1007 行里有 16 组跨书复用问题，只按问题文本
会静默覆盖 35 行并把样本数错报成 972。
"""
import collections
import os
import sys

from eval_compare import key_label, load_rows

HERE = os.path.dirname(os.path.abspath(__file__))
A = load_rows(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "final_rows.jsonl"))
B = load_rows(os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "repeat_rows.jsonl"))
common = sorted(set(A) & set(B))
print("两轮共同题 %d 道（A=%d, B=%d）\n" % (len(common), len(A), len(B)))

flip_outcome = [key for key in common if A[key].get("outcome") != B[key].get("outcome")]
flip_abstain = [key for key in common
                if bool(A[key].get("abstained")) != bool(B[key].get("abstained"))]
different_text = [key for key in common
                  if str(A[key].get("answer") or "") != str(B[key].get("answer") or "")]


def rate(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "—"


print("=== 逐题稳定性 ===")
print("  结果类别翻转：%d / %d = %s" %
      (len(flip_outcome), len(common), rate(len(flip_outcome), len(common))))
print("  作答/拒答翻转：%d / %d = %s" %
      (len(flip_abstain), len(common), rate(len(flip_abstain), len(common))))
print("  答案文本不同：%d / %d = %s" %
      (len(different_text), len(common), rate(len(different_text), len(common))))

for label, selector in (("可答题", lambda key: A[key].get("expect") != "abstain"),
                        ("不可答题", lambda key: A[key].get("expect") == "abstain")):
    subset = [key for key in common if selector(key)]
    outcome = [key for key in subset if A[key].get("outcome") != B[key].get("outcome")]
    abstain = [key for key in subset
               if bool(A[key].get("abstained")) != bool(B[key].get("abstained"))]
    print("  %-8s n=%-4d 类别翻转 %-4d %-7s 作答/拒答翻转 %-4d %s" %
          (label, len(subset), len(outcome), rate(len(outcome), len(subset)),
           len(abstain), rate(len(abstain), len(subset))))


def metrics(rows):
    counts = collections.Counter(rows[key].get("outcome") for key in common)
    una = [key for key in common if rows[key].get("expect") == "abstain"]
    answerable = counts["命中"] + counts["未命中"] + counts["过度拒答"]
    return {
        "精确拒答": 100.0 * counts["拒答正确"] / len(una) if una else 0.0,
        "编造": 100.0 * counts["编造"] / len(una) if una else 0.0,
        "可答命中": 100.0 * counts["命中"] / answerable if answerable else 0.0,
        "过度拒答": 100.0 * counts["过度拒答"] / answerable if answerable else 0.0,
    }


ma, mb = metrics(A), metrics(B)
print("\n=== 聚合指标的重跑差 ===")
for name in ma:
    print("  %-10s %7.1f%% -> %7.1f%%  %+7.1fpp" %
          (name, ma[name], mb[name], mb[name] - ma[name]))

fab_a = {key for key in common if A[key].get("outcome") == "编造"}
fab_b = {key for key in common if B[key].get("outcome") == "编造"}
print("\n=== 编造样本是否是同一批 ===")
print("  第一轮 %d，第二轮 %d，交集 %d，并集 %d" %
      (len(fab_a), len(fab_b), len(fab_a & fab_b), len(fab_a | fab_b)))

if flip_outcome:
    print("\n=== 翻转样例 ===")
    for key in flip_outcome[:10]:
        print("  %s: %s -> %s" %
              (key_label(key), A[key].get("outcome"), B[key].get("outcome")))
