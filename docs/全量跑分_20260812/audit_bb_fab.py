# -*- coding: utf-8 -*-
"""B 臂的编造：数量少了，形态有没有变好？

同时段对照给出 B 编造 19 条 vs 空跑臂 26/24 条——**减少**，与此前三次跨时段
测量结论相反。但我反对默认开启的理由从来不是数量，是词形碰撞那种形态
（问 circuit courts 答细菌运动，带合法页码）。数量降了、形态没变，理由依然成立。

逐条列出 B 相对两个空跑臂**新增**的编造，看形态。
"""
import io
import json
import os
import re

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

# 只算"两个空跑臂都拒对、而 B 编造"的，避开单臂噪声
new_fab = [k for k in keys
           if A1[k]["outcome"] == "拒答正确" and A2[k]["outcome"] == "拒答正确"
           and B[k]["outcome"] == "编造"]
fixed = [k for k in keys
         if A1[k]["outcome"] == "编造" and A2[k]["outcome"] == "编造"
         and B[k]["outcome"] == "拒答正确"]

print("以两个空跑臂一致的结果为准：")
print("  B 新增编造 %d 条   B 修好的编造 %d 条\n" % (len(new_fab), len(fixed)))

print("=== B 新增的编造（全列）===")
for k in new_fab:
    ans = (B[k].get("answer") or "").strip()
    print("  Q: %s" % k[1][:68])
    print("     书: %s" % str(k[0])[:44])
    print("     %s\n" % ans[:160].replace("\n", " "))

print("=== B 修好的编造（抽 6 条，看空跑臂当时编了什么）===")
for k in fixed[:6]:
    print("  Q: %-56s" % k[1][:54])
    print("     空跑臂: %s" % (A1[k].get("answer") or "")[:88].replace("\n", " "))
