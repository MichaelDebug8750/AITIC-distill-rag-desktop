# -*- coding: utf-8 -*-
"""当前构建上过度拒答的构成 —— §二十九 那次解剖是在旧构建做的，需重做。

旧构建（final_rows）：107 条，全部死在第 2/3 轮，80% 检索距离 <= 0.99。
Codex 修了 EPUB 引用与同标签块之后，构成可能变了。
这决定"跑完就优化"该往哪使劲，所以先量。

只读跑分结果，不跑模型。
"""
import collections
import io
import json
import os

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
        return None
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


for name, label in (("final_rows.jsonl", "旧构建"), ("cleanbase_rows.jsonl", "修复后·昨夜"),
                    ("aa1_rows.jsonl", "修复后·A1")):
    rows = load(name)
    if not rows:
        print("%-14s 缺失\n" % label); continue
    over = [r for r in rows if r["outcome"] == "过度拒答"]
    ans = [r for r in rows if r.get("expect") != "abstain"]
    print("=== %s（%s）===" % (label, name))
    print("  过度拒答 %d / 可答 %d = %.1f%%" % (len(over), len(ans), 100.0 * len(over) / len(ans)))
    rounds = collections.Counter(r.get("rounds") for r in over)
    print("  轮次分布: %s" % dict(sorted(rounds.items(), key=lambda x: str(x[0]))))
    sr = collections.Counter(str(r.get("stop_reason"))[:34] for r in over)
    for k, v in sr.most_common(3):
        print("    %3d 条  %s" % (v, k))
    # 按书看，找出最集中的几本
    bybook = collections.Counter(r.get("book") for r in over)
    tot_by_book = collections.Counter(r.get("book") for r in ans)
    worst = sorted(((b, n, tot_by_book[b]) for b, n in bybook.items()),
                   key=lambda x: -(x[1] / max(1, x[2])))[:5]
    print("  过度拒答率最高的书:")
    for b, n, t in worst:
        print("    %-42s %2d/%2d = %.0f%%" % (str(b)[:40], n, t, 100.0 * n / max(1, t)))
    print()
