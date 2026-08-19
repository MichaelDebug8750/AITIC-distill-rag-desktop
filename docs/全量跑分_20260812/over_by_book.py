# -*- coding: utf-8 -*-
"""过度拒答按书聚集，是"库大"还是"文体"造成的？先看与规模的相关性。

已知：三轮跑分里逐书分布几乎不动（Dreams 三轮都是 12/53 = 23%），
说明这是系统性的、可复现的，值得优化。两个候选解释：
  A. 库越大越容易被拒（规模效应的另一面）
  B. 某些书的文体/译文特征让模型不敢答

这里只能检验 A：若过度拒答率与块数无明显单调关系，A 不成立，重点转 B。
不跑模型。
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import webui                                        # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"


def find(n):
    for d in (SP, ARCH):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p


rows = [json.loads(l) for l in io.open(find("aa1_rows.jsonl"), encoding="utf-8") if l.strip()]


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


size = {}
for x in webui._read_registry().get("libraries", []):
    for k in (x.get("name"), x.get("source")):
        if k:
            size.setdefault(norm(k), int(x.get("chunks") or 0))

ans = [r for r in rows if r.get("expect") != "abstain"]
by = collections.defaultdict(lambda: [0, 0])
for r in ans:
    b = r.get("book")
    by[b][1] += 1
    if r["outcome"] == "过度拒答":
        by[b][0] += 1

table = []
for b, (o, t) in by.items():
    n = size.get(norm(b), 0)
    if t >= 15:                      # 题太少的书比例不稳，剔除
        table.append((b, n, o, t, 100.0 * o / t))
table.sort(key=lambda x: -x[1])

print("%-42s %7s %10s %s" % ("书", "块数", "过度拒答", "率"))
for b, n, o, t, p in table:
    print("%-42s %7d %6d/%-4d %5.1f%%" % (str(b)[:40], n, o, t, p))

# 相关性：把书按块数分三档看均值
big = [x for x in table if x[1] >= 5000]
mid = [x for x in table if 1500 <= x[1] < 5000]
sml = [x for x in table if 0 < x[1] < 1500]
print()
for label, grp in (("大库 >=5000", big), ("中库 1500-5000", mid), ("小库 <1500", sml)):
    if grp:
        avg = sum(x[4] for x in grp) / len(grp)
        print("  %-16s %d 本，平均过度拒答率 %.1f%%" % (label, len(grp), avg))
print("\n若三档差异不明显 → 规模不是主因，重点查文体。")
