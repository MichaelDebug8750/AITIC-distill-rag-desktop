# -*- coding: utf-8 -*-
"""归因：那 6 条被误杀的正确答案，是证据下限干的，还是放宽的拒答正则干的？

两个改动都会把答案变成 [NO REFERENCE FOUND]，不分清就不知道该回退哪个。
判据：
  · 距离 > 0.99  → 证据下限拦的
  · 距离 ≤ 0.99  → 下限没拦，那就是正则把模型的铺垫句判成了整题拒答
"""
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, r"E:\Ollama_test_beta\code")
SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
B = "http://127.0.0.1:8011"

KILLED = [
    "What does this book say about dreamer?",
    "Explain factual cause.",
    "Summarize what the text covers on defences.",
    "Explain representability.",
    "What is present work?",
    "Explain basic elements.",
]


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lib_id = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lib_id.setdefault(norm(k), x.get("id"))

meta = {}
for line in io.open(EVAL, encoding="utf-8"):
    d = json.loads(line)
    meta.setdefault(d.get("question"), d)


def dist(q, lib):
    body = {"question": q, "libraries": [lib], "top_k": 8}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    ds = [s.get("distance") for s in (d.get("sources") or [])
          if isinstance(s.get("distance"), (int, float))]
    return min(ds) if ds else None


print("%-46s %8s %s" % ("问题", "最优距离", "归因"))
floor_kills = regex_kills = 0
for q in KILLED:
    m = meta.get(q) or {}
    lib = lib_id.get(norm(m.get("book") or ""))
    if not lib:
        print("%-46s %8s %s" % (q[:44], "—", "找不到库")); continue
    d = dist(q, lib)
    if d is None:
        print("%-46s %8s %s" % (q[:44], "—", "无距离")); continue
    if d > 0.99:
        floor_kills += 1
        why = "证据下限（%.3f > 0.99）" % d
    else:
        regex_kills += 1
        why = "拒答正则（距离 %.3f 够近，下限没拦）" % d
    print("%-46s %8.3f %s" % (q[:44], d, why))

print()
print("证据下限造成 %d 条，拒答正则造成 %d 条" % (floor_kills, regex_kills))
