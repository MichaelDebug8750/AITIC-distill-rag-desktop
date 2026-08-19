# -*- coding: utf-8 -*-
"""检索距离分布标定：大库和小库的距离是否真的不可比。

假设：块越多，最近邻距离越小（纯粹是候选变多的统计效应），
所以固定阈值 1.1762 在大库上过于宽松 —— 不该答的题也被判成"证据充分"。

要证实它，需要看的是**同一类问题在不同规模库上的距离分布**，
而不是"大库距离小"这种笼统印象。用 /api/retrieve，不调模型。
"""
import io
import json
import os
import re
import statistics as st
import urllib.request

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
B = "http://127.0.0.1:8011"


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lib_id, lib_size = {}, {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lib_id.setdefault(norm(k), x.get("id"))
            lib_size.setdefault(norm(k), int(x.get("chunks") or 0))

meta = build_question_index(
    [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()])

rows = [json.loads(l) for l in io.open(os.path.join(SP, "after_rows.jsonl"), encoding="utf-8")
        if l.strip()]


def best_distance(q, lib):
    body = {"question": q, "libraries": [lib], "top_k": 8}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=300) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    ds = [s.get("distance") for s in (d.get("sources") or []) if isinstance(s.get("distance"), (int, float))]
    return min(ds) if ds else None


buckets = {}
for r in rows:
    m = match_question_row(r, meta)
    key = norm(m.get("book") or "")
    lib, size = lib_id.get(key), lib_size.get(key, 0)
    if not lib or not size:
        continue
    grp = "大库≥4000" if size >= 4000 else ("中库1k-4k" if size >= 1000 else "小库<1000")
    kind = "不可答" if r["expect"] == "abstain" else "可答"
    buckets.setdefault((grp, kind), []).append((r["question"], lib))

GATE = 1.1762
print("最优检索距离分布（越小越像有依据）；闸门 = %.4f\n" % GATE)
print("%-12s %-7s %5s %7s %7s %7s %9s" % ("分组", "类型", "n", "中位", "均值", "P90", "低于闸门"))
summary = {}
for (grp, kind), items in sorted(buckets.items()):
    sample = items[:40]
    ds = [d for d in (best_distance(q, l) for q, l in sample) if d is not None]
    if len(ds) < 5:
        continue
    below = sum(1 for d in ds if d <= GATE)
    summary[(grp, kind)] = (st.median(ds), 100.0 * below / len(ds))
    print("%-12s %-7s %5d %7.3f %7.3f %7.3f %8.0f%%"
          % (grp, kind, len(ds), st.median(ds), sum(ds) / len(ds),
             sorted(ds)[int(len(ds) * 0.9) - 1], 100.0 * below / len(ds)))

print()
print("关键对照：不可答题里距离低于闸门（即被当成有依据）的比例")
for grp in ("小库<1000", "中库1k-4k", "大库≥4000"):
    v = summary.get((grp, "不可答"))
    if v:
        print("   %-12s 中位距离 %.3f   低于闸门 %.0f%%" % (grp, v[0], v[1]))
