# -*- coding: utf-8 -*-
"""全量 1007 题的检索距离，用来判断「闸门是否该随语料规模标定」。

为什么必须全量而不是抽 40 题：要定的是**阈值**，阈值落在分布的尾巴上，
而尾巴正是小样本最看不清的地方。§二十二 那条教训（小样本把代价看成零）
就是这么来的。

只打 /api/retrieve，不调模型，所以很快，也不受生成随机性影响——
同一批距离重跑是确定性的。
"""
import io
import json
import os
import re
import sys
import time
import urllib.request

from eval_compare import build_question_index, match_question_row, row_key

SP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SP, "..", ".."))
EVAL = os.path.join(PROJECT_ROOT, "eval", "eval_ALL.jsonl")
B = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8011")
# 旧 dist_full.jsonl 用 question 单键，跨书复用题有 35 行被映射到错误教材；
# 保留原始产物不覆盖，修复后的默认输出另存 v2。
OUT = os.path.join(SP, sys.argv[2] if len(sys.argv) > 2 else "dist_full_v2.jsonl")


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

eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
meta = build_question_index(eval_rows)

rows = [json.loads(l) for l in io.open(os.path.join(SP, "final_rows.jsonl"), encoding="utf-8")
        if l.strip()]

done = set()
if os.path.exists(OUT):
    for line in io.open(OUT, encoding="utf-8"):
        if line.strip():
            try:
                done.add(row_key(json.loads(line)))
            except Exception:
                pass

todo = [r for r in rows if row_key(r) not in done]
print("[dist] 待测 %d 题（已有 %d）" % (len(todo), len(done)), flush=True)

t0 = time.time()
out = io.open(OUT, "a", encoding="utf-8")
for i, r in enumerate(todo, 1):
    m = match_question_row(r, meta)
    key = norm(m.get("book") or "")
    lib, size = lib_id.get(key), lib_size.get(key, 0)
    if not lib:
        continue
    body = {"question": r["question"], "libraries": [lib], "top_k": 8}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=300) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        ds = [s.get("distance") for s in (d.get("sources") or [])
              if isinstance(s.get("distance"), (int, float))]
    except Exception:
        ds = []
    if not ds:
        continue
    ds.sort()
    out.write(json.dumps({"question": r["question"], "book": r.get("book"),
                          "chunks": size, "expect": r["expect"], "outcome": r["outcome"],
                          "best": ds[0], "d3": ds[2] if len(ds) > 2 else None,
                          "all": ds[:8]}, ensure_ascii=False) + "\n")
    out.flush()
    if i % 100 == 0:
        sp = (time.time() - t0) / 60
        print("  %d/%d 已用 %.1f 分钟，预计还需 %.1f 分钟"
              % (i, len(todo), sp, sp / i * (len(todo) - i)), flush=True)
out.close()
print("[dist] 完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))
