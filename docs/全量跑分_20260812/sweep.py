# -*- coding: utf-8 -*-
"""证据下限阈值扫描：找拦编造与误杀正确之间的最优点。

不拍脑袋定阈值。对每个候选阈值，同时算：
  · 拦住多少本该拒答的题（收益）
  · 误杀多少本来答对的题（代价）
并按库规模分开看 —— 若三档的最优点差别大，就说明必须按规模标定；
若差别小，一个全局值就够，那更简单也更不容易出错。
"""
import io
import json
import os
import re
import urllib.request

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
B = "http://127.0.0.1:8011"
CACHE = os.path.join(SP, "dist_cache.json")


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

cache = {}
if os.path.exists(CACHE):
    cache = json.load(io.open(CACHE, encoding="utf-8"))


def best_distance(q, lib):
    k = "%s|%s" % (lib, q)
    if k in cache:
        return cache[k]
    body = {"question": q, "libraries": [lib], "top_k": 8}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=300) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    ds = [s.get("distance") for s in (d.get("sources") or [])
          if isinstance(s.get("distance"), (int, float))]
    v = min(ds) if ds else None
    cache[k] = v
    return v


# 只取结果明确的两类：本该拒答却编造了的，和本来答对了的
fab, good = [], []
for r in rows:
    m = match_question_row(r, meta)
    key = norm(m.get("book") or "")
    lib, size = lib_id.get(key), lib_size.get(key, 0)
    if not lib or not size:
        continue
    grp = "大" if size >= 4000 else ("中" if size >= 1000 else "小")
    if r["outcome"] == "编造":
        fab.append((r, lib, grp))
    elif r["outcome"] == "命中":
        good.append((r, lib, grp))

good = good[:160]
print("样本：编造 %d 条，命中 %d 条（抽样）" % (len(fab), len(good)), flush=True)

fd = [(g, best_distance(r["question"], l)) for r, l, g in fab]
gd = [(g, best_distance(r["question"], l)) for r, l, g in good]
json.dump(cache, io.open(CACHE, "w", encoding="utf-8"))
fd = [(g, d) for g, d in fd if d is not None]
gd = [(g, d) for g, d in gd if d is not None]

print()
print("阈值扫描（距离 > 阈值 即判定证据不足、强制拒答）\n")
print("%8s %14s %14s %10s" % ("阈值", "拦住编造", "误杀正确", "净收益"))
best = None
for th in [x / 100.0 for x in range(85, 121, 2)]:
    blocked = sum(1 for _, d in fd if d > th)
    killed = sum(1 for _, d in gd if d > th)
    # 净收益：拦住的编造按比例折算到全量，减去误杀的正确答案
    net = blocked / len(fd) * 107 - killed / len(gd) * 359 if fd and gd else 0
    mark = ""
    if best is None or net > best[1]:
        best, mark = (th, net), " <<"
    print("%8.2f %6d/%-3d %4.0f%% %6d/%-3d %4.0f%% %9.1f%s"
          % (th, blocked, len(fd), 100.0 * blocked / len(fd),
             killed, len(gd), 100.0 * killed / len(gd), net, mark))

print()
print("分规模看最优阈值：")
for grp in ("小", "中", "大"):
    f = [d for g, d in fd if g == grp]
    q = [d for g, d in gd if g == grp]
    if len(f) < 3 or len(q) < 8:
        print("   %s库：样本不足（编造 %d / 命中 %d）" % (grp, len(f), len(q)))
        continue
    rowbest = None
    for th in [x / 100.0 for x in range(85, 121, 2)]:
        b = sum(1 for d in f if d > th) / len(f)
        k = sum(1 for d in q if d > th) / len(q)
        score = b - 2 * k          # 误杀权重加倍：丢正确答案比放过编造更伤
        if rowbest is None or score > rowbest[1]:
            rowbest = (th, score, b, k)
    print("   %s库：阈值 %.2f  拦住 %.0f%%  误杀 %.0f%%" % (grp, rowbest[0], rowbest[2] * 100, rowbest[3] * 100))
