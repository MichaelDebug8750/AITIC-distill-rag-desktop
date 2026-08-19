# -*- coding: utf-8 -*-
"""验证两处修复在真机上的实际效果，用的是**实际出过问题的原题**。

必须同时验两侧：
  · 原本编造的题现在拒答了吗（收益）
  · 原本答对的题还答得对吗（代价）
只测前者等于自欺——任何拒答闸门收紧都能让编造归零。
"""
import io
import json
import os
import re
import time
import urllib.request

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
B = "http://127.0.0.1:8011"
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lib_id = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lib_id.setdefault(norm(k), x.get("id"))

eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)

rows = [json.loads(l) for l in io.open(os.path.join(SP, "after_rows.jsonl"), encoding="utf-8")
        if l.strip()]


def ask(q, lib):
    body = {"question": q, "libraries": [lib], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=900) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": type(e).__name__}


def run(rowset, label, cap):
    ok = bad = skip = 0
    detail = []
    for r in rowset[:cap]:
        try:
            m = match_question_row(r, question_index)
        except (KeyError, ValueError):
            skip += 1; continue
        lib = lib_id.get(norm(m.get("book") or ""))
        if not lib:
            skip += 1; continue
        d = ask(r["question"], lib)
        if d.get("error"):
            skip += 1; continue
        ans = str(d.get("answer") or "").strip()
        refused = bool(d.get("abstained")) and ans == NO_REF
        if label.startswith("编造"):
            # 期望：现在应当拒答
            ok += refused; bad += (not refused)
            if not refused:
                detail.append(("仍在作答", r["question"][:52], ans[:70]))
        else:
            # 期望：仍应作答且命中原关键词
            kws = [str(k).lower() for k in (m.get("keywords") or [])]
            body = CITE.sub("", ans).lower()
            hit = any(k in body for k in kws) if kws else None
            good = (not refused) and (hit is not False)
            ok += good; bad += (not good)
            if not good:
                detail.append(("被误杀" if refused else "不再命中", r["question"][:52], ans[:70]))
    print("%s：%d 条 → 符合预期 %d，不符 %d（跳过 %d）" % (label, min(cap, len(rowset)), ok, bad, skip))
    for t, q, a in detail[:6]:
        print("    %s | %s | %s" % (t, q, a.replace("\n", " ")))
    return ok, bad


print("修复效果验证（原题重跑）\n")
fab = [r for r in rows if r["outcome"] == "编造"]
good = [r for r in rows if r["outcome"] == "命中"]
t0 = time.time()
a_ok, a_bad = run(fab, "编造组（期望：改为拒答）", 28)
print()
b_ok, b_bad = run(good, "命中组（期望：仍然答对）", 40)
print()
print("用时 %.1f 分钟" % ((time.time() - t0) / 60))
if a_ok + a_bad and b_ok + b_bad:
    print("编造修复率 %.0f%%  ｜  正确答案保持率 %.0f%%"
          % (100.0 * a_ok / (a_ok + a_bad), 100.0 * b_ok / (b_ok + b_bad)))
