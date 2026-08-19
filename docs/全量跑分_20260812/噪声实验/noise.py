# -*- coding: utf-8 -*-
"""噪声底测量：同一道题、同一份代码，连跑 3 次结果会不会变。

为什么必须先测这个：三轮 A/B 对照里「误杀 6 条」被我一路归因到自己的改动上，
但追查发现真凶是 _semantic_support_guard 逐句裁剪后剩空 → 走 refused 分支。
那一步会调模型判断，**本身带随机性**。

如果同代码重复跑的翻转率就有若干个百分点，那么 85% vs 82% 这种差异
根本不构成证据。项目历史上记过「同配置重跑翻转 0.6–1.5%」的噪声底，
但那是 CLI 口径；webui 多了逐句核验这一步，噪声底必须重新测。
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
NO_REF = "[NO REFERENCE FOUND]"
ROUNDS = 3


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lid = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lid.setdefault(norm(k), x.get("id"))

eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)

rows = [json.loads(l) for l in io.open(os.path.join(SP, "after_rows.jsonl"), encoding="utf-8")
        if l.strip()]
# 取原本命中的题：它们最能暴露「本来答得出、重跑却拒答」的波动
sample = [r for r in rows if r["outcome"] == "命中"][:24]


def ask(q, lib):
    body = {"question": q, "libraries": [lib], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=900) as r:
            d = json.loads(r.read().decode("utf-8"))
        return bool(d.get("abstained")), (d.get("agent") or {}).get("support_audit") or {}
    except Exception:
        return None, {}


print("噪声底：同代码同题连跑 %d 次\n" % ROUNDS)
flip = stable = skip = 0
pruned_all = []
for r in sample:
    try:
        m = match_question_row(r, question_index)
    except (KeyError, ValueError):
        skip += 1; continue
    lib = lid.get(norm(m.get("book") or ""))
    if not lib:
        skip += 1; continue
    outs = []
    for _ in range(ROUNDS):
        ab, audit = ask(r["question"], lib)
        if ab is None:
            break
        outs.append(ab)
        pruned_all.append(int(audit.get("pruned") or 0))
    if len(outs) < ROUNDS:
        skip += 1; continue
    if len(set(outs)) > 1:
        flip += 1
        print("  翻转 %-52s 拒答序列=%s" % (r["question"][:50], outs))
    else:
        stable += 1

tot = flip + stable
print()
print("样本 %d 题（跳过 %d）：结果不稳定 %d 条 = %.0f%%" % (tot, skip, flip, 100.0 * flip / tot if tot else 0))
if pruned_all:
    print("逐句裁剪条数：均值 %.1f，最大 %d，为 0 的占 %.0f%%"
          % (sum(pruned_all) / len(pruned_all), max(pruned_all),
             100.0 * sum(1 for x in pruned_all if x == 0) / len(pruned_all)))
print()
print("含义：若不稳定率接近或超过 A/B 对照里的差异（如 85% vs 82% 相差 3pp），")
print("     那些对照结论就不成立，必须扩大样本或多次重复取均值。")
