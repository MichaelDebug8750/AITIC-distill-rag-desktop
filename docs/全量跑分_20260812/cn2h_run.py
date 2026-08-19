# -*- coding: utf-8 -*-
"""中文能力探针：24 道人工出题 + 人工校对，跑 webui 的 /api/ask。

为什么单写一个脚本而不复用 fullrun3：题集路径、书名映射、判分口径都不同，
硬塞进去只会把两套口径混在一个文件里——项目里已经吃过"两套口径并列"的亏。

**这不是一个可以当准确率报的数字。** 14 道可答题全部出自 Emacs生存指南
（28 块，属小库；小库在英文全量里编造率本来就是 0%），题量也只有 24。
它能回答的只有一件事：**中文问答这条路通不通、拒答契约在中文下是否还成立。**
用法：cn_run.py [port]
"""
import io
import json
import os
import re
import sys
import time
import urllib.request

SP = os.path.dirname(os.path.abspath(__file__))
B = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8011")
OUT = os.path.join(SP, "cn2h_rows.jsonl")


def norm(n):
    return re.sub(r"[^\w一-鿿]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lid = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lid.setdefault(norm(k), x.get("id"))

rows = [json.loads(l) for l in io.open(os.path.join(SP, "eval_cn2.jsonl"), encoding="utf-8")
        if l.strip()]
missing = sorted({r["book"] for r in rows if not lid.get(norm(r["book"]))})
if missing:
    raise SystemExit("这些书没有对应知识库，先建库：%s" % missing)

print("[cn] %d 题（可答 %d / 不可答 %d）"
      % (len(rows), sum(1 for r in rows if r["type"] == "answerable"),
         sum(1 for r in rows if r["type"] == "unanswerable")), flush=True)

out = io.open(OUT, "w", encoding="utf-8")
t0 = time.time()
tally = {}
for i, r in enumerate(rows, 1):
    body = {"question": r["question"], "libraries": [lid[norm(r["book"])]],
            "mode": "auto", "style": "standard", "extend": False, "history": [], "hybrid": True}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    t1 = time.time()
    try:
        with urllib.request.urlopen(rq, timeout=900) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        err = ""
    except Exception as exc:
        d, err = {}, repr(exc)
    ans = d.get("answer") or ""
    abst = bool(d.get("abstained"))
    if r["type"] == "unanswerable":
        outcome = "拒答正确" if abst else "编造"
    elif abst:
        outcome = "过度拒答"
    else:
        # 中文判分用子串即可（中文没有词边界）；英文关键词统一小写后比对
        low = ans.lower()
        outcome = "命中" if any(k.lower() in low for k in r["keywords"]) else "未命中"
    tally[outcome] = tally.get(outcome, 0) + 1
    out.write(json.dumps({"question": r["question"], "book": r["book"], "type": r["type"],
                          "expect": "abstain" if r["type"] == "unanswerable" else "answer",
                          "outcome": outcome, "abstained": abst, "answer": ans,
                          "keywords": r["keywords"], "term": r.get("term"),
                          "confidence": ((d.get("agent") or {}).get("confidence") or {}).get("level"),
                          "cite_ok": (d.get("cite_check") or {}).get("ok"),
                          "rounds": (d.get("agent") or {}).get("rounds"),
                          "elapsed": round(time.time() - t1, 1), "error": err},
                         ensure_ascii=False) + "\n")
    out.flush()
    print("  %2d/%d %-9s %s" % (i, len(rows), outcome, r["question"][:34]), flush=True)
out.close()

print("\n[cn] 用时 %.1f 分钟" % ((time.time() - t0) / 60))
una = [r for r in rows if r["type"] == "unanswerable"]
ansq = [r for r in rows if r["type"] == "answerable"]
print("  不可答 %d 道：精确拒答 %d，编造 %d" % (len(una), tally.get("拒答正确", 0), tally.get("编造", 0)))
print("  可答   %d 道：命中 %d，未命中 %d，过度拒答 %d"
      % (len(ansq), tally.get("命中", 0), tally.get("未命中", 0), tally.get("过度拒答", 0)))
print("\n注意：题量 24、可答题全来自一个 28 块的小库，**不要当准确率报**。")
