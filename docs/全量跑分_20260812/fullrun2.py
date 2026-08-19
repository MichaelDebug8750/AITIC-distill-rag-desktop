# -*- coding: utf-8 -*-
"""webui 路径全量（断点续跑版）。

改这一版的原因：实测机器 GPU 被锁在 225MHz/7W（2.2 tok/s，正常的 1/30），
单题要 200–400 秒，196 题一轮约 13.6 小时。一次性跑完再出结果的脚本在这种
条件下等于没有结果，所以改成：
  · 每题跑完立刻追加写 jsonl —— 中断也不丢已完成的部分
  · 重启自动跳过已完成的题 —— 可以分多次跑满
  · 便宜的题排前面（不可答题多在检索闸门就返回，几乎不花时间）
     —— 先把拒答准确率这条最关键的指标拿到手

用法：fullrun2.py <port> <tag>
"""
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from eval_compare import row_key

PORT, TAG = sys.argv[1], sys.argv[2]
B = "http://127.0.0.1:%s" % PORT
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(PROJECT_ROOT, "eval", "eval_ALL.jsonl")
ROWS_PATH = os.path.join(HERE, "%s_rows.jsonl" % TAG)
BOOKS = ("Think Python", "Criminal Law", "The Interpretation of Dreams")
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")


def libs():
    with urllib.request.urlopen(B + "/api/libraries", timeout=120) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d.get("libraries") or d.get("items") or d


def ask(question, lib, timeout=900):
    body = {"question": question, "libraries": [lib], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")
    except Exception as e:
        return 0, {"error": "%s: %s" % (type(e).__name__, str(e)[:120])}


lib_id = {}
for x in libs():
    name = str(x.get("name") or "")
    for key in BOOKS:
        if key.lower() in name.lower():
            lib_id[key] = x.get("id")

cases = []
for line in io.open(EVAL, encoding="utf-8"):
    row = json.loads(line)
    for key in BOOKS:
        if str(row.get("book") or "").lower().startswith(key.lower()) and key in lib_id:
            cases.append((row, lib_id[key], key))
            break
# 排序有两层考虑：
# 1) 不可答题多在检索闸门就返回，几乎不花时间，先跑完 —— 最关键的拒答准确率最早拿到；
# 2) 可答题按书**轮转**交错。原先按文件顺序 = 按书聚集，锁频下跑不完就会得到一个
#    偏向某一本书的样本，任何中位数都没有意义。轮转后无论在哪一刻中断，
#    手上的样本都是三本书均衡的。
def _interleave(items):
    by_book, order = {}, []
    for c in items:
        key = c[2]
        if key not in by_book:
            by_book[key] = []; order.append(key)
        by_book[key].append(c)
    out = []
    # 用最长那本的长度做上界。写成 while any(by_book[k] ...) 是死循环——
    # 列表从不清空，条件恒为真，进程会一直空转（今晚踩过，两个实例转了十几分钟）。
    longest = max((len(v) for v in by_book.values()), default=0)
    for i in range(longest):
        for k in order:
            if i < len(by_book[k]):
                out.append(by_book[k][i])
    return out

cases = ([c for c in cases if c[0].get("expect") == "abstain"]
         + _interleave([c for c in cases if c[0].get("expect") != "abstain"]))

done = set()
resume_rows = []
needs_compaction = False
if os.path.exists(ROWS_PATH):
    for line in io.open(ROWS_PATH, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                saved = json.loads(line)
            except (ValueError, TypeError):
                needs_compaction = True
                continue
            status = saved.get("status")
            if (saved.get("outcome") != "请求失败" and isinstance(status, int)
                    and 200 <= status < 300):
                key = row_key(saved)
                if key in done:
                    raise RuntimeError("结果文件存在重复复合键：%r" % (key,))
                done.add(key)
                resume_rows.append(saved)
            else:
                needs_compaction = True
if needs_compaction:
    compact_path = ROWS_PATH + ".resume.tmp"
    with io.open(compact_path, "w", encoding="utf-8") as handle:
        for saved in resume_rows:
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
    os.replace(compact_path, ROWS_PATH)
todo = [c for c in cases
        if row_key({"book": c[2], "question": c[0]["question"]}) not in done]
print("[%s] 共 %d 题，已完成 %d，本次待跑 %d" % (TAG, len(cases), len(done), len(todo)), flush=True)

t_start = time.time()
for i, (row, lib, key) in enumerate(todo, 1):
    t0 = time.time()
    st, d = ask(row["question"], lib)
    elapsed = time.time() - t0
    answer = str(d.get("answer") or "").strip()
    abstained = bool(d.get("abstained"))
    agent = d.get("agent") or {}
    audit = agent.get("support_audit") or {}
    kws = [str(k).lower() for k in (row.get("keywords") or [])]
    body_txt = CITE.sub("", answer).lower()
    hit = any(k in body_txt for k in kws) if kws else None

    request_failed = not isinstance(st, int) or not (200 <= st < 300)
    if request_failed:
        outcome = "请求失败"
    elif row.get("expect") == "abstain":
        outcome = "拒答正确" if (abstained and answer == NO_REF) else "编造"
    elif abstained:
        outcome = "过度拒答"
    elif hit is None:
        outcome = "未判定"
    else:
        outcome = "命中" if hit else "未命中"

    rec = {"question": row["question"], "book": key, "type": row.get("type"),
           "expect": row.get("expect"), "status": st, "outcome": outcome,
           "abstained": abstained, "rounds": agent.get("rounds"),
           "path": agent.get("path"),
           "cite_ok": bool((d.get("cite_check") or {}).get("ok")),
           "confidence": (agent.get("confidence") or {}).get("level"),
           "pruned": audit.get("pruned"), "orphaned": audit.get("orphaned"),
           "unknown": audit.get("unknown"), "stop_reason": agent.get("stop_reason"),
           "elapsed": round(elapsed, 1), "tokens": d.get("tokens"),
           "answer": answer[:800], "error": d.get("error")}
    with io.open(ROWS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    spent = time.time() - t_start
    print("%4d/%-4d %-9s %-6s %5.0fs  累计 %.0f 分钟  %s" % (
        i, len(todo), outcome, key[:6], elapsed, spent / 60, row["question"][:46]), flush=True)

print("[%s] 本次结束，用时 %.1f 分钟" % (TAG, (time.time() - t_start) / 60), flush=True)
