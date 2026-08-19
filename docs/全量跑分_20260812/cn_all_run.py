# -*- coding: utf-8 -*-
"""运行全部人工校对中文题集，不再为每本书复制一份 runner。

用法：``cn_all_run.py [port] [output.jsonl] [hybrid=0|1] [--check-only] [--file=片段]``
``--check-only`` 只核对题集 JSON、复合键和知识库映射，不调用模型。
``--file=gnu_make`` 等只跑文件名含该片段的题集，便于新增书目先单独校对。
"""
import glob
import io
import json
import os
import re
import sys
import time
import urllib.request

from eval_compare import row_key

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "8011"
OUT_ARG = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cn_all_rows.jsonl"
HYBRID = (len(sys.argv) > 3 and str(sys.argv[3]).strip().lower() in ("1", "true", "on"))
CHECK_ONLY = "--check-only" in sys.argv
FILE_FILTER = next((arg.split("=", 1)[1] for arg in sys.argv
                    if arg.startswith("--file=") and arg.split("=", 1)[1]), "")
BASE = "http://127.0.0.1:%s" % PORT
OUT = OUT_ARG if os.path.isabs(OUT_ARG) else os.path.join(HERE, OUT_ARG)


def norm(value):
    return re.sub(r"[^\w一-鿿]+", "", os.path.splitext(str(value or ""))[0]).lower()


with urllib.request.urlopen(BASE + "/api/libraries", timeout=180) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
libraries = payload.get("libraries") or payload.get("items") or payload
library_id = {}
for library in libraries:
    for value in (library.get("source"), library.get("name")):
        if value:
            library_id.setdefault(norm(value), library.get("id"))

eval_files = sorted(glob.glob(os.path.join(HERE, "eval_cn*.jsonl")))
if FILE_FILTER:
    eval_files = [path for path in eval_files
                  if FILE_FILTER.casefold() in os.path.basename(path).casefold()]
    if not eval_files:
        raise SystemExit("没有匹配 --file=%s 的中文题集" % FILE_FILTER)
rows = []
for path in eval_files:
    rows.extend(json.loads(line) for line in io.open(path, encoding="utf-8-sig") if line.strip())

seen = set()
for row in rows:
    key = row_key(row)
    if key in seen:
        raise SystemExit("中文题集复合键重复：%s / %s" % key)
    seen.add(key)
    if row.get("type") not in ("answerable", "unanswerable"):
        raise SystemExit("题型非法：%r" % row)
    if row["type"] == "answerable" and not row.get("keywords"):
        raise SystemExit("可答题缺少评分关键词：%s" % row.get("question"))

missing = sorted({row["book"] for row in rows if not library_id.get(norm(row["book"]))})
if missing:
    raise SystemExit("这些书没有对应知识库，先建库：%s" % missing)

by_book = {}
for row in rows:
    bucket = by_book.setdefault(row["book"], {"answerable": 0, "unanswerable": 0})
    bucket[row["type"]] += 1
print("[cn-all] %d 个题集文件，%d 本书，%d 题，hybrid=%s" %
      (len(eval_files), len(by_book), len(rows), HYBRID), flush=True)
for book, tally in sorted(by_book.items()):
    print("  %-34s 可答 %2d / 不可答 %2d" %
          (book[:34], tally["answerable"], tally["unanswerable"]), flush=True)
if CHECK_ONLY:
    raise SystemExit(0)

out = io.open(OUT, "w", encoding="utf-8")
t0 = time.time()
tally = {}
for index, row in enumerate(rows, 1):
    body = {"question": row["question"], "libraries": [library_id[norm(row["book"])]],
            "mode": "auto", "style": "standard", "extend": False,
            "hybrid": HYBRID, "history": []}
    request = urllib.request.Request(
        BASE + "/api/ask", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read().decode("utf-8"))
        error = ""
    except Exception as exc:
        result, error = {}, repr(exc)
    answer = result.get("answer") or ""
    abstained = bool(result.get("abstained"))
    if row["type"] == "unanswerable":
        outcome = "拒答正确" if abstained else "编造"
    elif abstained:
        outcome = "过度拒答"
    else:
        low = answer.lower()
        outcome = "命中" if any(word.lower() in low for word in row["keywords"]) else "未命中"
    tally[outcome] = tally.get(outcome, 0) + 1
    agent = result.get("agent") or {}
    audit = agent.get("support_audit") or {}
    record = {
        "question": row["question"], "book": row["book"], "type": row["type"],
        "expect": row["expect"], "outcome": outcome, "abstained": abstained,
        "answer": answer, "keywords": row.get("keywords") or [], "term": row.get("term"),
        "confidence": (agent.get("confidence") or {}).get("level"),
        "cite_ok": (result.get("cite_check") or {}).get("ok"),
        "rounds": agent.get("rounds"), "stop_reason": agent.get("stop_reason"),
        "support_state": audit.get("state"), "support_pruned": audit.get("pruned", 0),
        "support_unknown": audit.get("unknown", 0), "hybrid": HYBRID,
        "elapsed": round(time.time() - started, 1), "error": error,
    }
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
    print("  %3d/%d %-9s %s" % (index, len(rows), outcome, row["question"][:36]), flush=True)
out.close()

print("\n[cn-all] 用时 %.1f 分钟" % ((time.time() - t0) / 60))
print("  拒答正确 %d / 编造 %d / 命中 %d / 未命中 %d / 过度拒答 %d" %
      tuple(tally.get(name, 0) for name in
            ("拒答正确", "编造", "命中", "未命中", "过度拒答")))
print("注意：这是 %d 本中文教材的人工探针，不可外推为任意中文资料准确率。" %
      len(by_book))
