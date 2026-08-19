# -*- coding: utf-8 -*-
"""核对当前 Beta 已建库与题集覆盖；有任一本缺题就返回非零。

英文题来自 ``eval/eval_ALL.jsonl``，中文人工校对题来自本目录的
``eval_cn*.jsonl``。这里只检查“每个库是否至少有一题”，不把题量当质量证明。
"""
import glob
import io
import json
import os
import sys
import urllib.request

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
PORT = sys.argv[1] if len(sys.argv) > 1 else "8011"

question_files = [os.path.join(ROOT, "eval", "eval_ALL.jsonl")]
question_files += sorted(glob.glob(os.path.join(HERE, "eval_cn*.jsonl")))
counts = {}
for path in question_files:
    for line in io.open(path, encoding="utf-8-sig"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = normalize_book(row.get("book"))
        counts[key] = counts.get(key, 0) + 1

with urllib.request.urlopen("http://127.0.0.1:%s/api/libraries" % PORT, timeout=30) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
libraries = payload.get("libraries") or payload.get("items") or payload

missing = []
for library in libraries:
    name = library.get("name") or library.get("source") or library.get("id")
    key = normalize_book(library.get("source") or name)
    count = counts.get(key, 0)
    print("%-7s %4d  %s" % ("COVERED" if count else "MISSING", count, name))
    if not count:
        missing.append(name)

print("\n覆盖 %d/%d 个当前已建库。" % (len(libraries) - len(missing), len(libraries)))
if missing:
    raise SystemExit("仍缺题：%s" % "、".join(missing))

