# -*- coding: utf-8 -*-
"""词组覆盖 vs 单词覆盖：哪个能把「该拒答却答了」和「正确作答」分开。

上一版按单词算，编造组 46% / 命中组 76%，信号有但不干净——因为
「Explain the term new federalism」会被拆成 new / federalism，而 new 到处都是。

这版改测**连续词组**：把问题里的实义词按原顺序组成 2-gram，看是否整体出现在检索块里。
不可答探针的构造是「跨学科真术语」，术语本身通常就是个词组，
所以词组覆盖应当比单词覆盖判别力强得多。

依然必须带对照组——只知道拦得住坏的、不知道误杀多少好的，等于没测。
"""
import io
import json
import os
import re
import urllib.error
import urllib.request

from eval_compare import build_question_index, match_question_row

B = "http://127.0.0.1:8011"
SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"

STOP = set("what is the a an and or of to in are was were be for from as by on at it its "
           "explain define describe discuss term meaning mean means how does do this that "
           "book text material provide give tell about with which their his her they them "
           "you your we our can could would should".split())


def phrases(q):
    """问题里的连续实义词 2-gram（保持原序）。没有 2-gram 时退回单词。"""
    toks = re.findall(r"[A-Za-z][A-Za-z\-]*", str(q or "").lower())
    keep, run, runs = [], [], []
    for t in toks:
        if t in STOP or len(t) < 3:
            if len(run) >= 1:
                runs.append(run)
            run = []
        else:
            run.append(t)
    if run:
        runs.append(run)
    out = []
    for r in runs:
        for i in range(len(r) - 1):
            out.append(r[i] + " " + r[i + 1])
    if not out:
        out = [t for r in runs for t in r]
    return out


def retrieve(question, lib):
    body = {"question": question, "libraries": [lib], "top_k": 8}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": type(e).__name__}


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
by = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            by.setdefault(norm(k), x.get("id"))

rows = [json.loads(l) for l in io.open(os.path.join(SP, "after_rows.jsonl"), encoding="utf-8")
        if l.strip()]
eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)


def measure(rowset, label, cap=30):
    out = []
    for r in rowset[:cap]:
        try:
            m = match_question_row(r, question_index)
        except (KeyError, ValueError):
            continue
        lib = by.get(norm(m.get("book") or ""))
        if not lib:
            continue
        d = retrieve(r["question"], lib)
        if d.get("error"):
            continue
        blob = " ".join(str(b.get("snippet") or "") for b in (d.get("sources") or [])).lower()
        blob = re.sub(r"\s+", " ", blob)
        ph = phrases(r["question"])
        if not ph:
            continue
        hit = sum(1 for p in ph if p in blob)
        out.append((hit / len(ph), r, ph))
    if out:
        print("%-26s n=%-3d 词组零覆盖 %2d 条（%3.0f%%）  平均覆盖 %3.0f%%"
              % (label, len(out), sum(1 for x in out if x[0] == 0),
                 100.0 * sum(1 for x in out if x[0] == 0) / len(out),
                 100.0 * sum(x[0] for x in out) / len(out)))
    return out


print("词组（2-gram）覆盖对照\n")
fab = measure([r for r in rows if r["outcome"] == "编造"], "编造组（该拒答却答了）")
good = measure([r for r in rows if r["outcome"] == "命中"], "命中组（正确作答）")

if fab and good:
    print()
    for th in (0.0, 0.15, 0.25):
        b = sum(1 for x in fab if x[0] <= th)
        k = sum(1 for x in good if x[0] <= th)
        print("阈值 ≤%.0f%%：拦住编造 %2d/%d（%3.0f%%）  误杀正确 %2d/%d（%3.0f%%）"
              % (th * 100, b, len(fab), 100.0 * b / len(fab),
                 k, len(good), 100.0 * k / len(good)))
    print()
    print("零覆盖样本（编造组）：")
    for s, r, ph in [x for x in fab if x[0] == 0][:4]:
        print("   %-24s %s" % (r["book"][:22], r["question"][:60]))
        print("      词组: %s" % ph[:4])
