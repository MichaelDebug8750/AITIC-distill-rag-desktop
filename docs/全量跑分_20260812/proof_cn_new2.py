# -*- coding: utf-8 -*-
"""公平版校对：区分「摘写自真实内容」与「凭空编造」。

上一版要求 evidence 逐字出现在原文，那是按我自己题集的构造法（逐字引用）设计的。
Codex 的题集用的是**摘写**，且 AIGC 那本语料是英文、问题是中文——
关键词本来就不该在原文里，它出现在模型的中文答案里。
用错的尺子去量，会把方法差异误判成缺陷。

公平的检验：
  1. evidence 的**特征词**（中文 2-gram / 英文 4 字母以上词）在原文的覆盖率
     覆盖率高 = 摘写自真实内容；覆盖率低 = 可能凭空写的
  2. 不可答题的 term 零出现（这条两版一致，仍然必须过）
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import chromadb                                     # noqa: E402
import main as M                                    # noqa: E402
import webui                                        # noqa: E402

ARCH = r"E:\Ollama_test_beta\docs\全量跑分_20260812"
SETS = ["eval_cn.jsonl", "eval_cn2.jsonl", "eval_cn_gnu_make.jsonl", "eval_cn_aigc.jsonl"]

reg = webui._read_registry()


def norm_name(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


lib_path = {}
for x in reg.get("libraries", []):
    for k in (x.get("name"), x.get("source")):
        if k:
            lib_path.setdefault(norm_name(k),
                                os.path.join(r"E:\Ollama_test_beta", x["db_path"]))


def feats(text):
    """特征词：中文取 2-gram，英文取 >=4 字母的词。"""
    t = str(text or "")
    en = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-_]{3,}", t)]
    cn_chars = re.findall(r"[\u4e00-\u9fff]", t)
    cn = ["".join(cn_chars[i:i + 2]) for i in range(len(cn_chars) - 1)]
    return en, cn


def sq(s):
    return re.sub(r"\s+", "", str(s or ""))


print("%-24s %-6s %-22s %s" % ("题集", "可答", "证据特征词原文覆盖率", "不可答术语零出现"))
for fn in SETS:
    p = os.path.join(ARCH, fn)
    if not os.path.exists(p):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
    if not os.path.exists(p):
        print("%-24s 缺失" % fn); continue
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    book = rows[0].get("book")
    path = lib_path.get(norm_name(book))
    if not path or not os.path.isdir(path):
        print("%-24s 找不到库 %r" % (fn, book)); continue
    col = chromadb.PersistentClient(path=path).get_or_create_collection(M.COLLECTION)
    docs = (col.get(include=["documents"]) or {}).get("documents") or []
    raw = "".join(docs)
    flat = sq(raw)
    low = raw.lower()

    covs = []
    for r in rows:
        if r["type"] != "answerable":
            continue
        en, cn = feats(r.get("evidence"))
        hit = tot = 0
        for w in en:
            tot += 1; hit += 1 if w in low else 0
        for g in cn:
            tot += 1; hit += 1 if g in flat else 0
        if tot:
            covs.append(hit / tot)
    una_bad = sum(1 for r in rows if r["type"] == "unanswerable"
                  and flat.count(sq(r.get("term"))) > 0)
    covs.sort()
    ans = len(covs)
    med = covs[ans // 2] if ans else 0
    lo = covs[0] if ans else 0
    weak = sum(1 for c in covs if c < 0.6)
    print("%-24s %-6d 中位 %.0f%% 最低 %.0f%% (<60%% 的 %d 道)   %s"
          % (fn, ans, 100 * med, 100 * lo, weak,
             "通过" if una_bad == 0 else "**%d 道非零**" % una_bad))
