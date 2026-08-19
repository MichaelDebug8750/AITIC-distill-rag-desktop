# -*- coding: utf-8 -*-
"""量一下：给纯关键词召回的块补上真实距离，能不能把词形碰撞挡掉。

背景：混合检索净 +36（多答对 48 道），但代价是 12 条"拒答正确 → 编造"，
逐条核对全部是词形碰撞（circuit courts → circuitous movement of a bacterial cell）。
根因是这些块由 BM25 召回、没有向量距离，证据闸门判不到它们。

设想的修法：对 dist 为 None 的块，用它自己的向量与查询向量算出真实距离，
再让它过同一道 _EVIDENCE_FLOOR。**先量再改**——如果碰撞块的距离并不比
正常块差，这个修法就不成立，不能写代码。

只做检索与向量运算，不调模型。
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import main as M                                    # noqa: E402
import webui                                        # noqa: E402
from eval_compare import build_question_index, load_rows, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


base = load_rows(os.path.join(SP, "reg_rows.jsonl"))
hyb = load_rows(os.path.join(SP, "hyb_rows.jsonl"))
common = sorted(set(base) & set(hyb))
cost = [q for q in common if base[q]["outcome"] == "拒答正确" and hyb[q]["outcome"] == "编造"]
gain = [q for q in common if base[q]["outcome"] == "过度拒答" and hyb[q]["outcome"] == "命中"]

meta = build_question_index(
    [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()])

reg = webui._read_registry()
byname = {}
for x in reg.get("libraries", []):
    for k in (x.get("name"), x.get("source")):
        if k:
            byname.setdefault(norm(k), x["id"])


def keyword_only_dists(key):
    """返回 (纯关键词块的真实距离列表, 向量侧最优距离)。"""
    question = key[1]
    eval_row = match_question_row(base[key], meta)
    lid = byname.get(norm(eval_row.get("book") or ""))
    if not lid:
        return None, None
    docs, metas, dists, targets = webui._retrieve_selected(question, [lid], True, None)
    kw_idx = [i for i, d in enumerate(dists) if d is None]
    vec = [d for d in dists if isinstance(d, (int, float))]
    if not kw_idx:
        return [], (min(vec) if vec else None)
    col = M._open_collection(targets[0]["path"]) if hasattr(M, "_open_collection") else None
    if col is None:
        import chromadb
        col = chromadb.PersistentClient(path=targets[0]["path"]).get_or_create_collection(M.COLLECTION)
    qv = M.embed([question])[0]
    got = col.get(where_document={"$contains": docs[kw_idx[0]][:20]}, include=["embeddings"])
    out = []
    for i in kw_idx:
        g = col.get(where_document={"$contains": docs[i][:40]}, include=["embeddings"], limit=1)
        embs = (g or {}).get("embeddings")
        if embs is None or len(embs) == 0:
            continue
        e = embs[0]
        dot = sum(a * b for a, b in zip(qv, e))
        na = sum(a * a for a in qv) ** 0.5
        nb = sum(b * b for b in e) ** 0.5
        out.append(1.0 - dot / (na * nb) if na and nb else None)
    return [d for d in out if d is not None], (min(vec) if vec else None)


print("证据下限 = %s\n" % webui._EVIDENCE_FLOOR)
for label, qs in (("代价侧（词形碰撞，应被挡掉）", cost[:8]),
                  ("收益侧（真答对，不该被误伤）", gain[:8])):
    print("===== %s =====" % label)
    for q in qs:
        kw, vbest = keyword_only_dists(q)
        if kw is None:
            print("  %-46s 无法映射到知识库" % q[1][:44]); continue
        if not kw:
            print("  %-46s 无纯关键词块" % q[1][:44]); continue
        over = sum(1 for d in kw if d > webui._EVIDENCE_FLOOR)
        print("  %-46s 关键词块 %d 个，距离 %.3f–%.3f，超下限 %d 个 | 向量最优 %s"
              % (q[1][:44], len(kw), min(kw), max(kw), over,
                 ("%.3f" % vbest) if vbest is not None else "无"))
    print()
