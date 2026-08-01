# -*- coding: utf-8 -*-
r"""
diag_raw.py —— 生成端原始输出诊断

背景：v6b 跑出来模糊题只吐 [p.XXX] 不写正文，但页码是对的（检索没问题）。
     代码与库都已用指纹证明与 v3 同源，所以问题在生成端。
     这个脚本绕过 main.py 的一切后处理（_strip_think / is_abstain / verify_citations），
     直接打印 ollama 返回的原始字符串，看清楚到底是"模型只说了这么多"
     还是"模型说了但被我们剥掉了"。

用法（必须在 data\ 目录下，且当前 vectordb 是 Microbiology 那本）：
    C:\\Users\\Seifer\\distill\\Scripts\\python.exe diag_raw.py
指定别的问题：
    ... diag_raw.py "你的问题"
"""
import sys, os, json, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import ollama

# Microbiology 库里三道确认失败的模糊题（答案应含 adaptive specific / acanthamoeba keratitis）
DEFAULT_QS = [
    "Helminths produce many other substances that suppress elements of both innate nonspecific and it host defenses.",
    "cornea blindness condition",
]


def one(question):
    print("=" * 74)
    print("Q: %s" % question)
    print("=" * 74)

    col = M.get_collection()
    qe = M.embed([question])[0]
    res = col.query(query_embeddings=[qe], n_results=M.TOP_K,
                    include=["documents", "metadatas", "distances"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    print("\n-- 检索命中 --")
    for i, (m, d) in enumerate(zip(metas, dists)):
        print("  [%d] %-14s dist=%.4f  %s" % (i, M._cite_tag(m), d, docs[i][:70].replace("\n", " ")))

    # 复刻 _run_once 的打包逻辑（不改任何参数）
    if M.RELEVANCE_TRIM:
        packed, idx = M._pack_relevance(docs, question, M.CONTEXT_BUDGET)
    else:
        packed, idx = M._pack_truncate(docs, M.CONTEXT_BUDGET)
    context = M._labeled_context(packed, idx, metas)
    tags = [M._cite_tag(metas[i]) for i in idx if i < len(metas)]
    uniq = list(dict.fromkeys(tags))[:2]
    tag_example = " or ".join("[%s]" % t for t in uniq) if uniq else "[p.112]"
    prompt = M.PROMPT.format(context=context, question=question, tag_example=tag_example)

    print("\n-- 打包 --")
    print("  打进上下文的块: %s" % idx)
    print("  tag_example   : %s" % tag_example)
    print("  上下文字符数  : %d (预算 %d)" % (len(context), M.CONTEXT_BUDGET))

    for think in (False, True):
        print("\n-- ollama.generate(think=%s) 原始返回 --" % think)
        try:
            out = ollama.generate(model=M.LLM_MODEL, prompt=prompt, think=think,
                                  options={"temperature": M.TEMPERATURE,
                                           "num_predict": M.NUM_PREDICT})
        except Exception as e:
            print("  调用失败：%s: %s" % (type(e).__name__, e))
            continue
        resp = out.get("response", "")
        thk = out.get("thinking", "") or ""
        print("  done_reason  : %s   <-- 'length' 说明被 num_predict=%d 截断了"
              % (out.get("done_reason"), M.NUM_PREDICT))
        print("  eval_count   : %s" % out.get("eval_count"))
        print("  response 长度: %d" % len(resp))
        print("  thinking 长度: %d" % len(thk))
        print("  response repr: %s" % repr(resp[:600]))
        if thk:
            print("  thinking 前200: %s" % repr(thk[:200]))
        print("  过 _strip_think 之后: %s" % repr(M._strip_think(resp)[:300]))


if __name__ == "__main__":
    qs = sys.argv[1:] or DEFAULT_QS
    print("模型 %s | 嵌入 %s | 库 %s" % (M.LLM_MODEL, M.EMBED_MODEL, os.path.abspath(M.DB_PATH)))
    fp = M.library_fingerprint()
    print("库指纹 %s | 块数 %s | gate %s\n" % (fp.get("library_chunk_sha"),
                                              fp.get("library_n_chunks"),
                                              fp["runtime"]["escalate_sim_gate"]))
    for q in qs:
        one(q)
        print()
