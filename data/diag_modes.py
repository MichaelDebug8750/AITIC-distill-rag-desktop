# -*- coding: utf-8 -*-
r"""
diag_modes.py —— 四种生成调用模式对照

目的：v3full 对 "cornea blindness condition" 的存档答案是
      "Acanthamoeba keratitis can lead to blindness if left untreated. [p.955]" (tok=347)
      现在只吐 "[p.955]" (7 token)。
      _generate 有主路径(/api/generate)和降级路径(/api/chat)两条，行为不同。
      这个脚本用完全相同的 prompt 分别打这四种模式，看哪一种能复现 v3full 的输出。

用法（在 data\ 下，当前 vectordb 须为 Microbiology）：
    C:\Users\Seifer\distill\Scripts\python.exe diag_modes.py
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import ollama

QS = [
    ("cornea blindness condition",
     "v3full: 'Acanthamoeba keratitis can lead to blindness if left untreated. [p.955]' tok=347"),
    ("Helminths produce many other substances that suppress elements of both innate nonspecific and it host defenses.",
     "v3full: 'Helminths produce many other substances that suppress ... adaptive specific host defenses. [p.704]' tok=312"),
]


def build_prompt(question, col):
    qe = M.embed([question])[0]
    res = col.query(query_embeddings=[qe], n_results=M.TOP_K,
                    include=["documents", "metadatas", "distances"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    if M.RELEVANCE_TRIM:
        packed, idx = M._pack_relevance(docs, question, M.CONTEXT_BUDGET)
    else:
        packed, idx = M._pack_truncate(docs, M.CONTEXT_BUDGET)
    context = M._labeled_context(packed, idx, metas)
    tags = [M._cite_tag(metas[i]) for i in idx if i < len(metas)]
    uniq = list(dict.fromkeys(tags))[:2]
    tag_example = " or ".join("[%s]" % t for t in uniq) if uniq else "[p.112]"
    return M.PROMPT.format(context=context, question=question, tag_example=tag_example)


OPTS = {"temperature": M.TEMPERATURE, "num_predict": M.NUM_PREDICT}


def mode_A(prompt):
    """主路径：ollama 库 -> /api/generate，think=False（当前 _generate 实际走的）"""
    r = ollama.generate(model=M.LLM_MODEL, prompt=prompt, think=False, options=OPTS)
    return r.get("response", ""), r.get("eval_count"), r.get("done_reason"), len(r.get("thinking") or "")


def mode_B(prompt):
    """降级路径：HTTP /api/chat + think:false（v3 时代很可能实际走的这条）"""
    out = M._post_json("/api/chat", {"model": M.LLM_MODEL, "stream": False, "think": False,
                                     "messages": [{"role": "user", "content": prompt}],
                                     "options": OPTS})
    msg = out.get("message", {})
    return msg.get("content", ""), out.get("eval_count"), out.get("done_reason"), len(msg.get("thinking") or "")


def mode_C(prompt):
    """HTTP /api/generate + think:false（绕开 ollama 库，直连服务端）"""
    out = M._post_json("/api/generate", {"model": M.LLM_MODEL, "prompt": prompt,
                                         "stream": False, "think": False, "options": OPTS})
    return out.get("response", ""), out.get("eval_count"), out.get("done_reason"), len(out.get("thinking") or "")


def mode_D(prompt):
    """拟议修法：think=True + num_predict=900（只在第一遍空壳时才用）"""
    o = dict(OPTS); o["num_predict"] = 900
    r = ollama.generate(model=M.LLM_MODEL, prompt=prompt, think=True, options=o)
    return r.get("response", ""), r.get("eval_count"), r.get("done_reason"), len(r.get("thinking") or "")


MODES = [("A  ollama.generate + think=False  (/api/generate)", mode_A),
         ("B  HTTP /api/chat + think=false", mode_B),
         ("C  HTTP /api/generate + think=false", mode_C),
         ("D  ollama.generate + think=True, np=900", mode_D)]


def main():
    print("ollama 服务 %s | 模型 %s" % (M._ollama_host(), M.LLM_MODEL))
    col = M.get_collection()
    for q, note in QS:
        print("\n" + "=" * 76)
        print("Q: %s" % q)
        print("   %s" % note)
        print("=" * 76)
        prompt = build_prompt(q, col)
        for name, fn in MODES:
            try:
                resp, ec, dr, tl = fn(prompt)
            except Exception as e:
                print("  %-46s 调用失败 %s: %s" % (name, type(e).__name__, str(e)[:90]))
                continue
            body = M._strip_think(resp)
            print("  %-46s eval=%-4s done=%-7s think_len=%-5s" % (name, ec, dr, tl))
            print("      -> %s" % repr(body[:200]))


if __name__ == "__main__":
    main()
