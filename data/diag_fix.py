# -*- coding: utf-8 -*-
r"""
diag_fix.py —— 两件事，一次跑完

第一部分【可复现性】
    同一个调用重复 N 次，看输出是否稳定。
    起因：diag_raw 与 diag_modes 两次 think=True 结果不同（temperature=0 但没设 seed）。
    在这个问题澄清前，任何单次实测都不能当证据。

第二部分【PROMPT 变体探针】
    现象是模型对"完形填空/关键词碎片"式问题只吐 [p.XXX] 不写正文。
    这里在**完全不改检索、不改打包、不改参数**的前提下，只试 PROMPT 的最小改动，
    看哪一版能让模型写出正文。
    关键：带对照组——两道当前已经答对的可答题，用来确认变体没有改坏已通过的部分。

用法（在 data\ 下，当前 vectordb 须为 Microbiology）：
    C:\Users\Seifer\distill\Scripts\python.exe diag_fix.py
    C:\Users\Seifer\distill\Scripts\python.exe diag_fix.py --repeat 5
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import ollama

# --- 探针题：当前失败（只吐标签）---
FAIL = [
    ("cornea blindness condition", ["acanthamoeba keratitis"]),
    ("Helminths produce many other substances that suppress elements of both innate nonspecific and it host defenses.",
     ["adaptive specific"]),
]
# --- 对照题：当前已答对，变体不得把它们改坏 ---
CTRL = [
    ("Explain microbiota.", ["microbiota"]),
    ("What does this book say about microbial?", ["microbial"]),
]

# PROMPT 变体。V0 是现网原版，一个字没动，作为基准。
# 其余每版只在原版基础上加一句，改动点单一，便于归因。
V0 = M.PROMPT

V1 = M.PROMPT.replace(
    "Question: {question}\nAnswer:",
    "Question: {question}\nWrite the answer as a complete sentence, then the tag. Never answer with a tag alone.\nAnswer:")

V2 = M.PROMPT.replace(
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".',
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".\n'
    'Otherwise your answer must contain the actual information in your own words BEFORE the tag. '
    'A tag by itself is not an answer.')

V3 = M.PROMPT.replace(
    "Question: {question}\nAnswer:",
    "Question: {question}\n"
    "The question may be a sentence with a missing term, or just keywords. "
    "In either case, state the relevant fact from the material in full, then cite the tag.\n"
    "Answer:")

VARIANTS = [("V0 原版（基准）", V0), ("V1 加一句：不许只给标签", V1),
            ("V2 改拒答句后面：标签本身不算答案", V2), ("V3 说明题面可能是填空/关键词", V3)]


def build_prompt(question, col, template):
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
    return template.format(context=context, question=question, tag_example=tag_example)


def gen(prompt, think=False, np_=None, seed=None):
    o = {"temperature": M.TEMPERATURE, "num_predict": np_ or M.NUM_PREDICT}
    if seed is not None:
        o["seed"] = seed
    r = ollama.generate(model=M.LLM_MODEL, prompt=prompt, think=think, options=o)
    return M._strip_think(r.get("response", "")), r.get("eval_count")


def body_of(ans):
    import re
    return re.sub(r"\[[^\]]*\]", "", ans).strip()


def hit(ans, kws):
    a = ans.lower()
    return any(k.lower() in a for k in kws)


def part1(col, repeat):
    print("\n" + "=" * 76)
    print("第一部分：可复现性（同一调用重复 %d 次）" % repeat)
    print("=" * 76)
    q, _ = FAIL[0]
    p = build_prompt(q, col, V0)
    for label, think, np_, seed in [("think=False np=300 无seed", False, 300, None),
                                    ("think=True  np=900 无seed", True, 900, None),
                                    ("think=True  np=900 seed=0", True, 900, 0)]:
        outs = []
        for _ in range(repeat):
            try:
                a, ec = gen(p, think, np_, seed)
            except Exception as e:
                a, ec = "调用失败:%s" % e, None
            outs.append((len(body_of(a)), ec, a[:60]))
        uniq = len(set(o[2] for o in outs))
        print("  %-26s 不同结果 %d/%d   正文长度 %s" %
              (label, uniq, repeat, [o[0] for o in outs]))
        if uniq > 1:
            for o in outs:
                print("        %s" % repr(o[2]))


def part2(col):
    print("\n" + "=" * 76)
    print("第二部分：PROMPT 变体（检索/打包/参数全部不动，只换模板）")
    print("=" * 76)
    for name, tpl in VARIANTS:
        print("\n-- %s --" % name)
        nf = nc = 0
        for tag, pool in (("失败题", FAIL), ("对照题", CTRL)):
            for q, kws in pool:
                p = build_prompt(q, col, tpl)
                try:
                    a, ec = gen(p, think=False)
                except Exception as e:
                    print("   %s 调用失败 %s" % (tag, e)); continue
                b = body_of(a)
                ok = hit(a, kws)
                if tag == "失败题" and ok:
                    nf += 1
                if tag == "对照题" and ok:
                    nc += 1
                print("   %s %-3s 正文%-4d eval=%-4s %s | %s" %
                      (tag, "命中" if ok else "未中", len(b), ec, q[:34], repr(a[:110])))
        print("   >>> 失败题救回 %d/%d ｜ 对照题保持 %d/%d" % (nf, len(FAIL), nc, len(CTRL)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    print("模型 %s | 库 %s" % (M.LLM_MODEL, os.path.abspath(M.DB_PATH)))
    col = M.get_collection()
    part1(col, a.repeat)
    part2(col)
    print("\n判读：")
    print("  第一部分若\"不同结果\">1，说明生成不可复现，须先加 seed 再谈修法。")
    print("  第二部分只认\"失败题救回 2/2 且对照题保持 2/2\"的变体；")
    print("  任何以对照题退化为代价的变体，都是 v2 那次 PROMPT 翻车的重演。")
