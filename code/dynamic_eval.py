# -*- coding: utf-8 -*-
"""
dynamic_eval.py — 动态预算 对照评测

目的：量化「动态预算」对 20% 过度拒答的缓解，并证明其 Token 代价可控。
在可答集（eval_books.jsonl 的 GT 题，本应作答）上，对比三种模式：
  · 固定 900   —— 现生产配置：Token 低，但对被截断的题会误拒
  · 固定 1800  —— 无脑加预算：误拒少，但每题 Token 都涨
  · 动态       —— 首答 900；仅当「检索有命中却拒答」时升配 1800 重答一次
指标：过度拒答率（越低越好）、平均 Token/题（越低越好）、动态升配触发次数。

复用 hallucination.py 的建库/嵌入/拒答判定，配置与 main.py 生产一致。
从当前目录读 cs.pdf / med.pdf / bizlaw.pdf 及 eval_books.jsonl。
用法：python dynamic_eval.py
"""
import json
import os
import chromadb
import ollama
import hallucination as H   # 复用 load/build/embed/semantic_chunks/is_abstain/BOOKS/config

BUDGET_LOW, BUDGET_HIGH = 900, 1800


def run_at(col, question, budget):
    """按预算打包 + 生成，返回 (答案, tokens, 检索命中数)。"""
    qv = H.embed([question])[0]
    docs = col.query(query_embeddings=[qv], n_results=H.TOP_K)["documents"][0]
    packed, used = [], 0
    for doc in docs:
        if used + len(doc) > budget:
            if budget - used > 120:
                packed.append(doc[:budget - used])
            break
        packed.append(doc); used += len(doc)
    context = "\n---\n".join(packed)
    out = ollama.generate(model=H.LLM_MODEL,
                          prompt=H.PROMPT.format(context=context, question=question),
                          options={"temperature": H.TEMPERATURE, "num_predict": H.NUM_PREDICT})
    toks = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
    return out["response"].strip(), toks, len(docs)


def main():
    answerable = {}
    with open("eval_books.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                answerable.setdefault(r["book"], []).append(r["question"])

    client = chromadb.Client()
    agg = {"low": [0, 0], "high": [0, 0], "dyn": [0, 0]}  # 每项 [误拒数, token累计]
    n_total = 0
    n_escalated = 0
    rescued = []   # 被动态预算救回的题

    for disp, path in H.BOOKS.items():
        if not os.path.exists(path):
            print("[跳过] 找不到 %s" % path); continue
        recs = H.load(path, H.MAX_PAGES)
        col, nchunks = H.build(client, "dyn_%s" % disp, recs)
        qs = answerable.get(disp, [])
        print("\n========== %s（%s，%d 块，%d 题）==========" % (disp, path, nchunks, len(qs)))

        for q in qs:
            n_total += 1
            a900, t900, ndocs = run_at(col, q, BUDGET_LOW)
            a1800, t1800, _ = run_at(col, q, BUDGET_HIGH)
            r900, r1800 = H.is_abstain(a900), H.is_abstain(a1800)

            # 动态：首答 900；命中却拒答则升配 1800
            escalated = bool(ndocs) and r900
            if escalated:
                n_escalated += 1
                t_dyn = t900 + t1800
                r_dyn = r1800
                if r900 and not r1800:
                    rescued.append((disp, q))
            else:
                t_dyn = t900
                r_dyn = r900

            agg["low"][0] += r900;  agg["low"][1] += t900
            agg["high"][0] += r1800; agg["high"][1] += t1800
            agg["dyn"][0] += r_dyn;  agg["dyn"][1] += t_dyn

            mark = "↑升配" if escalated else "     "
            print("  %s 900:%s 1800:%s | %s" %
                  (mark, "拒" if r900 else "答", "拒" if r1800 else "答", H.short(q, 50)))

    print("\n" + "=" * 62)
    print("%-12s%14s%16s" % ("模式", "过度拒答率", "平均 token/题"))
    print("-" * 62)
    for key, name in [("low", "固定 900"), ("high", "固定 1800"), ("dyn", "动态")]:
        ref, tok = agg[key]
        print("%-12s%12d/%d%13.1f%%%12.1f" %
              (name, ref, n_total, 100.0 * ref / n_total, tok / n_total))
    print("-" * 62)
    print("动态升配触发：%d/%d 题" % (n_escalated, n_total))
    if rescued:
        print("被动态预算救回（900拒答→1800答出）：")
        for disp, q in rescued:
            print("  · [%s] %s" % (disp, H.short(q, 60)))

    json.dump({"n_total": n_total, "n_escalated": n_escalated,
               "low": agg["low"], "high": agg["high"], "dyn": agg["dyn"],
               "rescued": rescued},
              open("dynamic_eval_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n结果已存 dynamic_eval_result.json")


if __name__ == "__main__":
    main()
