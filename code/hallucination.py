# -*- coding: utf-8 -*-
"""
hallucination.py — 幻觉率实测（无引用生成率）

口径（对标任务书"幻觉率(无引用生成)≤15%"）：
  · 不可答集：问语料里根本没有答案的问题（跨学科 + 常识探针，已逐一验证关键词
    在该书前120页 0 命中）。faithful 系统应输出 [NO REFERENCE FOUND]；
    若编造实质答案 → 记一次幻觉。  幻觉率 = 编造数 / 不可答题数。
  · 可答集（对照）：eval_books.jsonl 里的 GT 题，系统应正常作答；
    若对这些也拒答 → 过度拒答。  拒答率应≈0，用以证明低幻觉率不是"无脑拒答"刷出来的。

配置与 main.py 生产一致：TOP_K=8 / CONTEXT_BUDGET=900 / 同一 PROMPT / temp 0 / num_predict 300。
自包含、纯文本建库（不走 VL，拒答行为与 VL 无关），从当前目录读 cs.pdf/med.pdf/bizlaw.pdf
与 eval_books.jsonl，结果写 hallucination_metrics.json。
跑法：在 data 目录下  python ..\\code\\hallucination.py
"""
import json, re, os
import fitz, ollama, chromadb

LLM_MODEL  = "qwen3:8b"
EMBED_MODEL = "bge-m3"
BOOKS = {"CS": "cs.pdf", "Medicine": "med.pdf", "Law": "bizlaw.pdf"}

MAX_PAGES = 120
P_TARGET, P_MAX, TOP_K, BUDGET = 450, 650, 8, 900     # 与 main.py 生产一致
NUM_PREDICT, TEMPERATURE = 300, 0.0
HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*[\.\)]|[A-Z][A-Z ]{3,}$|Chapter\s+\d+|CHAPTER\s+\d+)")

# main.py 的原版 PROMPT（含拒答指令）
PROMPT = """Answer the question using ONLY the material below. Cite the source page like [p.X] when possible.
If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""

# ---- 不可答集（每书7题，关键词均已验证在该书前120页 0 命中）----
UNANSWERABLE = {
    "CS": [
        "What are the elements of a valid contract?",
        "What is a tort?",
        "How do antibiotics affect bacteria?",
        "What is the difference between prokaryotic and eukaryotic cells?",
        "What is the role of the immune system?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ],
    "Medicine": [
        "What does an operating system do?",
        "What is a data structure and why is it useful?",
        "What is the difference between hardware and software?",
        "What is an algorithm in computer science?",
        "What is liability in business law?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ],
    "Law": [
        "What is the difference between prokaryotic and eukaryotic cells?",
        "How do antibiotics affect bacteria?",
        "What is the role of the immune system?",
        "What does an operating system do?",
        "What is a neural network?",
        "What is the capital of France?",
        "What is photosynthesis?",
    ],
}


def load(path, max_pages):
    d = fitz.open(path); recs = []; n = min(len(d), max_pages)
    for i in range(n):
        t = d[i].get_text("text").strip()
        if t: recs.append((t, i + 1))
    d.close(); return recs


def split_sents(text):
    return [x.strip() for x in re.split(r"(?<=[.!?。！？;\n])", text) if x.strip()]


def semantic_chunks(text):
    chunks, buf, blen, heading = [], [], 0, ""
    def flush():
        nonlocal buf, blen
        if buf:
            chunks.append((heading + " " + "".join(buf)).strip()); buf, blen = [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        if HEADING_RE.match(line) and len(line) < 60:
            flush(); heading = line; continue
        for s in split_sents(line):
            if blen + len(s) > P_MAX and buf: flush()
            buf.append(s); blen += len(s)
            if blen >= P_TARGET: flush()
    flush(); return chunks


def embed(texts):
    try:
        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    except Exception:
        return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


def build(client, name, recs):
    col = client.create_collection(name); ids, docs = [], []; idx = 0
    for text, page in recs:
        for ch in semantic_chunks(text):
            ids.append("c%d" % idx); docs.append(ch); idx += 1
    for i in range(0, len(docs), 64):
        col.add(ids=ids[i:i+64], embeddings=embed(docs[i:i+64]), documents=docs[i:i+64])
    return col, len(docs)


def ask(col, question):
    """复刻 main.py 的 ask：top-8 → 打包到 budget 900 → 生成"""
    qv = embed([question])[0]
    docs = col.query(query_embeddings=[qv], n_results=TOP_K)["documents"][0]
    packed, used = [], 0
    for doc in docs:
        if used + len(doc) > BUDGET:
            if BUDGET - used > 120: packed.append(doc[:BUDGET - used])
            break
        packed.append(doc); used += len(doc)
    context = "\n---\n".join(packed)
    out = ollama.generate(model=LLM_MODEL, prompt=PROMPT.format(context=context, question=question),
                          options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    return out["response"].strip()


_ABSTAIN_RE = re.compile(r"no\s+(?:\w+\s+){0,2}references?\s+(?:found|available|provided|in)", re.I)
def is_abstain(answer):
    """鲁棒判定拒答：认 [NO REFERENCE FOUND] 及其常见变体
    （NO REFERENCES FOUND / NO RELEVANT REFERENCE FOUND / NO REFERENCE AVAILABLE / 括号式…）
    与 main.py 的 is_abstain 保持一致：空/纯空白也视为拒答
    （Qwen3 有时只输出 <think> 块、正式回答为空，此时既非答案也非编造）。"""
    if answer is None or not str(answer).strip():
        return True
    a = answer.lower()
    if "no reference found" in a:
        return True
    if re.search(r"\[[^\]]*\bno\b[^\]]*\breferences?\b", a):   # [ ... NO ... REFERENCE(S) ... ]
        return True
    if _ABSTAIN_RE.search(a):
        return True
    return False


def short(s, n=70):
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def main():
    # 可答集（对照）从 eval_books.jsonl 读
    answerable = {}
    with open("eval_books.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); answerable.setdefault(r["book"], []).append(r["question"])

    client = chromadb.Client()
    results = {}
    tot_hallu = tot_unans = tot_refuse = tot_ans = 0

    for disp, path in BOOKS.items():
        if not os.path.exists(path):
            print("[跳过] 找不到 %s" % path); continue
        print("\n========== %s（%s）==========" % (disp, path))
        recs = load(path, MAX_PAGES)
        col, nchunks = build(client, "hallu_%s" % disp, recs)
        print("  建库 %d 块" % nchunks)

        # --- 不可答集：应拒答 ---
        print("  --- 不可答集（应输出 [NO REFERENCE FOUND]）---")
        hallu = 0; un = UNANSWERABLE[disp]; un_log = []
        for q in un:
            a = ask(col, q)
            ok = is_abstain(a)
            if not ok: hallu += 1
            un_log.append({"q": q, "abstain": ok, "answer": a})
            print("    %s %-55s" % ("✓拒答" if ok else "✗幻觉", short(q, 52)))
            if not ok:   # 疑似幻觉：打印完整答案供审核
                print("        ↳ 完整答案：%s" % short(a, 300))
        # --- 可答集：应作答 ---
        refuse = 0; ans = answerable.get(disp, []); ans_log = []
        for q in ans:
            a = ask(col, q)
            wrong = is_abstain(a)
            if wrong: refuse += 1
            ans_log.append({"q": q, "abstain": wrong, "answer": a})
            if wrong:   # 误拒：打印是哪道题 + 完整答案
                print("    ✗误拒（本应作答却拒答）：%s" % short(q, 52))
                print("        ↳ 完整答案：%s" % short(a, 200))
        nun, nan = len(un), len(ans)
        hr = hallu / nun if nun else 0
        rr = refuse / nan if nan else 0
        results[disp] = {"unanswerable": nun, "hallucinations": hallu, "hallu_rate": round(hr, 3),
                         "answerable": nan, "wrong_refusals": refuse, "refuse_rate": round(rr, 3),
                         "unanswerable_log": un_log, "answerable_log": ans_log}
        print("  小结：不可答 %d 题，幻觉 %d → 幻觉率 %.1f%% | 可答 %d 题，误拒 %d → 拒答率 %.1f%%"
              % (nun, hallu, hr*100, nan, refuse, rr*100))
        tot_hallu += hallu; tot_unans += nun; tot_refuse += refuse; tot_ans += nan

    print("\n\n================= 幻觉率总表 =================")
    print("%-10s%12s%12s%14s%12s" % ("学科", "不可答", "幻觉数", "幻觉率", "误拒率"))
    for disp, r in results.items():
        print("%-10s%12d%12d%13.1f%%%11.1f%%" % (disp, r["unanswerable"], r["hallucinations"],
                                                 r["hallu_rate"]*100, r["refuse_rate"]*100))
    overall_hr = tot_hallu / tot_unans if tot_unans else 0
    overall_rr = tot_refuse / tot_ans if tot_ans else 0
    print("-" * 60)
    print("总体幻觉率：%d/%d = %.1f%%   %s 目标≤15%%"
          % (tot_hallu, tot_unans, overall_hr*100, "OK 达标" if overall_hr <= 0.15 else "!! 未达标"))
    print("总体误拒率：%d/%d = %.1f%%   （越低越好，证明非无脑拒答）" % (tot_refuse, tot_ans, overall_rr*100))

    results["_overall"] = {"hallu_rate": round(overall_hr, 3), "refuse_rate": round(overall_rr, 3),
                           "hallucinations": tot_hallu, "unanswerable": tot_unans,
                           "wrong_refusals": tot_refuse, "answerable": tot_ans}
    json.dump(results, open("hallucination_metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[完成] 已存到 hallucination_metrics.json")


if __name__ == "__main__":
    main()
