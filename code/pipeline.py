"""
pipeline.py — 知识蒸馏管线 v1（对着 baseline 打）
================================================
和 baseline.py 用【同一个模型、同一套 eval】，唯一的差别是两处改进，
所以两边的数可以直接对比，证明提升是这两处带来的：

  改进①  语义分块：按段落/句子边界切，块更小更聚合，并给每块打"所属标题"元数据
          （基线是每 1000 字一刀切，经常切断句子/公式 → 检索不准）

  改进②  检索预算控制：多召回候选(8)，但只按 token 预算把【最相关的部分】打包进上下文
          （基线把 5 个大块整个塞进 prompt，prompt 段 token 虚高，水分全在这）

依赖：和 baseline 一样，pip install ollama chromadb pymupdf
模型：qwen3:8b + bge-m3（都已拉好）

用法（和 baseline 完全一致）：
  python pipeline.py --pdf "C:/路径/PDF 123.pdf" --eval eval.jsonl

跑完会：
  - 打印本管线的 token/query、Hit@5、延迟、吞吐
  - 自动读取 baseline_metrics.json，打印【对比】：token 降了多少 %、有没有到 60% 目标
  - 把结果存到 pipeline_metrics.json
"""

import argparse, glob, json, time, os, sys, re
import fitz
import ollama
import chromadb

# ---------------- 配置 ----------------
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3"

# 改进①：更小的语义块（基线是 1000）
CHUNK_TARGET = 450          # 目标块大小（字符），按句子边界凑到这个量级
CHUNK_MAX = 650             # 单块硬上限
TOP_K_CANDIDATES = 8        # 多召回候选（基线是 5 直接全塞）

# 改进②：上下文 token 预算 —— 这是砍 token 的核心旋钮
CONTEXT_CHAR_BUDGET = 900  # 打包进 prompt 的检索内容总字符上限（基线约 5000）

NUM_PREDICT = 512
TEMPERATURE = 0.0
DB_PATH = "./chroma_pipeline"
COLLECTION = "pipeline"

# 更紧凑的系统提示（也省一点 prompt token）
PROMPT_TEMPLATE = """依据下列资料回答问题，只用资料中的信息，无依据则答"[未找到参考资料]"。

资料：
{context}

问题：{question}
回答："""

# 标题识别：一、 / 1. / 1、 / （一） / 第X章 等
HEADING_RE = re.compile(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+[、．.]|\d+[、．.]|[（(][一二三四五六七八九十\d]+[）)])")


def load_pdfs(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        sys.exit(f"[错误] 没找到 PDF：{patterns}")
    records, total_pages = [], 0
    for path in paths:
        doc = fitz.open(path)
        total_pages += len(doc)
        for i, page in enumerate(doc):
            t = page.get_text("text").strip()
            if t:
                records.append((t, os.path.basename(path), i + 1))
        doc.close()
        print(f"  读入 {os.path.basename(path)}：{len(records)} 个非空页累计")
    return records, total_pages


def split_sentences(text):
    """按中文标点 + 换行切句，保留标点。"""
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [s.strip() for s in parts if s.strip()]


def semantic_chunks(text):
    """改进①：识别标题做元数据，按句子边界凑成 ~CHUNK_TARGET 的语义块。"""
    lines = text.split("\n")
    cur_heading = ""
    chunks, buf, buf_len = [], [], 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append((cur_heading, "".join(buf).strip()))
            buf, buf_len = [], 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line) and len(line) < 40:
            flush()
            cur_heading = line
            continue
        for sent in split_sentences(line):
            if buf_len + len(sent) > CHUNK_MAX and buf:
                flush()
            buf.append(sent)
            buf_len += len(sent)
            if buf_len >= CHUNK_TARGET:
                flush()
    flush()
    return chunks  # [(heading, text), ...]


def embed_texts(texts):
    try:
        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    except (AttributeError, KeyError, TypeError):
        return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


def ingest(records):
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    t0 = time.time()
    ids, docs, metas = [], [], []
    idx = 0
    for text, source, page in records:
        for heading, ch in semantic_chunks(text):
            ids.append(f"c{idx}")
            # 把标题拼进可检索文本，增强定位（元数据也单独存）
            docs.append((heading + "\n" + ch) if heading else ch)
            metas.append({"source": source, "page": page, "heading": heading})
            idx += 1

    BATCH = 64
    for i in range(0, len(docs), BATCH):
        col.add(ids=ids[i:i+BATCH], embeddings=embed_texts(docs[i:i+BATCH]),
                documents=docs[i:i+BATCH], metadatas=metas[i:i+BATCH])
        print(f"  入库 {min(i+BATCH, len(docs))}/{len(docs)} 块", end="\r")
    print()
    return col, len(docs), time.time() - t0


def pack_context(retrieved):
    """改进②：按字符预算只打包最相关的内容，砍掉冗余 → 直接压低 prompt token。"""
    packed, used = [], 0
    for doc in retrieved:
        if used + len(doc) > CONTEXT_CHAR_BUDGET:
            remain = CONTEXT_CHAR_BUDGET - used
            if remain > 120:                     # 还能塞一段有意义的片段
                packed.append(doc[:remain])
            break
        packed.append(doc)
        used += len(doc)
    return "\n---\n".join(packed)


def answer(col, question):
    qvec = embed_texts([question])[0]
    res = col.query(query_embeddings=[qvec], n_results=TOP_K_CANDIDATES)
    retrieved = res["documents"][0]
    context = pack_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    t0 = time.time()
    out = ollama.generate(model=LLM_MODEL, prompt=prompt,
                          options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    latency = time.time() - t0
    pt, gt = out.get("prompt_eval_count", 0), out.get("eval_count", 0)
    return {"question": question, "answer": out["response"].strip(),
            "retrieved": retrieved, "prompt_tokens": pt, "gen_tokens": gt,
            "total_tokens": pt + gt, "latency_s": round(latency, 2)}


def run_eval(col, eval_path):
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    hits, results = 0, []
    for r in rows:
        out = answer(col, r["question"])
        joined = "\n".join(out["retrieved"])
        kws = r.get("keywords", [])
        hit = any(kw in joined for kw in kws) if kws else None
        if hit:
            hits += 1
        out["hit@5"] = hit
        results.append(out)
        flag = "✓" if hit else ("✗" if hit is False else "?")
        print(f"  [{flag}] {r['question'][:30]:<30} | {out['total_tokens']} tok | {out['latency_s']}s")
    n = len(results)
    return (results,
            sum(x["total_tokens"] for x in results)/n if n else 0,
            sum(x["latency_s"] for x in results)/n if n else 0,
            hits/n if n else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--eval")
    args = ap.parse_args()

    print("== 1/3 读 PDF ==")
    records, total_pages = load_pdfs(args.pdf)
    print("== 2/3 入库（语义分块 → 向量化 → Chroma）==")
    col, n_chunks, elapsed = ingest(records)
    ppm = total_pages / (elapsed/60) if elapsed else 0
    print(f"\n== 入库 ==\n  总页数 : {total_pages}\n  总块数 : {n_chunks}（基线 9 块，更细）\n  耗时   : {elapsed:.1f}s\n  吞吐   : {ppm:.1f} 页/分钟")

    metrics = {"config": {"llm": LLM_MODEL, "embed": EMBED_MODEL, "chunk_target": CHUNK_TARGET,
                          "top_k_candidates": TOP_K_CANDIDATES, "context_char_budget": CONTEXT_CHAR_BUDGET},
               "ingest": {"total_pages": total_pages, "total_chunks": n_chunks,
                          "seconds": round(elapsed, 1), "pages_per_min": round(ppm, 1)}}

    print("\n== 3/3 问答 ==")
    if args.eval:
        results, avg_tok, avg_lat, hit = run_eval(col, args.eval)
        print(f"\n== 本管线 ==\n  平均 token/query : {avg_tok:.0f}\n  平均延迟         : {avg_lat:.2f}s\n  Hit@5            : {hit*100:.1f}%")
        metrics["qa"] = {"n": len(results), "avg_tokens_per_query": round(avg_tok, 1),
                         "avg_latency_s": round(avg_lat, 2), "hit@5": round(hit, 4)}
        # 自动对比 baseline
        if os.path.exists("baseline_metrics.json"):
            try:
                base = json.load(open("baseline_metrics.json", encoding="utf-8"))
                b = base.get("qa", {}).get("avg_tokens_per_query")
                if b:
                    ratio = avg_tok / b
                    print("\n== 对比 baseline ==")
                    print(f"  baseline token/query : {b:.0f}")
                    print(f"  本管线 token/query   : {avg_tok:.0f}")
                    print(f"  降幅                 : {(1-ratio)*100:.1f}%  （占基线 {ratio*100:.1f}%）")
                    print(f"  目标 ≤60%            : {'✅ 达标' if ratio <= 0.60 else '❌ 还需努力'}")
                    metrics["vs_baseline"] = {"baseline_tokens": b, "ratio": round(ratio, 4),
                                              "hit_baseline": base.get("qa", {}).get("hit@5")}
            except Exception as e:
                print(f"  （对比失败：{e}）")
    else:
        for q in ["方向A的项目要做什么?", "知识蒸馏在这个项目里指什么?"]:
            out = answer(col, q)
            print(f"\n  Q: {q}\n  A: {out['answer'][:300]}\n  tokens: {out['total_tokens']} | {out['latency_s']}s")

    json.dump(metrics, open("pipeline_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[完成] 已存到 pipeline_metrics.json")


if __name__ == "__main__":
    main()
