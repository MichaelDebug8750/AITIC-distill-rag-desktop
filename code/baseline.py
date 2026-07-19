"""
baseline.py — 知识蒸馏管线 · 裸 RAG 基线
================================================
这是任务书要求的「参照系」:一个【故意做得很笨】的标准 RAG。
你后面的蒸馏管线要在同一套测试集上,把 token 砍到这个基线的 60% 以下、
Hit@5 顶到 85%+。所以这里的目标不是"做好",而是"做标准、可对比"。

它会吐出三个核心基线数:
  1. tokens/query   —— 每次问答的总 token(prompt + 生成),这是 ≤60% 的【分母】
  2. pages/min      —— 入库吞吐(解析+切块+向量化的速度)
  3. Hit@5          —— top-5 检索里是否命中答案关键词(召回代理指标)

依赖(在你的 conda 环境里装):
  pip install ollama chromadb pymupdf

前置:Ollama 已起、且这两个模型已拉好:
  ollama pull qwen3:8b      # 基线 LLM(纯文本,故意不用 VL,基线就该弱)
  ollama pull bge-m3        # 向量模型

用法:
  # 只入库 + 跑几个手测问题
  python baseline.py --pdf "C:/path/教材.pdf"

  # 入库 + 用 eval.jsonl 自动算 Hit@5 / 平均 token
  python baseline.py --pdf "C:/path/教材.pdf" --eval eval.jsonl

  # 多个 PDF
  python baseline.py --pdf "C:/books/*.pdf" --eval eval.jsonl

eval.jsonl 每行一条(关键词命中即算 Hit):
  {"question": "什么是动态规划?", "keywords": ["最优子结构", "重叠子问题"]}
"""

import argparse
import glob
import json
import time
import os
import sys

import fitz  # PyMuPDF
import ollama
import chromadb

# ----------------------------- 配置(故意保守的"标准"参数)-----------------------------
LLM_MODEL = "qwen3:8b"          # 基线生成模型
EMBED_MODEL = "bge-m3"          # 向量模型
CHUNK_SIZE = 1000               # 字符级固定切块(基线就用最笨的,不做语义边界)
CHUNK_OVERLAP = 100
TOP_K = 5                       # 检索 top-k
NUM_PREDICT = 512               # 固定最大生成长度(token 口径要统一)
TEMPERATURE = 0.0               # 固定温度(同上,保证可复现对比)
DB_PATH = "./chroma_baseline"
COLLECTION = "baseline"

# 基线刻意不做 Citation Grounding,所以会"瞎编"——这正是你管线要改进的地方。
PROMPT_TEMPLATE = """你是一个学科问答助手。请只依据下面提供的资料回答问题。

[资料]
{context}

[问题]
{question}

[回答]"""


# ----------------------------- 工具函数 -----------------------------
def load_pdfs(patterns):
    """读 PDF,返回 [(text, source, page_no), ...] 和总页数。基线只抽纯文本,不碰图表。"""
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        sys.exit(f"[错误] 没找到任何 PDF,检查路径:{patterns}")

    records = []
    total_pages = 0
    for path in paths:
        doc = fitz.open(path)
        total_pages += len(doc)
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                records.append((text, os.path.basename(path), i + 1))
        doc.close()
        print(f"  读入 {os.path.basename(path)}:{len(records)} 个非空页累计")
    return records, total_pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """最笨的固定长度切块(字符级),不做语义边界——基线就该这样。"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed_texts(texts):
    """用 bge-m3 批量向量化。兼容 ollama 新旧 API。"""
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=texts)
        return resp["embeddings"]
    except (AttributeError, KeyError, TypeError):
        # 旧版本逐条
        return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


# ----------------------------- 入库 -----------------------------
def ingest(records):
    """切块 → 向量化 → 存 Chroma,并计时算 pages/min。"""
    client = chromadb.PersistentClient(path=DB_PATH)
    # 每次重跑都清掉旧库,保证基线干净
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    t0 = time.time()
    ids, docs, metas, all_chunks = [], [], [], []
    idx = 0
    for text, source, page in records:
        for ch in chunk_text(text):
            ids.append(f"c{idx}")
            docs.append(ch)
            metas.append({"source": source, "page": page})
            all_chunks.append(ch)
            idx += 1

    # 分批向量化 + 写入(一批 64,省显存也快)
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vecs = embed_texts(batch)
        col.add(
            ids=ids[i:i + BATCH],
            embeddings=vecs,
            documents=docs[i:i + BATCH],
            metadatas=metas[i:i + BATCH],
        )
        print(f"  入库 {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} 块", end="\r")
    elapsed = time.time() - t0
    print()
    return col, len(all_chunks), elapsed


# ----------------------------- 检索 + 生成 -----------------------------
def answer(col, question, top_k=TOP_K):
    """检索 top-k → 拼 prompt → 调 qwen3:8b → 返回答案 + token 统计 + 命中文档。"""
    qvec = embed_texts([question])[0]
    res = col.query(query_embeddings=[qvec], n_results=top_k)
    retrieved = res["documents"][0]
    context = "\n\n---\n\n".join(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    t0 = time.time()
    out = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    )
    latency = time.time() - t0

    prompt_tok = out.get("prompt_eval_count", 0)
    gen_tok = out.get("eval_count", 0)
    return {
        "question": question,
        "answer": out["response"].strip(),
        "retrieved": retrieved,
        "prompt_tokens": prompt_tok,
        "gen_tokens": gen_tok,
        "total_tokens": prompt_tok + gen_tok,
        "latency_s": round(latency, 2),
    }


# ----------------------------- 评测 -----------------------------
def run_eval(col, eval_path):
    """读 eval.jsonl,逐题算 Hit@5(关键词是否出现在 top-5 检索块里)+ 平均 token / 延迟。"""
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
        out["keywords"] = kws
        results.append(out)
        flag = "✓" if hit else ("✗" if hit is False else "?")
        print(f"  [{flag}] {r['question'][:30]:<30} | {out['total_tokens']} tok | {out['latency_s']}s")

    n = len(results)
    avg_tok = sum(x["total_tokens"] for x in results) / n if n else 0
    avg_lat = sum(x["latency_s"] for x in results) / n if n else 0
    hit_rate = hits / n if n else 0
    return results, avg_tok, avg_lat, hit_rate


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True, help="PDF 路径,支持通配符")
    ap.add_argument("--eval", help="eval.jsonl 评测集(可选)")
    args = ap.parse_args()

    print("== 1/3 读 PDF ==")
    records, total_pages = load_pdfs(args.pdf)

    print("== 2/3 入库(切块→向量化→Chroma)==")
    col, n_chunks, elapsed = ingest(records)
    pages_per_min = total_pages / (elapsed / 60) if elapsed else 0

    print("\n== 入库基线 ==")
    print(f"  总页数      : {total_pages}")
    print(f"  总块数      : {n_chunks}")
    print(f"  耗时        : {elapsed:.1f}s")
    print(f"  吞吐        : {pages_per_min:.1f} 页/分钟")

    metrics = {
        "config": {
            "llm": LLM_MODEL, "embed": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "top_k": TOP_K,
        },
        "ingest": {
            "total_pages": total_pages, "total_chunks": n_chunks,
            "seconds": round(elapsed, 1), "pages_per_min": round(pages_per_min, 1),
        },
    }

    print("\n== 3/3 问答 ==")
    if args.eval:
        results, avg_tok, avg_lat, hit_rate = run_eval(col, args.eval)
        print("\n== 问答基线 ==")
        print(f"  平均 token/query : {avg_tok:.0f}   <-- 这是 ≤60% 的【分母】")
        print(f"  平均延迟         : {avg_lat:.2f}s")
        print(f"  Hit@5            : {hit_rate*100:.1f}%")
        metrics["qa"] = {
            "n": len(results),
            "avg_tokens_per_query": round(avg_tok, 1),
            "avg_latency_s": round(avg_lat, 2),
            "hit@5": round(hit_rate, 4),
        }
        metrics["samples"] = results
    else:
        # 没给 eval 就跑两个手测问题,先看链路通不通
        for q in ["这本教材主要讲什么?", "请举一个书里的核心概念并解释。"]:
            out = answer(col, q)
            print(f"\n  Q: {q}")
            print(f"  A: {out['answer'][:300]}")
            print(f"  tokens: {out['total_tokens']} (prompt {out['prompt_tokens']} + gen {out['gen_tokens']}) | {out['latency_s']}s")

    with open("baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n[完成] 基线数已存到 baseline_metrics.json")


if __name__ == "__main__":
    main()
