"""bench.py — 跨学科基准（三本 × 两模式）"""
import argparse, json, time, os, re
import fitz, ollama, chromadb

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3"
BOOKS = {"CS": "cs.pdf", "Medicine": "med.pdf", "Law": "bizlaw.pdf"}
B_CHUNK, B_OVERLAP, B_TOPK = 1000, 100, 5
P_TARGET, P_MAX, P_TOPK, P_BUDGET = 450, 650, 8, 1300
NUM_PREDICT, TEMPERATURE = 512, 0.0
HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*[\.\)]|[A-Z][A-Z ]{3,}$|Chapter\s+\d+|CHAPTER\s+\d+)")
PROMPT = """Answer the question using ONLY the material below. If there is no basis, answer "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""

def load(path, max_pages):
    d = fitz.open(path); recs = []; n = min(len(d), max_pages)
    for i in range(n):
        t = d[i].get_text("text").strip()
        if t: recs.append((t, i + 1))
    d.close(); return recs, n

def fixed_chunks(text):
    out, s = [], 0
    while s < len(text):
        out.append(text[s:s + B_CHUNK]); s += B_CHUNK - B_OVERLAP
    return out

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

def build(client, name, recs, mode):
    col = client.create_collection(name); ids, docs = [], []; idx = 0
    for text, page in recs:
        pieces = fixed_chunks(text) if mode == "baseline" else semantic_chunks(text)
        for ch in pieces:
            ids.append("c%d" % idx); docs.append(ch); idx += 1
    for i in range(0, len(docs), 64):
        col.add(ids=ids[i:i+64], embeddings=embed(docs[i:i+64]), documents=docs[i:i+64])
        print("    入库 %d/%d" % (min(i+64, len(docs)), len(docs)), end="\r")
    print(" " * 30, end="\r"); return col, len(docs)

def answer(col, q, mode):
    qv = embed([q])[0]
    k = B_TOPK if mode == "baseline" else P_TOPK
    retr = col.query(query_embeddings=[qv], n_results=k)["documents"][0]
    if mode == "baseline":
        ctx = "\n---\n".join(retr)
    else:
        packed, used = [], 0
        for doc in retr:
            if used + len(doc) > P_BUDGET:
                if P_BUDGET - used > 120: packed.append(doc[:P_BUDGET - used])
                break
            packed.append(doc); used += len(doc)
        ctx = "\n---\n".join(packed)
    out = ollama.generate(model=LLM_MODEL, prompt=PROMPT.format(context=ctx, question=q),
                          options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    pt, gt = out.get("prompt_eval_count", 0), out.get("eval_count", 0)
    return {"tokens": pt + gt, "retrieved": retr, "latency": out.get("total_duration", 0)/1e9}

def eval_book(col, questions, mode):
    hits, toks, lat = 0, [], []
    for r in questions:
        a = answer(col, r["question"], mode)
        joined = "\n".join(a["retrieved"])
        if any(kw.lower() in joined.lower() for kw in r.get("keywords", [])): hits += 1
        toks.append(a["tokens"]); lat.append(a["latency"])
    n = len(questions); return sum(toks)/n, hits/n, sum(lat)/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--eval", default="eval_books.jsonl")
    args = ap.parse_args()
    eval_by_book = {}
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); eval_by_book.setdefault(r["book"], []).append(r)
    client = chromadb.Client(); results = {}
    for disp, path in BOOKS.items():
        if not os.path.exists(path):
            print("[跳过] 找不到 %s" % path); continue
        print("\n========== %s（%s）==========" % (disp, path))
        recs, npages = load(path, args.max_pages)
        qs = eval_by_book.get(disp, [])
        if not qs:
            print("  [警告] eval 里没有 %s 的题" % disp); continue
        row = {"pages": npages}
        for mode in ["baseline", "pipeline"]:
            col, nchunks = build(client, "%s_%s" % (disp, mode), recs, mode)
            tok, hit, lat = eval_book(col, qs, mode)
            row[mode] = {"chunks": nchunks, "tokens": round(tok, 1), "hit@5": round(hit, 3), "lat": round(lat, 2)}
            print("  %-9s: %4d块 | %6.0f tok | Hit@5 %5.1f%% | %.2fs" % (mode, nchunks, tok, hit*100, lat))
        ratio = row["pipeline"]["tokens"] / row["baseline"]["tokens"]; row["ratio"] = round(ratio, 3)
        print("  -> token 占基线 %.1f%%（降 %.1f%%）%s 目标<=60%%" % (ratio*100, (1-ratio)*100, "OK" if ratio<=0.6 else "!!"))
        results[disp] = row
    print("\n\n================= 跨学科总表 =================")
    print("%-8s%14s%14s%9s%9s%9s" % ("学科", "base_tok", "pipe_tok", "占基线", "baseHit", "pipeHit"))
    ratios = []
    for disp, r in results.items():
        b, p = r["baseline"], r["pipeline"]; ratios.append(r["ratio"])
        print("%-8s%14.0f%14.0f%8.1f%%%8.0f%%%8.0f%%" % (disp, b["tokens"], p["tokens"], r["ratio"]*100, b["hit@5"]*100, p["hit@5"]*100))
    if ratios:
        avg = sum(ratios)/len(ratios)
        print("\n三学科平均：token 占基线 %.1f%%  %s" % (avg*100, "OK 达标(<=60%)" if avg<=0.6 else "!! 未达标"))
    json.dump(results, open("bench_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[完成] 已存到 bench_metrics.json")

if __name__ == "__main__":
    main()