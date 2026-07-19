"""ablation.py — 预算消融：同一套书/题，P_BUDGET 取 900/1300/1800 各跑一遍看权衡"""
import json, re
import fitz, ollama, chromadb

EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen3:8b"
BOOKS = {"CS": "cs.pdf", "Medicine": "med.pdf", "Law": "bizlaw.pdf"}
MAX_PAGES = 120
P_TARGET, P_MAX, P_TOPK = 450, 650, 8
BUDGETS = [900, 1300, 1800]          # 要对比的三个预算
NUM_PREDICT, TEMPERATURE = 512, 0.0
HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*[\.\)]|[A-Z][A-Z ]{3,}$|Chapter\s+\d+|CHAPTER\s+\d+)")
PROMPT = """Answer the question using ONLY the material below. If there is no basis, answer "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""

def split_sents(t):
    return [x.strip() for x in re.split(r"(?<=[.!?。！？;\n])", t) if x.strip()]

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

# 读题，按 book 分组
eval_by_book = {}
for line in open("eval_books.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line); eval_by_book.setdefault(r["book"], []).append(r)

client = chromadb.Client()
# 每本书只建一次库（消融只改打包预算，不改入库）
cols = {}
for disp, path in BOOKS.items():
    d = fitz.open(path); docs = []
    for i in range(min(len(d), MAX_PAGES)):
        t = d[i].get_text("text").strip()
        if t: docs += semantic_chunks(t)
    d.close()
    col = client.create_collection("abl_%s" % disp)
    for i in range(0, len(docs), 64):
        col.add(ids=["c%d" % j for j in range(i, min(i+64, len(docs)))],
                embeddings=embed(docs[i:i+64]), documents=docs[i:i+64])
    cols[disp] = col
    print("建库 %s：%d 块" % (disp, len(docs)))
print()

def run(col, q, budget):
    retr = col.query(query_embeddings=[embed([q])[0]], n_results=P_TOPK)["documents"][0]
    packed, used = [], 0
    for doc in retr:
        if used + len(doc) > budget:
            if budget - used > 120: packed.append(doc[:budget - used])
            break
        packed.append(doc); used += len(doc)
    ctx = "\n---\n".join(packed)
    out = ollama.generate(model=LLM_MODEL, prompt=PROMPT.format(context=ctx, question=q),
                          options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    hit = any(kw.lower() in "\n".join(retr).lower() for kw in [])  # 占位，下面重算
    return out.get("prompt_eval_count", 0) + out.get("eval_count", 0), retr

# 跑消融
results = {}  # budget -> {book -> (tok, hit)}
for budget in BUDGETS:
    results[budget] = {}
    for disp, col in cols.items():
        qs = eval_by_book[disp]; toks, hits = [], 0
        for r in qs:
            retr = col.query(query_embeddings=[embed([r["question"]])[0]], n_results=P_TOPK)["documents"][0]
            packed, used = [], 0
            for doc in retr:
                if used + len(doc) > budget:
                    if budget - used > 120: packed.append(doc[:budget - used])
                    break
                packed.append(doc); used += len(doc)
            ctx = "\n---\n".join(packed)
            out = ollama.generate(model=LLM_MODEL, prompt=PROMPT.format(context=ctx, question=r["question"]),
                                  options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
            toks.append(out.get("prompt_eval_count", 0) + out.get("eval_count", 0))
            if any(kw.lower() in "\n".join(retr).lower() for kw in r.get("keywords", [])): hits += 1
        n = len(qs)
        results[budget][disp] = (sum(toks)/n, hits/n)
    print("预算 %d 跑完" % budget)

# 出表
print("\n================ 预算消融对比 ================")
print("%-9s" % "预算", end="")
for disp in BOOKS: print("%18s" % ("%s(tok/Hit)" % disp), end="")
print("%14s" % "三科均tok")
for budget in BUDGETS:
    print("%-9d" % budget, end="")
    avg = []
    for disp in BOOKS:
        tok, hit = results[budget][disp]; avg.append(tok)
        print("%18s" % ("%.0f / %.0f%%" % (tok, hit*100)), end="")
    print("%14.0f" % (sum(avg)/len(avg)))
print("\n（基线三科约 1582 tok；看预算往下调时 token 降多少、Hit@5 何时开始掉）")

json.dump({str(b): {d: {"tok": round(v[0],1), "hit": round(v[1],3)} for d,v in r.items()}
           for b,r in results.items()},
          open("ablation_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("[完成] 已存到 ablation_metrics.json")