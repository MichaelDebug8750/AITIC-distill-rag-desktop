# -*- coding: utf-8 -*-
r"""
quant_eval.py - 量化消融实验：Q4_K_M vs Q8_0（修订版）
对应任务书交付物④「消融实验（如关闭图表解析/改用 Q4 量化）」

本版相对旧版的五处修订：
  1. 检索只做一次、两档共用   -> 结构上保证检索完全相同（控制变量）
  2. 强制确定性              -> 包住 _generate，塞 temperature=0 + seed
  3. 确定性自检              -> 同题跑两遍，不一致则打印 BROKEN（不让采样噪声混入）
  4. 预热轮 + 中位数延迟      -> 剔除冷启动，报中位数抗离群
  5. 自动记显存              -> shell 出 ollama ps 抓 SIZE，无需另开窗口

题集：只取 eval_books.jsonl 里 book=Medicine 的题（复用真实 keywords）
      + 3 道库外幻觉探针（两档都应拒答，验证 Q8 不改变防幻觉行为）
      当前库为纯 med.pdf，CS/Law 题在语料里无答案，已剔除。

跑法（在 data 目录下）：
    C:\Users\Seifer\distill\Scripts\python.exe ..\code\quant_eval.py

前置：
    ollama pull qwen3:8b-q8_0
"""
import json, time, sys, os, re, subprocess, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M

MODELS = [
    ("Q4_K_M", "qwen3:8b"),
    ("Q8_0",   "qwen3:8b-q8_0"),
]

EVAL_FILE = "eval_books.jsonl"
SEED = 42

# 库外幻觉探针：答案不在 med.pdf 里，两档都应拒答。
# 若模型输出里出现这些关键词 = 编造 = LEAK（坏事）。
PROBES = [
    {"question": "What is the time complexity of quicksort?",
     "keywords": ["quicksort", "n log n", "nlogn", "logn", "logarithm"]},
    {"question": "Who won the 2024 Nobel Prize in Physics?",
     "keywords": ["nobel", "2024", "hopfield", "hinton"]},
    {"question": "What is the capital of Australia?",
     "keywords": ["canberra"]},
]

# ---- 修订2：强制确定性。无论 main.py 传不传 options，都钉死 temperature/seed ----
_orig_generate = M._generate
def _det_generate(model, prompt, system=None, options=None):
    opts = dict(options or {})
    opts["temperature"] = 0
    opts["seed"] = SEED
    return _orig_generate(model, prompt, system=system, options=opts)
M._generate = _det_generate


def load_medicine():
    rows = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r.get("book", "")).lower().startswith("med"):
                rows.append(r)
    return rows


def retrieve(col, q):
    emb = M.embed([q])
    res = col.query(query_embeddings=emb, n_results=M.TOP_K)
    return res["documents"][0]


def refused(ans):
    a = (ans or "").strip()
    return (not a) or ("NO REFERENCE FOUND" in a.upper())


def covered(ans, kws):
    low = (ans or "").lower()
    return any(k.lower() in low for k in kws)


def vram_snapshot(model):
    """修订5：模型加载后 shell 出 ollama ps，抓该模型的 SIZE 列。"""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception as e:
        return "?", "ollama ps failed: %s" % e
    size = "?"
    stub = model.split(":")[0]
    for line in out.splitlines():
        if stub in line:
            m = re.search(r"(\d+(?:\.\d+)?)\s*GB", line)
            if m:
                size = m.group(0)
    return size, out


def run_one(cache, q):
    docs = cache[q]
    t0 = time.time()
    try:
        ans, toks, _ = M._run_once(docs, q, M.CONTEXT_BUDGET)
    except Exception as e:
        ans, toks = "[ERROR] " + str(e), 0
    return ans, toks, time.time() - t0


def main():
    med = load_medicine()
    if not med:
        print("!! eval_books.jsonl 里没找到 book=Medicine 的题，检查 EVAL_FILE 路径")
        return
    col = M.get_collection()
    print("语料库: %d chunks (med.pdf)  |  医学答题 %d 道  |  幻觉探针 %d 道"
          % (col.count(), len(med), len(PROBES)))

    # ---- 修订1：检索只做一次，缓存，两档共用 → 检索链路结构上完全一致 ----
    cache = {}
    for r in med + PROBES:
        cache[r["question"]] = retrieve(col, r["question"])
    print("检索已完成并缓存（两档共用同一检索结果 = 控制变量）\n")

    results = {}
    for tag, model in MODELS:
        print("=" * 22 + " %s (%s) " % (tag, model) + "=" * 22)
        M.LLM_MODEL = model

        # ---- 修订4：预热轮，装权重进显存，把冷启动从计时里剔掉 ----
        run_one(cache, med[0]["question"])

        # ---- 修订5：此刻模型已加载，抓显存 ----
        vram, _ps = vram_snapshot(model)

        # ---- 修订3：确定性自检，同题两遍必须一致 ----
        a1, _, _ = run_one(cache, med[0]["question"])
        a2, _, _ = run_one(cache, med[0]["question"])
        det = "OK" if a1.strip() == a2.strip() else "BROKEN(采样噪声未消除,需在Modelfile里钉temperature)"

        cov_hits = 0
        probe_refused = 0
        toks_all, lats_all = [], []

        for r in med:
            q, kws = r["question"], r.get("keywords", [])
            ans, tk, dt = run_one(cache, q)
            ok = covered(ans, kws)
            cov_hits += 1 if ok else 0
            toks_all.append(tk); lats_all.append(dt)
            print("  %-6s [MED ] %5d tok %5.1fs  %s"
                  % ("OK" if ok else "MISS", tk, dt, q[:42]))

        for r in PROBES:
            q = r["question"]
            ans, tk, dt = run_one(cache, q)
            ref = refused(ans)
            probe_refused += 1 if ref else 0
            toks_all.append(tk); lats_all.append(dt)
            print("  %-6s [PROB] %5d tok %5.1fs  %s"
                  % ("REFUSE" if ref else "LEAK!", tk, dt, q[:42]))

        results[tag] = {
            "cov_hits": cov_hits,
            "cov_total": len(med),
            "cov_rate": cov_hits / max(len(med), 1) * 100,
            "probe_refused": probe_refused,
            "probe_total": len(PROBES),
            "avg_tok": sum(toks_all) / max(len(toks_all), 1),
            "mean_s": statistics.mean(lats_all),
            "median_s": statistics.median(lats_all),
            "vram": vram,
            "determinism": det,
        }
        print("  [显存] %s   [确定性] %s\n" % (vram, det))

    # ---------------- 汇总表 ----------------
    print("=" * 78)
    print("%-9s%10s%9s%12s%11s%10s%8s"
          % ("量化档", "覆盖率", "拒答", "平均Token", "中位延迟", "均值延迟", "显存"))
    print("-" * 78)
    for tag, r in results.items():
        print("%-9s%8d/%d%7d/%d%12.1f%10.2fs%9.2fs%8s"
              % (tag, r["cov_hits"], r["cov_total"],
                 r["probe_refused"], r["probe_total"],
                 r["avg_tok"], r["median_s"], r["mean_s"], r["vram"]))
    print("=" * 78)

    if len(results) == 2:
        a, b = results["Q4_K_M"], results["Q8_0"]
        print("\nQ8_0 相对 Q4_K_M：")
        print("  Token     %+.1f%%" % ((b["avg_tok"] / a["avg_tok"] - 1) * 100))
        print("  中位延迟   %+.1f%%" % ((b["median_s"] / a["median_s"] - 1) * 100))
        print("  覆盖率     %+d 题" % (b["cov_hits"] - a["cov_hits"]))
        print("  拒答行为   %s" % ("一致" if a["probe_refused"] == b["probe_refused"] else "不一致(需排查)"))
        print("\n说明：检索链路两档完全相同（控制变量），上述差异全部来自生成端。")
        print("     检索 Hit@5 不随量化变化，故不在本表比较。")

    with open("quant_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已存 quant_eval_result.json")


if __name__ == "__main__":
    main()
