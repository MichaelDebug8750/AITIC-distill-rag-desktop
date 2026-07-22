"""
引用准确度评测（Citation Accuracy Eval）
------------------------------------------------
用题集逐题跑管线，用 verify_citations 核对：正文引用的页码，是否都落在实际检索到的来源里。
输出【引用准确率】= 命中检索来源的引用数 / 正文总引用数，以及有多少题出现"疑似编造"。

用法（在 data 目录、且当前库=对应学科时运行）：
    python ..\\code\\cite_eval.py                       # 默认 eval_med.jsonl，跑当前库
    python ..\\code\\cite_eval.py eval_med.jsonl
    python ..\\code\\cite_eval.py eval_books.jsonl Medicine   # 从合集里按 book 过滤

三学科完整测：分别 build 该学科库后各跑一次（med / cs / law）。
"""
import json, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M

# 确定性：钉死 temperature=0 + seed，保证结果可复现（与 quant_eval 同风格）
SEED = 42
_orig_generate = M._generate
def _det_generate(model, prompt, system=None, options=None):
    opts = dict(options or {}); opts["temperature"] = 0; opts["seed"] = SEED
    return _orig_generate(model, prompt, system=system, options=opts)
M._generate = _det_generate


def load_questions(path, book=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if book and not str(r.get("book", "")).lower().startswith(book.lower()):
                continue
            q = r.get("question") or r.get("q") or r.get("query")
            if q:
                rows.append(q)
    return rows


def run_one(col, q):
    """跑一题：检索 → 生成(带真页标签) → 引用校验。返回 (answer, cite_check, abstained)。"""
    emb = M.embed([q])
    res = col.query(query_embeddings=emb, n_results=M.TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    ans, _, pidx = M._run_once(docs, q, M.CONTEXT_BUDGET, metas)
    # 动态升配：与 ask 同策略（首答拒答且检索有命中 → 升到 1800 重答）
    if M.DYNAMIC_BUDGET and docs and M.is_abstain(ans):
        a2, _, pidx2 = M._run_once(docs, q, M.BUDGET_ESCALATED, metas)
        if not M.is_abstain(a2):
            ans, pidx = a2, pidx2
    cc = M.verify_citations(ans, pidx, metas)
    return ans, cc, M.is_abstain(ans)


def main():
    eval_file = sys.argv[1] if len(sys.argv) > 1 else "eval_med.jsonl"
    book = sys.argv[2] if len(sys.argv) > 2 else None
    if not os.path.exists(eval_file):
        print("!! 找不到题集: %s（请在 data 目录运行，或传入正确路径）" % eval_file)
        return
    qs = load_questions(eval_file, book)
    if not qs:
        print("!! 题集为空: %s%s" % (eval_file, ("  book=%s" % book) if book else ""))
        return

    col = M.get_collection()
    print("=" * 56)
    print("引用准确度评测  |  题集: %s%s" % (eval_file, ("  book=%s" % book) if book else ""))
    print("语料库: %d chunks  |  题数: %d  |  determinism: temp=0, seed=%d" % (col.count(), len(qs), SEED))
    print("=" * 56)

    n_abstain = 0
    n_with_cite = 0
    sum_total = sum_hit = sum_bad = 0
    bad_questions = []       # 有编造引用的题
    nocite_questions = []    # 作答但没标引用的题
    t0 = time.time()

    for i, q in enumerate(qs, 1):
        try:
            ans, cc, abstained = run_one(col, q)
        except Exception as e:
            print("  [%2d] ERROR: %s" % (i, str(e)[:80]))
            continue
        if abstained:
            n_abstain += 1
            print("  [%2d] 拒答  | %s" % (i, q[:52]))
            continue
        sum_total += cc["total"]; sum_hit += len(cc["hit"]); sum_bad += len(cc["fabricated"])
        if cc["total"] == 0:
            nocite_questions.append(q)
            print("  [%2d] 作答·无引用            | %s" % (i, q[:52]))
        else:
            n_with_cite += 1
            flag = "OK " if cc["ok"] else "编造!"
            print("  [%2d] 作答·引用%d/%d %s | %s" % (i, len(cc["hit"]), cc["total"], flag, q[:44]))
            if not cc["ok"]:
                bad_questions.append((q, cc["fabricated"]))

    n_answered = len(qs) - n_abstain
    print("\n" + "=" * 56)
    print("结果汇总")
    print("-" * 56)
    print("题数: %d  |  作答: %d  |  拒答: %d" % (len(qs), n_answered, n_abstain))
    print("作答题中: 含引用 %d  |  无引用 %d" % (n_with_cite, len(nocite_questions)))
    print("-" * 56)
    print("正文总引用数: %d  |  命中检索来源: %d  |  疑似编造: %d" % (sum_total, sum_hit, sum_bad))
    if sum_total:
        print(">>> 引用准确率（命中/总引用）: %d/%d = %.1f%%" % (sum_hit, sum_total, 100.0 * sum_hit / sum_total))
    else:
        print(">>> 引用准确率: 无引用，N/A")
    if n_with_cite:
        clean = n_with_cite - len(bad_questions)
        print(">>> 无编造题占比（含引用的题中）: %d/%d = %.1f%%" % (clean, n_with_cite, 100.0 * clean / n_with_cite))
    print("耗时: %.1fs" % (time.time() - t0))

    if bad_questions:
        print("\n[需人工核查] 出现疑似编造引用的题:")
        for q, bad in bad_questions:
            print("  - \"%s\" → 编造 %s" % (q[:60], "、".join(bad)))
    else:
        print("\n✓ 无一题出现编造引用（所有正文引用都命中检索来源）")

    # 存结果
    out = {
        "eval_file": eval_file, "book": book, "n_questions": len(qs),
        "n_answered": n_answered, "n_abstain": n_abstain, "n_with_cite": n_with_cite,
        "total_citations": sum_total, "hit": sum_hit, "fabricated": sum_bad,
        "citation_accuracy": (sum_hit / sum_total) if sum_total else None,
        "clean_rate": ((n_with_cite - len(bad_questions)) / n_with_cite) if n_with_cite else None,
        "bad_questions": [{"q": q, "fabricated": b} for q, b in bad_questions],
    }
    res_name = "cite_eval_result_%s.json" % os.path.splitext(os.path.basename(eval_file))[0]
    with open(res_name, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已存: %s" % res_name)


if __name__ == "__main__":
    main()
