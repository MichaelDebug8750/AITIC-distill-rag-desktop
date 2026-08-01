# -*- coding: utf-8 -*-
r"""
verify_vl.py —— _vl_to_prose 的三段式验证

背景：
    VL 输出形如「1. **Anterior fontanelle**」的编号标签行，会被 semantic_chunks
    的 HEADING_RE 判为标题而整行丢弃，图内标签全部进不了检索空间。
    _vl_to_prose 把这类行合并成「Labels shown in this figure: A; B; C.」再入库。
    该改动至今没有被端到端验证过——全量评测跑的都是 --no-vl。

为什么不能只靠现有评测：
    4432 道题全部从**文本**抽取，一道图题都没有。跑 --use-vl 只能证明
    "VL 没把文本题弄坏"，证明不了"图内标签真的进了检索空间、真的能被检索到"。
    所以这里分三段：

    第一段 转写存活率（离线，零 VL 调用）
        直接读 vl_cache.json 里的 VL 原文，分别过
          semantic_chunks(原文)          <- 关掉转写
          semantic_chunks(_vl_to_prose)  <- 打开转写
        统计标签文本的存活率。这是 _vl_to_prose 本身的判据。

    第二段 检索可达性（需要一个开了 VL 的库）
        从存活的标签里抽样，对库做检索，看能否召回对应的 FIGURE 块。
        存活 ≠ 可检索，必须单独验。

    第三段 回归保护（交给 run_eval_batch）
        同一本书 --no-vl vs --use-vl 跑同一套题，判据是文本题不得退化。
        命令见文末。

用法（在 data\ 目录下跑，那里有 vl_cache.json 和 vectordb）：
    C:\Users\Seifer\distill\Scripts\python.exe verify_vl.py                 # 第一段
    C:\Users\Seifer\distill\Scripts\python.exe verify_vl.py --retrieval     # 第一+二段
    C:\Users\Seifer\distill\Scripts\python.exe verify_vl.py --book Anatomy  # 只看某本
"""
import os, re, sys, json, argparse, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M

LABEL_LINE = re.compile(r"^\s*\d+[\.\)]\s+(.*)$")


def labels_in(vtxt):
    """VL 原文里的编号标签项（清掉 markdown 星号）。"""
    out = []
    for line in (vtxt or "").split("\n"):
        m = LABEL_LINE.match(line.strip())
        if m:
            it = re.sub(r"\*+", "", m.group(1)).strip(" -–—:")
            if it:
                out.append(it)
    return out


def chunk_text(vtxt, page, use_prose):
    """复刻 build() 第 613 行的入库路径，返回拼接后的块文本。"""
    body = M._vl_to_prose(vtxt) if use_prose else vtxt
    return "\n".join(M.semantic_chunks("FIGURE p%d: %s" % (page, body)))


def survived(label, blob):
    """标签是否存活在块文本里。宽松比对：忽略大小写与多余空白。"""
    a = re.sub(r"\s+", " ", label.lower()).strip()
    b = re.sub(r"\s+", " ", blob.lower())
    if not a:
        return False
    if a in b:
        return True
    # split_sentences 把 "Fig." 当句末，重拼时不补空格，导致 "Fig. 1.4" -> "Fig.1.4"。
    # 内容没丢，只是空格没了，所以再做一次忽略全部空白的比对，避免测量假阳性。
    if re.sub(r"\s+", "", label.lower()) in re.sub(r"\s+", "", blob.lower()):
        return True
    # 标签可能被分块切断，退一步看主要词是否都在
    ws = [w for w in re.split(r"[^a-z0-9]+", a) if len(w) > 3]
    return bool(ws) and all(w in b for w in ws)


def part1(cache, book_filter):
    print("=" * 74)
    print("第一段：转写存活率（离线，零 VL 调用）")
    print("=" * 74)
    keys = sorted(k for k in cache if "::p" in k)
    if book_filter:
        keys = [k for k in keys if book_filter.lower() in k.lower()]
    if not keys:
        print("  vl_cache.json 里没有匹配的条目。")
        print("  需要先跑一次开 VL 的建库，例如：")
        print("    python main.py build --pdf \"...\\Anatomy_and_Physiology_2e.pdf\" --max-pages 999999 --vl-limit 15")
        return []

    per_book = {}
    rows = []
    for k in keys:
        book, ptxt = k.rsplit("::p", 1)
        try:
            page = int(ptxt)
        except ValueError:
            continue
        vtxt = cache[k]
        labs = labels_in(vtxt)
        if not labs:
            continue
        off = chunk_text(vtxt, page, use_prose=False)
        on = chunk_text(vtxt, page, use_prose=True)
        s_off = sum(1 for l in labs if survived(l, off))
        s_on = sum(1 for l in labs if survived(l, on))
        rows.append(dict(book=book, page=page, n_labels=len(labs),
                         surv_off=s_off, surv_on=s_on,
                         chunks_off=len(M.semantic_chunks("FIGURE p%d: %s" % (page, vtxt))),
                         chunks_on=len(M.semantic_chunks("FIGURE p%d: %s" % (page, M._vl_to_prose(vtxt)))),
                         labels=labs, blob_on=on))
        d = per_book.setdefault(book, dict(pages=0, labs=0, off=0, on=0))
        d["pages"] += 1; d["labs"] += len(labs); d["off"] += s_off; d["on"] += s_on

    print("\n  %-40s %5s %7s %9s %9s" % ("书", "含图页", "标签数", "关转写存活", "开转写存活"))
    T = dict(pages=0, labs=0, off=0, on=0)
    for b, d in sorted(per_book.items()):
        print("  %-40s %5d %7d %6d(%3.0f%%) %6d(%3.0f%%)" % (
            b[:40], d["pages"], d["labs"],
            d["off"], 100.0 * d["off"] / max(1, d["labs"]),
            d["on"], 100.0 * d["on"] / max(1, d["labs"])))
        for k in T:
            T[k] += d[k]
    if T["labs"]:
        print("  " + "-" * 72)
        print("  %-40s %5d %7d %6d(%3.0f%%) %6d(%3.0f%%)" % (
            "合计", T["pages"], T["labs"],
            T["off"], 100.0 * T["off"] / T["labs"],
            T["on"], 100.0 * T["on"] / T["labs"]))
        gain = 100.0 * (T["on"] - T["off"]) / T["labs"]
        print("\n  标签存活率提升 %+.1fpp（丢失率 %.1f%% -> %.1f%%）" % (
            gain, 100.0 - 100.0 * T["off"] / T["labs"], 100.0 - 100.0 * T["on"] / T["labs"]))
        print("  判据：关转写时丢失率应显著为正（对应文档里记的 35.5%），开转写后应接近 0。")

    lost = [r for r in rows if r["surv_on"] < r["n_labels"]]
    if lost:
        print("\n  -- 开转写后仍有丢失的页（最多 5 页）--")
        for r in lost[:5]:
            miss = [l for l in r["labels"] if not survived(l, r["blob_on"])]
            print("     %s p%d  %d/%d 存活，丢失: %s" % (
                r["book"][:30], r["page"], r["surv_on"], r["n_labels"], miss[:4]))
    return rows


def part2(rows, n_probe):
    print("\n" + "=" * 74)
    print("第二段：检索可达性（需要当前 vectordb 是开了 VL 建的）")
    print("=" * 74)
    try:
        col = M.get_collection()
    except SystemExit as e:
        print("  取不到库：%s" % e); return
    fp = M.library_fingerprint()
    print("  库指纹 %s ｜ 块数 %s ｜ vl_prose=%s" % (
        fp.get("library_chunk_sha"), fp.get("library_n_chunks"), fp["runtime"].get("vl_prose")))
    parts = fp.get("library_parts") or []
    if parts and not any(p.get("use_vl") for p in parts):
        print("  !! 这个库是 --no-vl 建的，里面没有 FIGURE 块，第二段没有意义。")
        print("     先用 --vl-limit N 重建一次再跑。")
        return

    pool = [(r["book"], r["page"], l) for r in rows for l in r["labels"] if survived(l, r["blob_on"])]
    if not pool:
        print("  没有可探测的标签。"); return
    random.seed(0)
    probes = random.sample(pool, min(n_probe, len(pool)))
    hit_fig = hit_page = 0
    print("\n  %-34s %-6s %-6s %s" % ("标签（检索词）", "命中图块", "同页", "最优距离"))
    for book, page, lab in probes:
        try:
            qe = M.embed([lab])[0]
            res = col.query(query_embeddings=[qe], n_results=5,
                            include=["documents", "metadatas", "distances"])
        except Exception as e:
            print("  检索失败 %s" % e); return
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        is_fig = any(d.startswith("FIGURE p") for d in docs)
        same_pg = any(str(m.get("page")) == str(page) for m in metas)
        hit_fig += int(is_fig); hit_page += int(same_pg)
        print("  %-34s %-6s %-6s %.4f" % (lab[:34], "是" if is_fig else "否",
                                          "是" if same_pg else "否", dists[0]))
    n = len(probes)
    print("\n  Top5 含 FIGURE 块 %d/%d = %.0f%% ｜ 命中原页 %d/%d = %.0f%%" %
          (hit_fig, n, 100.0 * hit_fig / n, hit_page, n, 100.0 * hit_page / n))
    print("  判据：标签作为查询时应能召回其所在图的 FIGURE 块。")
    print("        命中率低说明标签虽然入了库，但语义检索够不到——存活 != 可检索。")


def explain(cache, spec):
    """--explain 书关键词:页号  打印该页的 VL 原文与两条分块路径的结果，逐条标注存活。"""
    bk, _, pg = spec.rpartition(":")
    hit = [k for k in cache if bk.lower() in k.lower() and k.endswith("::p" + pg)]
    if not hit:
        print("没找到 %s（缓存里的键形如 '书名.pdf::p27'）" % spec); return
    k = hit[0]
    vtxt = cache[k]
    page = int(pg)
    print("=" * 74); print("键: %s" % k); print("=" * 74)
    print("\n---- VL 原文 ----"); print(vtxt[:1800])
    labs = labels_in(vtxt)
    print("\n---- 抽出的编号标签 %d 个 ----" % len(labs)); print(labs)
    for tag, use in (("关转写", False), ("开转写", True)):
        body = M._vl_to_prose(vtxt) if use else vtxt
        chunks = M.semantic_chunks("FIGURE p%d: %s" % (page, body))
        blob = "\n".join(chunks)
        print("\n---- %s：%d 块 ----" % (tag, len(chunks)))
        for i, c in enumerate(chunks):
            print("  [%d] %s" % (i, c[:200].replace("\n", " / ")))
        print("  存活: %s" % ["%s=%s" % (l, "Y" if survived(l, blob) else "N") for l in labs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", default="", help="书关键词:页号，打印该页两条路径的完整对比")
    ap.add_argument("--book", default="", help="只看书名含此关键词的条目")
    ap.add_argument("--retrieval", action="store_true", help="同时跑第二段检索验证")
    ap.add_argument("--probes", type=int, default=12, help="第二段抽多少个标签做检索")
    ap.add_argument("--dump", default="", help="把逐页明细写到这个 json")
    a = ap.parse_args()

    print("VL 缓存 : %s" % os.path.abspath(M.VL_CACHE))
    print("VL_PROSE: %s（DISTILL_VL_PROSE=0 可关闭）" % M.VL_PROSE)
    if not M.VL_PROSE:
        print("!! 当前环境变量把转写关掉了，第一段的\"开转写\"列会和\"关转写\"一样。")
    cache = M.load_vl_cache()
    print("缓存条目: %d\n" % len(cache))
    if a.explain:
        explain(cache, a.explain); return

    rows = part1(cache, a.book)
    if a.retrieval and rows:
        part2(rows, a.probes)
    if a.dump and rows:
        for r in rows:
            r.pop("blob_on", None)
        json.dump(rows, open(a.dump, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n逐页明细已写入 %s" % a.dump)

    print("\n" + "=" * 74)
    print("第三段：回归保护（另跑，判据是文本题不得退化）")
    print("=" * 74)
    print(r"""  $py  = "C:\Users\Seifer\distill\Scripts\python.exe"
  $reb = "E:\Ollama_test\run_eval_batch.py"
  $c = @('--books','E:\Ollama_test\books','--eval','E:\Ollama_test\eval\eval_by_book',
         '--main','E:\Ollama_test\data\main.py','--workdir','E:\Ollama_test\data',
         '--max-pages','999999')
  # 对照组已有：eval_results_v7full（--no-vl）
  & $py $reb @c --out E:\Ollama_test\eval_results_v7_vl --use-vl --vl-limit 15 --only Anatomy
  & $py $reb @c --out E:\Ollama_test\eval_results_v7_vl --use-vl --vl-limit 15 --only Microbiology
  然后：python compare_runs.py eval_results_v7full eval_results_v7_vl --flips 20
  判据：可答/模糊/拒答的差异不超过 ±1.5pp（既往噪声底）。""")


if __name__ == "__main__":
    main()
