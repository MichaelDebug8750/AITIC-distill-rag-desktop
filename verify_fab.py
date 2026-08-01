# -*- coding: utf-8 -*-
r"""
verify_fab.py —— 把"判为编造"的题拉回原书核对

为什么要做：
    不可答探针的构造判据是 source="cross-domain (verified 0 occurrences)"，
    即"术语字符串在书里出现 0 次"。这验的是**字面**，不是**语义**。
    像 "indirect method"（基底节 indirect pathway）、"latency period"（磁盘延迟）、
    "tree diagram"（数据结构的树）这些词，字面可能确实不出现，
    但概念在书里是真实存在的——模型基于真实检索内容作答，会被判成编造。
    幻觉率 3.6% 里有多少是模型的问题、多少是题目的问题，必须回原文才能分清。

它做什么：
    对每道判为 FABRICATED 的题
      1) 在原书全文里搜术语（原形 / 小写 / 去连字符 / 逐词共现）
      2) 抽出模型引用的那一页（或那一章）的原文，看是否支撑答案
      3) 输出人工复核清单 fab_review.md + 机器统计 fab_review.json

用法（在 E:\Ollama_test 下）：
    C:\Users\Seifer\distill\Scripts\python.exe verify_fab.py ^
        --results eval_results_v7full --books books --data data
参数：
    --results  评测结果目录
    --books    书库根目录（下面按学科分子目录）
    --data     放 main.py 的目录（借它的 EPUB 解析）
"""
import os, re, io, sys, json, glob, argparse


def find_book(books_root, name):
    """按文件名在书库里递归找原书。"""
    for dirpath, _, files in os.walk(books_root):
        if name in files:
            return os.path.join(dirpath, name)
    # 退一步：去掉扩展名做模糊匹配（评测里书名可能被截断过）
    stem = os.path.splitext(name)[0][:30]
    for dirpath, _, files in os.walk(books_root):
        for f in files:
            if f.startswith(stem):
                return os.path.join(dirpath, f)
    return None


def load_pdf_pages(path):
    import fitz
    d = fitz.open(path)
    pages = []
    for i in range(len(d)):
        try:
            pages.append(d[i].get_text())
        except Exception:
            pages.append("")
    d.close()
    return pages


def load_epub_pages(path, data_dir):
    """复用 main.py 的 _epub_blocks，保证与建库时的切法一致。"""
    sys.path.insert(0, data_dir)
    import main as M
    blocks = M._epub_blocks(path)
    # 返回 {章序号: 文本}
    out = {}
    for text, ch_idx, loc in blocks:
        out.setdefault(ch_idx, []).append(text)
    return {k: "\n".join(v) for k, v in out.items()}


def term_variants(term):
    t = (term or "").strip()
    if not t:
        return []
    v = {t, t.lower(), t.replace("-", " "), t.replace("-", ""), t.replace(" ", "-")}
    if t.lower().endswith("s"):
        v.add(t.lower()[:-1])
    else:
        v.add(t.lower() + "s")
    return [x for x in v if x]


def search_term(fulltext_lower, term):
    """返回 (字面命中次数, 逐词共现命中的段落数)。
       逐词共现 = 术语的每个词都在同一段里出现过，用来抓"概念在但用词不同"。"""
    exact = 0
    for v in term_variants(term):
        exact += fulltext_lower.count(v.lower())
    words = [w for w in re.split(r"[^a-z0-9]+", (term or "").lower()) if len(w) > 3]
    co = 0
    if words:
        for para in fulltext_lower.split("\n\n"):
            if all(w in para for w in words):
                co += 1
    return exact, co


_STOP = set(("the a an and or of to in is are was were be been for on with that this it its as by "
             "from at not but if then than which who whom what when where how there here their his "
             "her they them we you i material provided context refers refer mentioned mention "
             "according also can may such these those into other more most some any all one two "
             "each both").split())


def _content_words(t):
    return set(w for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if w not in _STOP)


def _grounding(answer, term, cited_text):
    """答案内容词落在引用页里的比例。剔除术语本身的词，避免自证。"""
    aw = _content_words(re.sub(r"\[[^\]]*\]", "", answer or "")) - _content_words(term)
    if not aw:
        return 0.0, 0
    return round(len(aw & _content_words(cited_text)) / len(aw), 3), len(aw)


CITE_PAGE = re.compile(r"\[p\.(\d+)\]")
CITE_CH = re.compile(r"\[ch(\d+)[:\]]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--books", default="books")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="fab_review")
    a = ap.parse_args()

    rows = []
    for f in glob.glob(os.path.join(a.results, "*.jsonl")):
        if os.path.basename(f).startswith("_"):
            continue
        for l in io.open(f, encoding="utf-8"):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    fab = [r for r in rows if r["verdict"] == "FABRICATED"]
    print("判为编造 %d 道，开始回原文核对..." % len(fab))

    cache = {}
    recs = []
    for i, r in enumerate(fab, 1):
        book = r["book"]
        print("  [%d/%d] %s" % (i, len(fab), book[:44]), flush=True)
        if book not in cache:
            p = find_book(a.books, book)
            if not p:
                cache[book] = ("missing", None, "")
            elif p.lower().endswith(".epub"):
                try:
                    d = load_epub_pages(p, a.data)
                    cache[book] = ("epub", d, "\n\n".join(d.values()).lower())
                except Exception as e:
                    cache[book] = ("error:%s" % e, None, "")
            else:
                try:
                    pg = load_pdf_pages(p)
                    cache[book] = ("pdf", pg, "\n\n".join(pg).lower())
                except Exception as e:
                    cache[book] = ("error:%s" % e, None, "")
        kind, pages, full = cache[book]
        term = r.get("term") or ""
        exact, co = search_term(full, term) if full else (-1, -1)

        # 抽引用页原文
        cited, cited_text = [], ""
        ans = r.get("answer") or ""
        if kind == "pdf" and pages:
            for m in CITE_PAGE.finditer(ans):
                n = int(m.group(1))
                cited.append(n)
                if 1 <= n <= len(pages):
                    cited_text += pages[n - 1][:900] + "\n---\n"
        elif kind == "epub" and pages:
            for m in CITE_CH.finditer(ans):
                n = int(m.group(1))
                cited.append(n)
                if n in pages:
                    cited_text += pages[n][:900] + "\n---\n"

        # 接地率：答案的内容词有多少真的出现在引用页里。
        # 低 = 答案不来自引用页（参数记忆编造，引用只是装饰）
        # 高 = 答案确实复述了引用页（则该页可能真讲了这个概念，是探针的问题）
        grounding, n_ans_w = _grounding(ans, term, cited_text)

        # 自动初判
        if kind.startswith("error") or kind == "missing":
            flag = "无法核对(%s)" % kind
        elif exact > 0:
            flag = "题目有问题：术语字面出现 %d 次" % exact
        elif grounding >= 0.6:
            flag = "题目可能有问题：答案 %.0f%% 复述引用页（概念在书里，只是用词不同）" % (100 * grounding)
        elif co > 0 and grounding >= 0.3:
            flag = "存疑：逐词共现 %d 段且答案 %.0f%% 复述引用页" % (co, 100 * grounding)
        else:
            flag = "真幻觉：答案仅 %.0f%% 来自引用页，引用为装饰" % (100 * grounding)
        recs.append(dict(book=book, subject=r.get("subject"), question=r["question"],
                         term=term, answer=ans, cited=cited, exact=exact, cooccur=co,
                         grounding=grounding, n_ans_words=n_ans_w,
                         flag=flag, cited_text=cited_text[:1500], src=r.get("source")))

    # 统计
    from collections import Counter
    import statistics as _st
    c = Counter(x["flag"].split("：")[0].split("(")[0] for x in recs)
    print("\n== 自动初判 ==")
    for k, v in c.most_common():
        print("  %-24s %2d 道  (%.0f%%)" % (k, v, 100.0 * v / max(1, len(recs))))
    gs = [x["grounding"] for x in recs]
    if gs:
        print("\n== 接地率（答案内容词落在引用页里的比例）==")
        print("  中位 %.2f ｜ 均值 %.2f" % (_st.median(gs), _st.mean(gs)))
        b = Counter("高 >=0.6" if g >= 0.6 else ("中 0.3-0.6" if g >= 0.3 else "低 <0.3") for g in gs)
        for k in ("高 >=0.6", "中 0.3-0.6", "低 <0.3"):
            print("  %-12s %2d 道  (%.0f%%)" % (k, b.get(k, 0), 100.0 * b.get(k, 0) / len(gs)))
        print("\n  解读：低接地 = 引用只是装饰，答案来自模型参数记忆，是真幻觉；")
        print("        高接地 = 答案确实复述了引用页，说明该页讲了这个概念，是探针的问题。")

    json.dump(recs, io.open(a.out + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    with io.open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("# 编造判定人工复核清单\n\n")
        f.write("共 %d 道。自动初判仅供排序，**最终结论以引用页原文为准**。\n\n" % len(recs))
        f.write("判据：引用页原文是否真的支撑答案。支撑=题目有问题；不支撑=真幻觉。\n\n")
        for x in sorted(recs, key=lambda y: -y["grounding"]):
            f.write("---\n\n## %s\n\n" % x["question"])
            f.write("- 书：`%s`（%s）\n- 术语：`%s`\n- 构题来源：%s\n" %
                    (x["book"], x["subject"], x["term"], x["src"]))
            f.write("- 字面出现 %s 次 ｜ 逐词共现 %s 段 ｜ 接地率 %.2f（答案 %d 个内容词）\n- **初判：%s**\n\n" %
                    (x["exact"], x["cooccur"], x["grounding"], x["n_ans_words"], x["flag"]))
            f.write("**模型答案**\n\n> %s\n\n" % x["answer"].replace("\n", " ")[:600])
            if x["cited_text"]:
                f.write("**引用页原文（截断）**\n\n```\n%s\n```\n\n" % x["cited_text"])
            else:
                f.write("*（答案里没有可解析的引用标记，或引用页超出范围）*\n\n")
            f.write("判定：[ ] 题目有问题　[ ] 真幻觉　[ ] 待定\n\n")
    print("\n已生成：\n  %s.md    <- 人工复核用，按可疑度排序\n  %s.json  <- 机器可读" % (a.out, a.out))


if __name__ == "__main__":
    main()
