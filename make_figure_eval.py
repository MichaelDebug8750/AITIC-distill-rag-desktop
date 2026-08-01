# -*- coding: utf-8 -*-
r"""
make_figure_eval.py —— 从 VL 缓存生成"图题"题集

为什么需要它：
    现有 4432 题全部从**文本层**抽取，一道图题都没有。所以 --use-vl 与 --no-vl
    的同题对照只能证明"开 VL 不损害准确性"（实测 190 题逐项持平），
    永远证明不了"VL 有没有用"。题集在结构上就测不到 VL 的贡献。

    这个脚本用 VL 缓存里的图内标签造题，且只保留**文本层完全没有**的标签——
    那才是 VL 的真实增量。

【必读的局限，写报告时不能省】
    题目来源是 VL 自己的输出，而 VL 输出未经人工核对。
    所以本题集测的是「VL 抽到的内容能不能被检索到」（可检索性），
    **不是**「图里到底画了什么」（正确性）。
    若 VL 把某个标签读错了，据此造的题，其"正确答案"也是错的。
    ==> 用于报告前必须人工抽检，脚本会生成 figure_eval_review.md 与页面截图供核对。

过滤五层（顺序执行，逐层收紧）：
    L1 元描述剥离：VL 常写成「Deuterium (²H) – Label beneath the second circular diagram」，
                   破折号后是 VL 在解释这个标签是什么，不是标签本身 → 只取破折号前
    L2 出版样板：access for free / openstax / credit: / ISBN / Rice University ...
    L3 无信息量：图号(Fig. 2.5)、面板标记((a))、纯符号数字、过短过长
    L4 文本层去重：标签在原书 PDF 文本里出现过 → VL 无增量，丢弃（核心过滤）
    L5 去重限额：全局去重、每图最多 N 个标签、每书最多 M 道题

用法（在 E:\Ollama_test 下）：
    C:\Users\Seifer\distill\Scripts\python.exe make_figure_eval.py ^
        --cache data\vl_cache.json --books books --out eval\eval_figure ^
        --review figure_eval_review.md --dump-pages figure_pages
"""
import os, re, io, sys, json, glob, argparse

# ---------------- L1 元描述剥离 ----------------
# VL 惯用 "标签 – 解释" / "标签 - 解释" / "标签: 解释" 的写法，破折号后是解释不是标签
META_TAIL = re.compile(
    r"\s*[–—-]\s*(label|sub-?label|title|description|caption|note|text|indicat|denot|identif|"
    r"shows?|showing|marking|beneath|below|above|on the|for the)\b.*$", re.I)
PAREN_META = re.compile(
    r"\s*\((structural line label|anatomical label[^)]*|label[^)]*|on the [^)]*)\)\s*$", re.I)

# ---------------- L2 出版样板 ----------------
BOILER = re.compile(
    r"access for free|openstax|cnx\.org|credit\s*:|creative commons|©|\(c\)\s*20|isbn|"
    r"rice university|all rights reserved|download for free|chapter title|figure caption|"
    r"scale bar|source\s*:|adapted from|courtesy of|https?://", re.I)

# ---------------- L3 无信息量 ----------------
NOINFO = [
    re.compile(r"^\(?[a-z]\)?[\.\)]?$", re.I),            # (a) / b) / c
    re.compile(r"^fig(ure)?\.?\s*\d+([\.\-]\d+)*[a-z]?$", re.I),
    # 「FIGURE 2.5 Isotopes of Hydrogen」是图注标题，不是图内标签。
    # 图注通常也在文本层，L4 本会兜住，但显式丢弃更干净，也避免题干泄题。
    re.compile(r"^fig(ure)?\.?\s*\d+([\.\-]\d+)*[a-z]?\s+\S", re.I),
    re.compile(r"^table\s*\d", re.I),
    re.compile(r"^chapter\s*\d", re.I),
    re.compile(r"^[\W\d_]+$"),                            # 纯符号数字
    re.compile(r"^(panel|part|step|item|number|label)\s*\d*$", re.I),
]
LABEL_LINE = re.compile(r"^\s*\d+[\.\)]\s+(.*)$")
# VL 自己的脚手架标题，不是图题。用它当题干会造出「Extracted Text Labels from Figure
# labeled structures」这种垃圾题——题干必须来自图的真实主题，否则整道题没有语义。
# 输出端强校验：不管输入侧漏了什么，题干本身过不了这一关就不出题。
# 实测教训：只靠输入侧的 SCAFFOLD 名单，63% 的题干仍是 VL 脚手架
# （"Description of the Figure"、"Additional Context on Content Type"、
#  "Text Labels Extracted from Figures/Diagrams/Tables" …花样远超预期）。
# 白名单式校验比黑名单式过滤稳健：题干必须看起来像一个真实主题短语。
BAD_STEM = re.compile(
    r"description|extracted|text\s+labels?|labels?\s+(from|in|of)\s+figur|"
    r"additional\s+context|content\s+type|figure\s+context|brief\s+desc|"
    r"what\s+(the|each)\s+figure|identifies?\s+the\s+figure|this\s+label|"
    r"common\s+conven|user\s+interface|section\s+header|"
    # 以下为第二轮实测漏网的写法，VL 的标题是开放集合，这里只补已观测到的
    r"key\s+(context|clarification|point)|labels?\s+(by|within)\s+|overall\s+figure|"
    r"^location\s*:|^from\s+fig|figure\s+purpose|navigation\s+tools", re.I)


def valid_stem(q):
    """题干必须像个真实主题短语，而不是 VL 的排版脚手架。"""
    q = (q or "").strip()
    if len(q) < 12 or len(q) > 240:
        return False
    if BAD_STEM.search(q):
        return False
    if not re.match(r"^[A-Za-z]", q):          # 以标点/括号/数字开头的都是残片
        return False
    if re.match(r"^it\s*[:\-]", q, re.I):      # 挖空挖到了句首，成了 "it: ..."
        return False
    if len(re.findall(r"[A-Za-z]{3,}", q)) < 3:
        return False
    return True


SCAFFOLD = re.compile(
    r"^(#+\s*)?(extracted\s+)?(text\s+)?labels?\s+(from|in|of)\b|"
    r"^(#+\s*)?brief\s+descriptions?|^(#+\s*)?descriptions?\s+of\s+each|"
    r"^(#+\s*)?(figure|diagram)s?\s*/?\s*(diagram)?s?\s*:?\s*$|"
    r"^(#+\s*)?(image|figure)\s+(analysis|content|summary)|"
    r"^(#+\s*)?text\s+(content|elements)", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_label(raw):
    """L1：剥掉 VL 的元描述，留下标签本体。返回 '' 表示应丢弃。"""
    s = re.sub(r"\*+", "", raw or "").strip()
    s = s.strip(" \t-–—:;.")
    s = PAREN_META.sub("", s)
    s = META_TAIL.sub("", s)
    s = s.strip().strip('"“”').strip(" \t-–—:;.")
    return s


def keep_label(s):
    """L2 + L3。"""
    if not s or not (4 <= len(s) <= 55):
        return False
    if BOILER.search(s):
        return False
    for p in NOINFO:
        if p.match(s):
            return False
    words = re.findall(r"[A-Za-z]{3,}", s)
    if len(words) < 1:
        return False
    # 整条像一句解释而不是标签
    if len(s.split()) > 8:
        return False
    return True


def parse_vl(vtxt):
    """把一页 VL 原文拆成 (标题, 标签列表, 描述句列表)。"""
    title, labels, descs = "", [], []
    for line in (vtxt or "").split("\n"):
        raw, t = line.rstrip(), line.strip()
        if not t:
            continue
        m = LABEL_LINE.match(t)
        if m:
            lab = clean_label(m.group(1))
            if lab:
                labels.append(lab)
            continue
        if raw[:1] in " \t" or re.match(r"^\*\s*(description|desc|caption|note)\b", t, re.I):
            body = re.sub(r"^\*\s*\w+\s*:?\s*\**", "", t).strip()
            body = re.sub(r"\*+", "", body).strip().lstrip("-–—: ")
            # 描述句必须是完整句子：字母开头、够长。否则会造出
            # "- Description: Indicates the initial change..." 这种题干。
            if len(body) > 40 and re.match(r"^[A-Z]", body):
                descs.append(body)
            continue
        plain = re.sub(r"^#+\s*", "", t).strip()
        plain = re.sub(r"^[-*]\s*", "", plain)
        if (not title and len(plain) < 90 and not BOILER.search(plain)
                and not SCAFFOLD.match(plain)):
            title = re.sub(r"\*+", "", plain).strip()
        plain2 = re.sub(r"\*+", "", plain).strip().lstrip("-–—: ")
        if (len(plain2) > 40 and not BOILER.search(plain2)
                and re.match(r"^[A-Z]", plain2) and not BAD_STEM.search(plain2)):
            descs.append(plain2)
    return title, labels, descs


def cut(s, n):
    """按词边界截断，绝不切在词中间。
    实测教训：ctx[:70] / q[:300] 硬截断造出「six types o」「humerus bone, hi」
    这类题干，21 道题里 5 道因此报废——损耗主要在截断，不在过滤。"""
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cutp = s[:n]
    sp = cutp.rfind(" ")
    return (cutp[:sp] if sp > n * 0.5 else cutp).rstrip(" ,;:-–—")


# 挖空句常带的排版前缀，必须剥掉，否则题干变成「Figure 2.8(b): Depicts…」
STEM_PREFIX = re.compile(
    # [a-z]\b 而不是 [a-z]? —— 后者会把 "FIGURE 2.24 Structure" 的 S 一起吃掉
    r"^\s*((figure|fig\.?|table)\s*[\d\.\-]+\s*(?:[a-z]\b)?\s*(\([a-z]\))?\s*[:：]?\s*"
    r"|panel\s*\([a-z]\)\s*[:：]\s*|\([a-z]\)\s*[:：]\s*)", re.I)


def strip_prefix(q):
    prev = None
    while prev != q:
        prev = q
        q = STEM_PREFIX.sub("", q).strip()
    # 剥完可能剩下「Structure of an Amino Acid: This diagram…」，冒号前是图题，留正文
    return q


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


# ---------------- L4 语义级去重 ----------------
# 教训：只做精确字符串比对，等于重蹈不可答探针 verified-0-occurrences 的覆辙。
# 实例：图内标签 "Frontal (coronal) plane" 字面不出现，但正文明写
#       "The frontal plane ... often referred to as a coronal plane"。
#       结果 L4 把真正的解剖名词（矢状面/横断面）全滤掉，只留下 VL 加了括号
#       注解的那些——题集测的成了"VL 写法特殊的字符串"，不是"图内独有的信息"。
# 因此除字面比对外，再加两道：去括号后的核心短语、内容词同窗共现。
_L4_STOP = set("the a an and or of in on at to for with by from is are was were "
               "left right upper lower part parts view figure label labels".split())


def _core_words(label):
    t = re.sub(r"\([^)]*\)", " ", label or "")      # 去掉括号注解
    t = re.sub(r"[^A-Za-z0-9\s]", " ", t).lower()
    return [w for w in t.split() if len(w) > 2 and w not in _L4_STOP]


def in_text_layer(label, full, full_ns, win=260):
    """标签是否在正文里出现过——字面、去空格、去括号核心短语、内容词同窗共现。
       任一命中即认为正文已有该信息，该标签不能用来出'图内独有'的题。"""
    a = norm(label)
    if not a:
        return True
    if a in full or norm_ns(label) in full_ns:
        return True
    core = _core_words(label)
    if not core:
        return True                                   # 只剩符号数字，没信息量
    core_phrase = " ".join(core)
    if core_phrase in full:
        return True
    if not all(w in full for w in core):              # 有词根本没出现 -> 真独有
        return False
    # 所有词都出现了，再看是否挤在同一段窗口里（挤在一起=正文确实在讲这件事）
    anchor = max(core, key=len)
    for m in re.finditer(re.escape(anchor), full):
        seg = full[max(0, m.start() - win): m.start() + win]
        if all(w in seg for w in core):
            return True
    return False


def norm_ns(s):
    return re.sub(r"\s+", "", (s or "").lower())


def load_pdf_text(path):
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


def find_book(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    stem = os.path.splitext(name)[0][:30]
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith(stem):
                return os.path.join(dirpath, f)
    return None


def subject_of(path, books_root):
    rel = os.path.relpath(path, books_root)
    parts = rel.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("data", "vl_cache.json"))
    ap.add_argument("--books", default="books")
    ap.add_argument("--out", default=os.path.join("eval", "eval_figure"))
    ap.add_argument("--review", default="figure_eval_review.md")
    ap.add_argument("--dump-pages", default="", help="把出题页渲染成 PNG 存到这个目录，供人工核对")
    ap.add_argument("--max-per-figure", type=int, default=4, help="每张图最多取几个标签")
    ap.add_argument("--max-per-book", type=int, default=60)
    a = ap.parse_args()

    cache = json.load(io.open(a.cache, encoding="utf-8"))
    print("VL 缓存条目 %d" % len(cache))

    by_book = {}
    for k, v in cache.items():
        if "::p" not in k:
            continue
        book, ptxt = k.rsplit("::p", 1)
        try:
            by_book.setdefault(book, {})[int(ptxt)] = v
        except ValueError:
            pass

    stats = {"raw": 0, "L1L2L3": 0, "L4": 0, "final": 0}
    all_rows, review = [], []

    for book, pages in sorted(by_book.items()):
        path = find_book(a.books, book)
        if not path:
            print("  !! 找不到原书，跳过：%s" % book)
            continue
        print("  处理 %s（%d 个含图页）..." % (book[:44], len(pages)), flush=True)
        try:
            ptexts = load_pdf_text(path)
        except Exception as e:
            print("     解析失败：%s" % e)
            continue
        FULL, FULL_NS = norm(" ".join(ptexts)), norm_ns(" ".join(ptexts))
        subj = subject_of(path, a.books)

        rows, seen = [], set()
        for pg in sorted(pages):
            title, labels, descs = parse_vl(pages[pg])
            stats["raw"] += len(labels)
            kept = [l for l in labels if keep_label(l)]
            stats["L1L2L3"] += len(kept)
            # L4 文本层去重：这一步决定了题目是否真的只能靠图回答
            vl_only = [l for l in kept if not in_text_layer(l, FULL, FULL_NS)]
            stats["L4"] += len(vl_only)
            # L5 去重限额
            uniq = []
            for l in vl_only:
                if norm(l) in seen:
                    continue
                seen.add(norm(l))
                uniq.append(l)
                if len(uniq) >= a.max_per_figure:
                    break
            if not uniq:
                continue

            kws = [norm(l) for l in uniq]

            # 题型一：完形填空（有描述句才出，语境完整，与现有 fuzzy_desc 同构）
            made_cloze = False
            for l in uniq:
                for sent in descs:
                    if norm(l) in norm(sent) and len(sent) > 50:
                        # 连同前置冠词一起替换，否则会造出 "The it sits superior to..."
                        q = re.sub(r"\b(the|a|an)\s+" + re.escape(l), "it",
                                   sent, count=1, flags=re.I)
                        if norm(l) in norm(q):          # 没冠词，直接替换
                            q = re.sub(re.escape(l), "it", sent, count=1, flags=re.I)
                        q = strip_prefix(q.strip())
                        # 语法闸门：挖空落在句首或造出 "its are" 一律弃用，
                        # 残句题干答对答错都说明不了问题
                        if re.match(r"^(it|its)\b", q, re.I) and not re.match(
                                r"^it\s+(is|was|has|had|can|divides|shows|sits|contains|"
                                r"consists|refers|represents|serves|forms|acts|allows)\b", q, re.I):
                            continue
                        if re.search(r"\bits\s+are\b", q, re.I):
                            continue
                        q = cut(q, 260)
                        if not q.endswith((".", "?", "!")):
                            q += "."
                        if not valid_stem(q):
                            continue
                        rows.append(dict(book=book, subject=subj, type="fuzzy_desc",
                                         expect="answer", question=q.strip(),
                                         keywords=[norm(l)], evidence=sent[:400], page=pg,
                                         source="vl-figure-cloze (unverified)",
                                         term=l, figure_title=title))
                        made_cloze = True
                        break
                if made_cloze:
                    break

            # 题型二：每张图一道"图题检索题"，命中任一独有标签即算
            ctx = title or (descs[0][:70] if descs else "")
            ctx = re.sub(r"^(figure|fig\.?)\s*[\d\.\-]*\s*", "", ctx, flags=re.I).strip()
            # 没有真实图题就不出这道题——题干无语义的题，答对答错都说明不了什么
            ctx = strip_prefix(ctx).strip(" :：-–—")
            # 只有"短标题"才配当检索题的题干。描述句当题干会造出
            # 「This figure shows a labeled anatomical diagram of the humerus bone, hi labeled structures」
            ok_ctx = (ctx and 2 <= len(ctx.split()) <= 8 and not SCAFFOLD.match(ctx)
                      and not BAD_STEM.search(ctx) and not ctx.endswith(","))
            stem = "%s labeled structures" % ctx
            if ok_ctx and valid_stem(stem):
                rows.append(dict(book=book, subject=subj, type="fuzzy_kw",
                                 expect="answer",
                                 question=stem,
                                 keywords=kws, evidence=" ".join(uniq)[:400], page=pg,
                                 source="vl-figure-labels (unverified)",
                                 term="; ".join(uniq), figure_title=title))
            review.append(dict(book=book, page=pg, title=title,
                               labels=uniq, all_labels=labels, descs=descs[:2]))

        rows = rows[:a.max_per_book]
        stats["final"] += len(rows)
        if not rows:
            print("     无可用标签，跳过")
            continue
        os.makedirs(a.out, exist_ok=True)
        fn = re.sub(r"[^A-Za-z0-9]+", "_", book)[:60] + ".jsonl"
        io.open(os.path.join(a.out, fn), "w", encoding="utf-8").write(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print("     出题 %d 道 -> %s" % (len(rows), fn))
        all_rows += rows

    print("\n== 过滤漏斗 ==")
    print("  VL 原始编号项            %5d" % stats["raw"])
    print("  L1-L3 剥元描述/去样板后   %5d  (%.0f%%)" % (stats["L1L2L3"], 100.0 * stats["L1L2L3"] / max(1, stats["raw"])))
    print("  L4 文本层去重后（VL独有） %5d  (%.0f%%)" % (stats["L4"], 100.0 * stats["L4"] / max(1, stats["raw"])))
    print("  L5 限额后最终出题         %5d 道" % stats["final"])

    # 页面截图，供人工核对"图里到底有没有这个标签"
    if a.dump_pages and review:
        try:
            import fitz
            os.makedirs(a.dump_pages, exist_ok=True)
            done = 0
            for r in review:
                p = find_book(a.books, r["book"])
                if not p or not p.lower().endswith(".pdf"):
                    continue
                d = fitz.open(p)
                if 1 <= r["page"] <= len(d):
                    png = os.path.join(a.dump_pages, "%s_p%d.png" % (
                        re.sub(r"[^A-Za-z0-9]+", "_", r["book"])[:40], r["page"]))
                    d[r["page"] - 1].get_pixmap(dpi=110).save(png)
                    done += 1
                d.close()
            print("  页面截图 %d 张 -> %s\\" % (done, a.dump_pages))
        except Exception as e:
            print("  截图失败（不影响题集）：%s" % e)

    # 人工复核清单
    with io.open(a.review, "w", encoding="utf-8") as f:
        f.write("# 图题人工复核清单\n\n")
        f.write("**这批题来自 VL 自己的输出，VL 输出未经核对。**\n")
        f.write("本题集测的是「VL 抽到的内容能否被检索到」，不是「图里到底画了什么」。\n")
        f.write("用于报告前，请对照页面截图核对下面的标签是否真的出现在图中。\n\n")
        f.write("抽检建议：每本随机看 5 页，若标签准确率 <80%，这批题不可用于对外报告。\n\n")
        f.write("过滤漏斗：VL 原始 %d 项 -> 去噪 %d -> VL独有 %d -> 最终出题 %d 道\n\n"
                % (stats["raw"], stats["L1L2L3"], stats["L4"], stats["final"]))
        for r in review:
            f.write("---\n\n## %s p%d\n\n" % (r["book"], r["page"]))
            if r["title"]:
                f.write("- 图题：%s\n" % r["title"])
            f.write("- **入选标签（文本层没有，据此出题）**：%s\n" % "、".join(r["labels"]))
            dropped = [l for l in r["all_labels"] if l not in r["labels"]]
            if dropped:
                f.write("- 被过滤掉的：%s\n" % "、".join(dropped[:10]))
            f.write("\n核对：[ ] 标签准确　[ ] 标签有误　[ ] 图里没有\n\n")
    print("  人工复核清单 -> %s" % a.review)

    if all_rows:
        from collections import Counter
        print("\n题型分布：%s" % dict(Counter(r["type"] for r in all_rows)))
        print("\n跑法（两轮，VL 组与对照组跑同一批题）：")
        tpl = ("  & $py E:\\Ollama_test\\run_eval_batch.py "
               "--books E:\\Ollama_test\\books --eval %s "
               "--main E:\\Ollama_test\\data\\main.py --workdir E:\\Ollama_test\\data "
               "--max-pages 999999 --out %s%s")
        print(tpl % (os.path.abspath(a.out), "E:\\Ollama_test\\eval_results_fig_vl",
                     " --use-vl --vl-limit 30"))
        print(tpl % (os.path.abspath(a.out), "E:\\Ollama_test\\eval_results_fig_novl", ""))
        print("  然后：python compare_runs.py eval_results_fig_novl eval_results_fig_vl")
        print("  VL 若真有增量，差异应显著为正；若接近 0，说明 VL 块检索不到。")


if __name__ == "__main__":
    main()
