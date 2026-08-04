# -*- coding: utf-8 -*-
"""标准检索 Hit@5 评测。

与答案关键词命中率不同，本脚本不调用生成模型，只检查问题的 top-5 检索结果
是否包含题集标注的 gold term。同时报告来源单元与证据文本诊断，并为每题
保存 top-5 文档、距离、来源和证据重合度。
"""
import argparse
import glob
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict


STOP = set((
    "the a an and or of to in is are was were be been for on with that this it its as by "
    "from at not but if then than which who whom what when where how there here their his "
    "her they them we you i does describe explain define book material about into other"
).split())


def load_main(path):
    spec = importlib.util.spec_from_file_location("hit5_main", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv[:]
    try:
        sys.argv = [path]
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


def read_jsonl(path):
    return [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def phrase_in_text(phrase, text):
    """单词按 token 边界匹配；词组容忍 PDF 抽取造成的首尾粘连。"""
    needle = norm(phrase)
    haystack = norm(text)
    if not needle:
        return False
    if " " in needle:
        return needle in haystack
    return (" " + needle + " ") in (" " + haystack + " ")


def content_words(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    useful = [w for w in words if w not in STOP and (len(w) >= 3 or w.isdigit())]
    return set(useful or words)


def evidence_score(evidence, document):
    en, dn = norm(evidence), norm(document)
    exact = bool(en and en in dn)
    ew = content_words(evidence)
    recall = (len(ew & content_words(document)) / len(ew)) if ew else 0.0
    return exact, round(recall, 4)


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_rank(flags):
    for i, flag in enumerate(flags, 1):
        if flag:
            return i
    return None


def worker(args):
    M = load_main(args.main)
    M.DB_PATH = os.path.abspath(args.db)
    col = M.get_collection()
    rows = [r for r in read_jsonl(args.eval_file) if r.get("type") != "unanswerable"]
    out = []
    batch_size = max(1, args.batch_size)
    n_results = min(5, col.count())
    if n_results < 1:
        raise RuntimeError("向量库为空")

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        embeddings = M.embed([r["question"] for r in batch])
        result = col.query(
            query_embeddings=embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        for j, row in enumerate(batch):
            ids = result["ids"][j]
            docs = result["documents"][j]
            metas = result["metadatas"][j]
            dists = result["distances"][j]
            gold_page = as_int(row.get("page"))
            primary = (row.get("keywords") or [row.get("term") or ""])[0]
            locator_flags, strict_flags, relaxed_flags, term_flags = [], [], [], []
            top5 = []
            for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
                exact, recall = evidence_score(row.get("evidence", ""), doc)
                locator = gold_page is not None and as_int((meta or {}).get("page")) == gold_page
                strict = exact or recall >= 0.80
                relaxed = exact or recall >= 0.60
                term_hit = phrase_in_text(primary, doc)
                locator_flags.append(locator)
                strict_flags.append(strict)
                relaxed_flags.append(relaxed)
                term_flags.append(term_hit)
                top5.append({
                    "rank": rank,
                    "id": doc_id,
                    "distance": round(float(dist), 6),
                    "page": (meta or {}).get("page"),
                    "loc": (meta or {}).get("loc"),
                    "type": (meta or {}).get("type"),
                    "source": (meta or {}).get("source"),
                    "gold_locator": locator,
                    "evidence_exact": exact,
                    "evidence_recall": recall,
                    "primary_term_hit": term_hit,
                    "document": (doc or "")[:1200],
                })
            lr = first_rank(locator_flags)
            sr = first_rank(strict_flags)
            rr = first_rank(relaxed_flags)
            tr = first_rank(term_flags)
            out.append({
                "subject": row.get("subject"),
                "book": row.get("book"),
                "type": row.get("type"),
                "source_kind": row.get("source"),
                "question": row.get("question"),
                "term": row.get("term"),
                "keywords": row.get("keywords"),
                "gold_page_or_chapter": gold_page,
                "evidence": row.get("evidence"),
                "source_hit_at_5": lr is not None,
                "source_first_rank": lr,
                "evidence_strict_hit_at_5": sr is not None,
                "evidence_strict_first_rank": sr,
                "evidence_relaxed_hit_at_5": rr is not None,
                "evidence_relaxed_first_rank": rr,
                "primary_term_hit_at_5": tr is not None,
                "primary_term_first_rank": tr,
                "top5": top5,
            })
        print("    检索 %d/%d" % (min(start + batch_size, len(rows)), len(rows)), flush=True)

    with io.open(args.worker_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


def index_books(root):
    found = {}
    for ext in ("*.pdf", "*.epub"):
        for path in glob.glob(os.path.join(root, "**", ext), recursive=True):
            found[os.path.basename(path)] = os.path.abspath(path)
    return found


def safe_name(text):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text)[:100]


def inside(parent, child):
    parent, child = os.path.abspath(parent), os.path.abspath(child)
    try:
        return os.path.commonpath([parent, child]) == parent and child != parent
    except ValueError:
        return False


def run_checked(cmd, cwd, timeout):
    p = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        raise RuntimeError("命令失败(%d): %s\n%s" % (p.returncode, " ".join(cmd), p.stdout[-2000:]))
    return p.stdout


def aggregate(rows):
    def one(group):
        n = len(group)
        src = sum(bool(r["source_hit_at_5"]) for r in group)
        strict = sum(bool(r["evidence_strict_hit_at_5"]) for r in group)
        relaxed = sum(bool(r["evidence_relaxed_hit_at_5"]) for r in group)
        term = sum(bool(r["primary_term_hit_at_5"]) for r in group)
        source_rr = sum((1.0 / r["source_first_rank"]) if r["source_first_rank"] else 0.0 for r in group)
        term_rr = sum((1.0 / r["primary_term_first_rank"]) if r["primary_term_first_rank"] else 0.0
                      for r in group)
        term_hit_1 = sum(bool(r["primary_term_first_rank"] and r["primary_term_first_rank"] <= 1)
                         for r in group)
        term_hit_3 = sum(bool(r["primary_term_first_rank"] and r["primary_term_first_rank"] <= 3)
                         for r in group)
        return {
            "n": n,
            "hits_at_1": term_hit_1,
            "hit_at_1": round(term_hit_1 / n, 6) if n else 0,
            "hits_at_3": term_hit_3,
            "hit_at_3": round(term_hit_3 / n, 6) if n else 0,
            "hits": term,
            "hit_at_5": round(term / n, 6) if n else 0,
            "mrr_at_5": round(term_rr / n, 6) if n else 0,
            "source_hits": src,
            "source_hit_at_5": round(src / n, 6) if n else 0,
            "source_mrr_at_5": round(source_rr / n, 6) if n else 0,
            "evidence_strict_hits": strict,
            "evidence_strict_hit_at_5": round(strict / n, 6) if n else 0,
            "evidence_relaxed_hits": relaxed,
            "evidence_relaxed_hit_at_5": round(relaxed / n, 6) if n else 0,
            "primary_term_hits": term,
            "primary_term_hit_at_5": round(term / n, 6) if n else 0,
        }

    by_subject, by_type, by_book, by_subject_type = (
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    )
    for row in rows:
        subject = row.get("subject") or "Unknown"
        row_type = row.get("type") or "Unknown"
        by_subject[subject].append(row)
        by_type[row_type].append(row)
        by_book[row.get("book") or "Unknown"].append(row)
        by_subject_type[(subject, row_type)].append(row)
    return {
        "overall": one(rows),
        "by_subject": {k: one(v) for k, v in sorted(by_subject.items())},
        "by_type": {k: one(v) for k, v in sorted(by_type.items())},
        "by_book": {k: one(v) for k, v in sorted(by_book.items())},
        "by_subject_type": {
            subject: {
                row_type: one(by_subject_type[(subject, row_type)])
                for row_type in sorted({t for s, t in by_subject_type if s == subject})
            }
            for subject in sorted({s for s, _ in by_subject_type})
        },
    }


def pct(value):
    return "%.1f%%" % (100 * value)


def write_report(path, summary, build_stats, config):
    lines = [
        "# 标准检索 Hit@5 评测（v8final）", "",
        "> 本报告只评检索，不调用生成模型。主指标检查 top-5 检索块是否包含题集标注的 gold term。", "",
        "## 口径", "",
        "- **Gold-term Hit@5（主指标）**：top-5 至少一个检索块包含题集的首个 GT 关键词。它是对检索块而非模型答案做判定。",
        "- Hit@K 的计算方式是标准的逐查询二元相关命中；gold-term 自动匹配是本题集采用的弱相关性标签，不等同于逐块人工语义标注。",
        "- 单词按规范化 token 边界匹配；多词术语允许首尾与 PDF 抽取文本粘连，避免版面抽取造成假阴性。",
        "- **Source-unit Hit@5（定位诊断）**：top-5 至少一个块来自构题时标注的页/章。术语可能在书中多处出现，因此它不是主指标。",
        "- **Evidence-strict Hit@5（严格诊断）**：top-5 至少一个块与标注证据精确匹配，或证据内容词召回不低于 80%。",
        "- **Evidence-relaxed Hit@5（宽松诊断）**：阈值为 60%。",
        "- 不可答题没有相关文档，不进入 Hit@5 分母；answerable 与 fuzzy 分开并合计报告。",
        "- 每题 top-5 原文、距离、页/章和证据重合度保存在明细 JSONL。", "",
        "## 配置", "",
        "- Embedding：`%s`" % config.get("embed_model", ""),
        "- top-k：5（生产代码 TOP_K=%s；本指标按任务书取前 5）" % config.get("production_top_k", ""),
        "- 分块指纹：`%s`" % config.get("chunk_fingerprint", ""),
        "- Query expansion：%s；VL quota：%s" % (config.get("query_expand"), config.get("vl_quota")),
        "- 全部书均以当前代码、纯文本模式从空库重建。", "",
        "## 总结果", "",
        "| 范围 | 题数 | Hit@1 | Hit@3 | Gold-term Hit@5 | MRR@5 | Source unit | Evidence strict | Evidence relaxed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("全部", summary["overall"])] + list(summary["by_type"].items())
    for name, s in groups:
        lines.append("| %s | %d | %d/%d（%s） | %d/%d（%s） | %d/%d（%s） | %.3f | %d/%d（%s） | %d/%d（%s） | %d/%d（%s） |" % (
            name, s["n"],
            s["hits_at_1"], s["n"], pct(s["hit_at_1"]),
            s["hits_at_3"], s["n"], pct(s["hit_at_3"]),
            s["hits"], s["n"], pct(s["hit_at_5"]), s["mrr_at_5"],
            s["source_hits"], s["n"], pct(s["source_hit_at_5"]),
            s["evidence_strict_hits"], s["n"], pct(s["evidence_strict_hit_at_5"]),
            s["evidence_relaxed_hits"], s["n"], pct(s["evidence_relaxed_hit_at_5"])))
    lines += ["", "## 分学科（全部可检索题）", "",
              "| 学科 | 题数 | Hit@1 | Hit@3 | Gold-term Hit@5 | MRR@5 | Source unit | Evidence strict |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary["by_subject"].items():
        lines.append("| %s | %d | %d/%d（%s） | %d/%d（%s） | %d/%d（%s） | %.3f | %d/%d（%s） | %d/%d（%s） |" % (
            name, s["n"],
            s["hits_at_1"], s["n"], pct(s["hit_at_1"]),
            s["hits_at_3"], s["n"], pct(s["hit_at_3"]),
            s["hits"], s["n"], pct(s["hit_at_5"]), s["mrr_at_5"],
            s["source_hits"], s["n"], pct(s["source_hit_at_5"]),
            s["evidence_strict_hits"], s["n"], pct(s["evidence_strict_hit_at_5"])))
    lines += ["", "## 分学科（answerable 验收口径）", "",
              "| 学科 | 题数 | Hit@1 | Hit@3 | Gold-term Hit@5 | MRR@5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, groups_by_type in summary["by_subject_type"].items():
        s = groups_by_type.get("answerable")
        if not s:
            continue
        lines.append("| %s | %d | %d/%d（%s） | %d/%d（%s） | %d/%d（%s） | %.3f |" % (
            name, s["n"],
            s["hits_at_1"], s["n"], pct(s["hit_at_1"]),
            s["hits_at_3"], s["n"], pct(s["hit_at_3"]),
            s["hits"], s["n"], pct(s["hit_at_5"]), s["mrr_at_5"]))
    misses = summary["overall"]["n"] - summary["overall"]["hits"]
    lines += ["", "## 完整性", "",
              "- 完成书籍：%d。" % len(build_stats),
              "- Gold-term 主指标未命中：%d 道。" % misses,
              "- 建库总耗时：%.1f 分钟。" % (sum(x["seconds"] for x in build_stats) / 60.0),
              "", "## 边界", "",
              "Gold-term 判据适合本题集的术语型问题，但不能证明检索块包含构题时那一句证据；因此必须同时披露 Source-unit 与 Evidence 指标。",
              "Source-unit 可能低估：同一术语在全书多处出现，检索到其他可回答页面也会被判为未命中原标注页。", ""]
    io.open(path, "w", encoding="utf-8").write("\n".join(lines))


def controller(args):
    eval_files = sorted(glob.glob(os.path.join(os.path.abspath(args.eval), "*.jsonl")))
    books = index_books(args.books)
    if args.only:
        eval_files = [f for f in eval_files if args.only.lower() in os.path.basename(f).lower()
                      or args.only.lower() in ((read_jsonl(f) or [{}])[0].get("book", "").lower())]
    if not eval_files:
        raise RuntimeError("没有匹配的题集")

    runtime = os.path.abspath(args.runtime or os.path.join(tempfile.gettempdir(), "aitic_hit5"))
    os.makedirs(runtime, exist_ok=True)
    output = os.path.abspath(args.out)
    os.makedirs(output, exist_ok=True)
    detail_path = os.path.join(output, "hit5_v8final_details.jsonl")
    build_path = os.path.join(output, "hit5_v8final_builds.json")

    existing = []
    if args.resume and os.path.exists(detail_path):
        existing = read_jsonl(detail_path)
    done = Counter(r["book"] for r in existing)
    all_rows = list(existing)
    build_stats = []
    if args.resume and os.path.exists(build_path):
        build_stats = json.load(io.open(build_path, encoding="utf-8"))

    for index, eval_file in enumerate(eval_files, 1):
        source_rows = [r for r in read_jsonl(eval_file) if r.get("type") != "unanswerable"]
        if not source_rows:
            continue
        book = source_rows[0]["book"]
        if done[book] == len(source_rows):
            print("[%d/%d] %s 已完成，跳过" % (index, len(eval_files), book), flush=True)
            continue
        if book not in books:
            raise RuntimeError("找不到原书：%s" % book)

        work = os.path.join(runtime, "%03d_%s" % (index, safe_name(book)))
        os.makedirs(work, exist_ok=True)
        db = os.path.join(work, "vectordb")
        print("[%d/%d] %s：重建库" % (index, len(eval_files), book), flush=True)
        t0 = time.time()
        if books[book].lower().endswith(".epub"):
            cmd = [sys.executable, args.main, "build", "--epub", books[book]]
        else:
            cmd = [sys.executable, args.main, "build", "--pdf", books[book],
                   "--max-pages", str(args.max_pages), "--no-vl"]
        build_log = run_checked(cmd, work, args.build_timeout)
        seconds = time.time() - t0
        match = re.search(r"共\s*(\d+)\s*块", build_log)
        n_chunks = int(match.group(1)) if match else None

        worker_out = os.path.join(work, "worker.json")
        worker_cmd = [
            sys.executable, os.path.abspath(__file__), "--worker", "--main", args.main,
            "--db", db, "--eval-file", eval_file, "--worker-out", worker_out,
            "--batch-size", str(args.batch_size),
        ]
        run_checked(worker_cmd, work, args.query_timeout)
        rows = json.load(io.open(worker_out, encoding="utf-8"))
        if len(rows) != len(source_rows):
            raise RuntimeError("%s 检索明细数量不一致：%d != %d" % (book, len(rows), len(source_rows)))
        all_rows = [r for r in all_rows if r["book"] != book] + rows
        build_stats = [r for r in build_stats if r["book"] != book]
        build_stats.append({"book": book, "seconds": round(seconds, 3), "chunks": n_chunks})
        with io.open(detail_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        json.dump(build_stats, io.open(build_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        s = aggregate(rows)["overall"]
        print("  %d 题 | Gold-term Hit@5 %s | Source unit %s | %.1fs" % (
            s["n"], pct(s["hit_at_5"]), pct(s["source_hit_at_5"]), seconds), flush=True)
        if inside(runtime, work):
            shutil.rmtree(work, ignore_errors=True)

    all_rows.sort(key=lambda r: (r.get("subject") or "", r.get("book") or "", r.get("type") or "", r["question"]))
    summary = aggregate(all_rows)
    M = load_main(args.main)
    config = {
        "embed_model": M.EMBED_MODEL,
        "production_top_k": M.TOP_K,
        "chunk_fingerprint": M.chunking_fingerprint(),
        "query_expand": M.QUERY_EXPAND,
        "vl_quota": M.VL_QUOTA,
    }
    payload = {"config": config, "summary": summary, "builds": build_stats}
    json.dump(payload, io.open(os.path.join(output, "hit5_v8final_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_report(os.path.join(output, "hit5_v8final_report.md"), summary, build_stats, config)
    print("\n完成：%d 题，Gold-term Hit@5=%s" % (
        summary["overall"]["n"], pct(summary["overall"]["hit_at_5"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="books")
    ap.add_argument("--eval", default=os.path.join("eval", "eval_by_book"))
    ap.add_argument("--main", default=os.path.join("code", "main.py"))
    ap.add_argument("--out", default="eval")
    ap.add_argument("--runtime", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-pages", type=int, default=999999)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--build-timeout", type=int, default=1800)
    ap.add_argument("--query-timeout", type=int, default=600)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--db", default="", help=argparse.SUPPRESS)
    ap.add_argument("--eval-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
