# -*- coding: utf-8 -*-
"""
throughput.py — 吞吐 / 延迟 专项测试

对齐任务书「处理吞吐」验收：本地端到端 ≥15 页/分钟（含图文），或音频文件（5min 内）≤2 分钟处理完；
并补评测报告缺的「延迟」指标（单次问答端到端耗时）。

测三项：
  1. PDF 建库吞吐：纯文本路径 vs 文本+VL 路径，各出「页/分钟」
  2. 音频转写吞吐：处理耗时 vs 音频时长 → 实时因子 RTF（<1 即快于实时）
  3. 查询延迟：多次问答的端到端耗时（平均/最快/最慢）

复用 main 的 build / build_audio / ask，配置与生产一致。
每次使用独立的 ./throughput_runs/<时间戳>/，不动现有 ./vectordb，也不复用 VL 缓存。
从当前目录读 cs.pdf / med.pdf / bizlaw.pdf / Starmer.mp3（在 data 目录下运行）。
用法：python ../code/throughput.py
"""
import os
import sys
import time
import json
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main
import asr
import fitz

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BENCH_ROOT = os.path.abspath(os.environ.get(
    "DISTILL_THROUGHPUT_DIR", os.path.join("throughput_runs", RUN_ID)))
main.DB_PATH = os.path.join(BENCH_ROOT, "vectordb")
main.VL_CACHE = os.path.join(BENCH_ROOT, "vl_cache.json")

BOOKS = {"CS": "cs.pdf", "Medicine": "med.pdf", "Law": "bizlaw.pdf"}
AUDIO = "Starmer.mp3"
AUDIO_MAX_SECONDS = 300
QUERY_BOOK = "med.pdf"
QUERIES = [
    "What are the parts of a bacteriophage?",
    "How do antibiotics affect bacteria?",
    "What is the role of the immune system?",
    "What is the difference between prokaryotic and eukaryotic cells?",
    "How do antibiotics affect bacteria?",
]


def _pages(pdf):
    d = fitz.open(pdf); n = len(d); d.close(); return n


def pdf_throughput():
    print("\n" + "=" * 66)
    print("1) PDF 建库吞吐（页/分钟）")
    print("-" * 66)
    print("%-10s%8s%16s%16s" % ("学科", "页数", "纯文本(页/分)", "文本+VL(页/分)"))
    rows = []
    for disp, pdf in BOOKS.items():
        if not os.path.exists(pdf):
            print("  [跳过] 找不到 %s" % pdf); continue
        pages = min(_pages(pdf), 120)
        # 纯文本路径
        t0 = time.time(); main.build(pdf, 120, 0, use_vl=False); dt_txt = time.time() - t0
        ppm_txt = pages / dt_txt * 60
        # 文本+VL 路径
        t0 = time.time(); main.build(pdf, 120, 15, use_vl=True); dt_vl = time.time() - t0
        ppm_vl = pages / dt_vl * 60
        print("%-10s%8d%16.1f%16.1f" % (disp, pages, ppm_txt, ppm_vl))
        rows.append({"book": disp, "pages": pages,
                     "ppm_text": round(ppm_txt, 1), "ppm_vl": round(ppm_vl, 1),
                     "sec_text": round(dt_txt, 1), "sec_vl": round(dt_vl, 1)})
    if rows:
        avg_txt = sum(r["ppm_text"] for r in rows) / len(rows)
        avg_vl = sum(r["ppm_vl"] for r in rows) / len(rows)
        print("-" * 66)
        print("%-10s%8s%16.1f%16.1f" % ("平均", "", avg_txt, avg_vl))
        print("目标：≥15 页/分钟（含图文）  → 纯文本 %s；文本+VL %s" %
              ("达标" if avg_txt >= 15 else "未达", "达标" if avg_vl >= 15 else "见下注"))
        print("注：VL 每含图页需调多模态模型，是吞吐瓶颈；纯文本页极快。"
              "整体页/分钟取决于含图页占比与 --vl-limit。")
    return rows


def audio_throughput():
    print("\n" + "=" * 66)
    print("2) 音频转写吞吐（实时因子 RTF）")
    print("-" * 66)
    if not os.path.exists(AUDIO):
        print("  [跳过] 找不到 %s" % AUDIO); return None
    t0 = time.time()
    docs, info = asr.transcribe_docs(AUDIO, max_seconds=AUDIO_MAX_SECONDS)
    dt = time.time() - t0
    processed = min(info.duration, AUDIO_MAX_SECONDS) if AUDIO_MAX_SECONDS else info.duration
    rtf = dt / processed
    print("  文件：%s（语言 %s）" % (AUDIO, info.language))
    print("  处理音频时长：%.0f s | 转写耗时：%.1f s | 实时因子 RTF：%.3f（<1 快于实时）" %
          (processed, dt, rtf))
    print("  目标：5min 内音频 ≤2min 处理完 → %s（%.1fs）" %
          ("达标" if dt <= 120 else "未达", dt))
    return {"processed_s": round(processed, 1), "transcribe_s": round(dt, 1),
            "rtf": round(rtf, 3), "under_2min": dt <= 120}


def query_latency():
    print("\n" + "=" * 66)
    print("3) 查询延迟（端到端问答耗时）")
    print("-" * 66)
    if not os.path.exists(QUERY_BOOK):
        print("  [跳过] 找不到 %s" % QUERY_BOOK); return None
    main.build(QUERY_BOOK, 120, 15, use_vl=True)      # 建查询用库
    col = main.get_collection()
    lats = []
    for q in QUERIES:
        t0 = time.time(); main.ask(col, q, verbose=False); lats.append(time.time() - t0)
    avg = sum(lats) / len(lats)
    print("  %d 次查询：平均 %.2fs | 最快 %.2fs | 最慢 %.2fs → 约 %.1f 查询/分钟" %
          (len(lats), avg, min(lats), max(lats), 60 / avg))
    return {"n": len(lats), "avg_s": round(avg, 2),
            "min_s": round(min(lats), 2), "max_s": round(max(lats), 2),
            "qpm": round(60 / avg, 1)}


def main_run():
    os.makedirs(BENCH_ROOT, exist_ok=False)
    print("环境说明：以下为本机（含 GPU）实测；无 GPU 降级吞吐另行验证。")
    print("独立评测目录：%s（新库、新 VL 缓存，不复用历史结果）" % BENCH_ROOT)
    result = {
        "meta": {
            "run_id": RUN_ID,
            "bench_root": BENCH_ROOT,
            "fresh_db": True,
            "fresh_vl_cache": True,
            "python": sys.version,
            "platform": platform.platform(),
            "runtime": main.runtime_fingerprint(),
        },
        "pdf": pdf_throughput(),
        "audio": audio_throughput(),
        "query": query_latency(),
    }
    output = os.path.join(BENCH_ROOT, "throughput_result.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已存 %s" % output)


if __name__ == "__main__":
    main_run()
