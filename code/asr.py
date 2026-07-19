# -*- coding: utf-8 -*-
"""
asr.py — 音频本地转写模块（faster-whisper）

职责：把音频（MP3/WAV/FLAC）用本地 faster-whisper 转写，按 ~CHUNK_TARGET 字符
      分组成带时间戳的文本块，供 main.py 走同一套 semantic 入库 + Citation Grounding。

设计与 PDF 路径同构：
  PDF 块  "p{N}: ..."            元数据 {page:N,   type:"text/figure"}   引用 [p.X]
  音频块  "[audio mm:ss] ..."     元数据 {page:段号, type:"audio", time}  引用 [audio mm:ss]

完全本地：模型走本地目录（modelscope 下好的 ./models/faster-whisper-small），
CPU + int8（顺带覆盖"无 GPU 降级"验收路径），不联网。
"""
import os

CHUNK_TARGET = 450  # 与 main.py 保持一致
# 模型目录候选：按 CWD 常见位置依次尝试，避免"从 code/ 还是 data/ 跑"导致找不到
MODEL_CANDIDATES = [
    "./models/faster-whisper-small",
    "../data/models/faster-whisper-small",
    "./data/models/faster-whisper-small",
    "../models/faster-whisper-small",
]


def _resolve_model(model_dir):
    """给定路径存在就用；否则在候选位置里找第一个存在的。"""
    if model_dir and os.path.isdir(model_dir):
        return model_dir
    for c in ([model_dir] if model_dir else []) + MODEL_CANDIDATES:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "找不到 faster-whisper 本地模型目录。请确认已用 modelscope 下载到 "
        "./models/faster-whisper-small（在 data 目录下运行），或用 --asr-model 指定路径。"
    )


def _mmss(sec):
    m, s = divmod(int(sec), 60)
    return "%02d:%02d" % (m, s)


def transcribe_chunks(audio_path, model_dir=None, max_seconds=None, chunk_target=CHUNK_TARGET):
    """转写音频 → 按 ~chunk_target 字符分组。
       返回 (chunks, info)，chunks = [(text, start_sec, end_sec), ...]，每组带起始时间。"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError("找不到音频文件：%s" % audio_path)
    from faster_whisper import WhisperModel  # 延迟导入，PDF 用户不受影响
    mdir = _resolve_model(model_dir)
    model = WhisperModel(mdir, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path)

    chunks = []
    buf, blen, cstart, clast = [], 0, None, 0.0
    for seg in segments:
        if max_seconds is not None and seg.start >= max_seconds:
            break
        txt = (seg.text or "").strip()
        if not txt:
            continue
        if cstart is None:
            cstart = seg.start
        buf.append(txt)
        blen += len(txt)
        clast = seg.end
        if blen >= chunk_target:
            chunks.append((" ".join(buf), cstart, clast))
            buf, blen, cstart = [], 0, None
    if buf:
        chunks.append((" ".join(buf), cstart if cstart is not None else 0.0, clast))
    return chunks, info


def transcribe_docs(audio_path, model_dir=None, max_seconds=None):
    """返回可直接入库的文档元组列表 + info。
       每条 = (prefixed_text, page_int, "audio", mmss)
         prefixed_text : "[audio mm:ss] 正文"  —— 时间戳前缀，供模型引用
         page_int      : 分段序号（1,2,3…），保证 main.py ask() 的 [来源] 行 %d 正常
         mmss          : "mm:ss" 起始时间，存入元数据 time 字段
    """
    chunks, info = transcribe_chunks(audio_path, model_dir=model_dir, max_seconds=max_seconds)
    docs = []
    for idx, (text, start, end) in enumerate(chunks, 1):
        mmss = _mmss(start)
        docs.append(("[audio %s] %s" % (mmss, text), idx, "audio", mmss))
    return docs, info


# 独立自检：python asr.py 音频路径 [最大秒数]
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python asr.py <音频文件> [最大秒数]")
        sys.exit(1)
    path = sys.argv[1]
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else None
    docs, info = transcribe_docs(path, max_seconds=ms)
    print("语言 %s | 时长 %.0fs | 音频块 %d" % (info.language, info.duration, len(docs)))
    for d in docs[:3]:
        print("  ", d[0][:90], "…" if len(d[0]) > 90 else "")
