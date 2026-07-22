"""
main.py — 知识蒸馏管线（整合版 · 命令行入口）
================================================================
把之前散落的"混合路由入库 + 检索问答"整合成一个可复用主程序。
核心能力：
  · 混合路由：纯文字页走文本通道，含图页自动走 VL（qwen3-vl）解析
  · 多格式入库：PDF（混合路由）/ 音频（ASR）/ EPUB（转文本）/ 独立图片（VL），后三者 append 成混合库
  · 持久化向量库：build 一次存盘，ask 时直接加载，不重复 embedding
  · 引用锚定：回答带 [p.X]/[audio mm:ss]/[ch:标题]/[image:名] 溯源，无依据时输出 [NO REFERENCE FOUND]
  · 动态上下文裁剪：按相关度整块保留（RELEVANCE_TRIM），预算内优先保住最相关的完整块
  · VL 缓存 + 失败重试：解析过的页不重算，网络抖动自动重试

依赖：pip install -r requirements.txt
模型：ollama pull qwen3:8b  /  ollama pull qwen3-vl:8b  /  ollama pull bge-m3

用法：
  # 1) 建库（对一本书做混合路由入库，结果存到 ./vectordb）
  python main.py build --pdf med.pdf

  # 只取前 60 页、最多 10 个含图页走 VL；纯文本模式加 --no-vl
  python main.py build --pdf med.pdf --max-pages 60 --vl-limit 10
  python main.py build --pdf cs.pdf --no-vl

  # 2) 提问（加载已建好的库）
  python main.py ask "What are the parts of a bacteriophage?"

  # 3) 进入连续问答（输入 exit 退出）
  python main.py chat
"""

import argparse, base64, json, os, re, sys, time
import fitz
import chromadb

try:
    import ollama
except ImportError:
    print("缺少 ollama 库，请先 pip install -r requirements.txt"); sys.exit(1)

# ----------------------------- 配置 -----------------------------
LLM_MODEL = "qwen3:8b"
VL_MODEL = "qwen3-vl:8b"
EMBED_MODEL = "bge-m3"

CHUNK_TARGET, CHUNK_MAX = 450, 650
TOP_K, CONTEXT_BUDGET = 8, 900          # 消融得到的最优预算
NUM_PREDICT, TEMPERATURE = 300, 0.0
BUDGET_ESCALATED = 1800                  # 动态预算：首答拒答且检索有命中时升配重答一次
DYNAMIC_BUDGET = True                    # 关掉则恒用 900（用于对照评测）
RELEVANCE_TRIM = True                    # 动态裁剪：按相关度整块保留（关掉=旧的按检索序字符截断，用于对照）
VL_DPI = 150
VL_RETRY = 3

DB_PATH = "./vectordb"
COLLECTION = "knowledge_base"
VL_CACHE = "./vl_cache.json"

HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*[\.\)]|[A-Z][A-Z ]{3,}$|Chapter\s+\d+|CHAPTER\s+\d+)")

PROMPT = """Answer the question using ONLY the material below. Each block starts with a source tag like [p.112]. When you cite, reuse the tag shown above that block (e.g. [p.112]).
If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""


# ----------------------------- 工具函数 -----------------------------
def split_sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?。！？；\n])", text) if x.strip()]


def semantic_chunks(text):
    """语义分块：识别标题做前缀，按句子边界凑成 ~CHUNK_TARGET 的块。"""
    chunks, buf, blen, heading = [], [], 0, ""

    def flush():
        nonlocal buf, blen
        if buf:
            chunks.append((heading + " " + "".join(buf)).strip())
            buf, blen = [], 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line) and len(line) < 60:
            flush(); heading = line; continue
        for s in split_sentences(line):
            if blen + len(s) > CHUNK_MAX and buf:
                flush()
            buf.append(s); blen += len(s)
            if blen >= CHUNK_TARGET:
                flush()
    flush()
    return chunks


def _embed_http(texts):
    """直接 HTTP 调 Ollama /api/embed（等价 curl），绕过 ollama 库版本不兼容。"""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["embeddings"]


def embed(texts):
    # 优先用 ollama 库；库与服务端版本不匹配（如 502）时自动降级到 HTTP 直连
    try:
        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    except (AttributeError, KeyError, TypeError):
        try:
            return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]
        except Exception:
            return _embed_http(texts)
    except Exception:
        return _embed_http(texts)


def _ollama_host():
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return ("http://" + host if not host.startswith("http") else host).rstrip("/")



def _strip_think(text):  # think_off_patch
    """剥离 <think>...</think> 思考块；若剩余为空则返回拒答标记。"""
    import re as _re
    if text is None:
        return "[NO REFERENCE FOUND]"
    t = _re.sub(r"<think>.*?</think>", "", str(text), flags=_re.S | _re.I)
    t = _re.sub(r"</?think>", "", t, flags=_re.I)
    t = t.strip()
    return t if t else "[NO REFERENCE FOUND]"


def _post_json(path, payload):
    import urllib.request
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_ollama_host() + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def _generate(model, prompt, system=None, options=None):
    """生成：优先 ollama 库；库不兼容（502 等）时降级 HTTP /api/generate。
       返回对象统一支持 out["response"] 与 out.get(key, default)。"""
    try:
        kw = {"model": model, "prompt": prompt, "think": False}   # think=False：避免推理块被 _strip_think 剥空后误判拒答
        if system is not None:
            kw["system"] = system
        if options is not None:
            kw["options"] = options
        _r = ollama.generate(**kw)
        try:
            _r["response"] = _strip_think(_r.get("response", ""))
        except Exception:
            pass
        return _r
    except Exception:
        # Ollama 0.31+ 的 /api/generate 对 prompt 收严会返 400，改走 /api/chat
        msgs = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": msgs, "stream": False, "think": False}
        if options is not None:
            payload["options"] = options
        out = _post_json("/api/chat", payload)
        text = _strip_think(out.get("message", {}).get("content", ""))
        # 统一成 generate 的返回形状，兼容上游 out["response"] / out.get(key)
        out["response"] = text
        return out


def _chat_vl(model, prompt, b64):
    """VL 多模态对话：优先 ollama 库；降级 HTTP /api/chat。返回 message.content 文本。"""
    try:
        out = ollama.chat(model=model, messages=[{"role": "user", "content": prompt, "images": [b64]}])
        return out["message"]["content"]
    except Exception:
        out = _post_json("/api/chat", {"model": model, "stream": False,
                                       "messages": [{"role": "user", "content": prompt, "images": [b64]}]})
        return out["message"]["content"]


def vl_parse(img_bytes):
    """对一页图调 VL，提取图表里的文字标签 + 描述；失败自动重试。"""
    b64 = base64.b64encode(img_bytes).decode()
    p = ("This is a textbook page. Extract ALL text labels from any figures, diagrams, or "
         "tables (e.g. component names, scale bars, axis labels). List every label, then "
         "briefly describe what each figure shows.")
    for attempt in range(VL_RETRY):
        try:
            return _chat_vl(VL_MODEL, p, b64)
        except Exception:
            print(" [VL重试%d]" % (attempt + 1), end="", flush=True); time.sleep(3)
    return ""


def load_vl_cache():
    if os.path.exists(VL_CACHE):
        try:
            return json.load(open(VL_CACHE, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_vl_cache(cache):
    """把 VL 描述缓存落盘。断点续传的关键：每处理完一页就调一次，中断也不丢已处理页。"""
    json.dump(cache, open(VL_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ----------------------------- 建库 -----------------------------
def build(pdf, max_pages, vl_limit, use_vl):
    if not os.path.exists(pdf):
        sys.exit("[错误] 找不到 PDF：%s" % pdf)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    cache = load_vl_cache()
    cache_key = lambda p: "%s::p%d" % (os.path.basename(pdf), p)
    _pref = os.path.basename(pdf) + "::"
    _resumed = sum(1 for kk in cache if kk.startswith(_pref))
    if _resumed:
        print("== 断点续传：检测到已缓存 %d 页 VL 描述，将跳过重复解析 ==" % _resumed)

    d = fitz.open(pdf)
    n = min(len(d), max_pages)
    text_chunks, vl_chunks, vl_done = [], [], 0
    t0 = time.time()
    print("== 混合路由入库：%s（前%d页，VL上限%d，%s）==" %
          (os.path.basename(pdf), n, vl_limit, "纯文本" if not use_vl else "文本+VL"))

    for i in range(n):
        page = d[i]
        txt = page.get_text("text").strip()
        if txt:
            for ch in semantic_chunks("p%d: %s" % (i + 1, txt)):
                text_chunks.append((ch, i + 1, "text"))
        if use_vl and page.get_images() and vl_done < vl_limit:
            k = cache_key(i + 1)
            if k in cache:
                vtxt = cache[k]
            else:
                print("  VL 第%d页..." % (i + 1), end="", flush=True)
                t1 = time.time()
                vtxt = vl_parse(page.get_pixmap(dpi=VL_DPI).tobytes("png"))
                cache[k] = vtxt
                save_vl_cache(cache)   # 断点续传：每处理完一页立即落盘，中断也不丢
                print(" %.0fs" % (time.time() - t1))
            for ch in semantic_chunks("FIGURE p%d: %s" % (i + 1, vtxt)):
                vl_chunks.append((ch, i + 1, "figure"))
            vl_done += 1
    d.close()
    save_vl_cache(cache)

    alldocs = text_chunks + vl_chunks
    for s in range(0, len(alldocs), 64):
        batch = alldocs[s:s + 64]
        col.add(
            ids=["c%d" % j for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": b[2], "source": os.path.basename(pdf)} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(alldocs)), len(alldocs)), end="\r")
    print()
    print("完成：文本块 %d | VL块 %d（来自%d个含图页）| 共 %d 块 | 耗时 %.0fs" %
          (len(text_chunks), len(vl_chunks), vl_done, len(alldocs), time.time() - t0))
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- 音频建库 -----------------------------
def build_audio(audio, max_seconds=None, asr_model=None):
    """音频转写入库：append 到现有 knowledge_base（可与 PDF 库共存成混合库）。"""
    import asr  # 延迟导入，避免 PDF 用户触发 faster_whisper
    if not os.path.exists(audio):
        sys.exit("[错误] 找不到音频：%s" % audio)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    print("== 音频转写入库：%s（faster-whisper-small, CPU/int8%s）==" %
          (os.path.basename(audio), ("，前%ds" % max_seconds) if max_seconds else ""))
    docs, info = asr.transcribe_docs(audio, model_dir=asr_model, max_seconds=max_seconds)
    print("  语言 %s | 时长 %.0fs | 生成音频块 %d 个" % (info.language, info.duration, len(docs)))

    base = os.path.basename(audio)
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    for s in range(0, len(docs), 64):
        batch = docs[s:s + 64]
        col.upsert(   # upsert：同名音频重跑不报错、直接覆盖
            ids=["audio_%s_%d" % (safe, j) for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": b[2], "source": base, "time": b[3]} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(docs)), len(docs)), end="\r")
    print()
    print("完成：音频块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- EPUB 建库 -----------------------------
def _epub_blocks(path, min_chars=40):
    """EPUB -> [(text, chapter_idx, loc)]。按 spine 顺序遍历，每章一块。
       loc 沿用引用锚定口径：ch{序号}:{章节标题}（EPUB 无固定页码）。"""
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)
    toc_title = {}
    def _walk(items):
        for it in items:
            if isinstance(it, tuple):
                link, children = it[0], it[1]
                if getattr(link, "href", None):
                    toc_title[link.href.split("#")[0]] = link.title
                _walk(children)
            elif getattr(it, "href", None):
                toc_title[it.href.split("#")[0]] = it.title
    try:
        _walk(book.toc)
    except Exception:
        pass

    out, idx = [], 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        fname = item.get_name()
        if "nav" in fname.lower():
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        head = soup.find(["h1", "h2"])
        if head:
            title = head.get_text(strip=True)
        elif toc_title.get(fname):
            title = toc_title.get(fname)
        else:
            title = os.path.splitext(os.path.basename(fname))[0]
        for t in soup(["script", "style"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if len(text) < min_chars:
            continue
        idx += 1
        out.append((text, idx, "ch%d:%s" % (idx, title or "?")))
    return out


def build_epub(epub_path, max_chapters=None):
    """EPUB 转文本入库：append 到现有 knowledge_base（与 PDF/音频共存成混合库）。"""
    if not os.path.exists(epub_path):
        sys.exit("[错误] 找不到 EPUB：%s" % epub_path)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    print("== EPUB 转文本入库：%s ==" % os.path.basename(epub_path))
    blocks = _epub_blocks(epub_path)
    if max_chapters:
        blocks = blocks[:max_chapters]
    # 每章再过语义分块，保持与 PDF 一致的块粒度
    docs = []
    for text, ch_idx, loc in blocks:
        for chk in semantic_chunks(text):
            docs.append((chk, ch_idx, loc))
    print("  章节 %d | 生成文本块 %d" % (len(blocks), len(docs)))

    base = os.path.basename(epub_path)
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    for s in range(0, len(docs), 64):
        batch = docs[s:s + 64]
        col.upsert(
            ids=["epub_%s_%d" % (safe, j) for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": "epub", "source": base, "loc": b[2]} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(docs)), len(docs)), end="\r")
    print()
    print("完成：EPUB 块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- 独立图片建库 -----------------------------
def build_image(img_path):
    """独立图片入库：走 VL 通道生成描述，append 到现有 knowledge_base。"""
    if not os.path.exists(img_path):
        sys.exit("[错误] 找不到图片：%s" % img_path)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    base = os.path.basename(img_path)
    print("== 独立图片入库：%s（走 VL 解析）==" % base)
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    p = ("Describe this image in detail for a knowledge base: list every text label, "
         "component, or figure element, then briefly explain what the image shows.")
    desc = ""
    for attempt in range(VL_RETRY):
        try:
            desc = _strip_think(_chat_vl(VL_MODEL, p, b64))
            break
        except Exception:
            print(" [VL重试%d]" % (attempt + 1), end="", flush=True); time.sleep(3)
    if not desc or desc == "[NO REFERENCE FOUND]":
        sys.exit("[错误] VL 解析图片失败（模型未返回描述）")

    loc = "image:%s" % base
    docs = [(chk, 0, loc) for chk in semantic_chunks(desc)] or [(desc, 0, loc)]
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    col.upsert(
        ids=["image_%s_%d" % (safe, j) for j in range(len(docs))],
        embeddings=embed([b[0] for b in docs]),
        documents=[b[0] for b in docs],
        metadatas=[{"page": 0, "type": "image", "source": base, "loc": loc} for _ in docs],
    )
    print("完成：图片块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- 智能体配置生成 -----------------------------
def _infer_subject(docs, book):
    """采样库内容，让 LLM 判断学科 + 生成助教角色；LLM 不可用则按书名降级。"""
    excerpt = "\n".join((d or "")[:300] for d in docs[:8])
    prompt = ('Based on these textbook excerpts, reply with ONLY a JSON object '
              '{"subject":"<2-4 word academic subject>","role":"<one-sentence teaching-assistant role>"}.\n\n'
              "Excerpts:\n" + excerpt)
    try:
        out = _generate(LLM_MODEL, prompt, options={"temperature": 0, "num_predict": 120})
        m = re.search(r"\{.*\}", out["response"], re.S)
        obj = json.loads(m.group(0))
        subj = (obj.get("subject") or "").strip()
        role = (obj.get("role") or "").strip()
        if subj and role:
            return subj, role
    except Exception:
        pass
    base = os.path.splitext(book)[0]
    return base, "a teaching assistant for %s" % base


def _compose_system(subject, role, book):
    return (
        'You are %s. You assist users with the material "%s" (domain: %s).\n'
        "Rules:\n"
        "- Answer ONLY using the retrieved material provided at query time.\n"
        "- Always cite sources: text/figure as [p.X], audio as [audio mm:ss].\n"
        '- If the retrieved material gives no basis, answer exactly "[NO REFERENCE FOUND]".\n'
        "- For arithmetic, a calculator tool computes the exact result; report it faithfully.\n"
        "- Be precise, concise, and educational."
    ) % (role, book, subject)


def gen_agent(pdf=None, max_pages=120, vl_limit=15, use_vl=True):
    """生成标准化智能体包：system prompt + 工具链(检索/计算/引用) + Ollama 运行配置。"""
    if pdf:
        build(pdf, max_pages, vl_limit, use_vl)      # 给了 PDF 就先建库
    col = get_collection()

    sample = col.get(limit=12)
    docs = (sample.get("documents") or [])[:12]
    metas = sample.get("metadatas") or []
    sources = sorted({m.get("source", "?") for m in metas}) or ["knowledge_base"]
    book = sources[0]
    print("== 生成智能体配置：采样 %d 块，判定学科中... ==" % len(docs))
    subject, role = _infer_subject(docs, book)
    system_text = _compose_system(subject, role, book)
    print("   学科：%s" % subject)

    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", os.path.splitext(book)[0]) or "agent"
    outdir = os.path.abspath("./agent_%s" % safe)
    os.makedirs(outdir, exist_ok=True)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    db_abs = os.path.abspath(DB_PATH)
    model_name = ("%s-assistant" % safe).lower()

    open(os.path.join(outdir, "system_prompt.txt"), "w", encoding="utf-8").write(system_text)

    modelfile = ('FROM %s\n\nSYSTEM """%s"""\n\n'
                 "PARAMETER temperature %s\nPARAMETER num_predict %s\n"
                 % (LLM_MODEL, system_text, TEMPERATURE, NUM_PREDICT))
    open(os.path.join(outdir, "Modelfile"), "w", encoding="utf-8").write(modelfile)

    runbat = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "echo == 1) 创建专属 Ollama 模型（system prompt + 参数）==\r\n"
        'ollama create %s -f "%s\\Modelfile"\r\n'
        "echo == 2) 启动智能体对话（工具：检索 + 计算器）==\r\n"
        'python "%s\\agent_runtime.py" --db "%s" --collection %s --prompt "%s\\system_prompt.txt"\r\n'
        % (model_name, outdir, code_dir, db_abs, COLLECTION, outdir)
    )
    open(os.path.join(outdir, "run.bat"), "w", encoding="utf-8", newline="").write(runbat)

    readme = (
        "# 智能体包：%s\n\n"
        "- 学科：%s\n"
        "- 工具链：检索（RAG + 引用溯源）、计算器（白名单 AST 安全求值）；规则路由\n"
        "- 一键启动：双击 `run.bat`（先 `ollama create` 专属模型，再进入对话）\n"
        "- 手动启动对话：\n"
        '  `python "%s\\agent_runtime.py" --db "%s" --prompt system_prompt.txt`\n\n'
        "对话中：普通问题走检索问答（带 [p.X] / [audio mm:ss] 引用）；"
        "输入算式（如 `3*(4+5)`、`sqrt(144)`）走计算器，返回精确结果。\n"
        % (model_name, subject, code_dir, db_abs)
    )
    open(os.path.join(outdir, "README.md"), "w", encoding="utf-8").write(readme)

    print("完成：智能体包已生成到 %s" % outdir)
    print("  ├─ system_prompt.txt   （按书定制 · 含引用约束）")
    print("  ├─ Modelfile           （Ollama 运行配置）")
    print("  ├─ agent_runtime.py    （运行时在 code/，检索+计算器+路由）")
    print("  ├─ run.bat             （一键：create 模型 + 启动对话）")
    print("  └─ README.md")
    print("启动：双击 %s\\run.bat" % outdir)


# ----------------------------- 提问 -----------------------------
def get_collection():
    if not os.path.exists(DB_PATH):
        sys.exit("[错误] 还没建库，请先运行：python main.py build --pdf 你的书.pdf")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        sys.exit("[错误] 向量库为空，请先 build。")


_ABSTAIN_RE = re.compile(r"no\s+(?:\w+\s+){0,2}references?\s+(?:found|available|provided|in)", re.I)


def is_abstain(answer):
    # 空/纯空白也视为拒答（模型只输出 think 块时 content 为空）
    if answer is None or not str(answer).strip():
        return True
    """鲁棒判定拒答：[NO REFERENCE FOUND] 及常见变体。"""
    a = answer.lower()
    return ("no reference found" in a
            or bool(re.search(r"\[[^\]]*\bno\b[^\]]*\breferences?\b", a))
            or bool(_ABSTAIN_RE.search(a)))


def _cosine(a, b):
    s = aa = bb = 0.0
    for x, y in zip(a, b):
        s += x * y; aa += x * x; bb += y * y
    if aa == 0 or bb == 0:
        return 0.0
    return s / (aa ** 0.5 * bb ** 0.5)


def _pack_relevance(docs, question, budget):
    """动态裁剪：沿检索相关度顺序整块贪心保留，绝不切半块。
       docs 已由 col.query 按相关度降序返回，直接信任该顺序（不再二次重排——
       二次余弦重排会与检索排序打架，在仅容一块时反而选错块）。
       相对字符截断的两点改进：①整块保留不砍半 ②跳过塞不下的大块、纳入后面更小的相关块。
       question 参数保留仅为兼容调用签名。
       返回 (拼好的块列表, 原始下标列表)——下标供引用溯源。"""
    if not docs:
        return [], []
    # 若连检索序最相关的第0块都整块塞不下 -> 截断第0块（与纯截断同行为），
    # 不跳过它去纳入后面更小的块（否则会捡了噪声小块、丢了最相关块=答案）。
    if len(docs[0]) > budget:
        return [docs[0][:budget]], [0]
    picked, used, sep = [], 0, len("\n---\n")
    for i in range(len(docs)):                # docs 已按相关度降序
        need = len(docs[i]) + (sep if picked else 0)
        if used + need <= budget:
            picked.append(i); used += need
    return [docs[i] for i in picked], picked  # picked 已升序（检索序）


def _pack_truncate(docs, budget):
    """旧策略：按检索序拼接，超预算按字符截断（保留用于对照评测）。
       返回 (块列表, 下标列表)。"""
    packed, idx, used = [], [], 0
    for i, doc in enumerate(docs):
        if used + len(doc) > budget:
            if budget - used > 120:
                packed.append(doc[:budget - used]); idx.append(i)
            break
        packed.append(doc); idx.append(i); used += len(doc)
    return packed, idx



# ============ 引用准确度：真页标签 + 引用校验（检验功能）============
def _cite_tag(m):
    """一个检索块的\"引用标签\"：喂进 context 让模型照标，也用于事后校验。
       页码型→ p.112；EPUB/图片→ 用 loc（ch2:Viruses / image:fig.png）；音频→ audio mm:ss。"""
    t = m.get("type")
    if t == "audio":
        return "audio %s" % m.get("time", "?")
    if t in ("epub", "image"):
        return m.get("loc", t)
    return "p.%d" % m["page"]


def _labeled_context(packed, packed_idx, metas=None):
    """给每块前缀真实来源标签，让模型引用真实页码而非瞎编。
       metas=None（评测脚本）时退回纯拼接，行为与旧版一致。"""
    if not metas:
        return "\n---\n".join(packed)
    blocks = []
    for pos, i in enumerate(packed_idx):
        tag = _cite_tag(metas[i]) if i < len(metas) else "?"
        blocks.append("[%s]\n%s" % (tag, packed[pos]))
    return "\n---\n".join(blocks)


def _norm_cite(s):
    """归一化一个引用/标签：'p.112'/'p112'/'P.112(text)' → 'p.112'；其余（ch../image../audio..）小写去空格。"""
    s = s.strip().lower().replace(" ", "").split("(")[0]
    mm = re.match(r"p\.?(\d+)", s)
    return ("p.%s" % mm.group(1)) if mm else s


def verify_citations(answer, packed_idx, metas):
    """检验功能：核对正文里的 [p.X] 等引用，是否都落在\"实际喂给模型的检索块\"里。
       引用了检索块里没有的页 = 疑似编造，标记出来。返回校验详情 + 引用准确率。"""
    metas = metas or []
    valid = {_norm_cite(_cite_tag(metas[i])) for i in packed_idx if i < len(metas)}
    cited = []
    for raw in re.findall(r"\[([^\]]+)\]", str(answer)):
        n = _norm_cite(raw)
        if re.match(r"(p\.\d+|ch|image|audio)", n):   # 只算引用型括号，忽略 [NO REFERENCE FOUND] 等
            cited.append(n)
    cset = list(dict.fromkeys(cited))                   # 去重保序
    hit = [c for c in cset if c in valid]
    bad = [c for c in cset if c not in valid]
    total = len(cset)
    return {
        "total": total,
        "hit": hit,
        "fabricated": bad,
        "valid_sources": sorted(valid),
        "ok": (total == 0) or (not bad),
        "rate": (len(hit) / total) if total else 1.0,
    }


def _cite_check_line(cc):
    if cc["total"] == 0:
        return "正文无显式引用"
    if cc["ok"]:
        return "%d/%d 引用全部命中检索来源 OK" % (len(cc["hit"]), cc["total"])
    return "%d/%d 命中；疑似编造: %s  <-- 警告" % (len(cc["hit"]), cc["total"], "、".join(cc["fabricated"]))


def _run_once(docs, question, budget, metas=None):
    """按给定预算打包检索块并生成一次。
       metas 提供时给每块前缀真实来源标签（让模型引用真页码，杜绝瞎编）；不提供则纯拼接（兼容评测脚本）。
       返回 (答案, tokens, 打包块的原始下标列表)——下标供引用溯源精确对齐。"""
    if RELEVANCE_TRIM:
        packed, packed_idx = _pack_relevance(docs, question, budget)
    else:
        packed, packed_idx = _pack_truncate(docs, budget)
    context = _labeled_context(packed, packed_idx, metas)
    out = _generate(LLM_MODEL, PROMPT.format(context=context, question=question),
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    toks = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
    return out["response"].strip(), toks, packed_idx


def ask(col, question, verbose=True, dynamic=DYNAMIC_BUDGET):
    qv = embed([question])[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]

    # 第一档：预算 900（常态，保 Token 效率）
    answer, toks, packed_idx = _run_once(docs, question, CONTEXT_BUDGET, metas)
    escalated = False
    # 动态升配：仅当"检索有命中却拒答"时，升到 1800 重答一次
    if dynamic and docs and is_abstain(answer):
        ans2, toks2, packed_idx2 = _run_once(docs, question, BUDGET_ESCALATED, metas)
        toks += toks2                      # 两次调用 token 累计，成本如实计
        escalated = True
        if not is_abstain(ans2):           # 升配后答出来了才采用
            answer, packed_idx = ans2, packed_idx2

    if verbose:
        def _src(m):
            t = m.get("type")
            if t == "audio":
                return "audio %s" % m.get("time", "?")
            if t in ("epub", "image"):          # 无页码，用 loc 定位（ch2:Viruses / image:fig.png）
                return m.get("loc", t)
            return "p%d(%s)" % (m["page"], m["type"])
        # 按实际打包的块下标取来源（相关度裁剪下打包的不一定是前几块）
        srcs = sorted({_src(metas[i]) for i in packed_idx if i < len(metas)})
        tag = "  (动态升配 %d)" % BUDGET_ESCALATED if escalated else ""
        cc = verify_citations(answer, packed_idx, metas)
        print("\n" + answer)
        print("\n[来源] %s  | tokens: %d%s" % ("、".join(srcs), toks, tag))
        print("[引用校验] %s" % _cite_check_line(cc))
    return answer


# ============ brief 文档生成模式（面向业务场景：综合资料生成结构化摘要）============
BRIEF_PROMPT = """You are writing a concise briefing document for a business/professional reader.
Using ONLY the material below, write a structured brief on the given TOPIC.
Requirements:
- Organize as: 【概要】one-sentence summary; 【要点】3-6 bullet points; 【依据】key supporting facts.
- Each block starts with a source tag like [p.112]. When you cite a fact, reuse the tag shown above that block (e.g. [p.112]).
- Do NOT invent anything not in the material. If the material is insufficient, say so plainly.
- Understanding-level accuracy is fine; you need not quote verbatim.

Material:
{context}

Topic: {topic}

Brief:"""


BRIEF_SYSTEM = ("You are a briefing writer. You MUST produce a structured brief based on the given material. "
                "Never output '[NO REFERENCE FOUND]' or refuse; if some detail is missing, write the brief with what IS present.")

def _gen_brief_raw(prompt):
    """brief 专用生成：think=False + 只剥 think 标签、不套 ask 那套\"空则拒答\"兜底，避免长文档被误判拒答。"""
    kw = {"model": LLM_MODEL, "prompt": prompt, "system": BRIEF_SYSTEM,
          "think": False, "options": {"temperature": TEMPERATURE, "num_predict": max(NUM_PREDICT, 700)}}
    try:
        r = ollama.generate(**kw)
    except Exception:
        # 与 _generate 一致的 /api/chat 降级
        payload = {"model": LLM_MODEL, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": BRIEF_SYSTEM},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": TEMPERATURE, "num_predict": max(NUM_PREDICT, 700)}}
        out = _post_json("/api/chat", payload)
        r = {"response": out.get("message", {}).get("content", ""),
             "prompt_eval_count": out.get("prompt_eval_count", 0), "eval_count": out.get("eval_count", 0)}
    text = re.sub(r"<think>.*?</think>", "", str(r.get("response", "")), flags=re.S | re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    toks = r.get("prompt_eval_count", 0) + r.get("eval_count", 0)
    return text, toks

def _run_once_brief(docs, topic, budget, metas=None):
    """同构复用检索/相关度打包/溯源；生成走 _gen_brief_raw（不套问答拒答兜底）。
       metas 提供时给每块前缀真实来源标签，让 brief 正文引用真页码。"""
    if RELEVANCE_TRIM:
        packed, packed_idx = _pack_relevance(docs, topic, budget)
    else:
        packed, packed_idx = _pack_truncate(docs, budget)
    context = _labeled_context(packed, packed_idx, metas)
    text, toks = _gen_brief_raw(BRIEF_PROMPT.format(context=context, topic=topic))
    return text, toks, packed_idx


def brief(col, topic, verbose=True):
    """面向业务场景：围绕 topic 综合检索资料，生成带出处的结构化 brief 文档。
       复用与 ask 相同的检索/相关度打包/引用溯源；直接用升配预算一步到位。"""
    qv = embed([topic])[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    answer, toks, packed_idx = _run_once_brief(docs, topic, BUDGET_ESCALATED, metas)
    if verbose:
        def _src(m):
            t = m.get("type")
            if t == "audio":
                return "audio %s" % m.get("time", "?")
            if t in ("epub", "image"):
                return m.get("loc", t)
            return "p%d(%s)" % (m["page"], m["type"])
        srcs = sorted({_src(metas[i]) for i in packed_idx if i < len(metas)})
        cc = verify_citations(answer, packed_idx, metas)
        print("\n" + answer)
        print("\n[来源] %s  | tokens: %d" % ("、".join(srcs), toks))
        print("[引用校验] %s" % _cite_check_line(cc))
    return answer
# ============ brief 模式结束 ============


# ----------------------------- 入口 -----------------------------
def main():
    ap = argparse.ArgumentParser(description="知识蒸馏管线 · 混合路由 RAG")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="对一本 PDF 或一段音频做入库")
    b.add_argument("--pdf", help="PDF 路径（混合路由，替换建库）")
    b.add_argument("--max-pages", type=int, default=120)
    b.add_argument("--vl-limit", type=int, default=15)
    b.add_argument("--no-vl", action="store_true", help="纯文本模式，跳过 VL")
    b.add_argument("--audio", help="音频路径 MP3/WAV/FLAC（转写后 append 入库）")
    b.add_argument("--max-seconds", type=int, default=None, help="仅取音频前 N 秒（对齐≤5min验收）")
    b.add_argument("--asr-model", default=None, help="faster-whisper 本地模型目录")
    b.add_argument("--epub", help="EPUB 路径（转文本后 append 入库）")
    b.add_argument("--max-chapters", type=int, default=None, help="仅取 EPUB 前 N 章")
    b.add_argument("--image", help="独立图片路径 PNG/JPG（走 VL 解析后 append 入库）")

    a = sub.add_parser("ask", help="向已建好的库提一个问题")
    a.add_argument("question")

    sub.add_parser("chat", help="连续问答，输入 exit 退出")

    g = sub.add_parser("agent", help="根据已建库/PDF 生成智能体配置包")
    g.add_argument("--pdf", default=None, help="给定则先建库再生成")
    g.add_argument("--max-pages", type=int, default=120)
    g.add_argument("--vl-limit", type=int, default=15)
    g.add_argument("--no-vl", action="store_true", help="纯文本模式，跳过 VL")

    br = sub.add_parser("brief", help="围绕一个主题生成带出处的 brief 文档")
    br.add_argument("topic")

    args = ap.parse_args()

    if args.cmd == "build":
        if args.audio:
            build_audio(args.audio, args.max_seconds, args.asr_model)
        elif args.epub:
            build_epub(args.epub, args.max_chapters)
        elif args.image:
            build_image(args.image)
        elif args.pdf:
            build(args.pdf, args.max_pages, args.vl_limit, use_vl=not args.no_vl)
        else:
            sys.exit("[错误] build 需要 --pdf / --audio / --epub / --image")
    elif args.cmd == "ask":
        ask(get_collection(), args.question)
    elif args.cmd == "chat":
        col = get_collection()
        print("进入问答（输入 exit 退出）")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            ask(col, q)
    elif args.cmd == "agent":
        gen_agent(args.pdf, args.max_pages, args.vl_limit, use_vl=not args.no_vl)
    elif args.cmd == "brief":
        brief(get_collection(), args.topic)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
