r"""
webui.py — 知识蒸馏 RAG 管线 · 本地 Web 界面（FastAPI 后端）
================================================================
设计原则：
  1. **不改 main.py**：本文件只 import 并复用其函数（embed / _run_once / is_abstain / ...），
     所有检索、打包、生成、动态预算逻辑与 CLI 完全一致，避免"演示和实测两套代码"。
  2. **结构化溯源**：CLI 的 ask() 把来源 print 出来，Web 需要结构化数据，
     故此处复刻 ask() 的流程但返回 dict（答案 + 来源 + tokens + 是否升配）。
  3. **真流式**：检索完成后逐 token 推送生成结果（SSE），不是"先算完再假装打字"。
     动态升配也会实时告知前端（"正在补充上下文重试…"）。

跑法（在 code 目录下）：
    uvicorn webui:app --host 127.0.0.1 --port 8000
然后浏览器打开 http://127.0.0.1:8000

依赖：
    pip install fastapi uvicorn
"""
import json
import os
import re
import sys
import time
import asyncio
import importlib.util
import threading
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# 文件上传依赖 python-multipart（可选）。没装也能正常问答，只是不能从网页上传文件。
try:
    from fastapi import UploadFile, File, Form
    import multipart  # noqa: F401  仅用于探测依赖是否存在
    _UPLOAD_OK = True
except Exception:
    _UPLOAD_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M   # noqa: E402  复用 CLI 的全部管线逻辑

app = FastAPI(title="知识蒸馏 RAG · 本地问答", version="1.0")

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "webui_index.html")
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
KB_ROOT = os.path.join(PROJECT_ROOT, "data", "webui_knowledge_bases")
REGISTRY_PATH = os.path.join(KB_ROOT, "registry.json")
MAX_UPLOAD_BYTES = int(os.environ.get("DISTILL_MAX_UPLOAD_MB", "512")) * 1024 * 1024

_state_lock = threading.RLock()
_pipeline_lock = threading.RLock()
_jobs = {}
_active_job_id = None


def _resolve_db_path():
    """定位向量库（解决 main.DB_PATH 是相对路径、随启动目录漂移的问题）。
       优先级：环境变量 DISTILL_DB > 当前工作目录 ./vectordb > ../data/vectordb > code/vectordb。
       找到后固化为绝对路径写回 M.DB_PATH，Web 与 CLI 从此指向同一个库。"""
    cands = []
    env = os.environ.get("DISTILL_DB")
    if env:
        cands.append(env)
    cands += [
        M.DB_PATH,                                        # ./vectordb（CLI 默认，跟随 cwd）
        os.path.join(HERE, "..", "data", "vectordb"),     # data 目录（建库常在这里跑）
        os.path.join(HERE, "vectordb"),                   # code/vectordb
        os.path.join(HERE, "..", "vectordb"),             # 项目根/vectordb
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(M.DB_PATH)                     # 都没有 → 保持默认，前端会提示未建库


M.DB_PATH = _resolve_db_path()                            # 固化绝对路径
LEGACY_DB_PATH = M.DB_PATH


def _now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _registry_default():
    return {"version": 1, "active_id": "legacy", "legacy_db_path": LEGACY_DB_PATH,
            "libraries": []}


def _read_registry():
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("libraries"), list):
            raise ValueError("知识库索引格式错误")
        data.setdefault("version", 1)
        data.setdefault("active_id", "legacy")
        data.setdefault("legacy_db_path", LEGACY_DB_PATH)
        return data
    except (OSError, ValueError, TypeError):
        return _registry_default()


def _write_registry(data):
    """原子写入索引；进程意外中断时不留下半个 JSON。"""
    os.makedirs(KB_ROOT, exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, REGISTRY_PATH)


def _manifest_info_for(db_path):
    info = {"subject": "", "sources": [], "built_at": ""}
    try:
        with open(os.path.join(db_path, "build_manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        sources = []
        for part in (manifest.get("parts") or []):
            source = str(part.get("source") or "").strip()
            if source and source not in sources:
                sources.append(source)
        info.update({"subject": str(manifest.get("subject") or "").strip(),
                     "sources": sources,
                     "built_at": str(manifest.get("built_at") or "").strip()})
    except (OSError, ValueError, TypeError):
        pass
    return info


def _resolve_db_ref(path):
    """新知识库存项目相对路径，复制整个项目后仍能恢复。"""
    if not path:
        return ""
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path))


def _restore_active_library():
    registry = _read_registry()
    changed = False
    for item in registry["libraries"]:
        if item.get("status") == "building":
            item.update({"status": "failed", "error": "服务在建库期间重启，请重新上传"})
            changed = True
    active_id = registry.get("active_id") or "legacy"
    if active_id == "legacy":
        path = registry.get("legacy_db_path") or LEGACY_DB_PATH
    else:
        item = next((x for x in registry["libraries"]
                     if x.get("id") == active_id and x.get("status") == "ready"), None)
        path = _resolve_db_ref(item.get("db_path")) if item else None
    if path and os.path.isdir(path):
        M.DB_PATH = os.path.abspath(path)
    else:
        M.DB_PATH = os.path.abspath(registry.get("legacy_db_path") or LEGACY_DB_PATH)
        registry["active_id"] = "legacy"
        changed = True
    if changed:
        _write_registry(registry)


def _safe_filename(name):
    name = os.path.basename(name or "document.pdf")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name:
        name = "document.pdf"
    stem, ext = os.path.splitext(name)
    return (stem[:100] or "document") + (ext[:10] or ".pdf")


def _job_update(job_id, **changes):
    with _state_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)
            _jobs[job_id]["updated_at"] = _now_text()


def _public_job(job):
    if not job:
        return None
    keys = ("id", "library_id", "filename", "status", "phase", "progress", "error",
            "chunks", "created_at", "updated_at", "elapsed_seconds")
    out = {k: job.get(k) for k in keys if k in job}
    if job.get("started_monotonic") and job.get("status") in ("queued", "running"):
        out["elapsed_seconds"] = round(time.monotonic() - job["started_monotonic"], 1)
    return out


def _load_builder_module(job_id):
    """为建库加载独立的 main 模块，避免半成品 DB_PATH 污染在线问答。"""
    spec = importlib.util.spec_from_file_location("distill_builder_%s" % job_id, M.__file__)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载建库模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _activate_library(library_id):
    with _pipeline_lock, _state_lock:
        registry = _read_registry()
        if library_id == "legacy":
            path = registry.get("legacy_db_path") or LEGACY_DB_PATH
        else:
            item = next((x for x in registry["libraries"] if x.get("id") == library_id), None)
            if not item or item.get("status") != "ready":
                raise ValueError("知识库不存在或尚未建好")
            path = _resolve_db_ref(item.get("db_path"))
            if os.path.commonpath([path, os.path.abspath(KB_ROOT)]) != os.path.abspath(KB_ROOT):
                raise ValueError("知识库路径越界")
        if not path or not os.path.isdir(path):
            raise ValueError("知识库目录不存在")
        client = M.chromadb.PersistentClient(path=os.path.abspath(path))
        col = client.get_collection(M.COLLECTION)
        if col.count() < 1:
            raise ValueError("知识库为空，不能切换")
        M.DB_PATH = os.path.abspath(path)
        registry["active_id"] = library_id
        _write_registry(registry)
        return col.count()


def _build_worker(job_id, library_id, source_path, db_path, max_pages, vl_limit, use_vl):
    global _active_job_id
    started = time.monotonic()
    try:
        _job_update(job_id, status="running", phase="正在解析 PDF", progress=18,
                    started_monotonic=started)
        builder = _load_builder_module(job_id)
        builder.DB_PATH = db_path
        builder.VL_CACHE = os.path.join(os.path.dirname(db_path), "vl_cache.json")
        # 浏览器上传目录不是学科名，不能把 uploads/随机目录误写成学科。
        try:
            if os.path.commonpath([os.path.abspath(source_path), os.path.abspath(KB_ROOT)]) == os.path.abspath(KB_ROOT):
                builder._subject_of = lambda _path: None
        except ValueError:
            pass

        original_embed = builder.embed
        original_vl_parse = builder.vl_parse
        embed_calls = {"n": 0}
        vl_calls = {"n": 0}

        def progress_vl(image_bytes):
            vl_calls["n"] += 1
            _job_update(job_id, phase="正在识别图表（%d/%d）" % (vl_calls["n"], max(vl_limit, 1)),
                        progress=min(46, 20 + vl_calls["n"] * max(1, 26 // max(vl_limit, 1))))
            return original_vl_parse(image_bytes)

        def progress_embed(texts):
            embed_calls["n"] += 1
            _job_update(job_id, phase="正在生成向量并写入知识库",
                        progress=min(90, 48 + embed_calls["n"] * 6))
            return original_embed(texts)

        builder.embed = progress_embed
        builder.vl_parse = progress_vl
        pages = max_pages if max_pages and max_pages > 0 else 1000000
        builder.build(source_path, pages, vl_limit, use_vl)
        _job_update(job_id, phase="正在校验知识库", progress=94)
        client = builder.chromadb.PersistentClient(path=db_path)
        chunks = client.get_collection(builder.COLLECTION).count()
        if chunks < 1:
            raise RuntimeError("建库完成但知识块为 0")
        manifest = _manifest_info_for(db_path)

        with _state_lock:
            registry = _read_registry()
            item = next((x for x in registry["libraries"] if x.get("id") == library_id), None)
            if item is None:
                raise RuntimeError("知识库索引项丢失")
            item.update({"status": "ready", "chunks": chunks,
                         "subject": manifest.get("subject", ""),
                         "built_at": manifest.get("built_at") or _now_text(), "error": ""})
            _write_registry(registry)

        _activate_library(library_id)
        _job_update(job_id, status="ready", phase="建库完成，已自动切换", progress=100,
                    chunks=chunks, elapsed_seconds=round(time.monotonic() - started, 1))
    except BaseException as e:
        message = str(e).strip() or type(e).__name__
        with _state_lock:
            registry = _read_registry()
            item = next((x for x in registry["libraries"] if x.get("id") == library_id), None)
            if item:
                item.update({"status": "failed", "error": message[:500]})
                _write_registry(registry)
        _job_update(job_id, status="failed", phase="建库失败", progress=100,
                    error=message[:500], elapsed_seconds=round(time.monotonic() - started, 1))
    finally:
        with _state_lock:
            if _active_job_id == job_id:
                _active_job_id = None


def _start_build_job(source_path, filename, max_pages, vl_limit, use_vl):
    global _active_job_id
    with _state_lock:
        if _active_job_id and _jobs.get(_active_job_id, {}).get("status") in ("queued", "running"):
            raise RuntimeError("已有建库任务正在运行，请等待完成")
        library_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job_id = uuid.uuid4().hex
        db_path = os.path.join(KB_ROOT, library_id, "vectordb")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        registry = _read_registry()
        registry["libraries"].insert(0, {
            "id": library_id, "name": os.path.splitext(filename)[0], "source": filename,
            "status": "building", "chunks": 0, "subject": "",
            "db_path": os.path.relpath(db_path, PROJECT_ROOT),
            "created_at": _now_text(), "built_at": "", "error": ""
        })
        _write_registry(registry)
        _jobs[job_id] = {"id": job_id, "library_id": library_id, "filename": filename,
                         "status": "queued", "phase": "文件已接收，等待建库", "progress": 10,
                         "error": "", "chunks": 0, "created_at": _now_text(),
                         "updated_at": _now_text()}
        _active_job_id = job_id
        thread = threading.Thread(target=_build_worker,
                                  args=(job_id, library_id, source_path, db_path,
                                        max_pages, vl_limit, use_vl), daemon=True)
        thread.start()
        return _public_job(_jobs[job_id])


_restore_active_library()
print("[webui] 向量库路径 →", M.DB_PATH)
print("[webui] Ollama     →", M._ollama_host())


# ----------------------------- 工具 -----------------------------
def _src_of(meta):
    """把一条 metadata 转成人类可读的来源标签 + 结构化字段（与 main.ask 的 _src 一致）。"""
    t = meta.get("type")
    if t == "audio":
        return {"label": "audio %s" % meta.get("time", "?"), "type": "audio",
                "loc": meta.get("time", "?"), "page": None}
    if t in ("epub", "image"):
        return {"label": meta.get("loc", t), "type": t,
                "loc": meta.get("loc", ""), "page": None}
    return {"label": "p%d" % meta["page"], "type": t or "text",
            "loc": "", "page": meta.get("page")}


def _collection():
    """取集合。注意：main.get_collection() 在没库时会调 sys.exit()，
       抛的是 SystemExit（继承 BaseException，不被 except Exception 捕获），
       故这里显式转成普通异常，避免整个请求 500。"""
    try:
        return M.get_collection()
    except SystemExit as e:
        raise RuntimeError(str(e) or "还没建库")


def _library_info():
    """读取建库清单，供演示页显示当前教材，避免示例问题与知识库错位。"""
    return _manifest_info_for(M.DB_PATH)


def _libraries_payload():
    with _state_lock:
        registry = _read_registry()
        active_id = registry.get("active_id") or "legacy"
        legacy_path = os.path.abspath(registry.get("legacy_db_path") or LEGACY_DB_PATH)
        legacy_info = _manifest_info_for(legacy_path)
        legacy_source = (legacy_info.get("sources") or [""])[0]
        libraries = [{"id": "legacy", "name": legacy_source or "原有知识库",
                      "source": legacy_source, "subject": legacy_info.get("subject", ""),
                      "status": "ready" if os.path.isdir(legacy_path) else "missing",
                      "chunks": None, "built_at": legacy_info.get("built_at", ""),
                      "active": active_id == "legacy"}]
        for item in registry["libraries"]:
            public = {k: item.get(k) for k in
                      ("id", "name", "source", "subject", "status", "chunks",
                       "created_at", "built_at", "error")}
            public["active"] = item.get("id") == active_id
            libraries.append(public)
        return {"active_id": active_id, "libraries": libraries,
                "build": _public_job(_jobs.get(_active_job_id)) if _active_job_id else None}


def _retrieve(question):
    """检索：返回 (docs, metas, dists)，完整复用 main 的扩写与 VL 配额。"""
    qv = M.embed([question])[0]
    return M._retrieve(_collection(), qv, question)


def _sources_from(metas, packed_idx, docs=None):
    """按实际打包的块下标取来源（相关度裁剪下打包的不一定是前几块）。"""
    out, seen = [], set()
    for i in packed_idx:
        if i >= len(metas):
            continue
        s = _src_of(metas[i])
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        if docs is not None and i < len(docs):
            s["snippet"] = docs[i][:220]      # 给前端展开看原文片段
        out.append(s)
    return sorted(out, key=lambda x: x["label"])


# ----------------------------- API -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(INDEX_HTML):
        return HTMLResponse(open(INDEX_HTML, encoding="utf-8").read())
    return HTMLResponse("<h3>webui_index.html 不存在，请与 webui.py 放在同一目录。</h3>")


@app.get("/api/status")
def status():
    """系统状态：模型、知识块数、是否离线、关键配置 + Ollama 探活诊断。"""
    host = M._ollama_host()
    info = {
        "ollama": "未连接",
        "ollama_host": host,          # 暴露出来，连不上时一眼看出地址对不对
        "chunks": 0,
        "llm_model": M.LLM_MODEL,
        "vl_model": M.VL_MODEL,
        "embed_model": M.EMBED_MODEL,
        "offline": True,
        "relevance_trim": bool(M.RELEVANCE_TRIM),
        "context_budget": M.CONTEXT_BUDGET,
        "budget_escalated": M.BUDGET_ESCALATED,
        "top_k": M.TOP_K,
        "db_path": os.path.abspath(M.DB_PATH),
        "library": _library_info(),
        "active_library_id": _read_registry().get("active_id", "legacy"),
        "build": _public_job(_jobs.get(_active_job_id)) if _active_job_id else None,
        "cwd": os.getcwd(),           # 相对路径 DB_PATH 跟随工作目录，暴露出来便于排查
        "ready": False,
        "upload_enabled": _UPLOAD_OK,
    }
    # 探活：直接打 Ollama 根路径
    try:
        import urllib.request
        with urllib.request.urlopen(host, timeout=3) as r:
            r.read(64)
        info["ollama"] = "已连接"
    except Exception as e:
        info["ollama"] = "连不上"
        info["ollama_error"] = "%s → %s" % (host, str(e)[:120])
    # 知识库
    try:
        info["chunks"] = _collection().count()
        info["ready"] = info["chunks"] > 0 and info["ollama"] == "已连接"
    except Exception as e:
        info["db_error"] = str(e)[:160]
    return info


@app.post("/api/ask")
def api_ask(payload: dict):
    """非流式问答：一次返回答案 + 结构化来源。逻辑与 CLI ask() 完全一致。"""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    t0 = time.time()
    docs, metas, dists = _retrieve(question)

    # 第一档：常态预算（保 Token 效率）
    answer, toks, packed_idx = M._run_once(docs, question, M.CONTEXT_BUDGET, metas)
    escalated = False
    # 动态升配：仅当"检索有命中却拒答"时升配重答（与 CLI 同策略）
    if M.should_escalate(answer, docs, dists, M.DYNAMIC_BUDGET):
        ans2, toks2, idx2 = M._run_once(docs, question, M.BUDGET_ESCALATED, metas)
        toks += toks2
        escalated = True
        if not M.is_abstain(ans2):
            answer, packed_idx = ans2, idx2

    return {
        "answer": answer,
        "abstained": M.is_abstain(answer),
        "sources": _sources_from(metas, packed_idx, docs),
        "tokens": toks,
        "escalated": escalated,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED if escalated else M.CONTEXT_BUDGET,
        "cite_check": M.verify_citations(answer, packed_idx, metas),
    }


@app.post("/api/brief")
def api_brief(payload: dict):
    """非流式简报生成：围绕 topic 综合检索资料，产出带出处的结构化 brief 文档。
       复用 main.brief 的生成逻辑（_run_once_brief），返回结构与 /api/ask 一致，前端可复用渲染。"""
    topic = (payload.get("topic") or payload.get("question") or "").strip()
    if not topic:
        return JSONResponse({"error": "主题为空"}, status_code=400)

    t0 = time.time()
    docs, metas, _dists = _retrieve(topic)
    answer, toks, packed_idx = M._run_once_brief(docs, topic, M.BUDGET_ESCALATED, metas)

    return {
        "answer": answer,
        "abstained": False,
        "sources": _sources_from(metas, packed_idx, docs),
        "tokens": toks,
        "escalated": False,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED,
        "cite_check": M.verify_citations(answer, packed_idx, metas),
        "mode": "brief",
    }


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


@app.get("/api/ask/stream")
async def api_ask_stream(q: str):
    """流式问答（SSE）。真流式：检索完先推来源，再逐 token 推生成结果。
       若首答拒答且检索有命中 → 推 escalate 事件，再流式重答（把动态预算演给用户看）。"""
    question = (q or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    async def gen():
        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            docs, metas, dists = await loop.run_in_executor(None, _retrieve, question)
        except Exception as e:
            msg = str(e)[:200]
            if "10061" in msg or "refused" in msg.lower() or "urlopen" in msg.lower():
                msg = ("连不上 Ollama（%s）。请确认：① Ollama 已启动（ollama list 能列出模型）；"
                       "② 若设过 OLLAMA_HOST 环境变量，需与 Ollama 实际监听地址一致。"
                       % M._ollama_host())
            yield _sse("error", {"msg": msg})
            return
        yield _sse("retrieved", {"n": len(docs)})

        async def run(budget, tag):
            """按预算打包 → 流式生成。返回 (完整答案, packed_idx, tokens)。"""
            if M.RELEVANCE_TRIM:
                packed, packed_idx = M._pack_relevance(docs, question, budget)
            else:
                packed, packed_idx = M._pack_truncate(docs, budget)
            context = M._labeled_context(packed, packed_idx, metas)   # 每块带真页标签，让模型引用真页码
            prompt = M._format_prompt(context, question, packed_idx, metas)

            yield ("meta", {"budget": budget, "tag": tag,
                            "sources": _sources_from(metas, packed_idx, docs)})

            q_out = asyncio.Queue()

            def produce():
                """在线程里跑 Ollama 流式，把分片塞进队列。

                ⚠️ 必须走 /api/generate（补全模式），与 main.py 的 _generate 一致。
                   走 /api/chat（对话模式）会套用聊天模板，模型天性"乐于助人"，
                   会脱离 Material 用自身知识作答、还硬安一个页码 —— 直接破坏
                   Citation Grounding 的防幻觉保证。这是实测踩过的坑，勿改。
                """
                try:
                    import ollama
                    stream = ollama.generate(
                        model=M.LLM_MODEL,
                        prompt=prompt,                    # prompt 以 "Answer:" 结尾 → 补全模式
                        think=False,
                        stream=True,
                        options={"temperature": M.TEMPERATURE,
                                 "num_predict": M.NUM_PREDICT},
                    )
                    for part in stream:
                        piece = part.get("response", "")
                        if piece:
                            loop.call_soon_threadsafe(q_out.put_nowait, ("d", piece))
                        if part.get("done"):
                            tk = (part.get("prompt_eval_count", 0) or 0) + \
                                 (part.get("eval_count", 0) or 0)
                            loop.call_soon_threadsafe(q_out.put_nowait, ("t", tk))
                except Exception as e:
                    loop.call_soon_threadsafe(q_out.put_nowait, ("e", str(e)[:200]))
                loop.call_soon_threadsafe(q_out.put_nowait, (None, None))

            loop.run_in_executor(None, produce)

            buf, toks = [], 0
            while True:
                kind, val = await q_out.get()
                if kind is None:
                    break
                if kind == "d":
                    buf.append(val)
                    yield ("delta", {"text": val})
                elif kind == "t":
                    toks = val
                elif kind == "e":
                    yield ("error", {"msg": val})
                    break
            full = M._strip_think("".join(buf)).strip()
            yield ("__done__", (full, packed_idx, toks))

        # ---- 第一档 ----
        answer = ""
        packed_idx, toks = [], 0
        async for ev, data in run(M.CONTEXT_BUDGET, "first"):
            if ev == "__done__":
                answer, packed_idx, toks = data
            else:
                yield _sse(ev, data)

        # ---- 动态升配（仅当检索有命中却拒答）----
        escalated = False
        if M.should_escalate(answer, docs, dists, M.DYNAMIC_BUDGET):
            escalated = True
            yield _sse("escalate", {"from": M.CONTEXT_BUDGET, "to": M.BUDGET_ESCALATED,
                                    "reason": "首答拒答且检索有命中，补充上下文重试"})
            a2, idx2, tk2 = "", [], 0
            async for ev, data in run(M.BUDGET_ESCALATED, "escalated"):
                if ev == "__done__":
                    a2, idx2, tk2 = data
                else:
                    yield _sse(ev, data)
            toks += tk2
            if not M.is_abstain(a2):          # 升配后答出来了才采用
                answer, packed_idx = a2, idx2

        yield _sse("done", {
            "answer": answer,
            "abstained": M.is_abstain(answer),
            "sources": _sources_from(metas, packed_idx, docs),
            "tokens": toks,
            "escalated": escalated,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "cite_check": M.verify_citations(answer, packed_idx, metas),
        })

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/libraries")
def api_libraries():
    """列出最近知识库，供左侧栏一键切换。"""
    return _libraries_payload()


@app.post("/api/libraries/{library_id}/activate")
def api_activate_library(library_id: str):
    try:
        chunks = _activate_library(library_id)
        return {"ok": True, "active_id": library_id, "chunks": chunks,
                "library": _library_info()}
    except (ValueError, OSError) as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=500)


@app.get("/api/build/jobs/{job_id}")
def api_build_job(job_id: str):
    with _state_lock:
        job = _public_job(_jobs.get(job_id))
    if job is None:
        return JSONResponse({"error": "建库任务不存在或服务已重启"}, status_code=404)
    return job


@app.get("/api/build/active")
def api_active_build():
    with _state_lock:
        return {"job": _public_job(_jobs.get(_active_job_id)) if _active_job_id else None}


@app.post("/api/build")
async def api_build(payload: dict):
    """兼容旧调用：按服务器本地路径建库。PDF 改为安全的后台独立建库。"""
    kind = (payload.get("kind") or "").strip().lower()
    path = (payload.get("path") or "").strip()
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "文件不存在：%s" % path}, status_code=400)
    if kind != "pdf":
        return JSONResponse({"error": "Web 管理器目前仅支持 PDF；其他格式请继续使用原 CLI"},
                            status_code=400)
    try:
        filename = _safe_filename(os.path.basename(path))
        job = _start_build_job(path, filename, int(payload.get("max_pages") or 0),
                               max(0, int(payload.get("vl_limit") or 15)),
                               bool(payload.get("use_vl", True)))
        return JSONResponse({"ok": True, "job": job}, status_code=202)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=500)


if _UPLOAD_OK:
    @app.post("/api/upload")
    async def api_upload(kind: str = Form("pdf"), max_pages: int = Form(0),
                         vl_limit: int = Form(15), use_vl: bool = Form(True),
                         file: UploadFile = File(...)):
        """流式保存 PDF 后启动后台建库。旧知识库直到新库校验成功前始终可用。"""
        if kind.lower() != "pdf":
            return JSONResponse({"error": "当前前端建库仅支持 PDF"}, status_code=400)
        filename = _safe_filename(file.filename or "document.pdf")
        if os.path.splitext(filename)[1].lower() != ".pdf":
            return JSONResponse({"error": "请选择 PDF 文件"}, status_code=400)
        try:
            with _state_lock:
                if _active_job_id and _jobs.get(_active_job_id, {}).get("status") in ("queued", "running"):
                    return JSONResponse({"error": "已有建库任务正在运行，请等待完成"}, status_code=409)
            up = os.path.join(KB_ROOT, "uploads", uuid.uuid4().hex)
            os.makedirs(up, exist_ok=True)
            path = os.path.join(up, filename)
            total = 0
            with open(path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError("PDF 超过上传上限（%d MB）" % (MAX_UPLOAD_BYTES // 1024 // 1024))
                    f.write(chunk)
            if total < 5:
                raise ValueError("PDF 文件为空或不完整")
            with open(path, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("文件内容不是有效 PDF")
            job = _start_build_job(path, filename, max(0, min(int(max_pages), 10000)),
                                   max(0, min(int(vl_limit), 100)), bool(use_vl))
            return JSONResponse({"ok": True, "job": job, "size": total}, status_code=202)
        except ValueError as e:
            if "path" in locals() and os.path.isfile(path):
                os.remove(path)
            return JSONResponse({"error": str(e)}, status_code=400)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except Exception as e:
            return JSONResponse({"error": str(e)[:300]}, status_code=500)
        finally:
            await file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
