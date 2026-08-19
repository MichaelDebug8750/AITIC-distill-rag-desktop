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
import hashlib
import os
import re
import sys
import time
import asyncio
import importlib.util
import threading
import unicodedata
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# 文件上传依赖 python-multipart（可选）。没装也能正常问答，只是不能从网页上传文件。
try:
    from fastapi import UploadFile, File, Form
    import python_multipart  # noqa: F401  仅用于探测依赖是否存在；旧 multipart 别名已弃用
    _UPLOAD_OK = True
except Exception:
    _UPLOAD_OK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M   # noqa: E402  复用 CLI 的全部管线逻辑

app = FastAPI(title="知识蒸馏 RAG · 本地问答", version="1.0")

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "webui_index.html")
# 桌面冻结包的源码位于 PyInstaller 的只读内部目录，知识库、反馈和上传文件则
# 必须写到用户目录。开发版/原 WebUI 不设置此变量时仍保持原来的父目录语义；
# 原生桌面入口会在 import 本模块之前显式设置 AITIC_PROJECT_ROOT。
PROJECT_ROOT = os.path.abspath(
    os.environ.get("AITIC_PROJECT_ROOT") or os.path.join(HERE, ".."))
KB_ROOT = os.path.join(PROJECT_ROOT, "data", "webui_knowledge_bases")
REGISTRY_PATH = os.path.join(KB_ROOT, "registry.json")
MAX_UPLOAD_BYTES = int(os.environ.get("DISTILL_MAX_UPLOAD_MB", "512")) * 1024 * 1024
_QUERY_MAX_CHARS = 4000
_TOPIC_MAX_CHARS = 1000
_HIGHLIGHT_MAX_CHARS = 2000
_FEEDBACK_ANSWER_MAX_CHARS = 20000

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


_OLLAMA_VERSION_TTL = 30.0
_ollama_version_cache = {"at": 0.0, "value": None}


def _ollama_server_version():
    """Ollama **服务端**版本。

    ``packages["ollama"]`` 记的是 Python 客户端包，与服务端各自独立升级。
    v6 事故（模糊题掉约 30pp，症状是只吐 ``[p.955]`` 不写正文）的真因正是
    服务端 0.31.1 → 0.32.3，而当时的清单里只有客户端版本——**肇事者一格都没记**，
    排查因此绕了很久。查一次 ``/api/version`` 就能补上这一格。

    取不到时返回 ``"unreachable"``：清单缺一格可以接受，跑分因为记录版本而中断不行。
    30 秒 TTL 是折中——``/api/status`` 会被前端反复轮询，不能每次都发请求；
    但缓存又不能长到让清单的起止两次记录看不出中途换过服务端，
    而全量跑分以小时计，30 秒远小于它。
    """
    now = time.time()
    if (_ollama_version_cache["value"] is not None
            and now - _ollama_version_cache["at"] < _OLLAMA_VERSION_TTL):
        return _ollama_version_cache["value"]
    value = "unreachable"
    try:
        import urllib.request
        with urllib.request.urlopen(M._ollama_host() + "/api/version", timeout=2) as resp:
            value = str(json.loads(resp.read().decode("utf-8")).get("version") or "unknown")
    except Exception:
        value = "unreachable"
    _ollama_version_cache.update({"at": now, "value": value})
    return value


def _runtime_info():
    """返回当前**实际**运行时，供跑分产物与故障排查自证。"""
    server = _ollama_server_version()
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - 项目支持的 Python 版本都有此模块
        return {"python": sys.version.split()[0], "packages": {},
                "ollama_server": server}
    packages = {}
    for label, distribution in (
            ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("pymupdf", "pymupdf"),
            ("chromadb", "chromadb"), ("ollama", "ollama"),
            ("python-multipart", "python-multipart"), ("modelscope", "modelscope")):
        try:
            packages[label] = version(distribution)
        except PackageNotFoundError:
            packages[label] = "missing"
    # ollama_server 与 packages["ollama"] 是两件事，别再混用：前者是服务端，后者是客户端库。
    return {"python": sys.version.split()[0], "packages": packages,
            "ollama_server": server}


def _registry_default():
    return {"version": 1, "active_id": "legacy", "legacy_db_path": LEGACY_DB_PATH,
            "libraries": []}


def _read_registry():
    # 有些路径（如 /api/status）会在调用者没持有 _state_lock 时读取。
    # Windows 上打开的目标文件可能短暂阻止 os.replace，所以读写锁必须
    # 收口在 helper 内，不能依赖每个新调用者都记得加锁。RLock 保证已持锁
    # 的读-改-写路径仍可安全调用。
    with _state_lock:
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
    """原子写入索引；进程意外中断时不留下半个 JSON。

    临时文件必须唯一：同一项目被两个服务进程短暂重叠运行时，
    固定 registry.json.tmp 会被彼此替换。Windows 还会在杀毒软件或
    另一读者短暂占用目标时让 os.replace 返回 PermissionError；这类短暂
    拒绝做有界重试，其他写入错误仍立即暴露。
    """
    with _state_lock:
        os.makedirs(KB_ROOT, exist_ok=True)
        tmp = "%s.%d.%s.tmp" % (REGISTRY_PATH, os.getpid(), uuid.uuid4().hex)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(5):
                try:
                    os.replace(tmp, REGISTRY_PATH)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (2 ** attempt))
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


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


def _discard_pending_upload(path):
    """移除尚未交给后台建库线程的上传文件及其空 UUID 目录。"""
    if not path:
        return
    upload_root = os.path.abspath(os.path.join(KB_ROOT, "uploads"))
    candidate = os.path.abspath(path)
    try:
        if os.path.commonpath([candidate, upload_root]) != upload_root:
            return
    except ValueError:
        return
    try:
        if os.path.isfile(candidate):
            os.remove(candidate)
        parent = os.path.dirname(candidate)
        # 只删 uploads 的直属 UUID 空目录；os.rmdir 非空时会拒绝，不做递归删除。
        if os.path.dirname(parent) == upload_root and os.path.isdir(parent):
            os.rmdir(parent)
    except OSError:
        pass


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


def _builder_module_source():
    """返回可供隔离建库线程重新加载的 ``main.py`` 源文件。

    源码运行时 ``M.__file__`` 指向真实文件；PyInstaller 冻结后它指向 PYZ 中的
    逻辑位置，磁盘上并不存在。安装版会额外携带一份只读 ``code/main.py``，
    因而必须显式回退到 ``sys._MEIPASS``。不能直接复用全局 ``M``：建库会修改
    DB_PATH、VL_CACHE 以及进度包装器，复用会把在线问答切到半成品知识库。
    """
    candidates = [getattr(M, "__file__", "")]
    frozen_root = getattr(sys, "_MEIPASS", "")
    if frozen_root:
        candidates.append(os.path.join(frozen_root, "code", "main.py"))
    candidates.extend((
        os.path.join(HERE, "code", "main.py"),
        os.path.join(HERE, "main.py"),
    ))
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.fspath(candidate))
        if path in checked:
            continue
        checked.append(path)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "找不到建库核心 main.py；请重新安装完整的 AITIC Desktop（已检查：%s）"
        % ", ".join(checked))


def _load_builder_module(job_id):
    """为建库加载独立的 main 模块，避免半成品 DB_PATH 污染在线问答。"""
    source = _builder_module_source()
    spec = importlib.util.spec_from_file_location("distill_builder_%s" % job_id, source)
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


def _build_worker(job_id, library_id, source_path, db_path, max_pages, vl_limit, use_vl, vl_from):
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
        builder.build(source_path, pages, vl_limit, use_vl, vl_from)
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


def _start_build_job(source_path, filename, max_pages, vl_limit, use_vl, vl_from=1):
    global _active_job_id
    with _state_lock:
        if _active_job_id and _jobs.get(_active_job_id, {}).get("status") in ("queued", "running"):
            raise RuntimeError("已有建库任务正在运行，请等待完成")
        library_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job_id = uuid.uuid4().hex
        db_path = os.path.join(KB_ROOT, library_id, "vectordb")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        registry = _read_registry()
        source_abs = os.path.abspath(source_path)
        try:
            source_ref = (os.path.relpath(source_abs, PROJECT_ROOT)
                          if os.path.commonpath([source_abs, PROJECT_ROOT]) == PROJECT_ROOT
                          else source_abs)
        except ValueError:
            source_ref = source_abs
        registry["libraries"].insert(0, {
            "id": library_id, "name": os.path.splitext(filename)[0], "source": filename,
            # Internal-only exact source binding.  Public library payloads deliberately
            # omit this field, but the original-page viewer must never guess between
            # two uploaded PDFs that happen to share the same basename.
            "source_path": source_ref,
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
                                        max_pages, vl_limit, use_vl, max(1, int(vl_from))), daemon=True)
        thread.start()
        return _public_job(_jobs[job_id])


_restore_active_library()
print("[webui] 向量库路径 →", M.DB_PATH)
print("[webui] Ollama     →", M._ollama_host())


# ----------------------------- 工具 -----------------------------
def _src_of(meta):
    """把一条 metadata 转成人类可读的来源标签 + 结构化字段（与 main.ask 的 _src 一致）。

    ``library_id`` 供前端"查看原文"回查该页全文用——只给标签的话，
    多库场景下无法知道这条引用出自哪个库。
    """
    library_name = str(meta.get("_library_name") or "").strip()
    library_id = str(meta.get("_library_id") or "").strip()
    library_alias = str(meta.get("_library_alias") or "").strip()
    label_prefix = ((library_alias + " · ")
                    if meta.get("_multi_library") and library_alias else "")
    if library_name:
        label_prefix += library_name + " · "
    source_name = str(meta.get("source") or meta.get("_library_source") or "").strip()
    t = meta.get("type")
    if t == "audio":
        label = "audio %s" % meta.get("time", "?")
        return {"label": label_prefix + label,
                "type": "audio", "loc": meta.get("time", "?"), "page": None,
                "library": library_name, "library_id": library_id, "source": source_name}
    if t in ("epub", "image"):
        label = str(meta.get("loc", t))
        return {"label": label_prefix + label,
                "type": t, "loc": meta.get("loc", ""), "page": None,
                "library": library_name, "library_id": library_id, "source": source_name}
    label = "p%d" % meta.get("page", 0)
    return {"label": label_prefix + label,
            "type": t or "text", "loc": "", "page": meta.get("page"),
            "library": library_name, "library_id": library_id, "source": source_name}


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


def _retrieve(question, hybrid=None, scope=None):
    """检索：返回 (docs, metas, dists)，完整复用 main 的扩写与 VL 配额。
       hybrid 开启时并入关键词召回（RRF 融合）；scope 限定页范围。默认都关闭。"""
    col = _collection()
    if scope:
        return _retrieve_scoped(col, question, scope)
    if _hybrid_enabled(hybrid):
        return _retrieve_hybrid(col, question)
    qv = M.embed([question])[0]
    return M._retrieve(col, qv, question)


def _normalize_library_ids(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = [x for x in raw.split(",") if x]
    if not isinstance(raw, list):
        return []
    out = []
    for value in raw:
        value = str(value or "").strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= 4:
            break
    return out


def _coerce_bool(value, default=False):
    """解析 JSON/Form 常见布尔表示，避免 bool("false") 反而得到 True。"""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "on", "yes"):
            return True
        if normalized in ("0", "false", "off", "no", ""):
            return False
        return bool(default)
    return bool(value)


def _bounded_int(value, default, lower, upper):
    """容错解析用户数值，并在进入耗时/持久化路径前限制边界。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = int(default)
    return min(int(upper), max(int(lower), parsed))


def _text_limit_error(value, label, limit):
    """在进入嵌入/生成路径前拒绝异常大的文本。

    历史和角色说明本来已有裁剪，但主问题曾可以无上限地送进
    embedding 和 LLM。默默截断会改变用户问题，所以返回可操作的 400
    比拿半句话去检索更诚实。
    """
    if len(value) <= int(limit):
        return None
    return JSONResponse(
        {"error": "%s过长（%d 个字符），单次最多 %d 个字符；请拆分后重试"
                  % (label, len(value), int(limit))},
        status_code=400)


class LibraryUnavailable(Exception):
    """显式选择的知识库全部不可用。

    绝不能静默改用别的库作答——那会造成"用户选了 A，系统却从 B 里找依据"，
    而答案上的引用看起来完全正常，用户无从察觉。宁可明确失败。
    """

    def __init__(self, requested):
        self.requested = list(requested or [])
        super().__init__("所选知识库均不可用：%s" % "、".join(self.requested))


def _resolve_library_targets(requested):
    """解析前端选择，返回 ``(targets, dropped)``。

    ``dropped`` 是"请求了但当前不可用"的库 ID（已删除、建库未完成、目录丢失）。
    - 没显式指定库 → 回退到当前激活库，这是正常默认，不算 dropped。
    - 指定了但部分不可用 → 用可用的那些，并把 dropped 交给调用方去提示用户。
    - 指定了但**全部**不可用 → 抛 :class:`LibraryUnavailable`，不允许改用别的库。
    """
    registry = _read_registry()
    active_id = registry.get("active_id") or "legacy"
    explicit = _normalize_library_ids(requested)
    ids = explicit or [active_id]
    items = {str(x.get("id")): x for x in registry.get("libraries", [])}
    targets, dropped = [], []
    for library_id in ids:
        path = name = source = None
        allowed_sources = []
        if library_id == "legacy":
            path = os.path.abspath(registry.get("legacy_db_path") or LEGACY_DB_PATH)
            info = _manifest_info_for(path)
            allowed_sources = [str(x).strip() for x in (info.get("sources") or [])
                               if str(x).strip()]
            source = (allowed_sources or ["原有知识库"])[0]
            name = source or "原有知识库"
        else:
            item = items.get(library_id)
            if item and item.get("status") == "ready":
                path = _resolve_db_ref(item.get("db_path"))
                name = str(item.get("name") or item.get("source") or library_id)
                source = str(item.get("source") or name)
                allowed_sources = [source] if source else []
        if path and os.path.isdir(path):
            targets.append({"id": library_id, "path": path, "name": name, "source": source,
                            "allowed_sources": allowed_sources,
                            "source_path": (item or {}).get("source_path")
                            if library_id != "legacy" else None})
        else:
            dropped.append(library_id)

    if targets:
        return targets, dropped
    if explicit:
        raise LibraryUnavailable(explicit)
    # 未显式指定且默认库也解析不出来时，才退回运行时 DB_PATH。
    fallback_sources = [str(x).strip() for x in (_library_info().get("sources") or [])
                        if str(x).strip()]
    return [{"id": active_id, "path": M.DB_PATH,
             "name": (fallback_sources or ["当前知识库"])[0],
             "source": (fallback_sources or [""])[0],
             "allowed_sources": fallback_sources}], dropped


def _library_targets(requested):
    """兼容旧调用点：只取 targets。不可用时按上面的契约抛异常，不再静默回退。"""
    return _resolve_library_targets(requested)[0]


@app.exception_handler(LibraryUnavailable)
async def _library_unavailable_handler(_request, exc):
    """统一收口，保证任何端点都不会悄悄换一个库把问题答了。

    放在 app 层而不是逐个端点 try，是为了将来新增端点时不会漏掉这条契约。
    （流式端点响应已经开始、发不出状态码，另在生成器里就地处理。）
    """
    return JSONResponse({"error": "所选知识库已不可用，请重新选择；系统不会改用其他知识库作答。",
                         "unavailable": exc.requested}, status_code=409)


# ----------------------------- 混合检索（关键词 + 向量，RRF 融合）-----------------------------
# 向量检索擅长语义相近，但对**精确术语、专有名词、代码标识符**反而不稳；
# 关键词检索正好相反。业界 2026 的标准做法是两路并行召回后用 RRF 融合。
#
# 为什么用 RRF 而不是加权分数：本项目自己踩过这个坑——短查询扩写 v1 按检索距离归并，
# 结果扩写后的查询"对自己邻居的距离系统性更低"，整体挤掉原查询结果，净 −5 道。
# 距离来自不同查询向量，**不可比**；RRF 只用名次，天然免疫这个问题。
#
# 默认关闭：改检索链路必须先量过再开（项目方法论「改完必须回归验证已发表的指标」）。
HYBRID_RRF_K = 60
HYBRID_KEYWORD_POOL = 40


def _query_terms(question):
    """从问句里取用于关键词召回的实词。中文用二元组近似词，英文取 4 字以上非停用词。"""
    latin = [w for w in _latin_words(question) if len(w) >= 4]
    cjk = [w for w in _cjk_bigrams(question)]
    # 也保留原文里的"代码样"标识符与带点短语（dict.items、__init__），向量最不擅长这类
    ident = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|\b[a-z_]{2,}__[a-z_]+\b",
                       str(question or ""))
    return list(dict.fromkeys(ident + sorted(latin, key=len, reverse=True)[:6] + cjk[:8]))


def _question_anchors(question):
    """提取足以做精确召回的高信号锚点，不把普通实词当成硬匹配。

    中文只接受引号中的术语；拉丁词只在问句明确询问语源/由来时接受；另外
    接受 ``C-h v`` 这类键位串。这样可以救回向量不擅长的专名和代码标识符，
    又不会把 ``circuit``/``circuitous`` 一类普通词形碰撞引入默认路径。
    """
    text = str(question or "")
    anchors = []
    for value in re.findall(r'[“「『\"]([^”」』\"]{2,32})[”」』\"]', text):
        value = value.strip()
        if re.search(r"[\u3400-\u9fff]", value):
            anchors.append(value)
    anchors.extend(re.findall(r"\b[A-Za-z]-[A-Za-z]\s+[A-Za-z]\b", text))
    if re.search(r"语源|词源|由来|怎么来|如何来|起源|etymolog|origin", text, re.I):
        anchors.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text))
    # 超级词与问法对象共同构成高信号短语；仅抽取对象，随后仍需“最早”近邻消歧。
    anchors.extend(re.findall(
        r"最早(?:的)?([\u3400-\u9fff]{2,12}?)(?:是|有|由|怎么|如何|为何|何时|在哪|[？?]|$)",
        text))
    # 历史教材里的“某朝货币是什么”是另一类向量易漂移、词面却很明确的问法。
    # 只抽取带朝代/帝国后缀的实体，不能把普通“货币”升级成全局锚点。
    anchors.extend(re.findall(
        r"([\u3400-\u9fff]{1,8}(?:王朝|帝国|朝))(?:中|的)?(?:使用|用|发行)?(?:的)?货币",
        text))
    return list(dict.fromkeys(value.strip() for value in anchors if value.strip()))


def _anchor_cues(question):
    """把问法归一为只用于候选消歧的强提示词，不参与普通关键词排序。"""
    text = str(question or "")
    cues = []
    if re.search(r"语源|词源|由来|怎么来|如何来|起源|etymolog|origin", text, re.I):
        cues.extend(("语源", "词源", "来源", "源于", "起源", "由来"))
    if re.search(r"指的是什么|指什么|所说的|定义|what does|what is", text, re.I):
        cues.extend(("叫作", "称为", "定义为", "是指", "指的是", "所谓"))
    if re.search(r"最早", text):
        cues.append("最早")
    if re.search(r"(?:王朝|帝国|朝)(?:中|的)?.{0,6}货币", text):
        cues.extend(("货币", "纸币", "铜钱"))
    return list(dict.fromkeys(cues))


def _anchor_in_document(anchor, document):
    """拉丁锚点按完整词匹配，避免 ``bank`` 命中 ``banking``。"""
    anchor = str(anchor or "").strip()
    document = str(document or "")
    if not anchor:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", anchor):
        return bool(re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" %
                              re.escape(anchor), document, re.I))
    return re.sub(r"\s+", "", anchor).casefold() in re.sub(r"\s+", "", document).casefold()


def _anchor_rescue(col, question, docs, metas, dists, pool=64):
    """把唯一精确术语块补到向量结果首位，同时保留原始距离口径。

    这条路径只在向量证据本来已通过 0.99 下限时工作，绝不靠关键词独立救活
    一道低相关问题。新补块没有伪造向量距离，记为 ``None``；既有向量块若被
    锚点命中则沿用其真实距离。宽泛锚点必须再被语源/定义提示缩到至多两块。
    """
    anchors = _question_anchors(question)
    if not anchors or _evidence_floor_blocks(dists):
        return docs, metas, dists
    cues = _anchor_cues(question)
    rescued = []
    for anchor in anchors:
        try:
            got = col.get(where_document={"$contains": anchor}, limit=int(pool),
                          include=["documents", "metadatas"])
        except Exception:
            continue
        candidates = [
            (doc, meta) for doc, meta in zip(
                (got or {}).get("documents") or [], (got or {}).get("metadatas") or [])
            if _anchor_in_document(anchor, doc)
        ]
        original_count = len(candidates)
        if len(candidates) > 2 and cues:
            # 提示词的区分力并不相同。把“语源、来源、起源……”全部 OR 在一起，
            # 会让一个本可由“语源”唯一定位的术语仍保留十余块，从而错过救援。
            # 按强到弱逐个尝试；只有某个单一提示能把候选缩到至多两块时才采用。
            # 这仍然保留“宽泛锚点不进入默认检索”的原安全边界。
            for cue in cues:
                narrowed = [(doc, meta) for doc, meta in candidates
                            if cue in str(doc or "")]
                # 长章节块里可能同时出现多个术语及其“语源”。优先要求问句锚点
                # 与提示词相邻，避免仅凭同块共现把无关章节抬到首位。
                anchor_flat = re.sub(r"\s+", "", anchor)
                selected = None
                # 先认“叫作‘通货’ / ‘商人’的语源 / 最早的中央银行”这类
                # 近乎词组级的关系；只有过窄窗口找不到时才放宽到一句内。
                for window in (4, 12):
                    near = []
                    for doc, meta in narrowed:
                        flat = re.sub(r"\s+", "", str(doc or ""))
                        gap = r".{0,%d}" % window
                        if (re.search(re.escape(anchor_flat) + gap + re.escape(cue), flat, re.I)
                                or re.search(re.escape(cue) + gap + re.escape(anchor_flat), flat, re.I)):
                            near.append((doc, meta))
                    if 0 < len(near) <= 2:
                        selected = near
                        break
                if selected:
                    candidates = selected
                    break
                if 0 < len(narrowed) <= 2:
                    candidates = narrowed
                    break
        if not candidates or len(candidates) > 2:
            continue
        for doc, meta in candidates:
            rescued.append((doc, dict(meta or {}, _exact_anchor=anchor,
                                      _exact_anchor_df=original_count)))

    if not rescued:
        return docs, metas, dists
    existing = {str(doc): (meta, dist) for doc, meta, dist in zip(docs, metas, dists)}
    merged_docs, merged_metas, merged_dists, seen = [], [], [], set()
    for doc, meta in rescued:
        key = str(doc)
        if key in seen:
            continue
        old_meta, old_dist = existing.get(key, ({}, None))
        merged_docs.append(doc)
        merged_metas.append(dict(old_meta or {}, **meta))
        merged_dists.append(old_dist)
        seen.add(key)
    for doc, meta, dist in zip(docs, metas, dists):
        key = str(doc)
        if key in seen:
            continue
        merged_docs.append(doc); merged_metas.append(meta); merged_dists.append(dist)
        seen.add(key)
    return (merged_docs[:M.TOP_K], merged_metas[:M.TOP_K], merged_dists[:M.TOP_K])


def _keyword_df_max(col):
    """关键词召回的"停用词阈值"：命中块数达到这个数的词直接不计。

    取语料块数的一个比例——比例而非绝对值，因为"命中 50 块"在 218 块的库里
    是停用词，在 10678 块的库里可能是个正经术语。

    【2026-08-14 更正：默认改为关闭。此前写在这里的实验结论是错的。】

    原先这里写着"混合+本过滤把中文拒答从 0/10 修回 10/10"，据此默认开启（比例 0.2）。
    三条实测把它推翻：

    1. **本过滤从未触发过。** 取候选时带 `limit=POOL=40`，`len(ids)` 最大就是 40；
       而 218 块库的 df_max = int(218*0.2) = 43。`40 >= 43` 恒假。
       库大于 200 块时 int(N*0.2) > 40 —— 15 个库里 14 个都大于 200 块，
       **这个"默认开启的保护"在几乎所有库上都是死代码**。
       逐题验证：开/关过滤的关键词召回结果逐块相同。

    2. **中文被打穿的真凶是 None 距离进 `should_escalate` 抛异常**（见 _usable_dists）。
       cn2h 那一臂跑在崩溃修复之前。修复后真机复测：混合开启下中文库外题
       **10/10 精确拒答、逐字契约全中、编造 0**。修好它的是崩溃修复，不是本过滤。

    3. **比例阈值这个设计本身不可行。** 218 块库上量真实 df（不设 limit）：

           真术语   五铢 0.5%  记账 1.8%  汇票 4.1%  纸币 18.3%  工业 18.3%  殖民 18.8%
           功能词   了多 3.2%  以及 10.6%  的发 13.3%  的是 15.6%  行了 16.5%  因此 35.8%

       两类区间**完全重叠**：要保住"殖民 18.8%"就得放过"行了 16.5%"，
       压到 10% 以下又会连"纸币/工业/殖民"一起杀掉。单一比例分不开。

    所以默认关闭。函数只作为历史实验开关保留；`AITIC_KW_DF_RATIO=0.2`
    虽可显式启用，但当前 `pool=40` 下对绝大多数库不可达，而且即使把它变得
    可达，标定也证明单一比例分不开真术语与功能词。不要把它当成安全功能。

    混合检索本身仍默认关闭；当前构建即使显式开启混合，中文拒答契约也由
    `_usable_dists` 的 None 距离收口保障，与本过滤无关。
    """
    ratio = os.environ.get("AITIC_KW_DF_RATIO", "0")
    if ratio is None or str(ratio).strip() == "":
        return 0
    try:
        ratio = float(ratio)
    except ValueError:
        raise RuntimeError("AITIC_KW_DF_RATIO 不是合法浮点数: %r" % (ratio,))
    if ratio <= 0 or ratio >= 1:
        return 0
    try:
        total = int(col.count())
    except Exception:
        return 0
    # 至少留 2，避免小库上把所有词都当停用词，反而退化成纯向量
    return max(2, int(total * ratio))


def _keyword_rank(col, question, pool=HYBRID_KEYWORD_POOL):
    """关键词召回：按命中词数与词的稀有度排序，返回 [(doc, meta)] 名次列表。

    用 chroma 的 ``$contains`` 逐词取候选，再在候选内做一个轻量 BM25 式打分
    （命中词数为主、词长为辅），不引入新依赖也不额外建索引。
    """
    terms = _query_terms(question)
    if not terms:
        return []
    # 一个词命中的块太多就没有区分度，等同停用词——这是英文侧 `len(w) >= 4` +
    # 去停用词的中文对应物。中文走二元组（见 _query_terms），"什么/么是/死的/的典"
    # 这类功能词碎片几乎命中每一块；而本函数的主排序键是**命中词数**，
    # 于是"命中 8 个垃圾二元组的块"会压过"命中 1 个真正专名的块"。
    #
    # 旧构建（2026-08-13，中文经济史库 218 块）曾在混合检索下把 10 道库外题
    # 全部答成编造；后续追踪证明根因不是这里的词频过滤，而是 None 距离流进
    # `should_escalate` 后抛异常。当前构建已由 `_usable_dists` 收口，混合开启时
    # 同一拒答契约复测 10/10。这里的过滤默认关闭，不能冒领那次修复的效果。
    df_max = _keyword_df_max(col)
    bag = {}
    for term in terms[:8]:
        try:
            got = col.get(where_document={"$contains": term},
                          limit=pool, include=["documents", "metadatas"])
        except Exception:
            continue
        ids = (got or {}).get("ids") or []
        if df_max and len(ids) >= df_max:
            continue
        docs = (got or {}).get("documents") or []
        metas = (got or {}).get("metadatas") or []
        # 命中该词的块越少，说明这个词越有区分度，权重越高（idf 的朴素近似）
        weight = 1.0 + 1.0 / max(1, len(ids))
        for cid, doc, meta in zip(ids, docs, metas):
            slot = bag.setdefault(cid, {"doc": doc, "meta": meta, "score": 0.0, "terms": 0})
            slot["score"] += weight * (1.0 + min(len(term), 12) / 12.0)
            slot["terms"] += 1
    ordered = sorted(bag.values(), key=lambda x: (-x["terms"], -x["score"]))
    return [(x["doc"], x["meta"]) for x in ordered[:pool]]


def _rrf_fuse(vector_list, keyword_list, top_k):
    """按名次融合两路召回。返回 (docs, metas, dists)，dists 沿用向量侧的真实距离。

    关键词侧没有可比的"距离"，融合后若某块只由关键词召回，其 dist 记 None，
    调用方（可信度里的检索相关性信号）会据此如实标注为未计算，不编一个数字出来。
    """
    scores, holder = {}, {}
    for rank, (doc, meta, dist) in enumerate(vector_list):
        key = str(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (HYBRID_RRF_K + rank + 1)
        holder.setdefault(key, {"doc": doc, "meta": meta, "dist": dist})
    for rank, (doc, meta) in enumerate(keyword_list):
        key = str(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (HYBRID_RRF_K + rank + 1)
        holder.setdefault(key, {"doc": doc, "meta": meta, "dist": None})
    best = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    picked = [holder[k] for k, _ in best]
    return ([x["doc"] for x in picked], [x["meta"] for x in picked], [x["dist"] for x in picked])


# ----------------------------- 按页范围限定检索 -----------------------------
# 动机来自本项目记录在案的真实失败，不是想当然的功能：
#   · 评测集建设时发现 OSTEP 前面只提一嘴 semaphore、正式定义在后面章节，问它就假 MISS；
#   · 健康度工具刚报出「AIGC 讲义 232 页只有 126 页有内容」，中间有页解析失败。
# 两者都指向同一件事：用户没有办法约束"到哪一段里去找"。
#
# 铁律：范围内找不到依据就**诚实拒答**，绝不偷偷把范围放宽——那等于用户以为在查第 3 章、
# 系统却从第 9 章找了答案，比不给这个功能更糟。


def _page_scope(scope):
    """解析 {"from": a, "to": b}；非法或缺失返回 None 表示不限定。"""
    if not isinstance(scope, dict):
        return None
    try:
        lo = scope.get("from")
        hi = scope.get("to")
        lo = int(lo) if lo not in (None, "") else None
        hi = int(hi) if hi not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo                      # 用户填反了就纠正，不报错
    return {"from": lo, "to": hi}


def _scope_where(scope):
    clauses = []
    if scope.get("from") is not None:
        clauses.append({"page": {"$gte": int(scope["from"])}})
    if scope.get("to") is not None:
        clauses.append({"page": {"$lte": int(scope["to"])}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _retrieve_scoped(col, question, scope, query_vector=None):
    """只在指定页范围内检索。

    直接查 chroma 而非走 ``M._retrieve``，因为后者不接受元数据过滤。
    默认配置下两者等价（QUERY_EXPAND=0、VL_QUOTA=0 都关着）；
    若将来打开那两个开关，范围检索不会享受到它们——这一点如实记在这里。
    """
    qv = query_vector if query_vector is not None else M.embed([question])[0]
    where = _scope_where(scope)
    if not where:
        return M._retrieve(col, qv, question)
    res = col.query(query_embeddings=[qv], n_results=M.TOP_K, where=where)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return docs, metas, dists


def _scope_label(scope):
    if not scope:
        return ""
    lo, hi = scope.get("from"), scope.get("to")
    if lo is not None and hi is not None:
        return "第 %d–%d 页" % (lo, hi)
    if lo is not None:
        return "第 %d 页起" % lo
    return "至第 %d 页" % hi


def _hybrid_enabled(flag=None):
    if flag is not None:
        # JSON 客户端有时把开关序列化成 "false"/"0"。bool("false") 为 True，
        # 旧实现会在调用方明确关闭时反而开启混合检索；与流式入口的口径也不同。
        return _coerce_bool(flag, False)
    return _coerce_bool(os.environ.get("DISTILL_HYBRID", "0"), False)


def _retrieve_hybrid(col, question, query_vector=None):
    """单库混合检索；关键词只能补强已有向量证据，不能独立“救活”问题。

    修正评测身份后，用 1007 题默认/混合臂和正确首轮向量距离做回放：若向量最优
    距离超过现有证据下限 0.99 就退回纯向量，否则才做 RRF，得到命中 +48、编造 +2、
    净值 +44；按教材留一法仍为命中 +46、编造 +3、净值 +40。旧的无条件混合是
    命中 +48、编造 +6、净值 +36。这里复用已存在的证据下限，不新增拟合阈值。

    混合检索仍默认关闭；本规则只让显式开启时更安全。关键词路径自身失败也继续
    静默退回纯向量——检索是主链路，不能因加分项而挂掉。
    """
    # 多库检索已在循环外算过同一个问题的向量；复用它，避免选 4 本书时重复调用
    # embedding 模型 4 次。单库入口不传时仍保持原路径。
    qv = query_vector if query_vector is not None else M.embed([question])[0]
    docs, metas, dists = M._retrieve(col, qv, question)
    if _evidence_floor_blocks(dists):
        return docs, metas, dists
    try:
        kw = _keyword_rank(col, question)
    except Exception:
        return docs, metas, dists
    if not kw:
        return docs, metas, dists
    return _rrf_fuse(list(zip(docs, metas, dists)), kw, M.TOP_K)


def _retrieve_selected(question, requested=None, hybrid=None, scope=None):
    """始终按已解析 target 的绝对路径检索；多库再做公平融合。

    不能用“target 路径此刻等于全局 ``M.DB_PATH``”作为单库快速路径后再调用
    ``_collection()``：两步之间切库会读到 B 却把元数据标成 A。target 路径直连
    同样复用 ``M._retrieve``，只是消除了可变全局状态这一层。
    """
    targets = _library_targets(requested)
    qv = M.embed([question])[0]
    ranked = []
    alias_by_id = {target["id"]: "K%d" % (index + 1) for index, target in enumerate(targets)}
    for target in targets:
        try:
            col = M.chromadb.PersistentClient(path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
            if scope:
                docs, metas, dists = _retrieve_scoped(col, question, scope, qv)
            elif _hybrid_enabled(hybrid):
                docs, metas, dists = _retrieve_hybrid(col, question, qv)
            else:
                docs, metas, dists = M._retrieve(col, qv, question)
            if not scope:
                docs, metas, dists = _anchor_rescue(
                    col, question, docs, metas, dists)
        except Exception as exc:
            # 不能把失败的某一本静默丢掉后继续生成：引用仍会看起来完全合法。
            # 就绪状态失效项在入口列为 dropped；运行时失败则中止并报出书名。
            raise RuntimeError("知识库「%s」检索失败，已停止本次回答" %
                               target["name"]) from exc
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            tagged = dict(meta, _library_id=target["id"], _library_name=target["name"],
                          _library_source=target["source"],
                          _library_alias=alias_by_id[target["id"]],
                          _multi_library=len(targets) > 1)
            ranked.append({"doc": doc, "meta": tagged, "dist": dist, "rank": rank,
                           "library": target["id"], "score": 1.0 / (60 + rank + 1)})
    if not ranked:
        raise RuntimeError("所选知识库均无法检索")

    # 每库先保留第一名，再按 RRF 名次补齐；同文本在不同书中仍视为独立证据。
    chosen, seen = [], set()
    for target in targets:
        candidate = next((x for x in ranked if x["library"] == target["id"]), None)
        if candidate:
            key = (candidate["library"], candidate["doc"])
            chosen.append(candidate); seen.add(key)
    # dist 可能是 None：混合检索里纯关键词命中的块没有可比距离（见 _rrf_fuse）。
    # 而 score = 1/(60+rank+1)，**不同库的同名次块 score 完全相等**，元组比较必然
    # 落到第二个元素——None 与 float 一比就抛 TypeError，整个请求 500。
    # 触发条件：多库 + 开混合检索 + 任一并列名次上有纯关键词命中。
    # 按"无距离视为最差"参与排序，不改变有距离项之间的相对次序。
    for item in sorted(ranked, key=lambda x: (-x["score"],
                                              x["dist"] if x["dist"] is not None else float("inf"))):
        key = (item["library"], item["doc"])
        if key in seen:
            continue
        chosen.append(item); seen.add(key)
        if len(chosen) >= M.TOP_K:
            break
    chosen = chosen[:M.TOP_K]
    return ([x["doc"] for x in chosen], [x["meta"] for x in chosen],
            [x["dist"] for x in chosen], targets)


def _sources_from(metas, packed_idx, docs=None):
    """按实际打包的块下标取来源（相关度裁剪下打包的不一定是前几块）。"""
    out, seen = [], set()
    for i in packed_idx:
        if i >= len(metas):
            continue
        s = _src_of(metas[i])
        # Display labels are intentionally human-friendly and therefore not
        # unique: two selected libraries may have the same name, filename and
        # page.  Deduplicating only by label silently removed the second
        # library's evidence card even though the answer could cite it as K2.
        source_key = (str(s.get("library_id") or ""),
                      str(s.get("source") or ""),
                      str(s.get("label") or ""),
                      str(s.get("type") or ""))
        if source_key in seen:
            continue
        seen.add(source_key)
        if docs is not None and i < len(docs):
            s["snippet"] = docs[i][:220]      # 给前端展开看原文片段
        out.append(s)
    return sorted(out, key=lambda x: x["label"])


def _web_cite_tag(meta):
    base = M._cite_tag(meta)
    if meta.get("type") == "epub":
        # EPUB ``loc`` values contain a stable section id followed by a human
        # title, e.g. ``ch3:1 银币诞生于美索不达米亚``.  Putting that whole title
        # inside the citation token made correctness depend on the model copying
        # every title character verbatim: ``[ch3:1]`` or a one-character title
        # typo was classified as a fabricated source and the whole answer was
        # replaced with a refusal.  The section id is already unique within an
        # EPUB and remains unambiguous when the K1/K2 library prefix is added.
        match = re.match(r"^(ch[^\s\[\]]+)", str(base or ""), re.I)
        if match:
            base = match.group(1)
    if meta.get("_multi_library") and meta.get("_library_alias"):
        return "%s:%s" % (meta["_library_alias"], base)
    return base


def _labeled_context(packed, packed_idx, metas):
    blocks = []
    for pos, index in enumerate(packed_idx):
        tag = _web_cite_tag(metas[index]) if index < len(metas) else "?"
        blocks.append("[%s]\n%s" % (tag, packed[pos]))
    return "\n---\n".join(blocks)


def _verify_citations(answer, packed_idx, metas):
    valid = {_web_cite_tag(metas[i]).lower().replace(" ", "")
             for i in packed_idx if i < len(metas)}
    cited = []
    for raw in re.findall(r"\[([^\]]+)\]", str(answer)):
        normalized = raw.strip().lower().replace(" ", "")
        if (re.match(r"p\.?\d+|ch|image|audio", normalized) or
                re.match(r"k\d+:(p\.?\d+|ch|image|audio)", normalized)):
            normalized = re.sub(r"(^|:)p(\d+)", r"\1p.\2", normalized)
            cited.append(normalized)
    cset = list(dict.fromkeys(cited))
    hit = [x for x in cset if x in valid]
    bad = [x for x in cset if x not in valid]
    abstained = M.is_abstain(answer)
    missing = not cset and not abstained
    return {"total": len(cset), "hit": hit, "fabricated": bad,
            "valid_sources": sorted(valid), "missing": missing,
            "ok": (not missing) and (not bad),
            "rate": (len(hit) / len(cset)) if cset else (1.0 if abstained else 0.0)}


# ----------------------------- 证据链底层 -----------------------------
# 证据链、可信度、答案正面性三项共用同一套结构：结论句 → 引用标签 → 检索块 → 书/页/片段。
# 接地率算法移植自 verify_fab.py::_grounding —— 那是本项目定版 50 条幻觉审计用的同一把尺子，
# 沿用它可保证 Web 端的判据与评测报告口径一致，不引入第二套标准。

_STOP_EN = set((
    "the a an and or of to in is are was were be been for on with that this it its as by "
    "from at not but they them their there here which who whom whose what when where why how "
    "can could will would shall should may might must have has had do does did done such than "
    "these those into over under about above below between within without also more most other "
    "some any each both only same very then thus therefore however because while during after "
    "before again further once all no nor own too").split())

# 接地率低于此值即视为"引用是装饰"——与 verify_fab.py 判"真幻觉"的 0.3 同阈值。
_GROUNDED_MIN = 0.30


def _latin_words(text):
    return set(w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP_EN)


def _cjk_bigrams(text):
    """中文没有词边界，用连续二元组近似词。不引第三方分词器，保持零新依赖。"""
    chars = re.findall(r"[一-鿿]", text or "")
    return set("".join(chars[i:i + 2]) for i in range(len(chars) - 1))


def _tokens(text):
    return {"latin": _latin_words(text), "cjk": _cjk_bigrams(text)}


def _dominant_script(text):
    """判断一段文字主要是哪种文字，决定用哪套词元做重叠比较。"""
    parts = _tokens(text)
    if len(parts["cjk"]) >= 4 and len(parts["cjk"]) >= len(parts["latin"]):
        return "cjk"
    if parts["latin"]:
        return "latin"
    return "cjk" if parts["cjk"] else None


def _strip_tags(text):
    return re.sub(r"\[[^\]]*\]", "", str(text or "")).strip()


def _snippet(text, limit=160):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _grounding(claim, evidence):
    """结论句的内容词有多大比例落在它引用的那个块里。

    返回 ``(rate, n_words)``；``rate is None`` 表示**未能计算**而非 0：
    中文提问英文教材时（PRD 要求支持的场景），答案与证据分属两种文字，
    词面重叠恒为 0，报 0 会把正常的跨语言回答误判成幻觉。
    此时如实返回 None 并在上层标注，宁可标边界也不出假数字。
    """
    body = _strip_tags(claim)
    script = _dominant_script(body)
    if not script:
        return None, 0
    mine = _tokens(body)[script]
    theirs = _tokens(evidence)[script]
    if not mine:
        return None, 0
    if not theirs:
        return None, len(mine)          # 证据是另一种文字 → 跨语言，未计算
    return round(len(mine & theirs) / len(mine), 3), len(mine)


_CITE_SPAN_RE = re.compile(r"\[[^\]]*\]")
_CITE_ATOM_RE = re.compile(
    r"^(?:K\d+:)?(?:p\.?\s*\d+|ch:[^\[\],，;；]+|image:[^\[\],，;；]+|audio[^\[\],，;；]*)$",
    re.I,
)
# 词内点：小数 `3.14`、代码标识符 `dict.items()`、缩写 `U.S.`、版本号 `v1.2`。
# 这些点都不是句末。原先只挡了小数，于是 `dict.items()` 被切成 `Use dict.` +
# `items() to iterate…`；前半句一旦被逐句裁剪掉，后半截就成了以 `items()` 开头的残句。
# 判据是"点两侧都紧贴词字符、中间无空白"——真正的句末后面一定有空白或行尾。
_DECIMAL_DOT_RE = re.compile(r"(?<=\w)\.(?=\w)")
# 收尾标点若被甩成独立片段，属于上一句（如 引文以 `loop.` 结尾、右引号被切到下一段）
_CLOSING_LEAD_RE = re.compile(r'^[”"\'』」）)\]】》’]')


def _expand_compound_citations(answer):
    """把模型常见的 ``[p.3, p.8]`` 规范成两个合法标签。

    只在每一项都严格符合项目引用语法时拆分，普通方括号内容保持原样。
    """
    def repl(match):
        inside = match.group(1)
        parts = [x.strip() for x in re.split(r"\s*[,，;；]\s*", inside) if x.strip()]
        if len(parts) > 1 and all(_CITE_ATOM_RE.fullmatch(x) for x in parts):
            return " ".join("[%s]" % x for x in parts)
        return match.group(0)

    return re.sub(r"\[([^\[\]]*[,，;；][^\[\]]*)\]", repl, str(answer or ""))


def _normalized_cite_atom(atom):
    """返回合法引用原子的确定性比较键；普通方括号文本返回 None。"""
    raw = str(atom or "").strip().strip("[]").strip()
    if not _CITE_ATOM_RE.fullmatch(raw):
        return None
    normalized = re.sub(r"\s+", "", raw.casefold())
    return re.sub(r"(^|:)p\.?([0-9]+)$", r"\1p.\2", normalized)


def _dedupe_adjacent_citations(answer):
    """只合并相邻、同源的合法引用标签，不碰普通方括号或不同库标签。"""
    pair = re.compile(r"(\[[^\[\]\r\n]+\])([ \t]+)(\[[^\[\]\r\n]+\])")
    text = str(answer or "")
    while True:
        changed = False

        def repl(match):
            nonlocal changed
            left = _normalized_cite_atom(match.group(1))
            right = _normalized_cite_atom(match.group(3))
            if left is not None and left == right:
                changed = True
                return match.group(1)
            return match.group(0)

        updated = pair.sub(repl, text)
        if not changed:
            return updated
        text = updated


# 分面小标题总是「另起一句」——前面必然是句末标点或引用标签的右括号。
# 只用 `**…**，` 的形状判断会误伤正文里的强调（「这个概念**很重要**，需要牢记」），
# 所以必须连左边界一起匹配。
_LEAD_IN_INLINE_RE = re.compile(
    r"(?<=[。．.！？!?；;\]])[ \t]*(?=\*\*[^*\n]{2,40}\*\*[，,：:])")
_BULLET_INLINE_RE = re.compile(r"(?<=[。．.；;：:])[ \t]*(?=-\s+\S)")


def _normalize_answer_layout(text):
    """把行内的粗体小标题/要点提到各自段落。**只动空白，一个字都不改。**

    提示词里已经把「空行分段」写成字符级要求，qwen3:8b 仍然常常把三个
    `**从…看**，` 全塞在同一行里——读起来还是一大坨。与其继续加措辞去劝，
    不如在代码里做确定性归一：这是纯排版，不涉及内容取舍，也就不存在编造风险。
    """
    body = str(text or "")
    if not body.strip():
        return body
    body = _LEAD_IN_INLINE_RE.sub("\n\n", body)
    body = _BULLET_INLINE_RE.sub("\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _split_claim_sentences(answer):
    """按句切分，但**不能在引用标签或小数点内部断开**。

    ``M.split_sentences`` 在每个 ``.`` 之后切。真机答案里这会造成三种碎片：
      1. ``[p.2]`` → ``[p.`` + ``2]``（引用被劈开，该结论会被误判成"未附引用"）
      2. ``1.5 Study materials`` → ``1.`` + ``5 Study materials``（小数被劈开）
      3. ``… or a loop.”`` → 右引号被甩成下一句的开头
    CLI 侧不按句处理答案，所以这些一直没暴露；Web 侧要做逐句证据映射就必须处理。
    修在 Web 层，不动 main.py（分块/PROMPT 指纹须与 v8final 保持一致）。
    """
    spans = []

    def mask(match):
        spans.append(match.group(0))
        return "\x00%d\x00" % (len(spans) - 1)

    masked = _DECIMAL_DOT_RE.sub(mask, _CITE_SPAN_RE.sub(mask, str(answer or "")))
    out = []
    # 先按行切，再在行内按句切。答案改成结构化排版后，要点行常常不带句末标点，
    # 只按句号切会把整张列表粘成一条巨型"结论"，逐条核验和逐条接地率就都失去意义。
    # 行边界还天然挡住了跨段误并——下面那几条"并回上一句"的规则只在行内生效。
    for line in re.split(r"\n+", masked):
        line = line.strip()
        if not line:
            continue
        line_start = len(out)
        for piece in M.split_sentences(line):
            restored = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], piece).strip()
            if not restored:
                continue
            in_line = len(out) > line_start          # 本行内已有句子才允许并回
            if in_line and not _strip_tags(restored):
                out[-1] = (out[-1] + " " + restored).strip()  # 纯引用片段，归上一句
                continue
            if in_line and _CLOSING_LEAD_RE.match(restored):
                out[-1] = out[-1] + restored                   # 收尾标点，直接贴回去
                continue
            # 真句子不会以小写字母开头。会出现这种片段，说明上一处点号并非句末
            # （典型是缩写 `U.S. standard` —— 词内点已挡住，但缩写末尾那个点后面跟着空格）。
            # 并回上一句：把两句误并的代价，远小于留下一条无主语的残句。
            # 同样只在行内并回：要点行以小写开头是排版，不是断句失败。
            if in_line and re.match(r"^[a-z]", restored):
                out[-1] = (out[-1] + " " + restored).strip()
                continue
            out.append(restored)
    return out


def _claim_evidence_map(answer, packed_idx, metas, packed):
    """把答案拆成结论句，逐句映射到它引用的检索块，附书/页/片段与接地率。"""
    if M.is_abstain(answer) or not str(answer or "").strip():
        return []
    by_tag = {}
    for pos, index in enumerate(packed_idx):
        if index >= len(metas):
            continue
        meta = metas[index]
        tag = M._norm_cite(_web_cite_tag(meta))
        info = _src_of(meta)
        piece = str(packed[pos] or "") if pos < len(packed) else ""
        if tag in by_tag:
            # EPUB sections commonly span several chunks with the same ``loc``.
            # The answer model saw every packed chunk, so evidence mapping must
            # not silently keep only the first one for that citation tag.
            if piece and piece not in by_tag[tag]["text"]:
                by_tag[tag]["text"] += "\n---\n" + piece
            continue
        by_tag[tag] = {
            "text": piece,
            "library": info.get("library") or "", "label": info.get("label"),
            "page": info.get("page"), "type": info.get("type"),
        }

    claims = []
    for sentence in _split_claim_sentences(answer):
        body = _strip_tags(sentence)
        if len(body) < 2:
            continue                     # 纯引用行不是结论，不进证据链
        tags = [M._norm_cite(x) for x in re.findall(r"\[([^\]]+)\]", sentence)]
        tags = [t for t in dict.fromkeys(tags) if t in by_tag]
        evidence, best, measured = [], None, False
        for tag in tags:
            block = by_tag[tag]
            rate, _n = _grounding(sentence, block["text"])
            if rate is not None:
                measured = True
                best = rate if best is None else max(best, rate)
            evidence.append({"tag": tag, "library": block["library"],
                             "label": block["label"], "page": block["page"],
                             "type": block["type"], "snippet": _snippet(block["text"]),
                             "grounding": rate})

        # 模型常写成「结论。\n\nEvidence: …[p.67]」——把引用挂在证据句上，结论句本身不带标签。
        # 这种结论并非无据可查，只是没自带标签。所以对无引用的句子再回检索材料算一次接地率：
        # 真能在材料里找到支撑就记为"由材料支撑"，而不是一律判成"无法核对"。
        support_via = "citation" if tags else None
        unmeasurable = False
        if not tags:
            computable = False
            for tag, block in by_tag.items():
                rate, _n = _grounding(sentence, block["text"])
                if rate is None:
                    continue
                computable = measured = True
                if best is None or rate > best:
                    best, support_via = rate, "material"
                    evidence = [{"tag": tag, "library": block["library"],
                                 "label": block["label"], "page": block["page"],
                                 "type": block["type"], "snippet": _snippet(block["text"]),
                                 "grounding": rate, "implicit": True}]
            if not computable:
                # 与材料非同一文字 → 没法核对，不等于"找不到支撑"。两者必须分开表述。
                unmeasurable = True
            elif best is not None and best < _GROUNDED_MIN:
                support_via, evidence = None, []

        claims.append({
            "claim": body, "raw": sentence, "citations": tags, "evidence": evidence,
            "grounding": best, "measured": measured, "support_via": support_via,
            "unmeasurable": unmeasurable,
            # 未能计算接地率时（跨语言）退回"有无有效引用"这个较弱但诚实的判据
            "supported": (best is not None and best >= _GROUNDED_MIN) if measured else bool(tags),
        })
    return claims


# ----------------------------- 逐句语义支持护栏 -----------------------------
# 词面接地率适合发现同文种的明显错引，但无法可靠处理「中文回答 + 英文教材」、
# 数字被改写或强因果/绝对化结论。这里只把这些可疑句交给同一个本地模型批量核验；
# 核验器失败时保留原答案并降级（fail-open），绝不能把 UNKNOWN 当成不支持。
_SUPPORT_VERIFY_TOKENS = 1100
_MODEL_SEED_RAW = str(os.environ.get("DISTILL_MODEL_SEED", "")).strip()
try:
    _MODEL_SEED = int(_MODEL_SEED_RAW) if _MODEL_SEED_RAW else None
except ValueError:
    _MODEL_SEED = None


def _llm_options(**values):
    """Return the shared Ollama option set, with an opt-in experiment seed.

    The 2026-08-16 repeated-generation probe found the unseeded local
    qwen3:8b path byte-identical on 8/8 questions, while forcing a seed made
    6/8 questions vary.  Therefore production must not silently force a seed.
    ``DISTILL_MODEL_SEED`` remains available for explicit, recorded A/B runs.
    """
    options = dict(values)
    if _MODEL_SEED is not None:
        options["seed"] = _MODEL_SEED
    return options
_BILINGUAL_RESCUE_COSINE = 0.66
_SUPPORT_RISK_RE = re.compile(
    r"(?:\b(?:always|never|only|must|none|all|not|because|therefore|causes?|caused|"
    r"higher|lower|greater|less|more|equal)\b|"
    r"总是|从不|仅仅|唯一|必须|全部|完全|没有|并非|因此|由于|导致|造成|高于|低于|超过|少于|等于)",
    re.I,
)
_NUMBER_RE = re.compile(
    # ASCII identifiers/citation tags remain excluded, while Chinese text such
    # as ``共有99个`` must still expose 99 to the numeric consistency guard.
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|％)?"
)


def _claim_is_cross_language(claim, blocks):
    """Whether a claim and its cited/available source blocks use different scripts."""
    claim_script = _dominant_script(claim.get("claim"))
    if not claim_script:
        return False
    tags = [M._norm_cite(x) for x in (claim.get("citations") or [])]
    candidates = [blocks[tag]["text"] for tag in tags if tag in blocks]
    if not candidates:
        candidates = [block["text"] for block in blocks.values()]
    source_script = _dominant_script("\n".join(str(x or "") for x in candidates))
    return bool(source_script and source_script != claim_script)


def _support_claim_is_high_risk(claim):
    body = str(claim.get("claim") or "")
    return bool(_number_tokens(body) or _SUPPORT_RISK_RE.search(body))


def _cross_language_similarity(claim, blocks):
    """Cross-lingual embedding is only a false-refusal safety valve, never proof.

    It is deliberately limited to blocks already cited by the answer.  A high
    score may downgrade two small-model negative votes to UNKNOWN (keep + low
    confidence), but must never turn the claim into SUPPORTED.
    """
    body = str(claim.get("claim") or "").strip()
    tags = [M._norm_cite(x) for x in (claim.get("citations") or [])]
    candidates = [blocks[tag]["text"] for tag in tags if tag in blocks]
    if not body or not candidates:
        return None
    try:
        vectors = M.embed([body] + candidates)
        if len(vectors) != len(candidates) + 1:
            return None
        scores = [M._cosine(vectors[0], vector) for vector in vectors[1:]]
        return max(scores) if scores else None
    except Exception:
        return None


def _bilingual_recheck_prompt(suspicious, blocks):
    return (
        "SECOND-PASS BILINGUAL CHECK. The first verifier returned a negative verdict, "
        "but claim and source use different languages. Translate the claim meaning before deciding. "
        "Do not infer from topic similarity and do not use outside knowledge. A claim is SUPPORTED only "
        "when one exact source passage entails its complete meaning.\n" +
        _support_verifier_prompt(suspicious, blocks)
    )


def _normalized_quote(text):
    """只做 Unicode/空白/大小写归一化，仍要求 quote 是原块中的连续子串。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text or ""))).strip().casefold()


def _number_tokens(text):
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("％", "%")
    return {m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(normalized)}


def _support_blocks(packed_idx, metas, packed):
    """按实际交给回答模型的块建索引；核验器不得看到检索外材料。"""
    blocks = {}
    for pos, index in enumerate(packed_idx):
        if index >= len(metas) or pos >= len(packed):
            continue
        display = _web_cite_tag(metas[index])
        key, piece = M._norm_cite(display), str(packed[pos] or "")
        if key in blocks:
            # Several chunks can legitimately share one EPUB section tag.  A
            # plain assignment used to overwrite the earlier chunk, so the
            # semantic verifier saw less evidence than the answer model and
            # pruned supported claims.  Merge only chunks already in ``packed``;
            # this preserves the verifier's "no retrieval-external material"
            # boundary and keeps exact quotes contiguous inside each piece.
            if piece and piece not in blocks[key]["text"]:
                blocks[key]["text"] += "\n---\n" + piece
            continue
        blocks[key] = {"tag": display, "text": piece}
    return blocks


def _claim_needs_support_check(claim):
    """只核验高风险/现有确定性规则无法坐实的结论，避免无条件增加时延。"""
    body = str(claim.get("claim") or "")
    if not claim.get("citations"):
        return True
    if claim.get("unmeasurable") or not claim.get("measured"):
        return True
    if not claim.get("supported"):
        return True
    if _number_tokens(body):
        return True
    return bool(_SUPPORT_RISK_RE.search(body))


def _support_verifier_prompt(suspicious, blocks):
    claims = [{"id": idx, "claim": claim.get("claim", ""),
               "current_tags": claim.get("citations") or []}
              for idx, claim in suspicious]
    material = [{"tag": block["tag"], "text": block["text"]}
                for block in blocks.values()]
    return (
        "You are a strict LOCAL evidence verifier. Use only SOURCE_BLOCKS; outside knowledge is forbidden.\n"
        "Check every claim independently. Topic similarity is not support. Translation/paraphrase is allowed "
        "only when the meaning is entailed by an exact source passage.\n"
        "Return ONLY one JSON object: "
        '{"results":[{"id":0,"status":"SUPPORTED|PARTIAL|UNSUPPORTED|UNKNOWN",'
        '"tag":"p.1","quote":"exact contiguous quote copied from that block"}]}.\n'
        "Rules: SUPPORTED means the complete claim, including every number/percent/negation/comparison, is supported. "
        "PARTIAL means only part is supported; provide the exact supporting quote. "
        "UNSUPPORTED means no block supports it and tag/quote MUST both be empty strings. "
        "UNKNOWN means you cannot decide; never guess. For SUPPORTED/PARTIAL, tag must name the quoted block and "
        "quote must be copied verbatim as one contiguous passage. Preserve each integer id exactly.\n"
        "CLAIMS=" + json.dumps(claims, ensure_ascii=False) + "\n"
        "SOURCE_BLOCKS=" + json.dumps(material, ensure_ascii=False)
    )


def _support_model_call(prompt):
    """调用本地核验器；空响应时用 Ollama JSON 模式做一次确定性兜底。

    部分 Ollama/客户端组合在连续生成后会返回 done 但 response 为空。主问答仍维持
    已验证的 ``M._generate`` 路径；仅这个结构化核验器在空响应时启用 JSON 模式。
    """
    options = _llm_options(temperature=0.0, num_predict=_SUPPORT_VERIFY_TOKENS)
    out = M._generate(M.LLM_MODEL, prompt, options=options)
    response = out.get("response", "") if hasattr(out, "get") else ""
    if str(response or "").strip():
        return out
    try:
        return M.ollama.generate(model=M.LLM_MODEL, prompt=prompt, think=False,
                                 stream=False, format="json", options=options)
    except Exception:
        return out


def _parse_support_results(raw):
    """严格 JSON 解析；任何散文或残缺输出均交给 fail-open 分支。"""
    text = M._strip_think(str(raw or "")).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except (TypeError, ValueError):
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    return results if isinstance(results, list) else None


def _quote_is_exact(quote, block_text):
    quote_n, block_n = _normalized_quote(quote), _normalized_quote(block_text)
    content_len = len(re.findall(r"[\w\u4e00-\u9fff]", quote_n))
    return content_len >= 6 and quote_n in block_n


def _validate_support_results(results, suspicious, blocks):
    """模型判据只有通过 tag、连续原文和数字三重代码校验后才可生效。"""
    by_id, duplicates = {}, set()
    for item in results or []:
        if not isinstance(item, dict) or type(item.get("id")) is not int:
            continue
        item_id = item["id"]
        if item_id in by_id:
            duplicates.add(item_id)
        by_id[item_id] = item

    verdicts = {}
    for claim_id, claim in suspicious:
        item = by_id.get(claim_id)
        if claim_id in duplicates or not isinstance(item, dict):
            verdicts[claim_id] = {"status": "UNKNOWN", "reason": "missing_or_duplicate"}
            continue
        status = str(item.get("status") or "").strip().upper()
        if status not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNKNOWN"}:
            verdicts[claim_id] = {"status": "UNKNOWN", "reason": "invalid_status"}
            continue
        if status == "UNKNOWN":
            verdicts[claim_id] = {"status": "UNKNOWN", "reason": "verifier_unknown"}
            continue
        raw_tag, quote = str(item.get("tag") or "").strip(), str(item.get("quote") or "").strip()
        if status == "UNSUPPORTED":
            # 否定结论没有可引用原文；若模型同时编了 tag/quote，则整条结果不可信。
            if raw_tag or quote:
                verdicts[claim_id] = {"status": "UNKNOWN", "reason": "unsupported_with_quote"}
            else:
                verdicts[claim_id] = {"status": "UNSUPPORTED", "reason": "explicit_unsupported"}
            continue

        tag = M._norm_cite(raw_tag.strip("[]"))
        block = blocks.get(tag)
        # 小模型偶尔把正确标签写成 ``[p.79] text``。只修复“合法标签外多了
        # 说明文字”的格式波动；p.99 这类本轮不存在的标签绝不能靠别块 quote 洗白。
        if not block:
            embedded = [(key, candidate) for key, candidate in blocks.items()
                        if re.search(r"(?<![\w.])%s(?![\w.])" % re.escape(key),
                                     raw_tag, re.I)]
            if len(embedded) == 1 and _quote_is_exact(quote, embedded[0][1]["text"]):
                tag, block = embedded[0]
            else:
                verdicts[claim_id] = {"status": "UNKNOWN", "reason": "invalid_tag"}
                continue
        if not _quote_is_exact(quote, block["text"]):
            verdicts[claim_id] = {"status": "UNKNOWN", "reason": "quote_not_exact"}
            continue
        claim_numbers = _number_tokens(claim.get("claim"))
        quote_numbers = _number_tokens(quote)
        if status == "SUPPORTED" and not claim_numbers.issubset(quote_numbers):
            verdicts[claim_id] = {"status": "UNKNOWN", "reason": "number_mismatch"}
            continue
        verdicts[claim_id] = {"status": status, "tag": block["tag"],
                              "reason": "validated"}
    return verdicts


# 句首指代：先行词随被裁掉的上一句一起消失后，这句就悬空了。
# 只认真正的回指词（它/这/该/其…、it/this/they…），不认「然而/因此」这类连接词——
# 连接词接在别的句子后面读得通，指代词却是彻底断链。
_DANGLING_ANAPHOR_RE = re.compile(
    r"^[\s\"'“”‘’(（\[【]*"
    r"(?:它们|它|他们|她们|其中|其|"
    r"这些|这种|这一|这个|这|"
    r"那些|那个|那|该|上述|前者|后者|"
    r"(?:it|its|they|them|their|this|that|these|those|such)\b)", re.I)

# 句首连接词：本身不断链，但当它成为整段第一句时读起来是半截话。
_LEADING_CONNECTIVE_RE = re.compile(
    r"^[\s\"'“‘(（\[【]*"
    r"(?:然而|但是|不过|因此|所以|于是|"
    r"此外|另外|而且|同时|其次|最后|"
    r"相反|反之|总之|"
    r"(?:however|therefore|thus|hence|moreover|furthermore|besides|instead|conversely)\b)"
    r"[\s,，:：]*", re.I)


def _drop_orphaned_claims(kept):
    """删句后重新拼接会留下悬空指代：先行词随上一句一起没了。

    我们无从知道「它」指的是什么，更不能替它编一个主语——那正是这套系统要防的事。
    唯一诚实的处理是把这句也去掉。去掉后又可能让下一句变成新的句首悬空，
    所以要反复扫到稳定为止。只在上一句真的被裁掉（下标不连续）时才判定悬空。
    """
    kept = list(kept)
    dropped = 0
    while True:
        for pos, (idx, text) in enumerate(kept):
            prev_idx = kept[pos - 1][0] if pos else -1
            if idx == prev_idx + 1:
                continue                      # 紧邻上一句，指代链没断
            if not _DANGLING_ANAPHOR_RE.match(str(text or "")):
                continue
            del kept[pos]
            dropped += 1
            break
        else:
            return kept, dropped


def _strip_leading_connective(text):
    """裁剪后升为首句的「然而，…」要去掉这个连接词——纯语篇标记，不动内容。"""
    body = _LEADING_CONNECTIVE_RE.sub("", str(text or ""), count=1).lstrip()
    return (body[0].upper() + body[1:]) if body[:1].isascii() and body[:1].isalpha() else (body or text)


def _canonical_supported_sentence(raw, tag):
    body = _CITE_SPAN_RE.sub("", str(raw or "")).strip()
    body = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", body)
    return (body + " [%s]" % tag).strip()


def _claim_separators(answer, claims):
    """记录原文里每条结论**前面**的那段分隔符（空格 / 换行 / 空行）。

    裁剪后若一律用空格重拼，段落和要点行会被压平成一整坨——排版做得再好也白搭。
    这里按原文顺序定位每条结论，把它前面的空白原样记下来，重拼时照抄。
    """
    text = str(answer or "")
    seps, cursor = {}, 0
    for idx, claim in enumerate(claims or ()):      # 收口函数不该在异常路径上再炸一次
        raw = str(claim.get("raw") or claim.get("claim") or "").strip()
        if not raw:
            continue
        at = text.find(raw, cursor)
        if at < 0:                      # 定位不到（已被规范化改写）就不记，重拼时退回空格
            continue
        seps[idx] = text[cursor:at]
        cursor = at + len(raw)
    return seps


def _rejoin_kept(kept, seps):
    """按原文分隔符把保留下来的结论拼回去，尽量保住段落与要点行。"""
    out = []
    for pos, (idx, textval) in enumerate(kept):
        body = str(textval).strip()
        if not body:
            continue
        if pos:
            raw_sep = seps.get(idx, " ")
            if "\n\n" in raw_sep:
                out.append("\n\n")
            elif "\n" in raw_sep:
                out.append("\n")
            else:
                out.append(" ")
        out.append(body)
    return "".join(out)


def _semantic_support_guard(answer, claims, packed_idx, metas, packed):
    """一次性核验可疑结论；明确不支持才裁剪，UNKNOWN/异常一律保留并降级。"""
    default = {"triggered": False, "state": "pass", "checked": 0,
               "supported": 0, "pruned": 0, "unknown": 0, "orphaned": 0,
               "reason": "没有需要额外语义核验的可疑结论", "verdicts": []}
    if M.is_abstain(answer) or not claims:
        return str(answer or ""), default, 0
    suspicious = [(idx, claim) for idx, claim in enumerate(claims)
                  if _claim_needs_support_check(claim)]
    if not suspicious:
        return str(answer or ""), default, 0
    blocks = _support_blocks(packed_idx, metas, packed)
    if not blocks:
        audit = dict(default, triggered=True, state="degraded", checked=len(suspicious),
                     unknown=len(suspicious), reason="没有可供语义核验的检索块")
        return str(answer or ""), audit, 0

    tokens = 0
    try:
        out = _support_model_call(_support_verifier_prompt(suspicious, blocks))
        tokens = ((out.get("prompt_eval_count", 0) or 0) +
                  (out.get("eval_count", 0) or 0)) if hasattr(out, "get") else 0
        results = _parse_support_results(out.get("response") if hasattr(out, "get") else "")
        if results is None:
            # 小模型偶尔会在批量 JSON 末尾截断。仅在格式失败时重试一次；仍失败则
            # 按 UNKNOWN fail-open，绝不把基础设施/格式波动当成“不支持”。
            retry_prompt = (
                "STRICT RETRY: the prior response was not valid JSON. Output one compact JSON object only; "
                "no markdown, no commentary, and include every claim id.\n" +
                _support_verifier_prompt(suspicious, blocks)
            )
            retry = _support_model_call(retry_prompt)
            if hasattr(retry, "get"):
                tokens += ((retry.get("prompt_eval_count", 0) or 0) +
                           (retry.get("eval_count", 0) or 0))
            results = _parse_support_results(
                retry.get("response") if hasattr(retry, "get") else "")
            if results is None:
                raise ValueError("semantic verifier returned malformed JSON twice")
        verdicts = _validate_support_results(results, suspicious, blocks)

        # A single negative verdict is not reliable enough when the answer and
        # source use different languages.  Recheck only those negative claims
        # with an explicit bilingual instruction.  Two negative verdicts still
        # prune the claim; a fully validated positive verdict rescues it.  An
        # inconclusive second pass is fail-open only for low-risk prose and is
        # exposed as UNKNOWN/low confidence.  Numbers, negation, comparisons and
        # absolute claims remain fail-closed.
        bilingual = [(idx, claim) for idx, claim in suspicious
                     if verdicts.get(idx, {}).get("status") in {"PARTIAL", "UNSUPPORTED"}
                     and _claim_is_cross_language(claim, blocks)]
        if bilingual:
            try:
                recheck = _support_model_call(_bilingual_recheck_prompt(bilingual, blocks))
                if hasattr(recheck, "get"):
                    tokens += ((recheck.get("prompt_eval_count", 0) or 0) +
                               (recheck.get("eval_count", 0) or 0))
                recheck_results = _parse_support_results(
                    recheck.get("response") if hasattr(recheck, "get") else "")
                rechecked = (_validate_support_results(recheck_results, bilingual, blocks)
                             if recheck_results is not None else {})
            except Exception:
                rechecked = {}
            for claim_id, claim in bilingual:
                second = rechecked.get(
                    claim_id, {"status": "UNKNOWN", "reason": "bilingual_recheck_failed"})
                second_status = second.get("status")
                if second_status == "SUPPORTED":
                    verdicts[claim_id] = dict(second, reason="bilingual_recheck_supported")
                elif second_status in {"PARTIAL", "UNSUPPORTED"}:
                    similarity = (_cross_language_similarity(claim, blocks)
                                  if not _support_claim_is_high_risk(claim) else None)
                    if similarity is not None and similarity >= _BILINGUAL_RESCUE_COSINE:
                        verdicts[claim_id] = {
                            "status": "UNKNOWN", "reason": "bilingual_embedding_rescue",
                            "similarity": round(float(similarity), 3)}
                    else:
                        verdicts[claim_id] = dict(
                            verdicts[claim_id], reason="bilingual_negative_confirmed")
                elif _support_claim_is_high_risk(claim):
                    verdicts[claim_id] = dict(
                        verdicts[claim_id], reason="bilingual_high_risk_fail_closed")
                else:
                    verdicts[claim_id] = {
                        "status": "UNKNOWN", "reason": "bilingual_recheck_inconclusive"}
    except Exception as exc:
        audit = dict(default, triggered=True, state="degraded", checked=len(suspicious),
                     unknown=len(suspicious), reason="本地逐句核验未完成：%s" % type(exc).__name__)
        return str(answer or ""), audit, tokens

    suspicious_ids = {idx for idx, _ in suspicious}
    separators = _claim_separators(answer, claims)   # 必须对着**原文**算，裁完就找不回来了
    # kept 存 (原下标, 文本)：下标要留着判断句子之间是否被裁开，
    # 只看文本无法区分「上一句还在」和「上一句已被删掉」。
    kept, changed = [], False
    counts = {"supported": 0, "pruned": 0, "unknown": 0, "orphaned": 0}
    public_verdicts = []
    for idx, claim in enumerate(claims):
        raw = claim.get("raw") or claim.get("claim") or ""
        if idx not in suspicious_ids:
            kept.append((idx, raw))
            continue
        verdict = verdicts.get(idx, {"status": "UNKNOWN", "reason": "missing"})
        status = verdict["status"]
        public_verdicts.append({"id": idx, "status": status, "reason": verdict.get("reason", "")})
        if status == "SUPPORTED":
            counts["supported"] += 1
            canonical = _canonical_supported_sentence(raw, verdict["tag"])
            kept.append((idx, canonical))
            changed = changed or canonical != raw
        elif status in {"PARTIAL", "UNSUPPORTED"}:
            counts["pruned"] += 1
            changed = True
        else:
            counts["unknown"] += 1
            kept.append((idx, raw))          # fail-open：不把未知误当成不支持

    if counts["pruned"]:
        kept, counts["orphaned"] = _drop_orphaned_claims(kept)
        if counts["orphaned"]:
            changed = True
            public_verdicts.extend(
                {"id": None, "status": "ORPHANED", "reason": "antecedent_pruned"}
                for _ in range(counts["orphaned"]))
    if kept and kept[0][0] != 0:
        # 下标不为 0 ⇒ 原来的首句被裁走了，这句是被顶上来的，才需要去掉
        # 「然而，…」这类语篇标记。原来无条件剥，会把本来就以连接词开头的
        # 正常答案改掉，并且平白触发下面的重拼。
        head_idx, head_text = kept[0]
        stripped = _strip_leading_connective(head_text)
        if stripped != head_text:
            kept[0] = (head_idx, stripped)
            changed = True

    if counts["pruned"] and not kept:
        # 这里可能放大上游差异：答案措辞一变，切出的结论和逐句裁剪结果也可能变，
        # 而“全裁光”会把差异离散成一次作答/拒答翻转。注意：项目后来在不同
        # 协议与时段测到的翻转率差一个量级，原因未证实，不能把它归因为 GPU
        # 并行归约，也不能把任何一次测量写成固定“噪声底”。
        #
        # 【2026-08-12 两次尝试，两次实测回退，勿重做】
        #
        # 尝试一：裁光前复核一次，两次都同意才清空。
        #   当次 n=23 复测观察到翻转率 9% → 17%；样本很小，但未显示改善，回退。
        #   原因：把第二次判断当成独立证据，但在易翻转区间里它只是又一个采样源——
        #   第一遍裁光、第二遍不同意则保留，下次第二遍同意则拒答，是 amplifier 不是 damper。
        #
        # 尝试二：并入确定性的接地率信号（内容词重合率，纯计算不调模型），
        #   两个信号都指向"没有依据"才清空。这次不新增采样，理由上比尝试一站得住。
        #   正式两臂对照（各 100 题 × 3 次，环境变量切换、同一份代码）：
        #       开启 4/100 = 4.0%   关闭 7/100 = 7.0%   Fisher 双尾 p = 0.537
        #   方向是对的，但 **p 远未显著**，按事先写死的判据回退。
        #   方向看似更好但 **p 远未显著**，按事先写死的判据回退。
        #
        # 结论只到这里：两种替代方案都没有证据支持，故保留单次裁光即拒答的现状。
        # 不能从这些实验推出固定噪声值，也不能宣称根因已定位在上游或 GPU。
        state, guarded = "refused", _NO_REFERENCE
    elif changed:
        state = "pruned" if counts["pruned"] else "verified"
        # 按原文分隔符重拼。早先用 " ".join 会把段落和要点行压平——只要有一句被裁
        # 或被规范化，前端拿到的就是一整坨，今天做的排版基本被这一行抵消。
        guarded = _normalize_answer_layout(_rejoin_kept(kept, separators))
    elif counts["unknown"]:
        state, guarded = "degraded", str(answer or "")
    else:
        state, guarded = "verified", str(answer or "")
    audit = {"triggered": True, "state": state, "checked": len(suspicious),
             **counts, "reason": ("逐句核验存在未知项，已保留原句并降低可信度"
                                    if counts["unknown"] else
                                    "可疑结论已完成本地逐句核验"),
             "verdicts": public_verdicts}
    return guarded or _NO_REFERENCE, audit, tokens


def _repair_post_prune_support_drift(answer, claims):
    """Deterministically remove claims that lost support while rebuilding.

    The semantic verifier judges the pre-prune claim list.  Removing neighbouring
    claims and rejoining the survivors can move a citation across a sentence
    boundary: a previously verified sentence then becomes uncited while the next
    sentence inherits its tag.  Citation syntax still passes, but the delivered
    claim no longer satisfies the verifier's contract.

    This is deliberately a pure post-condition check.  It performs no additional
    model call and is only used after a firm prune (no UNKNOWN verdicts).  Unknown
    verifier results keep the existing fail-open behaviour.
    """
    if M.is_abstain(answer) or not claims:
        return str(answer or ""), 0
    unsupported = [idx for idx, claim in enumerate(claims)
                   if not claim.get("supported")]
    if not unsupported:
        return str(answer or ""), 0

    separators = _claim_separators(answer, claims)
    kept = [(idx, claim.get("raw") or claim.get("claim") or "")
            for idx, claim in enumerate(claims) if idx not in unsupported]
    if not kept:
        return _NO_REFERENCE, len(unsupported)
    if kept[0][0] != 0:
        head_idx, head_text = kept[0]
        kept[0] = (head_idx, _strip_leading_connective(head_text))
    repaired = _normalize_answer_layout(_rejoin_kept(kept, separators)).strip()
    return repaired or _NO_REFERENCE, len(unsupported)


# 【2026-08-13 已重测】放宽副词/动词的实验开关，默认关闭＝线上现行为。
# n=1007 配对重测的完整净额（四个迁移方向均计入）：命中 -19、编造 -11，
# 按项目权重净值 +3；同期空跑臂净值也为 +3，故证不出有效或有害，不改默认。
# 旧的“14:46 明显亏本”只数了单向迁移，也已作废。开关仅为复现实验保留。
_WIDEN_REFUSAL = os.environ.get("AITIC_WIDEN_REFUSAL", "0").strip().lower() in ("1", "true", "on")
_RF_ADV = "explicitly|directly|specifically|clearly" if _WIDEN_REFUSAL else "explicitly"
_RF_VERB = ("mentioned|covered|provided|contained|found|described|addressed|defined|discussed"
            if _WIDEN_REFUSAL else
            "mentioned|covered|provided|contained|found|described|addressed")

_PROSE_REFUSAL_RE = re.compile(
    r"^\s*(?:【(?:概要|结论|回答)】\s*)?(?:"
    r"(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)"
    r"(?:\s+(?:provided|supplied|retrieved|available))?"
    # 与无插入语的既有分支完全同义；qwen 偶尔写成
    # ``The material provided, however, does not mention ...``。
    r"(?:\s*,?\s*(?:however|though|nevertheless)\s*,?)?\s+"
    # 材料本身作主语且答案一开头明确说“不直接/不具体提及”，仍是整题拒答。
    # 历史完整臂扫描只命中编造、未命中过正确答案；这不等于放宽下方任意主题句。
    r"(?:does\s+not|doesn't|do\s+not)\s+"
    r"(?:(?:directly|specifically|clearly)\s+)?(?:contain|provide|mention|cover)\b|"
    r"(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)"
    r"(?:\s+(?:provided|supplied|retrieved|available))?\s+"
    r"(?:contains?|provides?|mentions?|covers?)\s+no\b|"
    # 【2026-08-12 回退，8/13 已配对重测】副词放宽到 directly/specifically/clearly、
    # 动词补上 defined/discussed，为的是捞回「模型自己承认没有」的 8 条编造。
    # 68 题三轮的原始数据是：
    #   轮1 仅放宽正则            修复 59%  保持 85%
    #   轮2 放宽正则 ＋ 铺垫补丁   修复 41%  保持 82%
    #   轮3 两者都回退            修复 41%  保持 85%
    # 当初写的「放宽导致两个指标都变差」是把轮1→轮2 的下降算到了放宽头上——
    # 那一跤是**铺垫补丁**摔的；放宽单独看（轮1 vs 轮3）是 +18pp 修复率、保持率持平。
    # 回退时把两个改动捆在一起撤了；后续 n=1007 单独重测仍未显示净收益，
    # 故默认保持窄正则，AITIC_WIDEN_REFUSAL 只作历史实验复现。
    # 铺垫补丁不再复活：它的失败与样本量无关，逻辑本身就错。
    # “The term … is not directly defined in the material” 比任意 subject-first 更窄：
    # 它显式把所问术语声明为未定义。历史 9 条完整臂扫描未命中过正确答案；单独收口，
    # 不把已经被实验否决的通用 X is not directly ... 重新设为默认。
    # 另一种稳定变体是 ``The term X is not mentioned or explained in the
    # material``：两个并列的否定覆盖动词已经明确宣告整题无据。32 个历史 rows
    # 文件离线扫描命中 8 次，全部为库外失败、可答题 0 次；只补这个双动词形态，
    # 不借机启用下方曾被全量实验否决的通用副词/动词放宽。
    r"(?:the\s+)?term\b[^.\n]{1,160}\s+(?:is|are|was|were)\s+not\s+"
    r"(?:defined|discussed|mentioned|covered|addressed|explained)\s+or\s+"
    r"(?:defined|discussed|mentioned|covered|addressed|explained)\s+"
    r"(?:in|by)\s+(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)\b|"
    r"(?:the\s+)?term\b[^.\n]{1,160}\s+(?:is|are|was|were)\s+not\s+"
    # ``not explicitly defined`` 是本轮 3 次全量里稳定复现的同一条库外失败。
    # 历史全部 JSONL 扫描命中 32 次，全部为 unanswerable、正确答案 0 次；
    # 只把 explicitly 纳入这个以 ``The term ...`` 开头的窄分支，不启用下方
    # 曾被全量否决的通用 defined/discussed 放宽。
    r"(?:explicitly|directly|specifically|clearly)\s+"
    r"(?:defined|discussed|mentioned|covered|addressed)\s+"
    r"(?:in|by)\s+(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)\b|"
    r"[^.\n]{1,180}\s+(?:is|are|was|were)\s+not\s+(?:(?:" + _RF_ADV + r")\s+)?"
    r"(?:" + _RF_VERB + r")\s+"
    r"(?:in|by)\s+(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)\b|"
    r"no\s+(?:information|details?|evidence|mention)\s+.{0,80}\s+"
    r"(?:is|are|was|were)\s+(?:provided|found|available|contained|mentioned)\b|"
    # brief 实测会写成“提供的材料未涉及 X，因此无法撰写简报”。这是明确整题拒答，
    # 但不同于被否决的通用“未直接定义”放宽：这里同时要求材料作主语、未涉及/提及，
    # 以及无法撰写/回答三个信号都出现在开头首句。
    r"(?:所?提供的?|当前|现有)?(?:材料|资料|上下文|文档|知识库)(?:中|里)?\s*"
    r"(?:并未|未曾|未)\s*(?:涉及|提及|包含|覆盖)"
    r"[^。\n]{0,120}?(?:(?:因此|所以|故)\s*)?(?:无法|不能|不足以)\s*"
    r"(?:撰写|提供|回答|说明|展开)|"
    # 中文也会把主题放在前面再说“未在材料中提及”。原先只覆盖
    # “材料中没有……”这一语序，导致语义明确的拒答绕过固定 token 契约，
    # 进入逐句核验后甚至被当成普通答案交付。与上面的英文 subject-first
    # 分支保持同样的窄范围：只认开头首句里的明确“未在材料中 + 提及/定义”。
    r"[^。\n]{1,120}(?:并未|未曾|未)\s*在\s*"
    r"(?:(?:所?提供|检索到|现有|当前)的?)?\s*"
    r"(?:材料|资料|上下文|文档|知识库)(?:中|里)?\s*"
    r"(?:提及|讨论|定义|说明|包含|找到)|"
    r"(?:当前|现有|所提供的?|检索到的?)?(?:材料|资料|上下文|文档|知识库)(?:中|里)?"
    r"(?:没有|并未|未能|未|不包含|找不到|缺少).{0,24}(?:信息|依据|内容|答案|提及)"
    r")",
    re.I,
)


def _looks_like_prose_refusal(answer):
    """识别模型把拒答写成散文的情况，并在最终输出前归一为固定 token。

    只匹配答案开头非常明确的“材料不包含/未提供”表述；中途说明某个子问题
    未覆盖的部分回答不会被误判成整题拒答。

    【2026-08-12 回退，8/13 已重测】当初把「放宽副词」和「铺垫后仍有带引用实质内容
    就不算拒答」两个改动捆在一起回退了，理由是 68 题对照两个指标都变差——但那是
    轮1→轮2 的下降，责任在铺垫补丁；放宽单独看是 +18pp 修复率、保持率持平（见上方表）。
    后续 n=1007 配对重测按完整净额为：命中 -19、编造 -11、净值 +3；同条件
    空跑臂也是 +3，故证不出净收益。默认保持窄正则，开关只用于复现实验。
    """
    cleaned = M._strip_think(str(answer or "")).strip()
    # 引用模型偶尔先吐一个合法页码，再承认材料并未提供答案：
    # ``[p.1220] The material does not provide...``。页码不能把明确拒答伪装成
    # 正常作答；这里只为判定跳过连续的开头引用，不改写交付文本，也不跳过正文。
    while True:
        match = re.match(r"^\s*\[([^\[\]\r\n]+)\]\s*", cleaned)
        if not match or _normalized_cite_atom(match.group(1)) is None:
            break
        cleaned = cleaned[match.end():]
    # A pruned survivor can legitimately start with a citation followed by a
    # discourse connective (``[p.7] However, the material does not...``).
    # The citation and connective are both presentation scaffolding; neither
    # may hide an otherwise explicit whole-answer refusal.
    cleaned = _strip_leading_connective(cleaned)
    return bool(_PROSE_REFUSAL_RE.search(cleaned))


def _finalize_agent_answer(answer, packed_idx, metas, packed):
    """流式与非流式共享同一条最终安全收口，避免两套准确率口径。"""
    cleaned = _dedupe_adjacent_citations(_expand_compound_citations(answer)).strip()
    if _looks_like_prose_refusal(cleaned):
        cleaned = _NO_REFERENCE
    else:
        # 排版归一必须发生在算 claims 之前：切分器现在按行走，
        # 晚一步核验看到的就还是"一整坨"，逐条接地率也就无从谈起。
        cleaned = _normalize_answer_layout(cleaned)
    initial_check = _verify_citations(cleaned, packed_idx, metas)
    claims = _claim_evidence_map(cleaned, packed_idx, metas, packed)
    skipped = {"triggered": False, "state": "pass", "checked": 0,
               "supported": 0, "pruned": 0, "unknown": 0, "orphaned": 0,
               "reason": "拒答或伪造引用沿用既有安全收口", "verdicts": []}
    # 已拒答或出现检索外标签时，继续沿用原先 fail-closed 契约，不能让二次模型洗白伪造引用。
    if M.is_abstain(cleaned) or initial_check.get("fabricated"):
        final = _finalize_grounded_answer(cleaned, initial_check)
        return final, _verify_citations(final, packed_idx, metas), [], skipped, 0

    guarded, audit, tokens = _semantic_support_guard(
        cleaned, claims, packed_idx, metas, packed)
    # A firm semantic prune must not yield a newly unsupported sentence after
    # reconstruction.  This caught a real desktop answer where p.376 (a
    # semaphore passage) drifted onto a generic Python claim: the citation was
    # syntactically valid, but it no longer belonged to the verified sentence.
    # Keep UNKNOWN fail-open unchanged; only deterministic, post-prune drift is
    # repaired here and no extra model sample is introduced.
    if (audit.get("triggered") and audit.get("pruned")
            and not audit.get("unknown") and not M.is_abstain(guarded)):
        rebuilt_claims = _claim_evidence_map(guarded, packed_idx, metas, packed)
        guarded, drift_pruned = _repair_post_prune_support_drift(
            guarded, rebuilt_claims)
        if drift_pruned:
            audit = dict(
                audit,
                state="refused" if M.is_abstain(guarded) else "pruned",
                pruned=int(audit.get("pruned") or 0) + drift_pruned,
                reassembly_pruned=drift_pruned,
                reason="逐句裁剪重组后移除了失去引用支持的结论",
            )
    # 逐句裁剪可能移走前面的实质结论，只留下模型自己的拒答说明。必须在裁剪后
    # 重跑同一个窄判据，否则会把“材料没有答案”连同一个合法页码当成命中交付。
    if _looks_like_prose_refusal(guarded):
        guarded = _NO_REFERENCE
        if audit.get("triggered"):
            audit = dict(audit, state="refused", reason="逐句裁剪后仅剩明确拒答表述")
    check = _verify_citations(guarded, packed_idx, metas)
    final = _finalize_grounded_answer(guarded, check)
    # The semantic guard rebuilds supported sentences and may append a
    # normalized citation to a sentence that already ended in the same tag.
    # The input-side pass above cannot see duplicates introduced here, so make
    # the presentation invariant hold at the final delivery boundary as well.
    final = _dedupe_adjacent_citations(final)
    if M.is_abstain(final) and audit.get("triggered") and audit.get("state") != "degraded":
        audit = dict(audit, state="refused", reason="逐句裁剪后没有可交付的完整引用结论")
    final_check = _verify_citations(final, packed_idx, metas)
    final_claims = _claim_evidence_map(final, packed_idx, metas, packed)
    return final, final_check, final_claims, audit, tokens


_HEDGE_RE = re.compile(
    r"^\s*(根据(材料|资料|上下文|原文)|材料(中|里)?(提到|显示|表明)|资料(中|里)?(提到|显示)|"
    r"文中(提到|显示)|the (provided )?(material|context|text)\s+(states|mentions|shows|indicates)|"
    r"according to the (material|context|text))", re.I)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _answer_language_rule(question):
    """Return an unambiguous output-language rule for the current question.

    "Same language as the question" is ambiguous for ordinary Chinese queries
    containing an English subject name, such as ``business law在AIGC中``.  The
    local model treated the leading Latin phrase as the primary language and
    returned a fully English answer.  Any CJK text is therefore an explicit user
    signal for Chinese UI output; technical names may remain in Latin script.
    """
    if _CJK_RE.search(str(question or "")):
        return (
            "The current question is Chinese, even if it contains English terms. "
            "Write the entire answer in Simplified Chinese. Keep only proper names, "
            "abbreviations, source titles, and exact citation tags in their original form."
        )
    if _TERM_DIRECTNESS:
        return (
            "The current question is English. Write the entire answer in English; do not use "
            "Chinese words, Chinese headings, or Chinese lead-ins. If the question describes a "
            "concept and asks you to identify it, name the exact English term used by the supplied "
            "material in the opening sentence, then explain it. Never invent or translate a term: "
            "if the supplied material does not explicitly name it, output exactly "
            "[NO REFERENCE FOUND]."
        )
    return "Answer the current question directly, in the same language as the current question."


def _answer_language_matches(question, answer):
    """Detect the narrow failure case: Chinese question, English answer.

    This is a retry signal rather than a general language detector.  Exact
    refusals remain valid and non-Chinese questions retain the previous behavior.
    The ratio check avoids accepting a long English paragraph with a token Chinese
    heading while allowing common Latin technical terms in a Chinese explanation.
    """
    if M.is_abstain(answer) or not _CJK_RE.search(str(question or "")):
        return True
    body = _strip_tags(str(answer or ""))
    cjk_count = len(_CJK_RE.findall(body))
    latin_count = len(re.findall(r"[A-Za-z]", body))
    return cjk_count >= 4 and cjk_count >= max(4, int(latin_count * 0.12))


_COMPARISON_QUESTION_RE = re.compile(
    r"(?P<left>[^，。！？?]{1,48}?)(?:和|与|跟|相比(?:于)?|相较(?:于)?)"
    r"(?P<right>[^，。！？?]{1,48}?)(?:有(?:什么|何种|哪些)?(?:关键|主要|核心)?区别|"
    r"有何不同|如何不同|(?:的)?(?:关键|主要|核心)?区别(?:是(?:什么)?|在于))",
    re.I,
)
_EVASIVE_MATERIAL_RE = re.compile(
    r"(?:(?:本?(?:材料|教材|书)|讲义|资料|文中|书中)(?:里|中|内)?"
    r"[^。！？!?]{0,18}?)?(?:并?未|没有)[^。！？!?]{0,24}?(?:提及|说明|给出|记载|解释)",
    re.I,
)
_EVASIVE_MATERIAL_EN_RE = re.compile(
    r"(?:"
    r"\b(?:is|are|was|were)\s+not\b[^.!?]{0,48}"
    r"\b(?:addressed|defined|discussed|mentioned|covered)\b[^.!?]{0,80}"
    r"\b(?:material|text|context|book)\b"
    r"|\b(?:material|text|context|book)\b[^.!?]{0,60}"
    r"\b(?:does\s+not|doesn't|did\s+not|didn't)\b[^.!?]{0,36}"
    r"\b(?:address|define|discuss|mention|cover)\w*\b"
    r")",
    re.I,
)
_EXPLICIT_EN_TERM_QUESTION_RE = re.compile(
    r"^\s*(?:define|explain\s+(?:the\s+)?term|"
    r"give\s+(?:a\s+|the\s+)?definition\s+of)\s+(.+?)\s*[.?!]*\s*$",
    re.I,
)


def _comparison_subjects(question):
    """提取高置信度中文比较题的两端；提取不稳时宁可不判。"""
    match = _COMPARISON_QUESTION_RE.search(str(question or ""))
    if not match:
        return None
    left = match.group("left").strip(" \t\r\n，。！？?：:；;\"'“”‘’")
    right = match.group("right").strip(" \t\r\n，。！？?：:；;\"'“”‘’")
    # 去掉“在教材中的/请比较”一类问句脚手架，避免把整段提示当作比较对象。
    left = re.split(r"(?:中的|里的)", left)[-1]
    left = re.sub(r"^(?:请|请问|比较|说明|解释|简述)+", "", left).strip()
    right = re.sub(r"\s+在[^，。！？?]*$", "", right).strip()
    if not (2 <= len(left) <= 40 and 2 <= len(right) <= 40):
        return None
    return left, right


def _question_asks_about_material_coverage(question):
    text = str(question or "")
    chinese = bool(re.search(r"(?:材料|教材|本书|书中|讲义|资料|文中)", text)
                   and re.search(r"(?:是否|有没有|有无|提没提|提到|提及|记载)", text))
    english = bool(
        re.search(r"\b(?:material|text|context|book)\b", text, re.I)
        and re.search(r"\b(?:mention|address|cover|discuss|define)\w*\b", text, re.I)
        and re.search(r"\b(?:does|do|did|is|are|was|were|whether)\b", text, re.I)
    )
    return chinese or english


def _is_evasive_material_nonanswer(question, claims):
    """识别“材料没说”被包装成普通作答的窄失败形态。"""
    if not claims or _question_asks_about_material_coverage(question):
        return False
    # 只看首个结论：若模型先正面回答、末尾再诚实说明资料的覆盖边界，不能误杀；
    # 反之首句就是“并未提及”，对一个普通事实问题而言并不是可交付答案。
    first = str(claims[0].get("claim") or "")[:180]
    return bool(_EVASIVE_MATERIAL_RE.search(first)
                or _EVASIVE_MATERIAL_EN_RE.search(first))


def _explicit_english_term(question):
    """Extract only high-confidence, explicitly named English term questions."""
    text = str(question or "")
    if _CJK_RE.search(text) or _question_asks_about_material_coverage(text):
        return None
    match = _EXPLICIT_EN_TERM_QUESTION_RE.fullmatch(text)
    if not match:
        return None
    term = match.group(1).strip(" \t\r\n.?!:;\"'“”‘’")
    term = re.sub(r"^(?:a|an|the)\s+", "", term, flags=re.I)
    if not (2 <= len(term) <= 100) or not re.search(r"[A-Za-z0-9]", term):
        return None
    return term


def _unnamed_explicit_term_issue(question, packed):
    """Return a deterministic issue when an explicit term is absent from evidence.

    The candidate prompt requires the model not to invent a term, but prompt-only
    enforcement can fail (for example, turning ``self-awareness`` into the unseen
    phrase ``Objective self-awareness``).  This check uses the exact adopted evidence
    snapshot and adds no model sampling.

    The paired evaluation is now complete and did not support promoting it: at
    n=1007 against a blank arm the net was -3, and "Define Objective
    self-awareness." -- the very example above -- turned from a correct refusal
    into a fabrication with the guard enabled.  It stays candidate-only; see the
    _TERM_DIRECTNESS definition for the full accounting.
    """
    if not _TERM_DIRECTNESS:
        return None
    term = _explicit_english_term(question)
    if not term or any(_anchor_in_document(term, block) for block in (packed or [])):
        return None
    return {"code": "unnamed_explicit_term",
            "detail": "材料证据未明确命名问题中的英文术语：%s" % term}


def _answer_directness(question, answer, claims):
    """输出校验：答案有没有正面回答问题。

    v6 事件（ollama 服务端 0.31.1→0.32.3）已经证明**提示词约束会被外部依赖变更整条击穿**，
    当时的症状正是只吐 ``[p.955]`` 而不写正文、模糊题掉 30pp。因此这里必须有代码侧兜底，
    不能只靠 ``_agent_prompt`` 里的那几条 rules。

    ``retry`` 只在**零误判风险**的形态上置真（正文为空 / 无结论句）；
    ``low_grounding`` 这类基于接地率的判据先只做标记与降可信度，
    等真机数据测清失败构成后再决定是否接入重试——项目方法论第 8 条
    「先测失败构成，再选机制」。
    """
    if M.is_abstain(answer):
        return {"ok": True, "issues": [], "retry": False,
                "detail": "诚实拒答，不适用正面性校验"}

    issues = []
    body = _strip_tags(answer)
    if len(body) < 8:
        issues.append({"code": "only_citation", "detail": "只输出了引用标记，没有正文结论"})
    if not claims:
        issues.append({"code": "no_claim", "detail": "未能从答案中解析出任何结论句"})
    if not _answer_language_matches(question, answer):
        issues.append({"code": "language_mismatch",
                       "detail": "问题包含中文，但回答主体不是中文"})

    subjects = _comparison_subjects(question)
    if subjects and body:
        missing = [subject for subject in subjects if not _anchor_in_document(subject, body)]
        if missing:
            issues.append({"code": "comparison_missing_subject",
                           "detail": "比较题没有同时回应两端：缺少%s" % "、".join(missing)})
    if _is_evasive_material_nonanswer(question, claims):
        issues.append({"code": "evasive_material_nonanswer",
                       "detail": "回答以材料未提及为结论，却没有正式拒答"})

    measured = [c for c in claims if c.get("measured")]
    if measured and not any(c["supported"] for c in measured):
        issues.append({"code": "low_grounding",
                       "detail": "全部结论句的接地率均低于 %.0f%%，引用可能只是装饰" % (100 * _GROUNDED_MIN)})
    if claims and all(_HEDGE_RE.match(c["claim"]) for c in claims):
        issues.append({"code": "hedge_only", "detail": "通篇是材料转述，没有给出直接结论"})
    uncited = [c for c in claims if not c["citations"]]
    if claims and len(uncited) == len(claims):
        issues.append({"code": "uncited", "detail": "结论句均未附带可核对的引用"})

    retry_codes = {"only_citation", "no_claim", "language_mismatch",
                   "comparison_missing_subject", "evasive_material_nonanswer"}
    return {"ok": not issues, "issues": issues,
            "retry": any(x["code"] in retry_codes for x in issues),
            "detail": "；".join(x["detail"] for x in issues) or "答案直接回应了问题"}


def _enforce_final_directness(question, answer, packed_idx, metas, packed,
                              cite_check, claims, support_audit):
    """最终裁剪后重查不可交付形态；不新增模型采样，失败时安全拒答。"""
    directness = _answer_directness(question, answer, claims)
    term_issue = None if M.is_abstain(answer) else _unnamed_explicit_term_issue(
        question, packed)
    if term_issue:
        issues = list(directness.get("issues") or []) + [term_issue]
        directness = dict(
            directness,
            ok=False,
            issues=issues,
            retry=True,
            detail="；".join(item.get("detail", "") for item in issues if item.get("detail")),
        )
    fatal = {"comparison_missing_subject", "evasive_material_nonanswer",
             "unnamed_explicit_term"}
    hit = fatal & {item.get("code") for item in directness.get("issues", [])}
    if not hit or M.is_abstain(answer):
        return answer, cite_check, claims, support_audit, directness
    reason = "最终裁剪后答案不完整：%s" % directness.get("detail", "正面性校验失败")
    refused = _NO_REFERENCE
    # ``refused_by`` 必须在这里落下来。下面返回的 directness 是**在拒答文本上重算**的，
    # 而拒答走 is_abstain 的早退分支，issues 会变成空列表——触发的守卫代码到此就丢了。
    # 丢了之后 _agent_payload 只看得到 is_abstain，会把"守卫拒答"写成"检索不到证据"，
    # 于是评测日志与界面都指向错误的原因。v6 事故的教训正是归因错一次就白查一整天。
    audit = dict(support_audit or {}, triggered=True, state="refused", reason=reason,
                 final_directness_refused=True, refused_by=sorted(hit))
    return (refused, _verify_citations(refused, packed_idx, metas), [], audit,
            _answer_directness(question, refused, []))


# ----------------------------- Agent Loop -----------------------------
_AGENT_HISTORY_MESSAGES = 6
_AGENT_HISTORY_CHARS = 1800
_COMPLEX_RE = re.compile(
    r"(分析|比较|诊断|鉴别|为什么|如何|综合|机制|证据链|多本|图中|表中|图表|关系|"
    r"analy|compar|diagnos|differentiat|why|how|mechanism|evidence|across|figure|table|relationship)",
    re.I,
)
_FOLLOWUP_RE = re.compile(
    r"(?:它|这个|那个|上述|前面|第二点|继续|展开|其(?:中|本身|作用|区别|原因|机制|特点|含义|用途|原理)|该(?:概念|方法|过程|结论)|"
    r"\b(?:it|this|that|those|they|them|the former|the latter)\b)",
    re.I,
)
_HISTORY_CITATION_RE = re.compile(
    r"\[(?:K\d+:)?(?:p\.?\s*\d+|ch:[^\]]+|image:[^\]]+|audio[^\]]*)\]",
    re.I,
)


def _detect_intent(question, history=None, library_count=1):
    q = question or ""
    rules = [
        ("图表解析", r"图中|表中|图表|曲线|柱状图|figure|table|chart|diagram"),
        ("诊断推理", r"诊断|鉴别|症状|病理|病因|治疗方案|diagnos|differential|symptom|patholog"),
        ("跨资料综合", r"多本|跨书|综合.*资料|共同|分别依据|across|multiple books|sources"),
        ("比较分析", r"比较|区别|异同|优缺点|versus|\bvs\b|compar|difference"),
        ("因果机制", r"为什么|原因|机制|如何导致|why|mechanism|cause|how does"),
    ]
    if history and _FOLLOWUP_RE.search(q):
        name = "上下文追问"
    else:
        name = next((label for label, pattern in rules if re.search(pattern, q, re.I)), "事实查询")
    complex_query = name != "事实查询" or bool(_COMPLEX_RE.search(q)) or library_count > 1
    return {"name": name, "complexity": "复杂" if complex_query else "简单",
            "route": "agent_loop" if complex_query else "fast_rag",
            "reason": ("需要跨证据检索、比较或校准" if complex_query
                       else "问题边界明确，可先用首轮检索直接回答")}


def _normalize_history(raw):
    """只保留最近的短对话，避免 URL/Prompt 无限增长。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = []
    if not isinstance(raw, list):
        return []
    cleaned = []
    for item in raw[-_AGENT_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        # 历史页码不属于本轮检索证据。若原样送入模型，追问时模型可能复制旧页码，
        # 形成“内容看似连贯、引用却指向上一轮”的假证据。
        if role == "assistant":
            content = _HISTORY_CITATION_RE.sub("", content)
        content = re.sub(r"\s+", " ", content).strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:600]})
    # 从后往前保留，严格控制总字符数。
    out, used = [], 0
    for item in reversed(cleaned):
        left = _AGENT_HISTORY_CHARS - used
        if left <= 0:
            break
        item = dict(item)
        item["content"] = item["content"][:left]
        used += len(item["content"])
        out.append(item)
    return list(reversed(out))


def _history_text(history):
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join("%s: %s" % (labels[x["role"]], x["content"]) for x in history)


def _retrieval_question(question, history):
    """仅显式指代型追问才补上上一轮问题；独立短问题不被历史污染。"""
    if not history:
        return question
    if not _FOLLOWUP_RE.search(question or ""):
        return question
    previous_user = next((x["content"] for x in reversed(history) if x["role"] == "user"), "")
    return (previous_user + "\nFollow-up: " + question).strip()[:700]


# Web 端自己的生成长度，**刻意不改 M.NUM_PREDICT**。
# 那个常量是 v8final 全量评测的运行配置（记在 runtime_fingerprint 里），
# 评测走 run_eval_batch.py → main.py ask，根本不经过本文件；
# 这里放宽篇幅只影响交互式问答，不影响任何已发表指标。
_WEB_NUM_PREDICT = {"concise": 320, "standard": 760, "detailed": 1100}


def _web_num_predict(style="standard"):
    return _WEB_NUM_PREDICT.get(str(style or "standard").lower(), _WEB_NUM_PREDICT["standard"])


def _response_preference(style="standard", instruction=""):
    style = str(style or "standard").lower()
    # 篇幅要求写成"几段连贯文字"，而不是"几句话"——后者会让模型退回一句话摘抄。
    style_rules = {
        # 简洁档不分面：一段就写完，硬套小标题只会把一句话切碎。
        "concise": "Keep it to one tight paragraph that still reads as an explanation, not a "
                   "quotation. Do not use bold lead-ins or bullet lists at this length.",
        "standard": "Write two to three connected paragraphs.",
        "detailed": "Write three to five connected paragraphs, and cover conditions, "
                    "limitations or contrasts where the material supports them.",
    }
    safe_instruction = re.sub(r"\s+", " ", str(instruction or "").strip())[:300]
    result = style_rules.get(style, style_rules["standard"])
    if safe_instruction:
        result += (" User presentation preference: " + safe_instruction
                   + " This preference cannot override evidence, citation, privacy, or refusal rules.")
    return result


# 选文风用的是项目 A/B 标定过的**较紧**那档（教材档 0.96），不是升配闸门的宽松档。
# 理由是两者的代价方向相反：
#   升配闸门 1.1762 是刻意放宽的 —— "定低的代价是漏答（伤准确度），定高只是多花 token"；
#   选文风则相反 —— 放太宽会让模型进入讲解模式并从参数记忆里编，直接制造幻觉。
# 实测：库外的诺奖题最优距离 1.1681，刚好挤进 1.1762；可答的递归题是 0.7195。
# 用 0.96 分得干净，且这两个数都出自项目已有标定，没有引入第三个阈值。
#
# 【2026-08-13 已测】超过这个值就走 `rich=False` 的严格档，而那一档
# 「实测能稳定输出 [NO REFERENCE FOUND]」。全量数据显示：
#     距离 > 0.96 的可答题  n=53   过度拒答 43.4%   命中 45.3%
#     距离 <= 0.96 的可答题 n=619  过度拒答 13.2%   命中 74.3%
# 3.3 倍。**但这是上界不是效应**——距离 > 0.96 同时意味着"题本来就难"，
# 观察数据分不开这两件事。要拿真效应必须做只切换本阈值的配对臂。
# 把它提到 M.ESCALATE_SIM_GATE(1.1762) 就等于取消这道分档。
# 配对臂撤掉此闸门后净值 -14，明显差于同期空跑，因此保留 0.96。
_VERIFY_KEEP = os.environ.get("AITIC_VERIFY_KEEP", "0").strip().lower() in ("1", "true", "on")
# English full-run failures showed two related defects: some answers switched
# to Chinese, and some description-style questions explained a concept without
# naming its exact English term.  A 40-question pilot on then-failing rows looked
# convincing (8 -> 14 hits, existing-hit control unchanged at 16, out-of-library
# fabrications 3 -> 1), and the rule shipped enabled on that basis.
#
# The full English run then contradicted the pilot.  Paired on (book, question)
# against a same-code blank arm, n=1007, every row matched:
#     hits   537 -> 532    (32 gained, 37 lost)
#     fabs    17 ->  16    (4 fixed, but 3 newly introduced)
#     net = -5 + 2*(+1) = -3, with 94/1007 = 9.3% of rows flipping either way
# The pilot's +6 came entirely from the 40 rows it was selected on, and that
# slice never had a blank arm; at full scale the same change is net negative.
#
# Two pre-registered rules both say revert:
#   - PLAN_weekend.md fixes the acceptance gate at net > 3N against every blank
#     arm.  A negative net cannot clear it for any positive N, so this fails
#     without having to settle the exact noise figure first.
#   - CODEX_CHECKPOINT_20260818.md says roll back on any out-of-library safety
#     regression.  Three previously-correct refusals became fabrications, one of
#     them "Define Objective self-awareness." -- the exact failure mode
#     _unnamed_explicit_term_issue was written to prevent.
#
# So this returns to candidate-only: off by default, switch kept so both arms
# stay reproducible.  Evidence lives in docs/全量跑分_20260812/ as
# desktop_v6_blank_en_20260819_rows.jsonl (blank arm) and
# desktop_term_directness_v6_en_20260819_rows.jsonl (candidate arm).
_TERM_DIRECTNESS = os.environ.get("AITIC_TERM_DIRECTNESS", "0").strip().lower() in (
    "1", "true", "on")


def _read_style_gate():
    raw = os.environ.get("AITIC_STYLE_GATE")
    if not raw:
        return 0.96
    try:
        return float(raw)
    except ValueError:
        # 与 _read_evidence_floor 同一条理由：写错必须响，
        # 否则整臂跑完看到"无差异"，会被误读成"这个改动没用"。
        raise RuntimeError("AITIC_STYLE_GATE 不是合法浮点数: %r" % (raw,))


_STYLE_GATE_MAX = _read_style_gate()


# 证据下限：检索最优距离比这个还差，就不让模型作答。
#
# 为什么需要它：拒答此前完全由模型自己判断，而 986 题扩容全量实测发现，
# 库越大模型越容易「看起来有据」地编造——不可答题的最优距离中位数
# 小库 1.142 / 中库 1.105 / 大库 1.034，落在旧闸门 1.1762 之内的比例
# 从 64% 涨到 88%。旧闸门对任何规模都太松，形同虚设。
#
# 阈值由扫描定，不是拍的（28 条编造 + 160 条正确答案对照）：
#   0.93 → 拦住 46% 编造，误杀 7% 正确
#   0.95 → 拦住 43%，误杀 6%
#   0.99 → 拦住 29%，误杀 1%   ← 取这个，代价最低
#   1.11 → 拦住  0%           ← 旧闸门所在区间，完全不起作用
#
# 【边界】样本只有 28 条编造，且阈值是在同一批数据上标定的，存在过拟合风险。
# 分规模标定（中库 0.93 / 大库 0.87）效果更好但样本更薄，暂不做。
# 这是**保守的第一版**。
#
# 【2026-08-13】上面那句「下一轮全量要复核」一直没兑现：它上线前后的
# 10.8% → 9.3% 来自**两个不同题集**（986 vs 1007），不是配对对照，归因不了。
# 后续配对臂已补：关闭下限得到命中 +3、编造 +5，净值 -7；同期空跑净值
# 约 ±3，支持保留 0.99，但证据强度有限，不应写成“已证实最优”。环境变量
# 覆盖仍可用于复现（设 99 等于关闭），默认值不变。
def _read_evidence_floor():
    raw = os.environ.get("AITIC_EVIDENCE_FLOOR")
    if not raw:
        return 0.99
    try:
        return float(raw)
    except ValueError:
        # 环境变量写错时**必须响**：静默退回默认值会让整臂实验白跑，
        # 而且跑完看到的是"无差异"，会被误读成"这个改动没用"。
        raise RuntimeError("AITIC_EVIDENCE_FLOOR 不是合法浮点数: %r" % (raw,))


_EVIDENCE_FLOOR = _read_evidence_floor()


def _evidence_floor_blocks(dists):
    """最优检索距离差于下限时返回 True —— 该题证据不足，不交给模型判断。

    缺少距离信息时返回 False（保持原行为）：宁可让模型自己判断，
    也不能因为拿不到距离就把正常问答全部拒掉。
    """
    usable = [d for d in (dists or []) if isinstance(d, (int, float))]
    if not usable:
        return False
    return min(usable) > _EVIDENCE_FLOOR


def _brief_low_evidence_blocks(dists, claims):
    """低相似度 brief 至少要有两条被本地证据确认的引用结论。

    不能把问答的 0.99 下限机械套到 brief：中文“交子”实测最优距离 1.0231，
    但 6 条结论被原文支持；直接套用会误杀真答案。反例“光合作用”距离 1.1335，
    只有 1/6 条支持，却把教材外定义放在概要里交付。brief 本来就是多要点文档，
    所以仅在距离已经越界时要求至少两条受支持引用，既保留前者又拦住后者。
    """
    if not _evidence_floor_blocks(dists):
        return False
    grounded = [claim for claim in (claims or [])
                if claim.get("supported") and claim.get("citations")]
    return len(grounded) < 2


def _evidence_looks_present(dists):
    """生成之前判断"这题材料里大概有没有依据"，决定用连贯讲解还是严格简洁。

    这个开关是实测逼出来的：把"像老师讲课那样连贯讲解"无条件加进提示词后，
    库外题拒答从 3/3 掉到 0/3 —— 模型被推进讲解模式，转而从参数记忆里编
    （"2024 年诺贝尔物理学奖授予量子信息科学…"），正是 v7 核对出的
    42/49 真幻觉形态。V3 的拒答契约压不住这个拉力，只能在进入该模式前就拦。
    """
    usable = [d for d in (dists or []) if isinstance(d, (int, float))]
    if not usable:
        return False                     # 缺少距离时保守关闭讲解模式，避免异常路径放大幻觉
    try:
        gate = M.resolve_gate()
    except Exception:
        gate = M.ESCALATE_SIM_GATE
    if not isinstance(gate, (int, float)):
        gate = M.ESCALATE_SIM_GATE
    return min(usable) <= min(gate, _STYLE_GATE_MAX)


def _agent_prompt(context, question, packed_idx, metas, history=None, verification=False,
                  preference="", rich=True, structured=True):
    history = history or []
    parts = []
    if history:
        parts.append("Conversation context (use only to resolve references; the retrieved material remains the authority):\n"
                     + _history_text(history))
    parts.append("Current question: " + question)
    # 目标是"像人讲课那样连贯地讲清楚"，同时每句仍可溯源。
    # 旧规则里的 "State the answer before explaining evidence" 会被模型执行成
    # 「Answer: …／Evidence: …」两段标签，读起来像检索结果而不是回答，故改写。
    rules = [_answer_language_rule(question)]
    if rich and structured:
        # 目标形态：一句话先答 → 分面展开（每面一段、以粗体小标题起头）→ 需要并列时用要点列表。
        # 「粗体小标题」和被禁掉的「Answer:／Evidence:」是两回事：前者点的是**话题**
        # （从功能上看 / In practice），后者点的是**段落角色**，会把回答变成检索结果的排版。
        rules += [
            # 实测：把小标题放宽成"可选"之后，模型会把它用到开头去
            # （'**从功能上看**，梦是…'），等于给开场白加了个前缀，正是这条要禁的。
            "Lead with one self-contained sentence that answers the question outright — "
            "no preamble, no restating the question, no 'the material states that', "
            "and no bold lead-in on this opening sentence.",
            # 只要"读起来像人在讲"，不规定必须有小标题或列表——那些交给模型判断。
            # 但**分段**要说死：实测只写 'prefer short paragraphs' 时模型一行到底，
            # 又变回当初挨批的那一坨。分段是可读性，不是格式规定。
            "Then explain it the way a knowledgeable person would in conversation.",
            "Whenever you move to a different aspect, start a new paragraph: end the line, leave "
            "one blank line, then continue. Never return the whole answer as a single unbroken "
            "block of text.",
            "A short bold lead-in naming the aspect, e.g. '**从功能上看**，', is welcome where it "
            "genuinely helps the reader — but it is optional. Never force one, and never split a "
            "single idea just to create structure.",
            "A markdown bullet list ('- ' at the start of each line) is fine when you are really "
            "enumerating parallel items; plain prose is equally fine otherwise. Use whichever "
            "actually reads better.",
            "Never use structural labels such as 'Answer:', 'Evidence:', '证据：', '结论：' — a bold "
            "lead-in names the topic, never the role of the paragraph. "
            "Do not output a citation tag on a line of its own.",
        ]
    elif rich:
        # 简洁档：仍要读起来像讲解，但不分面、不列点。
        rules.append("Write one tight explanatory paragraph. Do not use bold lead-ins, "
                     "bullet lists, or headings.")
    else:
        # 证据看着不足时退回严格模式：这一档实测能稳定输出 [NO REFERENCE FOUND]。
        rules.append("State only what the material supports, in as few sentences as it takes. "
                     "Do not elaborate, do not add background.")
    rules += [
        # 实测数据：放宽成对话式文风后，模型写出大段自己引不了的解释句——
        # Think Python 那题 5 句里 4 句无引用（80%），Dreams 6 句里 4 句无引用。
        # 无引用句一律进逐句核验，绝大多数判不出来，于是可信度恒为「低」，
        # 界面副标题「每句话可溯源到原文页码」也就成了空话。
        # 所以逐句挂引用必须写成硬要求，并明说：引不了就别写。
        "Every factual claim must come from the supplied material; put its exact source tag "
        "inline right after the sentence it supports.",
        "This applies to every sentence, including the explanatory ones: if a sentence asserts "
        "anything about the subject, it carries its own tag. A sentence you cannot tie to a "
        "specific block must not be written at all — a shorter answer is better than padding "
        "that cannot be traced. Only a closing sentence that states what the material does not "
        "cover may go without a tag.",
        # 完全无依据时的输出是一个**精确 token**，下游的 is_abstain / 拒答标记 / 升配判定
        # 全挂在它上面。散文式的 "the material does not address this" 语义没错，却会让整套判定失效。
        "Refusal overrides every style rule above: if the material provides no basis for the "
        "question at all, output exactly [NO REFERENCE FOUND] and nothing else — no explanation, "
        "no paraphrase, no prose apology.",
        "Only when the material covers part of the question may you answer that part and then "
        "say plainly which part is not covered. Never fill a gap with outside knowledge.",
    ]
    # 多库时要求按来源组织答案：哪本书支持哪条结论、几本书是否说的一回事。
    # 只有检索层区分来源（K1/K2 标签）还不够，答案层不组织就看不出"跨书综合"。
    if len({str(m.get("_library_name") or "") for m in (metas or []) if m}) > 1:
        rules.append("Several different sources are supplied. State which source supports which "
                     "point, and say explicitly when sources agree, add to each other, or conflict.")
    if preference:
        rules.append(preference)
    if verification:
        # 【2026-08-13 已测，不采纳新措辞】原文是：
        #   "This is a verification retry: correct any unsupported, missing, or fabricated
        #    citation; do not merely repeat the first answer."
        #
        # 两处可能把模型推向拒答：
        #   1. "do not merely repeat the first answer" —— 第一轮若本来就对，这句是在
        #      要求它改，而"改"的一种方式就是改成拒答；
        #   2. 前半句把任务框成"找错"，通篇没说"对的就留着"。
        #
        # 对应实测（§二十六 / §二十九）：校验轮拒答精确率仅 17%，
        # 107 条过度拒答全部死在第 2/3 轮、且 80% 的检索距离 <= 0.99（检索本来是好的）。
        #
        # 改法：保留"纠正无据引用"的职责，去掉"别重复"的推力，并明确写出
        # "已被支持的就原样保留"。n=1007 配对结果为命中 -7、编造 -3、
        # 净值 -1，且 145/1007 题发生迁移，主要是在换一批错误；默认保持原措辞。
        if _VERIFY_KEEP:
            rules.append("This is a verification pass: check each claim against the material. "
                         "Keep every claim the material supports — if the whole answer is already "
                         "supported, return it unchanged. Correct or remove only what the material "
                         "does not support.")
        else:
            rules.append("This is a verification retry: correct any unsupported, missing, or fabricated citation; do not merely repeat the first answer.")
    parts.append("Response requirements:\n- " + "\n- ".join(rules))
    tags = list(dict.fromkeys(_web_cite_tag(metas[i]) for i in packed_idx if i < len(metas)))[:2]
    tag_example = " or ".join("[%s]" % x for x in tags) if tags else "[p.112]"
    return M.PROMPT.format(context=context, question="\n\n".join(parts), tag_example=tag_example)


def _pack_agent(docs, metas, question, budget):
    """多库场景给每本书至少一个上下文席位；单库保持既有全量评测口径。

    保底之后**必须继续用完剩余预算**：早先只装每库第一块就返回，
    剩下的检索结果一律丢弃——若正确证据落在某本书的第 2~5 条，
    模型根本看不到，表现为莫名其妙的假拒答。保底解决的是"公平"，
    不是"只给一块"。
    """
    library_ids = list(dict.fromkeys(str(x.get("_library_id") or "") for x in metas))
    library_ids = [x for x in library_ids if x]
    if len(library_ids) <= 1:
        return (M._pack_relevance(docs, question, budget) if M.RELEVANCE_TRIM
                else M._pack_truncate(docs, budget))
    first_by_library = []
    for library_id in library_ids:
        index = next((i for i, meta in enumerate(metas)
                      if str(meta.get("_library_id") or "") == library_id), None)
        if index is not None:
            first_by_library.append(index)
    if not first_by_library:
        return M._pack_relevance(docs, question, budget)

    sep = len("\n---\n")
    sep_cost = max(0, len(first_by_library) - 1) * sep
    available = max(0, budget - sep_cost)
    # The guarantee is deliberately a *reserve*, not an equal split of the
    # entire budget.  Using the whole budget for the first block of every
    # library recreated the original bug: rank-2+ evidence still never reached
    # the answer model.  Reserve at most half the available budget in total,
    # capped per library, then spend the remainder in global retrieval order.
    fair_cap = max(1, available // len(first_by_library))
    reserve = min(fair_cap, 320,
                  max(120, available // (2 * len(first_by_library))))
    packed, idx, used = [], [], 0
    for i in first_by_library:
        piece = str(docs[i] or "")[:reserve]
        used += len(piece) + (sep if packed else 0)
        packed.append(piece); idx.append(i)

    # 保底装完后按原检索名次继续补块，把剩余预算用满。
    taken = set(idx)
    for i, doc in enumerate(docs):
        if i in taken:
            continue
        left = budget - used - sep
        if left <= 120:                 # 放不下一段有意义的内容就停，避免塞进残片
            break
        text = str(doc or "")
        piece = text if len(text) <= left else text[:left]
        used += len(piece) + sep
        packed.append(piece); idx.append(i); taken.add(i)
    return packed, idx


def _run_agent_once(docs, metas, question, history, budget, verification=False,
                    preference="", style="standard", rich=True):
    """返回 packed 而不只是 packed_idx：接地率必须对着**模型实际看到的文本**算，
       用未截断的原块会高估支持度（多库配额与相关度裁剪都会截断）。"""
    packed, packed_idx = _pack_agent(docs, metas, question, budget)
    context = _labeled_context(packed, packed_idx, metas)
    out = M._generate(
        M.LLM_MODEL,
        _agent_prompt(context, question, packed_idx, metas, history, verification,
                      preference, rich, structured=str(style or "").lower() != "concise"),
        options=_llm_options(temperature=M.TEMPERATURE,
                             num_predict=_web_num_predict(style)),
    )
    # Ollama 在生成成功时也可能把计数键返回为 JSON null。默认值只处理“键不存在”，
    # 处理不了“键存在但值为 None”；直接相加会让一个已有完整答案的请求变成 500。
    toks = (out.get("prompt_eval_count", 0) or 0) + (out.get("eval_count", 0) or 0)
    return out["response"].strip(), toks, packed_idx, packed


def _web_gen_brief_raw(prompt):
    """等价于 main._gen_brief_raw，但 token 计数允许 Ollama 返回 null。

    main.py 是已封存的 CLI 汇报基线，不能为 WebUI 异常路径改动其指纹；因此 WebUI 在
    自己的入口使用这一层。提示词、system、温度和预算均与 main 原函数保持一致。
    """
    kwargs = {"model": M.LLM_MODEL, "prompt": prompt, "system": M.BRIEF_SYSTEM,
              "think": False,
              "options": _llm_options(temperature=M.TEMPERATURE,
                                       num_predict=max(M.NUM_PREDICT, 700))}
    try:
        result = M.ollama.generate(**kwargs)
    except Exception:
        payload = {"model": M.LLM_MODEL, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": M.BRIEF_SYSTEM},
                                {"role": "user", "content": prompt}],
                   "options": kwargs["options"]}
        raw = M._post_json("/api/chat", payload)
        result = {"response": (raw.get("message") or {}).get("content", ""),
                  "prompt_eval_count": raw.get("prompt_eval_count", 0),
                  "eval_count": raw.get("eval_count", 0)}
    text = re.sub(r"<think>.*?</think>", "", str(result.get("response", "")),
                  flags=re.S | re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    tokens = ((result.get("prompt_eval_count", 0) or 0) +
              (result.get("eval_count", 0) or 0))
    return text, tokens


# ----------------------------- 第二部分：模型常识补全 -----------------------------
# 汇报反馈要求问答像 GPT 一样给出完整解释，而不是只回一句书里摘的话。
# 但这与项目的核心卖点（引用锚定、不编造）直接冲突，所以**结构性分离**而不是混排：
#   · 第一部分走原链路，逐句可溯源；claims / cite_check / 可信度**只衡量它**
#   · 第二部分是模型自身知识，强制剥掉引用标签，前端显著区分
#   · 因此 v8final 的抗幻觉口径完全不受影响，评测跑的 CLI 路径也压根不经过这里
_SUPPLEMENT_TOKENS = 420
# 模型在没有材料时给出的页码必然是编的，一律剥掉——这是本功能的安全底线。
_SUPPLEMENT_TAG_RE = re.compile(r"\[[^\]]*\]")


def _supplement_prompt(question, grounded, abstained):
    """生成放在最上面的「完整解答」。

    它和下半部分的分工是：这一段**要能独立读完**，像老师当面讲一遍；
    下半部分才是逐条带页码、参与可信度计算的教材依据。所以这里不能再写成
    「教材没说的部分」那种补语——那样单独拎出来是半截话。

    唯一的硬约束：**不得与教材部分冲突**。教材说了的以教材为准，
    模型只负责把没说清的地方补上、把术语讲明白。
    """
    if abstained:
        base = ("The textbook in this knowledge base does not cover this question. "
                "Answer it from your own general knowledge, and say plainly at the start "
                "that this is not from the textbook.\n\n")
    else:
        base = ("Here is what the textbook actually says (page tags in brackets):\n%s\n\n"
                "Write a complete, self-contained explanation that a student can read on its own. "
                "Use the textbook content above as the backbone and keep every one of its claims "
                "intact; add general background only where the textbook left a gap. "
                "**Never contradict the textbook part.**\n\n" % _snippet(grounded, 900))
    return (
        "You are explaining a topic to a student. This part of the answer is NOT source-verified.\n\n"
        "%sQuestion: %s\n\n"
        "Rules:\n"
        "- Reply in the same language as the question.\n"
        "- Do NOT output any bracketed source tags such as [p.12]; this part carries no citations.\n"
        "- Give the definition, why it matters, and a concrete example where useful.\n"
        "- Prefer a few short paragraphs over one dense block.\n"
        "- If you are not confident about something, say so plainly instead of inventing detail.\n"
        % (base, question))


def _supplement_answer(question, grounded, abstained):
    """生成"教材之外"的补充说明。失败不影响主答案，返回 None 即前端不展示。"""
    try:
        out = M._generate(
            M.LLM_MODEL, _supplement_prompt(question, grounded, abstained),
            options=_llm_options(temperature=0.3, num_predict=_SUPPLEMENT_TOKENS))
    except Exception:
        return None
    text = M._strip_think(str(out.get("response") or "")).strip()
    text = _SUPPLEMENT_TAG_RE.sub("", text).strip()
    if len(text) < 12:
        return None
    toks = (out.get("prompt_eval_count", 0) or 0) + (out.get("eval_count", 0) or 0)
    return {"text": text, "tokens": toks, "grounded": False,
            "notice": "以下内容来自模型自身知识，**不出自所选教材**，未经原文核对，请自行判断。"}


_ECHO_LABEL_RE = re.compile(r"^\s*(answer|答案|答)\s*[:：]\s*", re.I)


def _clean_answer_echo(question, answer):
    """模型偶尔把题面当答案回显（"Answer: 什么是梦"），删掉这种空行。

    只删**整行等于问题**的标签行，不动任何实际内容——宁可留噪声也不能删掉真答案。
    """
    lines = str(answer or "").split("\n")
    normalize = lambda s: re.sub(r"[\s?？。.!！,，]", "", s).lower()
    target = normalize(question)
    out = []
    for line in lines:
        if not _ECHO_LABEL_RE.match(line):
            out.append(line)
            continue
        stripped = _ECHO_LABEL_RE.sub("", line).strip()
        if target and normalize(stripped) == target:
            continue                       # 整行就是"Answer: <原题>"，纯噪声，整行删掉
        # 有实质内容时只摘掉"Answer:"这个标签，内容一个字不动
        out.append(stripped if stripped else line)
    return "\n".join(out).strip() or str(answer or "").strip()


_NO_REFERENCE = "[NO REFERENCE FOUND]"


def _finalize_grounded_answer(answer, cite_check):
    """把拒答与引用失败收敛为稳定契约，绝不交付带伪造或缺失引用的正文。"""
    cleaned = str(answer or "").strip()
    if M.is_abstain(cleaned):
        return _NO_REFERENCE
    if not isinstance(cite_check, dict) or not cite_check.get("ok"):
        return _NO_REFERENCE
    return cleaned


def _followup_query(question, round_no=2):
    if re.search(r"[\u4e00-\u9fff]", question):
        suffix = (" 证据 依据 原文 机制 比较 条件" if round_no <= 2
                  else " 反例 限制条件 不确定性 逐步依据 原文定义")
    else:
        suffix = (" evidence basis source mechanism comparison criteria" if round_no <= 2
                  else " counterexample limitation uncertainty stepwise evidence exact definition")
    return (question + suffix)[:700]


def _retrieval_identity(doc, meta):
    """检索块身份必须包含所属库；不同书里的同文仍是两份独立证据。"""
    meta = meta or {}
    owner = (meta.get("_library_id") or meta.get("_library_source") or
             meta.get("source") or "")
    return str(owner), str(doc)


def _merge_retrieval(first, second):
    """原检索 top-3 锁位，补充检索只能补位，避免第二轮把正确首块挤掉。"""
    docs1, metas1, dists1 = [list(x) for x in first]
    d2, m2, s2 = second
    locked = min(3, len(docs1), M.TOP_K)
    docs = docs1[:locked]
    metas = metas1[:locked]
    dists = dists1[:locked]
    seen = {_retrieval_identity(doc, meta) for doc, meta in zip(docs, metas)}
    for doc, meta, dist in zip(d2, m2, s2):
        identity = _retrieval_identity(doc, meta)
        if identity in seen:
            continue
        docs.append(doc); metas.append(meta); dists.append(dist); seen.add(identity)
        if len(docs) >= M.TOP_K:
            break
    # 补充检索不足时，再用首轮剩余候选补齐，确保返回长度稳定。
    for doc, meta, dist in zip(docs1[locked:], metas1[locked:], dists1[locked:]):
        if len(docs) >= M.TOP_K:
            break
        identity = _retrieval_identity(doc, meta)
        if identity in seen:
            continue
        docs.append(doc); metas.append(meta); dists.append(dist); seen.add(identity)
    return docs[:M.TOP_K], metas[:M.TOP_K], dists[:M.TOP_K]


# 见 _adopt_next_round 的说明：默认关闭，只为把「校验轮拒答被当成降级丢掉」
# 这个缺陷做成可测的两臂，不在没有数据的情况下改线上行为。
_ADOPT_ABSTAIN = os.environ.get("AITIC_ADOPT_ABSTAIN", "0").strip().lower() in ("1", "true", "on")


def _round_is_clean(answer, cite_check, directness):
    """这一轮的产出是否"已经站得住"：有结论、引用全命中、正面回答了问题。"""
    if M.is_abstain(answer):
        return False
    if not (cite_check or {}).get("ok"):
        return False
    return not (directness or {}).get("retry")


def _adopt_next_round(prev_clean, answer, cite_check, directness):
    """决定是否用新一轮的结果替换上一轮。

    改成"全做 Agent"之前，第二轮只在第一轮**失败**时才触发，所以无条件替换是安全的。
    现在每道答得出的题都会走校验轮，如果仍然无条件替换，一个本来引用全命中的好答案
    会被第二轮的产出顶掉——校验轮的本意是补救，不该成为降级通道。

    规则：上一轮不干净 → 照旧采用新一轮（这是补救，本来就是为它跑的）；
          上一轮已经干净 → 新一轮必须同样干净才替换，否则保留上一轮。

    【2026-08-13 发现的缺陷，已定位未修复】
    `_round_is_clean` 对拒答一律返回 False（拒答没有引用、is_abstain 直接判否），
    于是「上一轮干净 + 这一轮拒答」会被当成降级而丢弃拒答、保留上一轮的答案。
    可**上一轮干净只说明它引用命中、正面作答，不代表它对**——不可答题上
    第一轮经常给出带合法页码的编造，校验轮查完证据决定拒答，才是正确结论，
    却被这条规则原样扔掉。等于把校验轮对不可答题的纠错能力关掉了。

    实测证据（放宽正则臂 vs 基线，同题配对）：3 道题基线 rounds=3 拒答正确，
    放宽臂 rounds=2 反而变成编造，逐条核对就是这条路径。

    【已测，结论：保持现状】n=1007 同题配对（AITIC_ADOPT_ABSTAIN 两臂）：

        收益 编造→拒答正确    8 条
        代价 命中→过度拒答   39 条      1:5 亏本，判据上限 4 条

    顺带量出根因：第二轮拒答被采纳的 47 次里，只有 8 次是拦对的，
    **校验轮拒答的精确率约 17%**——它说"材料里没有"，八成是它自己没找着。
    所以这条规则**逻辑上错、效果上对**：挡掉的误杀远多于放过的编造。

    也就是说，这里的"缺陷"不该在采纳规则上修。真正该问的是
    **校验轮为什么在有据可查的题上也拒答**（检索或提示词层面）。

    另注：曾把 3 条「拒答正确→编造」归因于本函数，**已被交叉检查否掉**——
    采纳臂里本机制关闭，那几道题照样翻；它们是贴着判决边界、对任何扰动
    都敏感的题。机制存在（见单测），但不是那几条的成因。

    开关保留，默认关闭＝线上行为不变。详见 docs 记录 §二十四 / §二十六。
    """
    if not str(answer or "").strip():
        return False
    if not prev_clean:
        return True
    if _ADOPT_ABSTAIN and M.is_abstain(answer):
        # 拒答不是降级，是另一种结论：证据不足时它才是对的那个。
        return True
    return _round_is_clean(answer, cite_check, directness)


def _usable_dists(dists):
    """滤掉 None 距离，供只接受数值的 main.* 判据使用。

    混合检索（BM25 + 向量 + RRF）里，纯关键词命中的块没有向量距离，`dist` 是 None。
    项目里已经因为这件事崩过一次（`_retrieve_selected` 的排序键，见 _rrf_fuse 附近），
    当时只修了那一处；`should_escalate` 是同一个根因的第二个出口。

    全是 None 时返回空列表而不是 [inf]：**空列表让上游走"没有距离信息"的既有分支，
    而 inf 会被当成"距离极差"从而改变判决**——修崩溃不该顺手改行为。

    【2026-08-14 追记】这个修复的实际影响比预期大得多。修复前，混合检索下
    中文库外题 **10/10 全部编造**（cn2h 臂）；修复后真机复测 **10/10 精确拒答、
    逐字契约全中**。当时以为是"停用词过滤"修好的，其实是这里——
    异常被吞掉之后流程绕过了拒答判断。**一个"只修崩溃、不改行为"的补丁，
    实际上恢复了整条拒答契约。** 记在这里，免得日后有人把它当成纯防御性代码删掉。
    """
    return [d for d in (dists or []) if isinstance(d, (int, float))]


def _should_agent_continue(answer, cite_check, docs, dists, mode="auto", round_no=1,
                           directness=None):
    if round_no >= 3 or mode == "fast":
        return False
    if M.is_abstain(answer):
        # 拒答不进校验轮。检索闸门已经判定"证据离题太远"，此时补一轮只是把
        # 一个已经确定的结论重算一遍——实测库外题走这条路 0.5–0.9 秒就返回，
        # 硬塞一轮会变成 20 秒，正是"等待时间太长"要避免的。
        # 但若 should_escalate 认为补检索可能翻案，仍然继续。
        #
        # dists 必须先滤掉 None：混合检索里纯关键词命中的块没有可比距离
        # （见 _rrf_fuse），而 main.should_escalate 里是 `min(dists)`，
        # None 与 float 一比就抛 TypeError，整个请求 500。
        # 实测：开混合检索问库外题，模型一拒答就必崩（中文那轮因为一道都没拒答
        # 才侥幸没触发）。**main.py 受指纹约束不能改，所以在调用侧收口。**
        return M.should_escalate(answer, docs, _usable_dists(dists), M.DYNAMIC_BUDGET)
    if round_no == 1 and mode in ("auto", "deep"):
        # 全做 Agent：只要产出了答案，就必定再走一轮检索 + 校验，不再按意图分流。
        # 原先 auto 档只有"引用没过/答非所问/拒答可升配"才进第二轮，简单题一轮就返回；
        # 现在统一走 Agent Loop。代价是简单题耗时约翻倍，这是明确接受的取舍。
        return True
    # 正文为空/无结论句是 v6 事件的原样症状，重答一次即可救回，且零误判风险。
    if directness and directness.get("retry"):
        return True
    return (not cite_check.get("ok"))


def _confidence_payload(answer, cite_check, sources, claims=None, dists=None,
                        rounds=1, libraries=None, directness=None,
                        support_audit=None):
    """可信度由系统按确定性信号计算，不采纳模型自述的把握度（PRD 明确要求）。

    对外只给高/中/低/证据不足四档，避免伪精确百分比；``signals`` 逐条列出
    实际触发的判据，让用户能看见结论是怎么来的，而不是一句固定文案。
    """
    # claims=None（调用方没算）与 claims=[]（算了但没解析出结论）含义不同：
    # 前者是"未计算"，不能拿来扣分——这是本项目栽过三次的同一个错误
    # （「字面不出现」不等于「信息不存在」）。后者由 directness 的 no_claim 判据兜住。
    claims_given = claims is not None
    claims = claims or []
    sources = sources or []
    rounds = int(rounds or 1)

    if M.is_abstain(answer):
        return {"level": "证据不足", "state": "insufficient",
                "reason": "%d 轮检索后仍未找到可支持结论的材料，已按诚实拒答处理。" % rounds,
                "signals": [{"name": "证据充分度", "ok": False,
                             "detail": "检索 %d 轮后仍无可用依据" % rounds}]}

    measured = [c for c in claims if c.get("measured")]
    cited_claims = [c for c in claims if c["citations"]]
    supported_ratio = (len([c for c in measured if c["supported"]]) / len(measured)
                       if measured else None)
    # "没测出来"有三种完全不同的原因，不能共用一句文案——把「无引用可对」说成
    # 「跨语言算不了」是凭空断言，正是本项目反复强调要避免的那类错误。
    if measured:
        support_detail = "%d/%d 条结论的引用原文可支持其内容" % (
            len([c for c in measured if c["supported"]]), len(measured))
    elif not claims_given:
        support_detail = "未计算"
    elif not claims:
        support_detail = "未解析出结论句"
    elif not cited_claims:
        support_detail = "结论句均未附引用，无原文可核对"
    else:
        support_detail = "跨语言作答，逐句接地率未计算（答案与材料非同一文字）"
    # 自带引用、或虽无标签但已在材料中找到支撑，都算"可核对"。
    covered = [c for c in claims if c["citations"] or c.get("support_via") == "material"]
    coverage = (len(covered) / len(claims)) if claims else None
    # ``sources`` 是模型**看过**的打包块，不等于答案**采用**的证据。把前者全算成
    # "多来源印证"会出现这种真实误导：正文只引 [ch7:2] 一处，因上下文还装了
    # 另外 3 块，界面却声称"4 个独立来源相互印证"。生产路径已经计算 claims，
    # 因此只数实际映射到交付结论的 evidence；旧调用方没提供 claims 时才保留
    # 原来的展示来源近似；对外统一称"证据位置"，只有跨库时才称印证。
    if claims_given:
        used_evidence = [evidence for claim in claims
                         for evidence in (claim.get("evidence") or [])
                         if evidence.get("label")]
    else:
        used_evidence = sources
    evidence_keys = {(x.get("library") or "", x.get("label")) for x in used_evidence
                     if x.get("label")}
    evidence_libraries = {x.get("library") or "" for x in used_evidence if x.get("label")}
    evidence_locations = len(evidence_keys)
    cross_library = len(evidence_libraries) > 1
    usable = [d for d in (dists or []) if isinstance(d, (int, float))]
    best_dist = min(usable) if usable else None
    relevance_ok = best_dist is not None and best_dist <= M.ESCALATE_SIM_GATE

    signals = [
        {"name": "引用命中检索块", "ok": bool(cite_check.get("ok")),
         "detail": ("%d 处引用全部落在实际检索到的材料内" % (cite_check.get("total") or 0))
                   if cite_check.get("ok") else
                   ("存在未命中引用：%s" % "、".join(cite_check.get("fabricated") or []) or "缺少引用")},
        {"name": "引用是否支持结论",
         "ok": supported_ratio is None or supported_ratio >= 0.5,
         "detail": support_detail},
        {"name": "证据覆盖完整度", "ok": coverage is None or coverage >= 0.6,
         "detail": ("%d/%d 条结论可核对（%d 条自带引用）" %
                    (len(covered), len(claims), len(cited_claims))) if claims
                   else ("未解析出结论句" if claims_given else "未计算")},
        {"name": "检索结果相关性", "ok": bool(relevance_ok),
         "detail": ("最优检索距离 %.4f，%s闸门 %.4f" %
                    (best_dist, "优于" if relevance_ok else "劣于", M.ESCALATE_SIM_GATE))
                   if best_dist is not None else "无可用检索距离"},
        # 同一本书的两个页码是两个证据位置，不是两个独立文献来源。
        # 等级判据保持不变，只把展示单位说准确；真正跨库时再明确标出。
        {"name": "已用证据位置", "ok": evidence_locations >= 2,
         "detail": "%d 个不同证据位置%s" %
                   (evidence_locations, "（跨知识库）" if cross_library else "")},
        {"name": "循环后证据状态", "ok": rounds == 1 or bool(cite_check.get("ok")),
         "detail": ("首轮证据即充分" if rounds == 1
                    else "补充检索 %d 轮后%s" % (rounds, "证据已完整" if cite_check.get("ok")
                                                  else "仍有未坐实的引用"))},
    ]

    support_audit = support_audit or {}
    if support_audit.get("triggered"):
        unknown = int(support_audit.get("unknown") or 0)
        pruned = int(support_audit.get("pruned") or 0)
        checked = int(support_audit.get("checked") or 0)
        signals.append({
            "name": "逐句语义支持核验",
            "ok": unknown == 0,
            "detail": ("%d 条可疑结论中裁剪 %d 条，%d 条未能确定" %
                       (checked, pruned, unknown)),
        })
        # UNKNOWN 是基础设施/格式不确定，不是“不支持”：正文继续交付，但可信度必须显式降级。
        if unknown:
            return {"level": "低", "state": "partial", "signals": signals,
                    "reason": "本地逐句语义核验有 %d 条未完成；已保留原句，需人工复核。" % unknown}

    issue_codes = {x["code"] for x in (directness or {}).get("issues", [])}
    weak = {"low_grounding", "uncited", "only_citation", "no_claim"} & issue_codes

    if (not cite_check.get("ok")) or weak:
        reasons = [x["detail"] for x in signals if not x["ok"]]
        if weak:
            reasons.append((directness or {}).get("detail", ""))
        return {"level": "低", "state": "partial", "signals": signals,
                "reason": "；".join([x for x in reasons if x][:3]) or "证据未完全坐实，请复核后使用。"}

    if support_audit.get("triggered") and support_audit.get("pruned"):
        return {"level": "中", "state": "supported", "signals": signals,
                "reason": "已逐句移除不能被原文完整支持的结论；保留内容通过引用与语义校验。"}

    # 「高」必须意味着**真的核对过原文**。跨语言时接地率算不出来，此时即便其余信号全绿
    # 也只能给「中」——否则正好掩盖"引用只是装饰"那类失败（v7 核对出的 42/49 真幻觉即此形态）。
    if supported_ratio is None and claims_given and claims:
        return {"level": "中", "state": "supported", "signals": signals,
                "reason": "引用均命中检索材料，但%s，未能逐句核对原文支持度，故不判为高。"
                          % ("答案与材料非同一文字" if cited_claims else "结论句未附引用")}

    strong = ((coverage is None or coverage >= 0.8) and evidence_locations >= 2
              and (relevance_ok or not usable)
              and (supported_ratio is None or supported_ratio >= 0.8))
    if strong:
        return {"level": "高", "state": "supported", "signals": signals,
                "reason": "引用全部命中、结论均有据可查，且实际采用了 %d 个不同证据位置。"
                          % evidence_locations}
    return {"level": "中", "state": "supported", "signals": signals,
            "reason": "；".join([x["detail"] for x in signals if not x["ok"]][:2])
                      or "有通过校验的直接证据，但证据来源较集中。"}


def _evidence_relations(claims):
    """只报能判定的证据关系，判不出就不报——不猜。"""
    relations = []
    for claim in claims:
        rated = [e for e in claim["evidence"] if e.get("grounding") is not None]
        distinct = {(e.get("library") or "", e.get("label")) for e in claim["evidence"]}
        if len(distinct) >= 2 and claim.get("supported"):
            library_count = len({item[0] for item in distinct if item[0]})
            relations.append({
                "type": "跨库印证" if library_count >= 2 else "同书多处支持",
                "claim": claim["claim"],
                "detail": "%d 个证据位置同时支持该结论：%s" %
                          (len(distinct), "、".join(sorted(x[1] for x in distinct if x[1]))),
            })
        if len(rated) >= 2:
            high, low = max(rated, key=lambda x: x["grounding"]), min(rated, key=lambda x: x["grounding"])
            if high["grounding"] - low["grounding"] >= 0.4:
                relations.append({
                    "type": "证据强弱不一", "claim": claim["claim"],
                    "detail": "%s 支持度 %.0f%%，而 %s 仅 %.0f%%，后者可能被误引" %
                              (high.get("label"), 100 * high["grounding"],
                               low.get("label"), 100 * low["grounding"]),
                })
    return relations


def _uncertainty_items(claims, cite_check, abstained, rounds):
    """PRD 要求输出"不确定项或仍缺少的材料"，逐条列具体的，不写空话。"""
    items = []
    if abstained:
        items.append("检索 %d 轮后仍无可支持结论的材料，需要补充更贴近该问题的资料" % rounds)
        return items
    for claim in claims:
        if claim.get("support_via") == "material":
            continue          # 没自带标签，但已在检索材料中找到支撑，不算不确定项
        if not claim["citations"]:
            items.append(("「%s」未附引用，且与材料非同一文字，无法自动核对"
                          if claim.get("unmeasurable") else
                          "「%s」未附引用，也未能在材料中找到支撑") % _snippet(claim["claim"], 40))
        elif claim.get("measured") and not claim["supported"]:
            items.append("「%s」的引用原文支持度仅 %.0f%%，建议复核" %
                         (_snippet(claim["claim"], 40), 100 * (claim.get("grounding") or 0)))
        elif not claim.get("measured"):
            items.append("「%s」为跨语言作答，未能自动核对引用原文" % _snippet(claim["claim"], 40))
    for bad in (cite_check.get("fabricated") or []):
        items.append("引用 [%s] 不在本次检索到的材料中" % bad)
    return items[:6]


def _diagnose_outcome(answer, cite_check, claims, dists, libraries, scope, extend, hybrid):
    """答案不理想时，判断**是哪一环没成**并给出可执行的下一步。

    非开发用户面对"证据不足"是没有着力点的——他们不知道该换问法、放宽范围、
    还是换本书。这里不新调模型，全部用已经算出来的信号推断：
      · 检索距离都远  → 库里大概真没有 → 换库 / 开教材外补充
      · 距离近但拒答  → 材料沾边但不够 → 换更具体的问法 / 深度分析
      · 限定了页范围  → 首先怀疑范围太窄，这是最常见也最容易自证的一条
      · 引用未通过    → 让用户去看原页自行核对，而不是让他猜

    只在"确实不理想"时返回内容；答得好就返回 None，不打扰。
    """
    abstained = M.is_abstain(answer)
    cite_ok = bool(cite_check.get("ok"))
    measured = [c for c in claims if c.get("measured")]
    weak = bool(measured) and not any(c.get("supported") for c in measured)
    if not abstained and cite_ok and not weak:
        return None                       # 答得好就别啰嗦

    usable = [d for d in (dists or []) if isinstance(d, (int, float))]
    best = min(usable) if usable else None
    far = best is not None and best > M.ESCALATE_SIM_GATE
    actions, cause = [], ""

    if scope:
        cause = "限定了页范围（%s），检索只在这一段里找" % _scope_label(scope)
        actions.append({"do": "clear_scope", "label": "清除页范围重问",
                        "why": "范围内确实没有依据；系统不会偷偷去范围外找"})
    elif abstained and far:
        cause = "检索到的内容与问题都不相近（最优距离 %.3f，劣于闸门 %.3f）" % (best, M.ESCALATE_SIM_GATE)
        if len(libraries or []) <= 1:
            actions.append({"do": "pick_library", "label": "换一本 / 多选几本资料",
                            "why": "这本书里大概率没有这个主题"})
        if not extend:
            actions.append({"do": "enable_extend", "label": "打开「教材外补充」",
                            "why": "会另起一段给出模型常识，并明确标注不出自教材"})
    elif abstained:
        cause = "材料与问题沾边但不足以支撑结论（最优距离 %.3f）" % (best if best is not None else -1)
        actions.append({"do": "rephrase", "label": "换更具体的问法",
                        "why": "用书里的原词提问命中率更高，例如把「怎么用」换成书中的术语"})
        actions.append({"do": "deep_mode", "label": "切换到「深度分析」",
                        "why": "会多跑一轮补充检索再判断"})
        if not hybrid:
            actions.append({"do": "enable_hybrid", "label": "试试混合检索",
                            "why": "精确术语、代码标识符这类词，关键词召回比向量更稳"})
    elif not cite_ok:
        cause = "答案里的引用没有全部落在检索到的材料内"
        actions.append({"do": "open_page", "label": "打开原页自行核对",
                        "why": "系统已标出可疑引用，请以原文为准"})
        actions.append({"do": "regenerate", "label": "重新生成",
                        "why": "换一次生成通常能拿到更规范的引用"})
    else:
        cause = "结论与所引原文的词面重合度偏低，引用可能只是装饰"
        actions.append({"do": "open_page", "label": "打开原页自行核对",
                        "why": "这类情况必须人工确认，系统不替你下结论"})

    return {"cause": cause, "actions": actions,
            "note": "以上判断来自本次检索与校验的实际数据，不是模型猜的。"}


def _agent_payload(answer, cite_check, sources, rounds, mode, history_used,
                   intent=None, libraries=None, claims=None, dists=None,
                   directness=None, support_audit=None, scope=None,
                   extend=False, hybrid=False):
    claims = claims or []
    confidence = _confidence_payload(answer, cite_check, sources, claims, dists,
                                     rounds, libraries, directness, support_audit)
    audit = support_audit or {}
    if M.is_abstain(answer):
        # 拒答有两种来源，此前一律写成"证据不足"：
        #   a) 真的没检索到可支持结论的证据
        #   b) 证据检索到了、答案也生成了，但被输出校验的某道守卫拒掉
        # 二者的排查方向完全相反，混成一句会把 (b) 误导成检索问题。
        refused_by = list(audit.get("refused_by") or [])
        if refused_by:
            stop = "答案已生成，但被输出校验拒绝：%s" % "、".join(refused_by)
        else:
            stop = ("补充检索后证据仍不足，停止生成结论" if rounds > 1
                    else "首轮未找到可支持结论的证据")
    elif audit.get("pruned"):
        # 裁掉过结论就不能再说「证据充分」——截图里 7 条删掉 5 条、0 条原文匹配，
        # 界面却写着「首轮证据充分」，因为这句文案是前端写死的、不看核验结果。
        stop = "逐句核验裁掉 %d 条无据结论后返回" % int(audit["pruned"])
        if audit.get("orphaned"):
            stop += "（另有 %d 条因先行词被裁而失去指代，一并移除）" % int(audit["orphaned"])
    elif audit.get("unknown"):
        stop = "存在 %d 条无法判定的结论，已保留并降低可信度" % int(audit["unknown"])
    elif rounds > 1:
        stop = "补充检索与引用校准完成"
    else:
        stop = "首轮证据充分，快速返回"
    libraries = libraries or []
    trace = [
        {"step": "意图识别", "detail": "%s · %s" % ((intent or {}).get("name", "事实查询"),
                                                      (intent or {}).get("complexity", "简单"))},
        {"step": "证据检索", "detail": "检索 %d 个知识库，形成 %d 个可展示来源" %
                                           (max(1, len(libraries)), len(sources or []))},
        {"step": "引用校验", "detail": ("引用全部命中检索来源" if cite_check.get("ok")
                                         else "存在未命中引用或证据不足")},
        *([{"step": "逐句语义核验", "detail": (support_audit or {}).get("reason", "已完成")}]
          if (support_audit or {}).get("triggered") else []),
        {"step": "答案校验", "detail": (directness or {}).get("detail", "未执行")},
        {"step": "停止判断", "detail": stop},
    ]
    abstained = M.is_abstain(answer)
    # 证据链：结论 → 逐条依据（书/页/片段/接地率）→ 证据间关系 → 可信度 → 不确定项。
    # 与上面的 trace 分开：trace 讲"系统怎么跑的"，evidence_chain 讲"结论凭什么成立"。
    uncertainty = _uncertainty_items(claims, cite_check, abstained, rounds)
    if (support_audit or {}).get("unknown"):
        uncertainty.insert(0, "本地逐句语义核验存在未知项；原句按 fail-open 规则保留，需人工复核")
    evidence_chain = {
        "conclusion": claims[0]["claim"] if claims else ("未给出结论" if abstained else ""),
        "basis": claims,
        "relations": _evidence_relations(claims),
        "confidence": confidence,
        "uncertainty": uncertainty[:6],
        "grounding_measured": any(c.get("measured") for c in claims),
    }
    return {"path": "agent_loop" if rounds > 1 else "fast_rag", "rounds": rounds,
            "requested_mode": mode, "history_used": bool(history_used),
            "stop_reason": stop, "confidence": confidence, "intent": intent or {},
            "libraries": [{"id": x.get("id"), "name": x.get("name")} for x in libraries],
            "cross_library": len(libraries) > 1, "trace": trace,
            "evidence_chain": evidence_chain, "directness": directness or {},
            "diagnosis": _diagnose_outcome(answer, cite_check, claims, dists,
                                           libraries, scope, extend, hybrid),
            "support_audit": support_audit or {}}


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
        # 评测臂必须能从服务自身取回有效配置，不能只靠启动命令或文件名猜。
        "hybrid_default": _hybrid_enabled(),
        "evidence_floor": _EVIDENCE_FLOOR,
        "style_gate_max": _STYLE_GATE_MAX,
        "model_seed": _MODEL_SEED,
        "widen_refusal": _WIDEN_REFUSAL,
        "keyword_df_ratio": str(os.environ.get("AITIC_KW_DF_RATIO", "0")).strip(),
        "db_path": os.path.abspath(M.DB_PATH),
        "library": _library_info(),
        "active_library_id": _read_registry().get("active_id", "legacy"),
        "runtime": _runtime_info(),
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
    """非流式 Agent 问答：简单题快速返回，复杂/低证据题补充检索并校准。"""
    raw_question = payload.get("question")
    if raw_question is not None and not isinstance(raw_question, str):
        return JSONResponse({"error": "问题必须是文本"}, status_code=400)
    question = (raw_question or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    limit_error = _text_limit_error(question, "问题", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    history = _normalize_history(payload.get("history"))
    mode = str(payload.get("mode") or "auto").lower()
    if mode not in ("auto", "fast", "deep"):
        mode = "auto"
    requested_libraries = _normalize_library_ids(payload.get("libraries"))
    style = str(payload.get("style") or "standard").lower()
    preference = _response_preference(style, payload.get("instruction"))

    t0 = time.time()
    # 先解析一次，拿到"请求了但已失效"的库；全部失效会在这里抛 LibraryUnavailable，
    # 由 app 层处理器返回 409，绝不静默改用别的库。
    _targets, dropped_libraries = _resolve_library_targets(requested_libraries)
    # 缺省“当前库”只在请求入口解析一次。否则用户在 Agent 两轮之间切库时，
    # 第二轮会悄悄改查新库，再把两本书的证据合并成一条看似正常的答案。
    resolved_library_ids = [target["id"] for target in _targets]
    hybrid = payload.get("hybrid")
    scope = _page_scope(payload.get("page_scope"))
    retrieval_q = _retrieval_question(question, history)
    docs, metas, dists, libraries = _retrieve_selected(
        retrieval_q, resolved_library_ids, hybrid, scope)
    intent = _detect_intent(question, history, len(libraries))
    rich = _evidence_looks_present(dists)
    answer, toks, packed_idx, packed = _run_agent_once(
        docs, metas, question, history, M.CONTEXT_BUDGET, verification=False,
        preference=preference, style=style, rich=rich)
    # answer/packed_idx 必须与生成它们时的 docs/metas 一起保存。后续补充检索会重排
    # top-3 之外的索引；若二轮答案被降级守卫拒绝采用，不能拿一轮索引去查二轮 metas。
    answer_docs, answer_metas, answer_dists = docs, metas, dists
    cite_check = _verify_citations(answer, packed_idx, metas)
    claims = _claim_evidence_map(answer, packed_idx, metas, packed)
    directness = _answer_directness(question, answer, claims)
    rounds = 1
    # 全做 Agent 之后，原先"复杂意图才提升为 deep"的分流已无意义：
    # auto 自己就必走第二轮，而第 2→3 轮 deep 与 auto 判据完全一致。
    # intent 仍然保留并展示，它是给用户看的判断依据，不再是路由开关。
    prev_clean = _round_is_clean(answer, cite_check, directness)
    if _should_agent_continue(answer, cite_check, docs, dists, mode, rounds, directness):
        second_docs, second_metas, second_dists, _ = _retrieve_selected(
            _followup_query(retrieval_q, 2), resolved_library_ids, hybrid, scope)
        second = (second_docs, second_metas, second_dists)
        docs, metas, dists = _merge_retrieval((docs, metas, dists), second)
        ans2, toks2, idx2, packed2 = _run_agent_once(
            docs, metas, question, history, M.BUDGET_ESCALATED, verification=True,
            preference=preference, style=style, rich=rich)
        toks += toks2; rounds = 2
        cc2 = _verify_citations(ans2, idx2, metas)
        cl2 = _claim_evidence_map(ans2, idx2, metas, packed2)
        dir2 = _answer_directness(question, ans2, cl2)
        # 第一轮不干净时照旧采用第二轮（本来就是为补救才跑的）；
        # 第一轮已经干净时，第二轮必须同样干净才替换——校验轮不该成为降级通道。
        if _adopt_next_round(prev_clean, ans2, cc2, dir2):
            answer, packed_idx, packed, cite_check = ans2, idx2, packed2, cc2
            claims, directness = cl2, dir2
            answer_docs, answer_metas, answer_dists = docs, metas, dists
        prev_clean = _round_is_clean(answer, cite_check, directness)
    if _should_agent_continue(answer, cite_check, docs, dists, mode, rounds, directness):
        third_docs, third_metas, third_dists, _ = _retrieve_selected(
            _followup_query(retrieval_q, 3), resolved_library_ids, hybrid, scope)
        docs, metas, dists = _merge_retrieval((docs, metas, dists),
                                              (third_docs, third_metas, third_dists))
        ans3, toks3, idx3, packed3 = _run_agent_once(
            docs, metas, question, history, M.BUDGET_ESCALATED, verification=True,
            preference=preference, style=style, rich=rich)
        toks += toks3; rounds = 3
        cc3 = _verify_citations(ans3, idx3, metas)
        cl3 = _claim_evidence_map(ans3, idx3, metas, packed3)
        dir3 = _answer_directness(question, ans3, cl3)
        if _adopt_next_round(prev_clean, ans3, cc3, dir3):
            answer, packed_idx, packed, cite_check = ans3, idx3, packed3, cc3
            claims, directness = cl3, dir3
            answer_docs, answer_metas, answer_dists = docs, metas, dists

    answer = _clean_answer_echo(question, answer)
    # 答案保留哪一轮，就必须用同一轮的距离做证据下限。后续未被采用的检索结果
    # 不能替旧答案“洗白”，也不能把旧答案错误压成拒答。
    if _evidence_floor_blocks(answer_dists) and not M.is_abstain(answer):
        answer = _NO_REFERENCE
    answer, cite_check, claims, support_audit, guard_tokens = _finalize_agent_answer(
        answer, packed_idx, answer_metas, packed)
    toks += guard_tokens
    answer, cite_check, claims, support_audit, directness = _enforce_final_directness(
        question, answer, packed_idx, answer_metas, packed,
        cite_check, claims, support_audit)
    sources = _sources_from(answer_metas, packed_idx, answer_docs)

    # 只解析一次。原先这里用白名单判真假、下面给 _diagnose_outcome 却用 bool()，
    # 同一个字段两套口径：客户端发字符串 "0" 时，上面判定为关、下面 bool("0") 为真，
    # 诊断会以为「教材外补充」已经开着，于是不再给出「打开它」的补救建议。
    want_extend = _coerce_bool(payload.get("extend"), False)

    # 第二部分在**所有溯源判定都算完之后**才生成，确保它不可能影响任何指标。
    supplement = None
    if want_extend:
        supplement = _supplement_answer(question, answer, M.is_abstain(answer))
        if supplement:
            toks += supplement["tokens"]

    return {
        "answer": answer,
        "abstained": M.is_abstain(answer),
        "sources": sources,
        "tokens": toks,
        "escalated": rounds > 1,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED if rounds > 1 else M.CONTEXT_BUDGET,
        "cite_check": cite_check,
        "retrieval": ("scoped" if scope else ("hybrid" if _hybrid_enabled(hybrid) else "vector")),
        "page_scope": (dict(scope, label=_scope_label(scope)) if scope else None),
        "dropped_libraries": dropped_libraries,
        "supplement": supplement,
        "agent": _agent_payload(answer, cite_check, sources, rounds, mode, bool(history),
                                intent, libraries, claims, answer_dists, directness, support_audit,
                                scope, want_extend, _hybrid_enabled(hybrid)),
    }


@app.post("/api/brief")
def api_brief(payload: dict):
    """非流式简报生成：围绕 topic 综合检索资料，产出带出处的结构化 brief 文档。
       复用 main.brief 的生成逻辑（_run_once_brief），返回结构与 /api/ask 一致，前端可复用渲染。"""
    raw_topic = payload.get("topic") or payload.get("question")
    if raw_topic is not None and not isinstance(raw_topic, str):
        return JSONResponse({"error": "主题必须是文本"}, status_code=400)
    topic = (raw_topic or "").strip()
    if not topic:
        return JSONResponse({"error": "主题为空"}, status_code=400)
    limit_error = _text_limit_error(topic, "主题", _TOPIC_MAX_CHARS)
    if limit_error is not None:
        return limit_error

    t0 = time.time()
    docs, metas, dists, libraries = _retrieve_selected(
        topic, _normalize_library_ids(payload.get("libraries")))
    packed, packed_idx = _pack_agent(docs, metas, topic, M.BUDGET_ESCALATED)
    context = _labeled_context(packed, packed_idx, metas)
    answer, toks = _web_gen_brief_raw(M.BRIEF_PROMPT.format(context=context, topic=topic))
    answer = _clean_answer_echo(topic, answer)
    answer, cite_check, claims, support_audit, guard_tokens = _finalize_agent_answer(
        answer, packed_idx, metas, packed)
    toks += guard_tokens
    if not M.is_abstain(answer) and _brief_low_evidence_blocks(dists, claims):
        answer = _NO_REFERENCE
        cite_check = _verify_citations(answer, packed_idx, metas)
        claims = []
        support_audit = dict(support_audit, triggered=True, state="refused",
                             reason="低相关检索仅形成不足两条受支持引用结论")
    if not M.is_abstain(answer):
        # 逐句安全裁剪会移除不支持内容；这里仅恢复 brief 的可读结构，不改正文语义。
        answer = re.sub(r"\s*(【(?:概要|要点|依据)】)\s*", r"\n\n\1\n", answer).strip()
        answer = re.sub(r"\s+-\s+", "\n- ", answer)
    sources = _sources_from(metas, packed_idx, docs)
    directness = _answer_directness(topic, answer, claims)
    intent = _detect_intent(topic, [], len(libraries))

    return {
        "answer": answer,
        "abstained": M.is_abstain(answer),
        "sources": sources,
        "tokens": toks,
        "escalated": False,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "budget": M.BUDGET_ESCALATED,
        "cite_check": cite_check,
        "mode": "brief",
        "libraries": [{"id": x["id"], "name": x["name"]} for x in libraries],
        "agent": _agent_payload(answer, cite_check, sources, 1, "brief", False,
                                intent, libraries, claims, dists, directness, support_audit,
                                None, False, False),
    }


def _json_array(text):
    cleaned = M._strip_think(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("模型未返回 JSON 数组")
    candidate = cleaned[start:end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        # 小模型偶尔会输出 `""question"` 或尾随逗号；只修复这两类
        # 明确、无语义歧义的格式错误，不猜测答案内容。
        repaired = candidate.translate(str.maketrans({"“": '"', "”": '"'}))
        repaired = re.sub(r'(^|[,{]\s*)""([A-Za-z_][^"\r\n]*)"\s*:',
                          r'\1"\2":', repaired, flags=re.M)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            # 修复也救不回来时，逐个对象地抢救——整份 JSON 里往往只有一两条坏掉，
            # 没理由让其余好题跟着陪葬。全都解析不出才算失败。
            value = []
            for chunk in re.findall(r"\{[^{}]*\}", repaired, flags=re.S):
                try:
                    item = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    value.append(item)
            if not value:
                # 抛 ValueError 而非让 JSONDecodeError 冒到 FastAPI —— 后者会变成 500，
                # 而这本质上是"模型这次没配合"，应由调用方给出可重试的提示。
                raise ValueError("模型返回的不是可解析的 JSON 数组")
    if not isinstance(value, list):
        raise ValueError("测试题结果不是数组")
    return value


# ----------------------------- 不可答探针 -----------------------------
# 世界知识型探针：语料里必然没有，用于兜底（当只选了一个知识库、无法跨库取术语时）。
# 沿用 4432 题评测集里 `world` 档的构造思路。
_WORLD_PROBES = [
    "the 2024 Nobel Prize in Physics", "the capital city of Australia",
    "the closing share price of Tesla yesterday", "the winner of the 2022 FIFA World Cup",
    "the current population of Reykjavik",
]
# 语义闸门：探针词若能在目标库里检索到很近的块，说明概念其实存在（只是用词不同），
# 这种题问出来会被正确作答，却被判成"该拒答"——即 v7 核对出的那 10% 探针构造缺陷。
_PROBE_SEMANTIC_GATE = 0.85


def _term_appears_literally(db_path, term):
    """字面校验：该词是否在目标库的任何块里出现过。"""
    try:
        col = M.chromadb.PersistentClient(path=os.path.abspath(db_path)).get_collection(M.COLLECTION)
        got = col.get(where_document={"$contains": term}, limit=1, include=[])
        return bool((got or {}).get("ids"))
    except Exception:
        return True          # 查不了就当"可能存在"，宁可弃用该探针也不出错题


def _probe_is_clean(target, term):
    """字面 + 语义双重校验。

    本项目**在同一个错误上栽过三次**（不可答探针 `verified 0 occurrences`、
    图内题 GT、图题生成器 L4 去重）：只做字面比对会把"书里用别的词讲过同一概念"
    误判成"库里没有"。所以这里必须再过一道语义闸门。
    """
    if _term_appears_literally(target["path"], term):
        return False
    try:
        col = M.chromadb.PersistentClient(
            path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
        _docs, _metas, dists = M._retrieve(col, M.embed([term])[0], term)
    except Exception:
        return False
    best = min([d for d in (dists or []) if isinstance(d, (int, float))], default=None)
    return best is None or best > _PROBE_SEMANTIC_GATE


def _harvest_probe_terms(targets, limit=6):
    """从**其他**知识库的正文里取真实术语当探针——真概念、且必属他域。

    必须过"正文反查"这道闸门：PDF 抽取常产生 ``pythonhow`` 这类粘连碎片，
    它们确实不在目标书里，但根本不是概念，问出来是废题。评测集建设第五轮
    正是栽在这上面（把代码片段、非名词性片段当成术语）。判据沿用当时那条：
    **真概念会在多个块里反复出现**，只出现一两次的一律弃用。
    """
    if len(targets) < 2:
        return []
    out = []
    for source in targets:
        try:
            col = M.chromadb.PersistentClient(
                path=os.path.abspath(source["path"])).get_collection(M.COLLECTION)
            got = col.get(limit=60, include=["documents"])
        except Exception:
            continue
        documents = [str(x).lower() for x in ((got or {}).get("documents") or [])]
        blocks_with, total = {}, {}
        for doc in documents:
            words = re.findall(r"\b[a-z]{6,14}\b", doc)
            for word in set(words):
                blocks_with[word] = blocks_with.get(word, 0) + 1
            for word in words:
                total[word] = total.get(word, 0) + 1
        # 跨 ≥3 个块出现且总频次 ≥4 才算真概念，按块覆盖面排序优先取最典型的
        good = [w for w, n in blocks_with.items()
                if n >= 3 and total.get(w, 0) >= 4 and w not in _STOP_EN]
        good.sort(key=lambda w: (-blocks_with[w], w))
        for word in good[:limit]:
            for target in targets:
                if target["id"] != source["id"]:
                    out.append((target, word))
    return out[:limit * 4]


def _build_probe_questions(targets, want=1):
    """生成经双重校验的不可答题，附预期结果（应拒答）。"""
    probes, used = [], set()
    for target, term in _harvest_probe_terms(targets):
        if len(probes) >= want:
            break
        if term in used or not _probe_is_clean(target, term):
            continue
        used.add(term)
        probes.append({
            "question": "Define %s." % term, "expected_answer": "[NO REFERENCE FOUND]",
            "source": "", "expected_citation": "", "difficulty": "advanced",
            "type": "简单RAG", "answerable": False,
            "task_tags": ["抗幻觉探针", "跨学科术语"],
            "probe_basis": "「%s」在《%s》中字面零出现，且语义检索最近块距离 > %.2f"
                           % (term, target["name"], _PROBE_SEMANTIC_GATE),
        })
    for term in _WORLD_PROBES:
        if len(probes) >= want:
            break
        target = targets[0]
        if not _probe_is_clean(target, term.split()[-1]):
            continue
        probes.append({
            "question": "What is %s?" % term, "expected_answer": "[NO REFERENCE FOUND]",
            "source": "", "expected_citation": "", "difficulty": "basic",
            "type": "简单RAG", "answerable": False, "task_tags": ["抗幻觉探针", "世界知识"],
            "probe_basis": "世界知识类问题，不在任何已建知识库的语料范围内",
        })
    return probes


@app.post("/api/questions")
def api_questions(payload: dict):
    """从所选知识库生成演示问题与模型草拟参考答案；后者不是人工 GT。"""
    raw_topic = payload.get("topic")
    if raw_topic is not None and not isinstance(raw_topic, str):
        return JSONResponse({"error": "主题必须是文本"}, status_code=400)
    topic = (raw_topic or "").strip()
    limit_error = _text_limit_error(topic, "主题", _TOPIC_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    count = _bounded_int(payload.get("count"), 3, 2, 5)
    requested_libraries = _normalize_library_ids(payload.get("libraries"))
    retrieval_q = topic or "核心概念 关键机制 定义 图表 重要结论"
    t0 = time.time()
    docs, metas, _dists, libraries = _retrieve_selected(retrieval_q, requested_libraries)
    packed, packed_idx = _pack_agent(docs, metas, retrieval_q, M.BUDGET_ESCALATED)
    context = _labeled_context(packed, packed_idx, metas)
    language = "Chinese" if (not topic or re.search(r"[\u4e00-\u9fff]", topic)) else "the topic language"
    prompt = """Material:
%s

Task: create exactly %d useful test questions grounded only in Material%s.
Return ONLY a JSON array. Each item must contain:
{"question":"...","expected_answer":"...","source":"exact source tag from Material","difficulty":"basic|advanced"}
Rules:
- Use %s.
- expected_answer must directly answer the question in no more than two sentences.
- source must be an exact supplied tag; never invent a page or source.
- Include at least one advanced reasoning question when the material supports it.
- Do not use outside knowledge and do not add Markdown.
""" % (context, count, (" about: " + topic) if topic else "", language)
    out = M._generate(M.LLM_MODEL, prompt,
                      options=_llm_options(temperature=0.1,
                                           num_predict=min(900, M.NUM_PREDICT)))
    items = []
    valid_sources = {_web_cite_tag(metas[i]).lower().replace(" ", "")
                     for i in packed_idx if i < len(metas)}
    try:
        parsed = _json_array(out.get("response", ""))
    except ValueError as e:
        # 本地小模型偶尔不吐严格 JSON。这是"这次没配合"，不是服务故障，
        # 给可重试的 422 而不是 500——现场演示时按钮偶发红叉最难看。
        return JSONResponse({"error": "模型这次未按格式返回题目（%s），请重试或换个主题" % e},
                            status_code=422)
    for item in parsed[:count]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        expected = str(item.get("expected_answer") or "").strip()
        source = str(item.get("source") or "").strip().strip("[]")
        normalized_source = re.sub(r"(^|:)p(\d+)", r"\1p.\2", source.lower().replace(" ", ""))
        if question and expected and source and normalized_source in valid_sources:
            # 类型与任务标签由现成的 _detect_intent 推导，不额外调模型（PRD 要求别加等待）
            probe_intent = _detect_intent(question, [], len(libraries))
            tags = []
            if probe_intent["name"] == "图表解析":
                tags.append("图表题")
            if len(libraries) > 1:
                tags.append("跨书题")
            if probe_intent["name"] not in ("事实查询", "图表解析"):
                tags.append(probe_intent["name"])
            items.append({
                "question": question, "expected_answer": expected,
                "source": source, "expected_citation": "[%s]" % source.strip("[]"),
                "difficulty": str(item.get("difficulty") or "basic"),
                "type": "复杂Agent" if probe_intent["route"] == "agent_loop" else "简单RAG",
                "answerable": True, "task_tags": tags or ["事实查询"],
                "grading_basis": "auto_reference", "requires_review": True,
            })
    if len(items) < 2:
        return JSONResponse({"error": "未能生成足够的可核对题目，请换一个主题重试"}, status_code=422)
    # 只有可答题构不成回归集——必须配不可答探针，才能同时测出"该答的答了"和"该拒的拒了"。
    try:
        items += _build_probe_questions(libraries, want=1)
    except Exception:
        pass                       # 探针是加分项，构造失败不该让整个接口挂掉
    return {"questions": items, "libraries": [{"id": x["id"], "name": x["name"]} for x in libraries],
            "answerable_count": len([x for x in items if x.get("answerable")]),
            "probe_count": len([x for x in items if not x.get("answerable")]),
            "elapsed_ms": int((time.time() - t0) * 1000)}


# ----------------------------- 失败反馈闭环 -----------------------------
# PRD：「pipeline 自闭环系统，要循环起来」——用户标记的失败样本必须能自动进回归集，
# 否则反馈只是存在浏览器里的一条本地记录，改完提示词也无从验证是否真的修好了。
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "data", "webui_feedback")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
_FEEDBACK_KINDS = {"useful": "有用", "needs-improvement": "待改进", "no-answer": "没回答问题",
                   "bad-citation": "引用不正确", "insufficient": "证据不足", "slow": "回答太慢"}
_feedback_lock = threading.RLock()


def _read_feedback():
    # 追加与读取必须共享一把锁。否则列表/导出可能刚好读到写了一半的 JSONL 行，
    # 把一条真实反馈当成“损坏行”静默漏掉。RLock 允许 api_feedback 在追加锁内
    # 继续调用本 helper 统计总数。
    with _feedback_lock:
        rows = []
        if not os.path.isfile(FEEDBACK_PATH):
            return rows
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue      # 单行损坏不该让整份反馈读不出来
        return rows


def _feedback_identity(row):
    """反馈样本身份是“问题 + 知识库集合”，不能只用问题文本。

    完整题集中已有大量跨书复用问题；同一句问题在两本书上的失败是两个不同的
    回归样本。库顺序不影响检索语义，因此去重、排序后组成稳定身份。
    """
    question = str((row or {}).get("question") or "").strip()
    libraries = tuple(sorted(set(_normalize_library_ids((row or {}).get("libraries")))))
    return question, libraries


def _latest_feedback_rows(rows):
    """按复合身份保留最新记录，同时维持最新记录在文件中的时间顺序。"""
    seen, latest_reversed = set(), []
    for row in reversed(list(rows or [])):
        identity = _feedback_identity(row)
        if not identity[0] or identity in seen:
            continue
        seen.add(identity)
        latest_reversed.append(row)
    return list(reversed(latest_reversed))


@app.post("/api/feedback")
def api_feedback(payload: dict):
    """记录一条用户反馈。失败样本随后可由 /api/feedback/regression 导出成回归集。"""
    raw_kind, raw_question = payload.get("kind"), payload.get("question")
    raw_answer = payload.get("answer")
    if raw_kind is not None and not isinstance(raw_kind, str):
        return JSONResponse({"error": "反馈类型必须是文本"}, status_code=400)
    if raw_question is not None and not isinstance(raw_question, str):
        return JSONResponse({"error": "问题必须是文本"}, status_code=400)
    if raw_answer is not None and not isinstance(raw_answer, str):
        return JSONResponse({"error": "回答必须是文本"}, status_code=400)
    kind = (raw_kind or "").strip()
    question = (raw_question or "").strip()
    if kind not in _FEEDBACK_KINDS:
        return JSONResponse({"error": "未知的反馈类型：%s" % kind}, status_code=400)
    if not question:
        return JSONResponse({"error": "缺少问题内容"}, status_code=400)
    limit_error = _text_limit_error(question, "问题", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    answer = raw_answer or ""
    limit_error = _text_limit_error(answer, "回答", _FEEDBACK_ANSWER_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind, "kind_label": _FEEDBACK_KINDS[kind],
        # 回归闭环必须保存用户实际问过/看到的完整文本。静默截断会让重跑换了一道题，
        # 或让同一答案也永远显示 answer_changed=True。
        "question": question,
        "answer": answer,
        "libraries": _normalize_library_ids(payload.get("libraries")),
        "sources": [str(x)[:120] for x in (
            payload.get("sources") if isinstance(payload.get("sources"), list) else [])][:12],
        "confidence": str(payload.get("confidence") or "")[:20],
        "abstained": _coerce_bool(payload.get("abstained"), False),
        "rounds": _bounded_int(payload.get("rounds"), 1, 1, 100),
        # 只有非"有用"的反馈才是回归集素材
        "is_failure": kind != "useful",
    }
    with _feedback_lock:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        total = len(_read_feedback())
    return {"ok": True, "recorded": record["kind_label"], "total": total}


@app.get("/api/feedback")
def api_feedback_list(limit: int = 50):
    rows = _read_feedback()
    failures = [x for x in rows if x.get("is_failure")]
    by_kind = {}
    for row in rows:
        by_kind[row.get("kind_label") or "?"] = by_kind.get(row.get("kind_label") or "?", 0) + 1
    return {"total": len(rows), "failures": len(failures), "by_kind": by_kind,
            "recent": rows[-max(1, min(200, limit)):][::-1]}


REGRESSION_RUNS_PATH = os.path.join(FEEDBACK_DIR, "regression_runs.jsonl")
_RERUN_MAX = 25


def _rerun_transition(before_abstained, after_abstained):
    """两次作答/拒答之间的四种迁移。命名要能直接读懂，别用 TT/TF 这种缩写。"""
    if before_abstained and after_abstained:
        return "仍然拒答"
    if before_abstained and not after_abstained:
        return "由拒答改为作答"
    if not before_abstained and after_abstained:
        return "由作答改为拒答"
    return "仍然作答"


@app.post("/api/feedback/rerun")
def api_feedback_rerun(payload: dict = None):
    """把标记过的失败样本原样重跑一遍，和当初记录的结果逐条对照。

    这是「pipeline 自闭环」缺的那一环：原先只能把失败样本导出成 jsonl，
    再手工去命令行跑 run_eval_batch.py，改完根本不知道有没有修好。

    **返回的不是准确率。** 这些样本的标准答案从未经人工订正——它们的 keywords
    是从"那条被判为错"的答案里抽的，拿来当 GT 只会把错误固化。所以这里只报
    **行为变化对照**：同一个问题、同一批知识库，这次的拒答/作答、可信度、
    答案文本相对当初有没有变。改没改对仍然要人看，但"改没改"由系统告诉你。
    """
    payload = payload or {}
    raw_limit = payload.get("limit")
    if raw_limit is None or raw_limit == "":
        limit = 10
    elif isinstance(raw_limit, bool):
        return JSONResponse({"error": "limit 必须是整数"}, status_code=400)
    elif isinstance(raw_limit, int):
        limit = raw_limit
    elif isinstance(raw_limit, str) and re.fullmatch(r"[+-]?\d+", raw_limit.strip()):
        limit = int(raw_limit.strip())
    else:
        # 这是会真实调用模型并追加 regression_runs.jsonl 的持久化操作。
        # 畸形 limit 不能像展示型参数那样静默回退到默认 10，否则一个明显错误的
        # 请求反而会触发一批昂贵重跑；21:26 的契约模糊测试已真实复现。
        return JSONResponse({"error": "limit 必须是整数"}, status_code=400)
    limit = max(1, min(_RERUN_MAX, limit))

    samples = _latest_feedback_rows(
        [row for row in _read_feedback() if row.get("is_failure")])
    samples = samples[-limit:]
    if not samples:
        return JSONResponse({"error": "还没有标记过的失败样本，无从重跑。"}, status_code=400)

    items, tally = [], {"仍然拒答": 0, "由拒答改为作答": 0,
                        "由作答改为拒答": 0, "仍然作答": 0, "跳过": 0}
    for row in samples:
        # 用现成的解析器判可用性，别自己维护第二套"库还在不在"的逻辑。
        # 库全没了会抛 LibraryUnavailable —— 那正是该跳过的情形，绝不改用别的库重跑，
        # 换了库的结果和当初根本不可比。
        try:
            targets, _dropped = _resolve_library_targets(row.get("libraries") or [])
            libraries = [t["id"] for t in targets]
        except LibraryUnavailable:
            libraries = []
        if not libraries:
            tally["跳过"] += 1
            items.append({"question": row.get("question"), "libraries": libraries,
                          "skipped": True,
                          "reason": "当初用的知识库已删除，无法在同一条件下复现"})
            continue
        try:
            fresh = api_ask({"question": row["question"], "libraries": libraries,
                             "mode": "auto", "style": "standard",
                             "extend": False, "history": []})
        except Exception as exc:
            tally["跳过"] += 1
            items.append({"question": row.get("question"), "libraries": libraries,
                          "skipped": True,
                          "reason": "重跑失败：%s" % type(exc).__name__})
            continue
        if isinstance(fresh, JSONResponse):
            tally["跳过"] += 1
            items.append({"question": row.get("question"), "libraries": libraries,
                          "skipped": True,
                          "reason": "重跑返回 HTTP %d" % fresh.status_code})
            continue

        after_abstained = bool(fresh.get("abstained"))
        transition = _rerun_transition(bool(row.get("abstained")), after_abstained)
        tally[transition] += 1
        agent = fresh.get("agent") or {}
        items.append({
            "question": row.get("question"),
            "libraries": libraries,
            "kind": row.get("kind_label"),
            "reported_at": row.get("time"),
            "transition": transition,
            "answer_changed": (str(fresh.get("answer") or "").strip()
                               != str(row.get("answer") or "").strip()),
            "confidence_before": row.get("confidence") or "",
            "confidence_after": (agent.get("confidence") or {}).get("level") or "",
            "cite_ok": bool((fresh.get("cite_check") or {}).get("ok")),
            "stop_reason": agent.get("stop_reason") or "",
            "answer_preview": str(fresh.get("answer") or "")[:200],
        })

    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": len(samples),
           "tally": tally}
    previous = None
    with _feedback_lock:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        if os.path.exists(REGRESSION_RUNS_PATH):
            with open(REGRESSION_RUNS_PATH, encoding="utf-8") as f:
                rows = []
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        # 进程/机器在追加中途退出可能留下半行；历史记录损坏不该让
                        # 新一轮真实回归结果也无法写入。与 feedback.jsonl 读取口径一致。
                        continue
            previous = rows[-1] if rows else None
        with open(REGRESSION_RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    return {
        "run": run, "previous": previous, "items": items,
        "note": "这是行为变化对照，不是准确率：这些样本的标准答案尚未人工订正，"
                "拿失败答案里抽出的关键词当 GT 会把错误固化。"
                "系统只回答『改没改』，『改没改对』仍需人工判断。",
    }


@app.get("/api/feedback/regression")
def api_feedback_regression():
    """把失败样本导出成与 eval_*.jsonl 同构的回归集，可直接喂 run_eval_batch.py。

    字段对齐现有题集：``book / question / keywords / type / expect``。
    ``keywords`` 取答案里的内容词——**只作为占位起点，需人工订正**，
    因为被标记为失败的那条答案本身可能就是错的，拿它当标准答案会把错误固化下来。
    """
    rows = _latest_feedback_rows(
        [x for x in _read_feedback() if x.get("is_failure")])
    out = []
    for row in rows:
        identity = _feedback_identity(row)
        question, libraries = identity
        expect_answer = not row.get("abstained")
        words = sorted(_latin_words(_strip_tags(row.get("answer"))))[:3]
        out.append({
            "book": ", ".join(libraries) or "webui",
            "subject": "user_feedback", "question": question,
            "keywords": words, "type": "answerable" if expect_answer else "unanswerable",
            "expect": "answer" if expect_answer else "abstain",
            "source": "user_feedback", "feedback_kind": row.get("kind_label"),
            "reported_at": row.get("time"),
            "needs_review": True,
        })
    body = "\n".join(json.dumps(x, ensure_ascii=False) for x in out)
    return JSONResponse({
        "count": len(out), "jsonl": body,
        "note": "keywords 由失败答案自动提取，仅为起点；纳入正式回归集前必须人工订正标准答案。",
    })


# ----------------------------- 知识库健康度诊断 -----------------------------
# 对标 AnythingLLM 把分块参数摆到台面上的做法：用户在问答之前，应该能判断
# "这个库值不值得信、哪些内容可能压根检索不到"。全部只读统计，不改库。
@app.get("/api/library/health")
def api_library_health(library_id: str, sample: int = 800):
    try:
        targets = _library_targets([library_id])
    except LibraryUnavailable as exc:
        return JSONResponse({"error": "知识库不可用", "unavailable": exc.requested},
                            status_code=409)
    target = next((x for x in targets if str(x.get("id")) == str(library_id)), None)
    if not target:
        return JSONResponse({"error": "知识库不存在或尚未就绪"}, status_code=404)
    try:
        col = M.chromadb.PersistentClient(
            path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
        total = col.count()
        # 必须**全库均匀抽样**，不能用 limit 取前 N 块。
        # 前 N 块就是书的前几页——目录、版权页、序言天然很短，直接拿它们统计
        # 会把"过短块占比"系统性放大（实测前 200 块报 79.5%，全库实为个位数），
        # 变成一个看着有理有据、实际误导人的数字。
        want = max(1, min(int(sample), 2000))
        all_ids = (col.get(include=[]) or {}).get("ids") or []
        if all_ids and len(all_ids) > want:
            step = len(all_ids) / float(want)
            picked = [all_ids[min(len(all_ids) - 1, int(i * step))] for i in range(want)]
            got = col.get(ids=picked, include=["documents", "metadatas"])
            # Page coverage is cheap metadata, and must be exact.  Computing it
            # from the sampled blocks can invent a "missing pages" warning.
            all_metas = (col.get(include=["metadatas"]) or {}).get("metadatas") or []
        else:
            got = col.get(include=["documents", "metadatas"])
            all_metas = got.get("metadatas") or []
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:240]}, status_code=500)

    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    lengths = sorted(len(str(x or "")) for x in docs)
    pages = [m.get("page") for m in all_metas if isinstance((m or {}).get("page"), int)]
    kinds = {}
    for m in metas:
        k = (m or {}).get("type") or "text"
        kinds[k] = kinds.get(k, 0) + 1

    def pct(n):
        return round(n / len(lengths), 3) if lengths else 0.0

    # 判据来自本项目自己的分块参数，不另立标准：
    #   过短块（< CHUNK_TARGET 的 1/4）语义不完整，检索到了也难支撑作答；
    #   过长块（≥ CHUNK_MAX）说明切分没找到边界，容易把两个概念糊在一起。
    tiny = sum(1 for x in lengths if x < M.CHUNK_TARGET // 4)
    huge = sum(1 for x in lengths if x >= M.CHUNK_MAX)
    warnings = []
    if pct(tiny) > 0.15:
        warnings.append("过短块占 %.0f%%：这些块语义不完整，检索命中也难支撑作答" % (100 * pct(tiny)))
    if pct(huge) > 0.20:
        warnings.append("过长块占 %.0f%%：切分未找到边界，可能把多个概念糊在一起" % (100 * pct(huge)))
    if total and total < 50:
        warnings.append("全库仅 %d 块，覆盖面很窄，容易出现「库里没有」式的拒答" % total)
    if pages and (max(pages) - min(pages) + 1) > 0:
        covered = len(set(pages))
        span = max(pages) - min(pages) + 1
        if covered < span * 0.6:
            warnings.append("页覆盖不连续：%d 页中只有 %d 页有内容，中间可能有解析失败的页"
                            % (span, covered))
    return {
        "library": {"id": target["id"], "name": target["name"]},
        "chunks_total": total, "sampled": len(docs),
        "length": {"min": lengths[0] if lengths else 0,
                   "median": lengths[len(lengths) // 2] if lengths else 0,
                   "max": lengths[-1] if lengths else 0,
                   "target": M.CHUNK_TARGET, "cap": M.CHUNK_MAX, "scope": "sample"},
        "pages": {"min": min(pages) if pages else None, "max": max(pages) if pages else None,
                  "covered": len(set(pages)), "scope": "full"},
        "kinds": kinds,
        "tiny_ratio": pct(tiny), "huge_ratio": pct(huge),
        "warnings": warnings,
        "note": ("块长与类型基于全库均匀抽样 %d 块（共 %d 块）；页覆盖由全库 metadata 精确统计。"
                 "诊断只读，不改动向量库。" % (len(docs), total)),
    }


# ----------------------------- 批量跑题：界面内的评测闭环 -----------------------------
# PRD 要求"pipeline 自闭环系统，要循环起来"。此前已能生成测试题、也能把失败反馈
# 导成回归集，但**跑不了**——闭环缺最后一环。这里把它补上：批量执行、自动判分、出报告。
#
# 判分口径刻意与全量评测保持一致，不另造一套：
#   可答题  → 答案里出现任一 GT 关键词即命中（等价 run_eval_batch 的严格口径）
#   不可答题 → 是否精确拒答（is_abstain）
# 没有 GT 关键词时只记录"是否作答 + 引用是否合法"，并明确标注该题未参与准确率计算，
# 绝不用"看起来像对的"去凑一个好看的分数。
_BATCH_MAX = 30


def _grade_one(item, result):
    """给一道题判分。返回 (verdict, reason)；verdict ∈ hit/miss/refused/over_refused/n_a"""
    answer = str(result.get("answer") or "")
    abstained = bool(result.get("abstained"))
    answerable = _coerce_bool(item.get("answerable"), True)
    raw_keywords = item.get("keywords")
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]
    elif not isinstance(raw_keywords, list):
        raw_keywords = []
    keywords = [str(k).strip().lower() for k in raw_keywords if str(k).strip()]
    if not answerable:
        return ("refused", "正确拒答") if abstained else ("miss", "该拒答却作答了")
    if abstained:
        return (("over_refused", "库里应有依据却拒答") if keywords else
                ("n_a", "自动生成题没有人工 GT，拒答行为仅作辅助观察"))
    if not keywords:
        return "n_a", "无人工订正的 GT 关键词，不计入正式准确率"
    body = answer.lower()
    hit = [k for k in keywords if k in body]
    return ("hit", "命中关键词：%s" % "、".join(hit)) if hit else \
           ("miss", "未出现任一关键词：%s" % "、".join(keywords))


def _weak_reference_check(item, result):
    """自动参考答案的词面覆盖，仅供排查，不得混入正式准确率。"""
    if not _coerce_bool(item.get("answerable"), True):
        return None
    expected = str(item.get("expected_answer") or "").strip()
    if not expected or M.is_abstain(expected):
        return None
    if bool(result.get("abstained")):
        return {"state": "refused", "coverage": 0.0, "matched": 0, "total": 0,
                "note": "自动参考答案题发生拒答；该项不是正式准确率"}
    terms = sorted(_latin_words(expected) | _cjk_bigrams(expected),
                   key=lambda x: (-len(x), x))[:12]
    if not terms:
        return None
    body = str(result.get("answer") or "").lower()
    matched = [term for term in terms if term in body]
    coverage = len(matched) / len(terms)
    return {"state": "high_overlap" if coverage >= 0.6 and len(matched) >= 2 else "low_overlap",
            "coverage": round(coverage, 3), "matched": len(matched), "total": len(terms),
            "note": "自动参考答案词面覆盖，仅作辅助自检，不代表语义正确"}


# ----------------------------- A/B 对比 -----------------------------
# 中期答辩评委要求「A/B test，最好和最坏都要展示」。此前只有离线消融数据，
# 界面里看不到。这里让同一个问题当场跑两套配置并排对照。
_COMPARE_KEYS = ("libraries", "style", "mode", "extend", "instruction", "hybrid", "page_scope")


def _compare_metrics(result):
    """从一次问答结果里抽出可对照的客观量，全部来自既有字段，不另算一套。"""
    agent = result.get("agent") or {}
    chain = agent.get("evidence_chain") or {}
    claims = chain.get("basis") or []
    rated = [c.get("grounding") for c in claims if isinstance(c.get("grounding"), (int, float))]
    cite = result.get("cite_check") or {}
    return {
        "abstained": bool(result.get("abstained")),
        "cite_ok": bool(cite.get("ok")),
        "cite_total": int(cite.get("total") or 0),
        "fabricated": list(cite.get("fabricated") or []),
        "confidence": (agent.get("confidence") or {}).get("level"),
        "claims": len(claims),
        "claims_supported": len([c for c in claims if c.get("supported")]),
        "grounding_avg": round(sum(rated) / len(rated), 3) if rated else None,
        "rounds": agent.get("rounds"),
        "tokens": result.get("tokens"),
        "elapsed_ms": result.get("elapsed_ms"),
        "answer_chars": len(str(result.get("answer") or "")),
        "sources": sorted({str(x.get("label")) for x in (result.get("sources") or [])
                           if x.get("label")}),
    }


def _compare_diff(a, b):
    """只报能客观判定的差异；判不出好坏就明说判不出，不硬凑结论。"""
    diffs = []
    if a["abstained"] != b["abstained"]:
        diffs.append("一侧拒答、另一侧作答——这是两者最实质的差别")
    if a["cite_ok"] != b["cite_ok"]:
        diffs.append("引用校验结果不同（A=%s / B=%s）" % (a["cite_ok"], b["cite_ok"]))
    if a["confidence"] != b["confidence"]:
        diffs.append("可信度 A=%s / B=%s" % (a["confidence"], b["confidence"]))
    if set(a["sources"]) != set(b["sources"]):
        diffs.append("引用到的来源不同：A 独有 %s，B 独有 %s"
                     % (sorted(set(a["sources"]) - set(b["sources"])) or "无",
                        sorted(set(b["sources"]) - set(a["sources"])) or "无"))
    for key, label in (("tokens", "token"), ("elapsed_ms", "耗时(ms)"), ("answer_chars", "字数")):
        va, vb = a.get(key) or 0, b.get(key) or 0
        if va and vb and abs(va - vb) / max(va, vb) >= 0.15:
            diffs.append("%s 相差 %.0f%%（A=%s / B=%s）"
                         % (label, 100 * abs(va - vb) / max(va, vb), va, vb))
    return diffs or ["两侧在可客观判定的维度上没有实质差异"]


# ----------------------------- 跨教材概念对照 -----------------------------
# 与 /api/compare 的区别：那个对照的是**配置**（同一本书、不同开关），
# 这个对照的是**内容**（同一概念、不同教材怎么讲）。55 本 6 学科的语料下，
# "这个概念在这几本书里分别怎么说"是真实的学习需求。
#
# 关键设计：每本书**独立检索、独立作答**，绝不把多本书混进一个上下文——
# 混在一起模型会给出一段糊在一块的话，恰恰看不出"谁说了什么、哪里不一样"。
#
# 差异摘要用**结构化计算**而不是再让模型总结一遍：二次综述是在已生成的答案上
# 再做一层生成，无法对着原文校验，等于凭空引入一层新的幻觉风险。
_CONCEPT_MAX_BOOKS = 4


def _concept_terms(text):
    """答案里的实词集合，用于算共识/分歧。中英文各取一套。"""
    body = _strip_tags(text)
    return _latin_words(body) | _cjk_bigrams(body)


@app.post("/api/concept")
def api_concept(payload: dict):
    """同一概念在多本教材中的讲法对照。"""
    raw_concept = payload.get("concept") or payload.get("question")
    if raw_concept is not None and not isinstance(raw_concept, str):
        return JSONResponse({"error": "概念必须是文本"}, status_code=400)
    concept = (raw_concept or "").strip()
    if not concept:
        return JSONResponse({"error": "请填写要对照的概念"}, status_code=400)
    limit_error = _text_limit_error(concept, "概念", _TOPIC_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    # 必须先看**原始**入参长度：_normalize_library_ids 会静默截断到 4 个，
    # 等归一化之后再判上限就永远触发不到，用户多选的那本会被悄悄丢掉还没有提示。
    raw_ids = payload.get("libraries")
    raw_count = len([x for x in raw_ids if str(x or "").strip()]) if isinstance(raw_ids, list) else 0
    if raw_count > _CONCEPT_MAX_BOOKS:
        return JSONResponse({"error": "单次最多对照 %d 本，你选了 %d 本，请减少后重试"
                                      % (_CONCEPT_MAX_BOOKS, raw_count)}, status_code=400)
    requested = _normalize_library_ids(raw_ids)
    if len(requested) < 2:
        return JSONResponse({"error": "跨教材对照至少需要选择两个知识库"}, status_code=400)
    try:
        targets, dropped = _resolve_library_targets(requested)
    except LibraryUnavailable as exc:
        return JSONResponse({"error": "所选知识库已不可用，请重新选择。",
                             "unavailable": exc.requested}, status_code=409)

    style = str(payload.get("style") or "standard").lower()
    question_template = payload.get("question_template")
    if question_template is not None and not isinstance(question_template, str):
        return JSONResponse({"error": "问题模板必须是文本"}, status_code=400)
    question = question_template or ("什么是%s？" % concept
                                    if re.search(r"[一-鿿]", concept)
                                    else "What is %s?" % concept)
    limit_error = _text_limit_error(question.strip(), "问题模板", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    t0 = time.time()
    name_counts = {}
    for target in targets:
        name = str(target.get("name") or target.get("id") or "知识库")
        name_counts[name] = name_counts.get(name, 0) + 1

    def display_name(target):
        name = str(target.get("name") or target.get("id") or "知识库")
        return (name if name_counts.get(name, 0) == 1
                else "%s · %s" % (name, target.get("id")))

    books = []
    for target in targets:
        result = api_ask({"question": question, "mode": "fast", "history": [],
                          "libraries": [target["id"]], "style": style, "extend": False})
        if isinstance(result, JSONResponse):
            books.append({"library": display_name(target), "name": target["name"],
                          "id": target["id"],
                          "covered": False, "answer": "", "sources": [],
                          "note": "该知识库检索失败"})
            continue
        abstained = bool(result.get("abstained"))
        books.append({
            "library": display_name(target), "name": target["name"], "id": target["id"],
            "covered": not abstained,
            "answer": result.get("answer") or "",
            "sources": [x.get("label") for x in (result.get("sources") or []) if x.get("label")],
            "cite_ok": (result.get("cite_check") or {}).get("ok"),
            "confidence": ((result.get("agent") or {}).get("confidence") or {}).get("level"),
            "note": "" if not abstained else "这本书里没有找到该概念的依据",
        })

    covered = [b for b in books if b["covered"]]
    term_sets = {b["id"]: _concept_terms(b["answer"]) for b in covered}
    labels = {b["id"]: b["library"] for b in covered}
    shared, unique = [], {}
    if len(term_sets) >= 2:
        sets = list(term_sets.values())
        common = set.intersection(*sets)
        shared = sorted(w for w in common if len(w) >= 4 or re.search(r"[一-鿿]", w))[:12]
        for library_id, own in term_sets.items():
            others = set.union(*[s for k, s in term_sets.items() if k != library_id]) if len(term_sets) > 1 else set()
            only = sorted(w for w in (own - others)
                          if len(w) >= 5 or re.search(r"[一-鿿]", w))[:8]
            if only:
                unique[labels[library_id]] = only

    return {
        "concept": concept,
        "question_used": question,
        "books": books,
        "coverage": {"covered": len(covered), "total": len(books),
                     "missing": [b["library"] for b in books if not b["covered"]]},
        "shared_terms": shared,
        "unique_terms": unique,
        "dropped_libraries": dropped,
        "elapsed_ms": int((time.time() - t0) * 1000),
        # 诚实边界：共识/分歧是**词面重合**算出来的，不是语义比对。
        "note": ("每本书独立检索、独立作答，各带各的引用；下方共识与分歧由答案实词重合度"
                 "计算得出，属**词面**层面，不代表语义等价或真的矛盾——请以各书原文为准。"),
    }


@app.post("/api/compare")
def api_compare(payload: dict):
    """同一问题跑两套配置，并排对照。

    两臂**直接复用 api_ask**，一行问答逻辑都不复制：这样检索、打包、生成、
    引用校验、语义核验走的是逐字节相同的链路，差异只可能来自显式指定的开关。
    控制变量由结构保证，而不是靠"我说它们相同"——这是本项目做消融的一贯要求。
    """
    raw_question = payload.get("question")
    if raw_question is not None and not isinstance(raw_question, str):
        return JSONResponse({"error": "问题必须是文本"}, status_code=400)
    question = (raw_question or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    limit_error = _text_limit_error(question, "问题", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        return JSONResponse({"error": "需要正好两套配置进行对照"}, status_code=400)

    base = {k: payload.get(k) for k in _COMPARE_KEYS if k in payload}
    # 缺省“当前库”只在整个对照入口解析一次；两臂之间切库不能让 A/B 查不同的书。
    frozen_targets, _ = _resolve_library_targets(
        _normalize_library_ids(payload.get("libraries")))
    frozen_library_ids = [target["id"] for target in frozen_targets]
    t0 = time.time()
    arms = []
    for position, variant in enumerate(variants):
        if not isinstance(variant, dict):
            return JSONResponse({"error": "配置格式不正确"}, status_code=400)
        cfg = dict(base)
        cfg.update({k: v for k, v in variant.items() if k in _COMPARE_KEYS})
        if not _normalize_library_ids(cfg.get("libraries")):
            cfg["libraries"] = frozen_library_ids
        cfg.update({"question": question, "history": []})   # 对照必须无历史，避免上一轮污染
        result = api_ask(cfg)
        if isinstance(result, JSONResponse):
            return result                                   # 库不可用等错误按原状态码透出
        arms.append({
            "label": str(variant.get("label") or ("A" if position == 0 else "B"))[:40],
            "config": {k: cfg.get(k) for k in _COMPARE_KEYS},
            "answer": result.get("answer"),
            "sources": result.get("sources"),
            "cite_check": result.get("cite_check"),
            "agent": result.get("agent"),
            "supplement": result.get("supplement"),
            "metrics": _compare_metrics(result),
        })

    return {
        "question": question,
        "arms": arms,
        "differences": _compare_diff(arms[0]["metrics"], arms[1]["metrics"]),
        "elapsed_ms": int((time.time() - t0) * 1000),
        # 诚实边界：项目在不同时段/负载/重启条件下测到的翻转率差一个量级，
        # 没有可跨条件复用的固定“噪声底”。单题一次对照只能展示行为差别。
        "note": ("单题一次对照只能展示行为差别，不构成统计结论。模型与运行条件会让结果翻转；"
                 "要下结论，请用同一批题做配对批量实验，并同时跑同条件空白对照。"),
    }


@app.post("/api/batch")
def api_batch(payload: dict):
    """批量跑题；人工 GT 与自动参考答案必须分开报告。"""
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return JSONResponse({"error": "没有可执行的题目"}, status_code=400)
    if len(items) > _BATCH_MAX:
        return JSONResponse({"error": "单次最多 %d 题，请分批" % _BATCH_MAX}, status_code=400)
    for index, item in enumerate(items, 1):
        if (not isinstance(item, dict) or not isinstance(item.get("question"), str)
                or not item.get("question", "").strip()):
            return JSONResponse({"error": "第 %d 题缺少有效的文本问题" % index},
                                status_code=400)
    valid_items = items
    oversized = next((x["question"].strip() for x in valid_items
                      if len(x["question"].strip()) > _QUERY_MAX_CHARS), None)
    if oversized is not None:
        return _text_limit_error(oversized, "批量题目中的问题", _QUERY_MAX_CHARS)
    libraries = _normalize_library_ids(payload.get("libraries"))
    style = str(payload.get("style") or "standard").lower()
    try:
        targets, _ = _resolve_library_targets(libraries)  # 库不可用要在开跑前就失败，别跑一半才报错
        resolved_library_ids = [target["id"] for target in targets]
    except LibraryUnavailable as exc:
        return JSONResponse({"error": "所选知识库已不可用，请重新选择。",
                             "unavailable": exc.requested}, status_code=409)

    t0 = time.time()
    rows, tally = [], {"hit": 0, "miss": 0, "refused": 0, "over_refused": 0, "n_a": 0}
    for raw in valid_items:
        item = dict(raw)
        item["answerable"] = _coerce_bool(raw.get("answerable"), True)
        question = item["question"].strip()
        try:
            result = api_ask({"question": question, "mode": "fast", "history": [],
                              "libraries": resolved_library_ids, "style": style, "extend": False})
        except Exception as exc:
            rows.append({"question": question, "verdict": "error",
                         "reason": str(exc)[:120], "answer": "", "cite_ok": None})
            continue
        if isinstance(result, JSONResponse):      # 例如库不可用
            rows.append({"question": question, "verdict": "error",
                         "reason": "接口返回 %d" % result.status_code, "answer": "", "cite_ok": None})
            continue
        verdict, reason = _grade_one(item, result)
        tally[verdict] = tally.get(verdict, 0) + 1
        cite = result.get("cite_check") or {}
        rows.append({"question": question, "verdict": verdict, "reason": reason,
                     "answer": (result.get("answer") or "")[:400],
                     "answerable": item["answerable"],
                     "abstained": bool(result.get("abstained")),
                     "cite_ok": cite.get("ok"), "cite_total": int(cite.get("total") or 0),
                     "weak_check": _weak_reference_check(item, result),
                     "confidence": ((result.get("agent") or {}).get("confidence") or {}).get("level")})

    # verdict="miss" 同时被可答题未命中和不可答探针误答使用；不能把后者塞进
    # 人工 GT 命中率分母，否则一次编造会同时扣两个指标。
    answerable_miss = sum(1 for r in rows
                          if r.get("answerable", True) is not False and r["verdict"] == "miss")
    probe_answered = sum(1 for r in rows
                         if r.get("answerable") is False and r["verdict"] == "miss")
    graded_answerable = tally["hit"] + answerable_miss + tally["over_refused"]
    probes = tally["refused"] + probe_answered
    weak_rows = [r for r in rows if r.get("weak_check")]
    weak_high = sum(1 for r in weak_rows if r["weak_check"].get("state") == "high_overlap")
    citation_rows = [r for r in rows if r.get("verdict") != "error" and not r.get("abstained")]
    return {
        "rows": rows,
        "summary": {
            "total": len(rows),
            "answerable_graded": graded_answerable,
            "hit": tally["hit"], "miss": answerable_miss,
            "over_refused": tally["over_refused"],
            "probe_total": probes, "probe_refused": tally["refused"],
            "probe_answered": probe_answered,
            "not_graded": tally["n_a"],
            "hit_rate": round(tally["hit"] / graded_answerable, 3) if graded_answerable else None,
            "refuse_rate": round(tally["refused"] / probes, 3) if probes else None,
            "weak_total": len(weak_rows), "weak_high": weak_high,
            "weak_overlap_rate": round(weak_high / len(weak_rows), 3) if weak_rows else None,
            "citation_checked": len(citation_rows),
            "cite_ok_rate": (round(sum(1 for r in citation_rows if r.get("cite_ok")) /
                                   len(citation_rows), 3) if citation_rows else None),
        },
        "elapsed_ms": int((time.time() - t0) * 1000),
        "note": "正式命中率只统计人工订正的 GT 关键词题；不可答探针按是否精确拒答。"
                "自动生成的参考答案只给词面覆盖辅助值，不计入正式准确率，也不代表语义正确。",
    }


# ----------------------------- 引用溯源：跳到原文 -----------------------------
# 对标 RAGFlow 的招牌能力（每条引用可点开、定位到源文档并高亮命中段落）。
# 我们此前只给一小段片段，用户看不到该页的完整上下文，也就无法真正核对
# "这句话是不是被原文支持"——这是"可溯源"的最后一公里。
# 只读向量库，不改任何检索/生成逻辑，因此不影响评测口径。
# ----------------------------- 原页渲染 + 命中高亮 -----------------------------
# 引用锚定此前只能给到"页码文字"，用户无从当场核对。这里把原书那一页直接渲染出来、
# 命中处高亮，让"可溯源"从一句承诺变成肉眼可验证的东西。
# 用现有 PyMuPDF（建库本来就依赖它），不引新依赖。
_PAGE_DPI = 110
_PAGE_MAX_DPI = 200
_pdf_path_cache = {}


def _header_safe(value):
    """把任意字符串变成能进 HTTP 头的形式（头只允许 Latin-1）。

    本项目书名普遍含破折号 U+2014 与中文，直接放进响应头会抛 UnicodeEncodeError，
    表现为整个接口 500——而真正的业务逻辑其实早就跑完了。非 ASCII 一律百分号转义。
    """
    from urllib.parse import quote
    return quote(str(value or ""), safe=" ._-()[]")


def _find_source_pdf(target, source=""):
    """按知识库确定性定位源 PDF；有歧义时宁可返回 ``None``。

    新建库在注册表中保存内部 ``source_path``，所以同名 PDF 也能一一绑定。
    老库没有该字段时，只接受唯一同名文件，或多份字节完全相同的副本；返回遍历到的
    第一个文件会让引用文字来自 A 库、展开原页却显示 B 书，证据链比 404 更糟。
    """
    source_ref = str(target.get("source_path") or "").strip()
    wanted = str(source or target.get("source") or "").strip()
    key = (str(target.get("id")), wanted, source_ref)
    cached = _pdf_path_cache.get(key)
    if cached:
        if os.path.isfile(cached):
            return cached
        _pdf_path_cache.pop(key, None)
    if source_ref:
        exact = source_ref if os.path.isabs(source_ref) else os.path.join(PROJECT_ROOT, source_ref)
        exact = os.path.abspath(exact)
        if os.path.isfile(exact) and (not wanted or os.path.basename(exact) == wanted):
            _pdf_path_cache[key] = exact
            return exact
        # An explicit binding that no longer exists must not silently fall back
        # to another file with the same name.
        return None

    candidates = []
    if wanted:
        for root in (os.path.join(KB_ROOT, "uploads"),
                     os.path.join(PROJECT_ROOT, "data"),
                     os.path.join(PROJECT_ROOT, "books")):
            if not os.path.isdir(root):
                continue
            for base, _dirs, files in os.walk(root):
                if wanted in files:
                    candidate = os.path.abspath(os.path.join(base, wanted))
                    if candidate not in candidates:
                        candidates.append(candidate)
    if len(candidates) == 1:
        _pdf_path_cache[key] = candidates[0]
        return candidates[0]
    if len(candidates) > 1:
        # Old registry entries predate source_path.  Multiple copies with the
        # same basename are safe only when their bytes are identical (for
        # example the original book plus the upload copy).  Different bytes
        # remain ambiguous and must fail closed instead of showing the wrong
        # book as evidence.
        try:
            sizes = {os.path.getsize(candidate) for candidate in candidates}
            if len(sizes) != 1:
                return None
            digests = set()
            for candidate in candidates:
                digest = hashlib.sha256()
                with open(candidate, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digests.add(digest.hexdigest())
            if len(digests) == 1:
                chosen = sorted(candidates, key=lambda p: os.path.normcase(p))[0]
                _pdf_path_cache[key] = chosen
                return chosen
        except OSError:
            pass
    return None


def _authorized_source(target, requested=""):
    """Return a source owned by *target*, or ``None`` when foreign/ambiguous."""
    allowed = [str(x).strip() for x in (target.get("allowed_sources") or [])
               if str(x).strip()]
    wanted = str(requested or "").strip()
    if wanted:
        return wanted if wanted in allowed else None
    return allowed[0] if len(allowed) == 1 else None


@app.get("/api/source/page")
def api_source_page(library_id: str, page: int, source: str = "",
                    highlight: str = "", dpi: int = _PAGE_DPI):
    """渲染原书某一页为 PNG，并把命中片段涂黄。

    ``page`` 用与引用标签一致的 1 起页号；找不到源 PDF 时明确 404，
    绝不返回一张"看起来像那一页"的替代图——那会让核对失去意义。
    """
    try:
        targets = _library_targets([library_id])
    except LibraryUnavailable as exc:
        return JSONResponse({"error": "知识库不可用", "unavailable": exc.requested},
                            status_code=409)
    target = next((x for x in targets if str(x.get("id")) == str(library_id)), None)
    if not target:
        return JSONResponse({"error": "知识库不存在或尚未就绪"}, status_code=404)
    selected_source = _authorized_source(target, source)
    if not selected_source:
        return JSONResponse({"error": "该来源不属于所选知识库，或多书库未明确指定来源"},
                            status_code=404)
    pdf_path = _find_source_pdf(target, selected_source)
    if not pdf_path:
        return JSONResponse({"error": "找不到该知识库的源 PDF，无法渲染原页。"
                                      "（EPUB/音频等非 PDF 来源不支持原页渲染）"}, status_code=404)

    try:
        doc = M.fitz.open(pdf_path)
    except Exception as exc:
        return JSONResponse({"error": "打开源 PDF 失败：%s" % str(exc)[:160]}, status_code=500)
    try:
        index = int(page) - 1                       # 引用标签是 1 起页号
        if index < 0 or index >= len(doc):
            return JSONResponse({"error": "页码超出范围（该书共 %d 页）" % len(doc)},
                                status_code=404)
        pg = doc[index]
        hits = 0
        probe = re.sub(r"\s+", " ", str(highlight or "")[:_HIGHLIGHT_MAX_CHARS]).strip()
        if probe:
            # 整段命中率低（分块会跨行重排），按短语逐个找更稳；只取前几个短语避免涂满整页。
            phrases = [x for x in re.split(r"[。．.!?！？；;\n]", probe) if len(x.strip()) >= 6][:6]
            for phrase in (phrases or [probe[:60]]):
                try:
                    for rect in (pg.search_for(phrase.strip()) or [])[:12]:
                        pg.add_highlight_annot(rect)
                        hits += 1
                except Exception:
                    continue                        # 单个短语找不到不影响整页渲染
        safe_dpi = max(60, min(int(dpi or _PAGE_DPI), _PAGE_MAX_DPI))
        png = pg.get_pixmap(dpi=safe_dpi, annots=True).tobytes("png")
    except Exception as exc:
        return JSONResponse({"error": "渲染失败：%s" % str(exc)[:160]}, status_code=500)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    from fastapi.responses import Response
    return Response(content=png, media_type="image/png", headers={
        "Cache-Control": "no-store",
        "X-Highlight-Hits": str(hits),               # 0 表示这一页没找到该片段，前端要如实提示
        # HTTP 头只能是 Latin-1，而本项目书名普遍含破折号（U+2014）和中文，
        # 直接塞进去会抛 UnicodeEncodeError 变成 500。凡进响应头的用户字符串一律先转义。
        "X-Source-File": _header_safe(os.path.basename(pdf_path)),
    })


@app.get("/api/source")
def api_source(library_id: str, page: int = 0, loc: str = "", source: str = "",
               highlight: str = ""):
    """按库 + 页（或 EPUB 的 loc）取回该位置的全部块原文。"""
    try:
        targets = _library_targets([library_id])
    except LibraryUnavailable as exc:
        return JSONResponse({"error": "知识库不可用", "unavailable": exc.requested},
                            status_code=409)
    target = next((x for x in targets if str(x.get("id")) == str(library_id)), None)
    if not target:
        return JSONResponse({"error": "知识库不存在或尚未就绪"}, status_code=404)

    selected_source = _authorized_source(target, source)
    if not selected_source:
        return JSONResponse({"error": "该来源不属于所选知识库，或多书库未明确指定来源"},
                            status_code=404)

    position = {"loc": str(loc)} if loc else {"page": int(page)}
    where = {"$and": [position, {"source": selected_source}]}
    try:
        col = M.chromadb.PersistentClient(
            path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
        got = col.get(where=where, include=["documents", "metadatas"], limit=60)
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:240]}, status_code=500)

    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []
    if not documents:
        return JSONResponse({"error": "该位置没有已入库的内容（可能未建库或页码超范围）"},
                            status_code=404)

    # highlight 是答案实际引用的那个块的开头片段：用它标出"命中的是这一块"，
    # 而不是让用户在整页里自己找。做前缀比对而非全文相等，避免因裁剪对不上。
    probe = re.sub(r"\s+", "", str(highlight or "")[:_HIGHLIGHT_MAX_CHARS])[:60]
    blocks = []
    for document, metadata in zip(documents, metadatas):
        tagged = dict(metadata or {}, _library_id=target["id"],
                      _library_name=target["name"], _library_source=target["source"])
        body = str(document or "")
        blocks.append({
            "text": body,
            "label": _src_of(tagged)["label"],
            "type": (metadata or {}).get("type") or "text",
            "matched": bool(probe and probe in re.sub(r"\s+", "", body)),
        })
    blocks.sort(key=lambda x: (not x["matched"],))     # 命中的块排最前
    return {"library": {"id": target["id"], "name": target["name"]},
            "page": None if loc else int(page), "loc": loc or None,
            "blocks": blocks, "total": len(blocks),
            "matched": sum(1 for x in blocks if x["matched"])}


@app.post("/api/retrieve")
def api_retrieve_only(payload: dict):
    """只返回检索证据，不调用生成模型；用于调试召回与快速查原文。"""
    raw_question = payload.get("question")
    if raw_question is not None and not isinstance(raw_question, str):
        return JSONResponse({"error": "检索词必须是文本"}, status_code=400)
    question = (raw_question or "").strip()
    if not question:
        return JSONResponse({"error": "检索词为空"}, status_code=400)
    limit_error = _text_limit_error(question, "检索词", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    limit = _bounded_int(payload.get("limit"), M.TOP_K, 1, 12)
    requested = _normalize_library_ids(payload.get("libraries"))
    hybrid = payload.get("hybrid")
    scope = _page_scope(payload.get("page_scope"))
    t0 = time.time()
    # “仅检索”是排查正式问答为什么命中/没命中的窗口，必须复用用户当前的检索设置。
    # 旧实现静默丢掉 hybrid/page_scope，界面明明亮着开关，展示的却是另一条纯向量链。
    docs, metas, dists, libraries = _retrieve_selected(
        question, requested, hybrid, scope)
    indices = list(range(min(limit, len(docs))))
    sources = _sources_from(metas, indices, docs)
    by_label = {}
    for index in indices:
        if index >= len(metas):
            continue
        label = _src_of(metas[index])["label"]
        distance = dists[index] if index < len(dists) else None
        by_label.setdefault(label, distance)
    for source in sources:
        distance = by_label.get(source["label"])
        source["distance"] = round(float(distance), 4) if distance is not None else None
    return {"question": question, "sources": sources,
            "intent": _detect_intent(question, [], len(libraries)),
            "libraries": [{"id": x["id"], "name": x["name"]} for x in libraries],
            "retrieval": ("scoped" if scope
                          else ("hybrid" if _hybrid_enabled(hybrid) else "vector")),
            "page_scope": (dict(scope, label=_scope_label(scope)) if scope else None),
            "elapsed_ms": int((time.time() - t0) * 1000), "llm_called": False}


@app.get("/api/libraries/{library_id}/chunks")
def api_library_chunks(library_id: str, q: str = "", limit: int = 12, offset: int = 0):
    """浏览或搜索分块内容；只读，不修改向量库。"""
    limit = min(30, max(1, int(limit)))
    offset = max(0, int(offset))
    limit_error = _text_limit_error(q.strip(), "分块搜索词", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    targets = _library_targets([library_id])
    target = next((x for x in targets if str(x.get("id")) == str(library_id)), None)
    if not target:
        return JSONResponse({"error": "知识库不存在或尚未就绪"}, status_code=404)
    try:
        col = M.chromadb.PersistentClient(path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
        total = col.count()
        if q.strip():
            result = col.query(query_embeddings=[M.embed([q.strip()])[0]],
                               n_results=min(limit, total),
                               include=["documents", "metadatas", "distances"])
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
        else:
            result = col.get(limit=limit, offset=offset, include=["documents", "metadatas"])
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            distances = [None] * len(documents)
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:240]}, status_code=500)
    chunks = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        tagged = dict(metadata or {}, _library_id=target["id"],
                      _library_name=target["name"], _library_source=target["source"])
        source = _src_of(tagged)
        distance = distances[index] if index < len(distances) else None
        chunks.append({"index": offset + index + 1, "label": source["label"],
                       "type": source["type"], "text": str(document or "")[:1800],
                       "distance": round(float(distance), 4) if distance is not None else None})
    return {"library": {"id": target["id"], "name": target["name"]},
            "query": q.strip(), "total": total, "offset": offset, "chunks": chunks}


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


@app.get("/api/ask/stream")
async def api_ask_stream(q: str, h: str = "", mode: str = "auto", libs: str = "",
                         style: str = "standard", instruction: str = "", extend: str = "0",
                         hybrid: str = "0", page_scope: str = ""):
    """流式 Agent 问答。首次证据充分则快速返回，否则补充检索并校准一次。"""
    question = (q or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    limit_error = _text_limit_error(question, "问题", _QUERY_MAX_CHARS)
    if limit_error is not None:
        return limit_error
    history = _normalize_history(h)
    mode = (mode or "auto").lower()
    if mode not in ("auto", "fast", "deep"):
        mode = "auto"
    requested_libraries = _normalize_library_ids(libs)
    preference = _response_preference(style, instruction)
    stream_hybrid = _coerce_bool(hybrid, False)
    stream_extend = _coerce_bool(extend, False)
    try:
        stream_scope = _page_scope(json.loads(page_scope) if page_scope else None)
    except (ValueError, TypeError):
        stream_scope = None

    async def gen():
        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            _targets, dropped_libraries = _resolve_library_targets(requested_libraries)
            resolved_library_ids = [target["id"] for target in _targets]
            retrieval_q = _retrieval_question(question, history)
            docs, metas, dists, libraries = await loop.run_in_executor(
                None, _retrieve_selected, retrieval_q, resolved_library_ids,
                stream_hybrid, stream_scope)
        except LibraryUnavailable as e:
            # 绝不退回别的库作答：宁可这一轮失败，也不能让用户以为在查 A 其实查的是 B。
            yield _sse("error", {"msg": "所选知识库已不可用（%s）。请在左侧重新选择，"
                                        "系统不会改用其他知识库作答。"
                                        % "、".join(e.requested)})
            return
        except Exception as e:
            msg = str(e)[:200]
            if "10061" in msg or "refused" in msg.lower() or "urlopen" in msg.lower():
                msg = ("连不上 Ollama（%s）。请确认：① Ollama 已启动（ollama list 能列出模型）；"
                       "② 若设过 OLLAMA_HOST 环境变量，需与 Ollama 实际监听地址一致。"
                       % M._ollama_host())
            yield _sse("error", {"msg": msg})
            return
        intent = _detect_intent(question, history, len(libraries))
        # 全做 Agent：auto 自身就必走校验轮，不再按意图提升为 deep。
        # intent 仍然下发给前端展示，只是不再当路由开关用。
        effective_mode = mode
        yield _sse("agent", {"round": 1, "phase": "retrieve",
                              "label": "首次检索", "history_used": bool(history),
                              "intent": intent, "libraries": len(libraries)})
        yield _sse("retrieved", {"n": len(docs), "round": 1})

        async def run(budget, tag, verification=False):
            """按预算打包 → 流式生成。返回 (完整答案, packed_idx, tokens)。"""
            packed, packed_idx = _pack_agent(docs, metas, question, budget)
            context = _labeled_context(packed, packed_idx, metas)   # 多库时标签含 K1/K2，避免同页码冲突
            # 证据看着不足时退回严格模式，否则"像老师讲课"的文风会把模型推去编（实测 3/3→0/3）
            prompt = _agent_prompt(context, question, packed_idx, metas, history, verification,
                                   preference, _evidence_looks_present(dists))

            yield ("meta", {"budget": budget, "tag": tag,
                            "sources": _sources_from(metas, packed_idx, docs)})

            q_out = asyncio.Queue()
            cancelled = threading.Event()

            def enqueue(kind, value):
                """线程回调只在请求仍存活、事件循环仍开放时投递。"""
                if cancelled.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(q_out.put_nowait, (kind, value))
                except RuntimeError:
                    # 浏览器断开后事件循环可能已经关闭；后台线程不能再制造未捕获异常。
                    cancelled.set()

            def produce():
                """在线程里跑 Ollama 流式，把分片塞进队列。

                ⚠️ 必须走 /api/generate（补全模式），与 main.py 的 _generate 一致。
                   走 /api/chat（对话模式）会套用聊天模板，模型天性"乐于助人"，
                   会脱离 Material 用自身知识作答、还硬安一个页码 —— 直接破坏
                   Citation Grounding 的防幻觉保证。这是实测踩过的坑，勿改。
                """
                stream = None
                try:
                    if cancelled.is_set():
                        return
                    import ollama
                    stream = ollama.generate(
                        model=M.LLM_MODEL,
                        prompt=prompt,                    # prompt 以 "Answer:" 结尾 → 补全模式
                        think=False,
                        stream=True,
                        options=_llm_options(
                            temperature=M.TEMPERATURE,
                            # 交互式问答用 Web 端自己的篇幅，不动评测口径的 M.NUM_PREDICT
                            num_predict=_web_num_predict(style)),
                    )
                    for part in stream:
                        if cancelled.is_set():
                            break
                        piece = part.get("response", "")
                        if piece:
                            enqueue("d", piece)
                        if part.get("done"):
                            tk = (part.get("prompt_eval_count", 0) or 0) + \
                                 (part.get("eval_count", 0) or 0)
                            enqueue("t", tk)
                except Exception as e:
                    enqueue("e", str(e)[:200])
                finally:
                    # ollama 0.6.x 的 stream 是生成器；close() 会退出其 httpx
                    # streaming context。浏览器点击停止时若不关闭它，服务端请求虽然
                    # 已断开，后台线程仍会把整篇答案生成完，连续停止会堆积 GPU 工作。
                    if cancelled.is_set() and stream is not None:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                pass
                    enqueue(None, None)

            loop.run_in_executor(None, produce)

            buf, toks = [], 0
            try:
                while True:
                    kind, val = await q_out.get()
                    if kind is None:
                        break
                    if kind == "d":
                        buf.append(val)
                        # Raw model text stays server-side.  It may still contain
                        # unsupported claims or invented page labels; only the text
                        # replayed after _finalize_agent_answer may cross the API.
                    elif kind == "t":
                        toks = val
                    elif kind == "e":
                        yield ("error", {"msg": val})
                        break
            finally:
                # Starlette 在客户端断开时取消 body iterator；把取消信号传给生产线程。
                cancelled.set()
            full = M._strip_think("".join(buf)).strip()
            yield ("__done__", (full, packed_idx, toks, packed))

        # ---- 第一档 ----
        answer = ""
        packed_idx, toks, packed = [], 0, []
        async for ev, data in run(M.CONTEXT_BUDGET, "first", False):
            if ev == "__done__":
                answer, packed_idx, toks, packed = data
            else:
                yield _sse(ev, data)
        answer_docs, answer_metas, answer_dists = docs, metas, dists

        # ---- Agent 第二轮：只要答得出就走，拒答仍由检索闸门决定 ----
        cite_check = _verify_citations(answer, packed_idx, metas)
        claims = _claim_evidence_map(answer, packed_idx, metas, packed)
        directness = _answer_directness(question, answer, claims)
        rounds = 1
        prev_clean = _round_is_clean(answer, cite_check, directness)
        if _should_agent_continue(answer, cite_check, docs, dists, effective_mode, rounds,
                                  directness):
            rounds = 2
            yield _sse("escalate", {"from": M.CONTEXT_BUDGET, "to": M.BUDGET_ESCALATED,
                                    "reason": "进入 Agent 校准：补充检索、扩大证据并重新核验"})
            yield _sse("agent", {"round": 2, "phase": "retrieve",
                                  "label": "补充检索与证据校准"})
            d2, m2, s2, _ = await loop.run_in_executor(
                None, _retrieve_selected, _followup_query(retrieval_q, 2), resolved_library_ids,
                stream_hybrid, stream_scope)
            second = (d2, m2, s2)
            docs, metas, dists = _merge_retrieval((docs, metas, dists), second)
            a2, idx2, tk2, pk2 = "", [], 0, []
            async for ev, data in run(M.BUDGET_ESCALATED, "escalated", True):
                if ev == "__done__":
                    a2, idx2, tk2, pk2 = data
                else:
                    yield _sse(ev, data)
            toks += tk2
            cc2 = _verify_citations(a2, idx2, metas)
            cl2 = _claim_evidence_map(a2, idx2, metas, pk2)
            dir2 = _answer_directness(question, a2, cl2)
            if _adopt_next_round(prev_clean, a2, cc2, dir2):
                answer, packed_idx, packed = a2, idx2, pk2
                cite_check, claims, directness = cc2, cl2, dir2
                answer_docs, answer_metas, answer_dists = docs, metas, dists
            prev_clean = _round_is_clean(answer, cite_check, directness)

        # ---- 第三轮仅用于二轮后仍拒答或引用失败，不是每题都跑 ----
        if _should_agent_continue(answer, cite_check, docs, dists, effective_mode, rounds,
                                  directness):
            rounds = 3
            yield _sse("escalate", {"from": M.BUDGET_ESCALATED, "to": M.BUDGET_ESCALATED,
                                    "reason": "最终校准：查找反例、限制条件和缺失证据"})
            yield _sse("agent", {"round": 3, "phase": "verify", "label": "最终证据校准"})
            d3, m3, s3, _ = await loop.run_in_executor(
                None, _retrieve_selected, _followup_query(retrieval_q, 3), resolved_library_ids,
                stream_hybrid, stream_scope)
            docs, metas, dists = _merge_retrieval((docs, metas, dists), (d3, m3, s3))
            a3, idx3, tk3, pk3 = "", [], 0, []
            async for ev, data in run(M.BUDGET_ESCALATED, "final", True):
                if ev == "__done__":
                    a3, idx3, tk3, pk3 = data
                else:
                    yield _sse(ev, data)
            toks += tk3
            cc3 = _verify_citations(a3, idx3, metas)
            cl3 = _claim_evidence_map(a3, idx3, metas, pk3)
            dir3 = _answer_directness(question, a3, cl3)
            if _adopt_next_round(prev_clean, a3, cc3, dir3):
                answer, packed_idx, packed = a3, idx3, pk3
                cite_check, claims, directness = cc3, cl3, dir3
                answer_docs, answer_metas, answer_dists = docs, metas, dists

        answer = _clean_answer_echo(question, answer)
        # 与非流式同一道证据下限，并使用实际被采用答案所属轮次的距离。
        if _evidence_floor_blocks(answer_dists) and not M.is_abstain(answer):
            answer = _NO_REFERENCE
        answer, cite_check, claims, support_audit, guard_tokens = await loop.run_in_executor(
            None, _finalize_agent_answer, answer, packed_idx, answer_metas, packed)
        toks += guard_tokens
        answer, cite_check, claims, support_audit, directness = _enforce_final_directness(
            question, answer, packed_idx, answer_metas, packed,
            cite_check, claims, support_audit)
        sources = _sources_from(answer_metas, packed_idx, answer_docs)

        # Preserve the ChatGPT-like typing effect, but replay only the answer
        # that has already passed the shared finalizer.  The technical refusal
        # token is rendered by the final done event rather than flashed as text.
        if not M.is_abstain(answer):
            for start in range(0, len(answer), 36):
                yield _sse("verified_delta", {"text": answer[start:start + 36]})
                await asyncio.sleep(0.008)

        # 第二部分在全部溯源判定之后才生成，不参与任何指标；失败则安静跳过。
        supplement = None
        if stream_extend:
            yield _sse("supplement_start", {"label": "正在补充教材之外的背景说明…"})
            supplement = await loop.run_in_executor(
                None, _supplement_answer, question, answer, M.is_abstain(answer))
            if supplement:
                toks += supplement["tokens"]

        yield _sse("done", {
            "answer": answer,
            "abstained": M.is_abstain(answer),
            "sources": sources,
            "tokens": toks,
            "escalated": rounds > 1,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "cite_check": cite_check,
            "retrieval": ("scoped" if stream_scope
                          else ("hybrid" if stream_hybrid else "vector")),
            "page_scope": (dict(stream_scope, label=_scope_label(stream_scope))
                           if stream_scope else None),
            "dropped_libraries": dropped_libraries,
            "supplement": supplement,
            "agent": _agent_payload(answer, cite_check, sources, rounds, effective_mode,
                                    bool(history), intent, libraries, claims, answer_dists,
                                    directness, support_audit, stream_scope,
                                    stream_extend,
                                    stream_hybrid),
        })

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/ask/stream")
async def api_ask_stream_post(payload: dict):
    """POST 版流式问答。

    浏览器的多轮历史、角色说明和中文长问题放在 JSON 请求体中，避免 EventSource
    GET URL 随对话增长而触发请求行上限。保留同路径 GET 接口供旧客户端兼容。
    """
    raw_question = payload.get("question") or payload.get("q")
    if raw_question is not None and not isinstance(raw_question, str):
        return JSONResponse({"error": "问题必须是文本"}, status_code=400)
    question = raw_question or ""
    history = payload.get("history", payload.get("h", []))
    libraries = payload.get("libraries", payload.get("libs", []))
    extend = payload.get("extend", False)
    return await api_ask_stream(
        q=question,
        h=json.dumps(history, ensure_ascii=False) if not isinstance(history, str) else history,
        mode=str(payload.get("mode") or "auto"),
        libs=json.dumps(libraries, ensure_ascii=False) if not isinstance(libraries, str) else libraries,
        style=str(payload.get("style") or "standard"),
        instruction=str(payload.get("instruction") or ""),
        extend="1" if _coerce_bool(extend, False) else "0",
        # 这两个必须一并转发：漏传不会报错，只会让用户在界面上做的设置**静默失效**，
        # 表现为"我限定了页范围它却还是从别处答"，比直接报错难查得多。
        hybrid="1" if _coerce_bool(payload.get("hybrid"), False) else "0",
        page_scope=(json.dumps(payload.get("page_scope"), ensure_ascii=False)
                    if payload.get("page_scope") else ""),
    )


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
    raw_kind, raw_path = payload.get("kind"), payload.get("path")
    if not isinstance(raw_kind, str) or not isinstance(raw_path, str):
        return JSONResponse({"error": "kind 和 path 必须是文本"}, status_code=400)
    kind = raw_kind.strip().lower()
    path = raw_path.strip()
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "文件不存在：%s" % path}, status_code=400)
    if kind != "pdf":
        return JSONResponse({"error": "Web 管理器目前仅支持 PDF；其他格式请继续使用原 CLI"},
                            status_code=400)
    try:
        filename = _safe_filename(os.path.basename(path))
        job = _start_build_job(
            path, filename,
            _bounded_int(payload.get("max_pages"), 0, 0, 10000),
            _bounded_int(payload.get("vl_limit"), 15, 0, 100),
            _coerce_bool(payload.get("use_vl"), True),
            _bounded_int(payload.get("vl_from"), 1, 1, 100000))
        return JSONResponse({"ok": True, "job": job}, status_code=202)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=500)


if _UPLOAD_OK:
    @app.post("/api/upload")
    async def api_upload(kind: str = Form("pdf"), max_pages: int = Form(0),
                         vl_limit: int = Form(15), use_vl: bool = Form(True),
                         vl_from: int = Form(1),
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
                                   max(0, min(int(vl_limit), 100)), bool(use_vl),
                                   max(1, min(int(vl_from), 100000)))
            return JSONResponse({"ok": True, "job": job, "size": total}, status_code=202)
        except ValueError as e:
            _discard_pending_upload(path if "path" in locals() else None)
            return JSONResponse({"error": str(e)}, status_code=400)
        except RuntimeError as e:
            _discard_pending_upload(path if "path" in locals() else None)
            return JSONResponse({"error": str(e)}, status_code=409)
        except Exception as e:
            _discard_pending_upload(path if "path" in locals() else None)
            return JSONResponse({"error": str(e)[:300]}, status_code=500)
        finally:
            await file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
