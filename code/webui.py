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
import unicodedata
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
                                        max_pages, vl_limit, use_vl, max(1, int(vl_from))), daemon=True)
        thread.start()
        return _public_job(_jobs[job_id])


_restore_active_library()
print("[webui] 向量库路径 →", M.DB_PATH)
print("[webui] Ollama     →", M._ollama_host())


# ----------------------------- 工具 -----------------------------
def _src_of(meta):
    """把一条 metadata 转成人类可读的来源标签 + 结构化字段（与 main.ask 的 _src 一致）。"""
    library_name = str(meta.get("_library_name") or "").strip()
    source_name = str(meta.get("source") or meta.get("_library_source") or "").strip()
    t = meta.get("type")
    if t == "audio":
        label = "audio %s" % meta.get("time", "?")
        return {"label": ((library_name + " · ") if library_name else "") + label,
                "type": "audio", "loc": meta.get("time", "?"), "page": None,
                "library": library_name, "source": source_name}
    if t in ("epub", "image"):
        label = str(meta.get("loc", t))
        return {"label": ((library_name + " · ") if library_name else "") + label,
                "type": t, "loc": meta.get("loc", ""), "page": None,
                "library": library_name, "source": source_name}
    label = "p%d" % meta.get("page", 0)
    return {"label": ((library_name + " · ") if library_name else "") + label,
            "type": t or "text", "loc": "", "page": meta.get("page"),
            "library": library_name, "source": source_name}


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


def _library_targets(requested):
    """把前端选择解析为可用集合；最多四库，失效项直接忽略。"""
    registry = _read_registry()
    active_id = registry.get("active_id") or "legacy"
    ids = _normalize_library_ids(requested) or [active_id]
    items = {str(x.get("id")): x for x in registry.get("libraries", [])}
    targets = []
    for library_id in ids:
        if library_id == "legacy":
            path = os.path.abspath(registry.get("legacy_db_path") or LEGACY_DB_PATH)
            info = _manifest_info_for(path)
            source = (info.get("sources") or ["原有知识库"])[0]
            name = source or "原有知识库"
        else:
            item = items.get(library_id)
            if not item or item.get("status") != "ready":
                continue
            path = _resolve_db_ref(item.get("db_path"))
            name = str(item.get("name") or item.get("source") or library_id)
            source = str(item.get("source") or name)
        if path and os.path.isdir(path):
            targets.append({"id": library_id, "path": path, "name": name, "source": source})
    return targets or [{"id": active_id, "path": M.DB_PATH,
                        "name": (_library_info().get("sources") or ["当前知识库"])[0],
                        "source": (_library_info().get("sources") or [""])[0]}]


def _retrieve_selected(question, requested=None):
    """单库走原路径；多库按库内名次做公平融合，避免块数量多的书霸占 Top-K。"""
    targets = _library_targets(requested)
    if len(targets) == 1 and os.path.abspath(targets[0]["path"]) == os.path.abspath(M.DB_PATH):
        docs, metas, dists = _retrieve(question)
        target = targets[0]
        metas = [dict(x, _library_id=target["id"], _library_name=target["name"],
                      _library_source=target["source"]) for x in metas]
        return docs, metas, dists, targets

    qv = M.embed([question])[0]
    ranked = []
    alias_by_id = {target["id"]: "K%d" % (index + 1) for index, target in enumerate(targets)}
    for target in targets:
        try:
            col = M.chromadb.PersistentClient(path=os.path.abspath(target["path"])).get_collection(M.COLLECTION)
            docs, metas, dists = M._retrieve(col, qv, question)
        except Exception:
            continue
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
    for item in sorted(ranked, key=lambda x: (-x["score"], x["dist"])):
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
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        if docs is not None and i < len(docs):
            s["snippet"] = docs[i][:220]      # 给前端展开看原文片段
        out.append(s)
    return sorted(out, key=lambda x: x["label"])


def _web_cite_tag(meta):
    base = M._cite_tag(meta)
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
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
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
    for piece in M.split_sentences(masked):
        restored = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], piece).strip()
        if not restored:
            continue
        if out and not _strip_tags(restored):
            out[-1] = (out[-1] + " " + restored).strip()      # 纯引用片段，归上一句
            continue
        if out and _CLOSING_LEAD_RE.match(restored):
            out[-1] = out[-1] + restored                       # 收尾标点，直接贴回去
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
        by_tag.setdefault(tag, {
            "text": packed[pos] if pos < len(packed) else "",
            "library": info.get("library") or "", "label": info.get("label"),
            "page": info.get("page"), "type": info.get("type"),
        })

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
_SUPPORT_RISK_RE = re.compile(
    r"(?:\b(?:always|never|only|must|none|all|not|because|therefore|causes?|caused|"
    r"higher|lower|greater|less|more|equal)\b|"
    r"总是|从不|仅仅|唯一|必须|全部|完全|没有|并非|因此|由于|导致|造成|高于|低于|超过|少于|等于)",
    re.I,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|％)?"
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
        blocks[M._norm_cite(display)] = {"tag": display, "text": str(packed[pos] or "")}
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
    options = {"temperature": 0.0, "num_predict": _SUPPORT_VERIFY_TOKENS}
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


def _canonical_supported_sentence(raw, tag):
    body = _CITE_SPAN_RE.sub("", str(raw or "")).strip()
    body = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", body)
    return (body + " [%s]" % tag).strip()


def _semantic_support_guard(answer, claims, packed_idx, metas, packed):
    """一次性核验可疑结论；明确不支持才裁剪，UNKNOWN/异常一律保留并降级。"""
    default = {"triggered": False, "state": "pass", "checked": 0,
               "supported": 0, "pruned": 0, "unknown": 0,
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
    except Exception as exc:
        audit = dict(default, triggered=True, state="degraded", checked=len(suspicious),
                     unknown=len(suspicious), reason="本地逐句核验未完成：%s" % type(exc).__name__)
        return str(answer or ""), audit, tokens

    suspicious_ids = {idx for idx, _ in suspicious}
    kept, changed = [], False
    counts = {"supported": 0, "pruned": 0, "unknown": 0}
    public_verdicts = []
    for idx, claim in enumerate(claims):
        raw = claim.get("raw") or claim.get("claim") or ""
        if idx not in suspicious_ids:
            kept.append(raw)
            continue
        verdict = verdicts.get(idx, {"status": "UNKNOWN", "reason": "missing"})
        status = verdict["status"]
        public_verdicts.append({"id": idx, "status": status, "reason": verdict.get("reason", "")})
        if status == "SUPPORTED":
            counts["supported"] += 1
            canonical = _canonical_supported_sentence(raw, verdict["tag"])
            kept.append(canonical)
            changed = changed or canonical != raw
        elif status in {"PARTIAL", "UNSUPPORTED"}:
            counts["pruned"] += 1
            changed = True
        else:
            counts["unknown"] += 1
            kept.append(raw)                 # fail-open：不把未知误当成不支持

    if counts["pruned"] and not kept:
        state, guarded = "refused", _NO_REFERENCE
    elif changed:
        state = "pruned" if counts["pruned"] else "verified"
        guarded = " ".join(x.strip() for x in kept if str(x).strip())
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


_PROSE_REFUSAL_RE = re.compile(
    r"^\s*(?:【(?:概要|结论|回答)】\s*)?(?:"
    r"(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)"
    r"(?:\s+(?:provided|supplied|retrieved|available))?\s+"
    r"(?:does\s+not|doesn't|do\s+not)\s+(?:contain|provide|mention|cover)\b|"
    r"(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)"
    r"(?:\s+(?:provided|supplied|retrieved|available))?\s+"
    r"(?:contains?|provides?|mentions?|covers?)\s+no\b|"
    r"[^.\n]{1,180}\s+(?:is|are|was|were)\s+not\s+(?:explicitly\s+)?"
    r"(?:mentioned|covered|provided|contained|found|described|addressed)\s+"
    r"(?:in|by)\s+(?:the\s+)?(?:(?:provided|supplied|retrieved|available|current)\s+)?"
    r"(?:material|context|documents?|sources?|knowledge\s+base)\b|"
    r"no\s+(?:information|details?|evidence|mention)\s+.{0,80}\s+"
    r"(?:is|are|was|were)\s+(?:provided|found|available|contained|mentioned)\b|"
    r"(?:当前|现有|所提供的?|检索到的?)?(?:材料|资料|上下文|文档|知识库)(?:中|里)?"
    r"(?:没有|并未|未能|未|不包含|找不到|缺少).{0,24}(?:信息|依据|内容|答案|提及)"
    r")",
    re.I,
)


def _looks_like_prose_refusal(answer):
    """识别模型把拒答写成散文的情况，并在最终输出前归一为固定 token。

    只匹配答案开头非常明确的“材料不包含/未提供”表述；中途说明某个子问题
    未覆盖的部分回答不会被误判成整题拒答。
    """
    cleaned = M._strip_think(str(answer or "")).strip()
    return bool(_PROSE_REFUSAL_RE.search(cleaned))


def _finalize_agent_answer(answer, packed_idx, metas, packed):
    """流式与非流式共享同一条最终安全收口，避免两套准确率口径。"""
    cleaned = _expand_compound_citations(answer).strip()
    if _looks_like_prose_refusal(cleaned):
        cleaned = _NO_REFERENCE
    initial_check = _verify_citations(cleaned, packed_idx, metas)
    claims = _claim_evidence_map(cleaned, packed_idx, metas, packed)
    skipped = {"triggered": False, "state": "pass", "checked": 0,
               "supported": 0, "pruned": 0, "unknown": 0,
               "reason": "拒答或伪造引用沿用既有安全收口", "verdicts": []}
    # 已拒答或出现检索外标签时，继续沿用原先 fail-closed 契约，不能让二次模型洗白伪造引用。
    if M.is_abstain(cleaned) or initial_check.get("fabricated"):
        final = _finalize_grounded_answer(cleaned, initial_check)
        return final, _verify_citations(final, packed_idx, metas), [], skipped, 0

    guarded, audit, tokens = _semantic_support_guard(
        cleaned, claims, packed_idx, metas, packed)
    check = _verify_citations(guarded, packed_idx, metas)
    final = _finalize_grounded_answer(guarded, check)
    if M.is_abstain(final) and audit.get("triggered") and audit.get("state") != "degraded":
        audit = dict(audit, state="refused", reason="逐句裁剪后没有可交付的完整引用结论")
    final_check = _verify_citations(final, packed_idx, metas)
    final_claims = _claim_evidence_map(final, packed_idx, metas, packed)
    return final, final_check, final_claims, audit, tokens


_HEDGE_RE = re.compile(
    r"^\s*(根据(材料|资料|上下文|原文)|材料(中|里)?(提到|显示|表明)|资料(中|里)?(提到|显示)|"
    r"文中(提到|显示)|the (provided )?(material|context|text)\s+(states|mentions|shows|indicates)|"
    r"according to the (material|context|text))", re.I)


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

    measured = [c for c in claims if c.get("measured")]
    if measured and not any(c["supported"] for c in measured):
        issues.append({"code": "low_grounding",
                       "detail": "全部结论句的接地率均低于 %.0f%%，引用可能只是装饰" % (100 * _GROUNDED_MIN)})
    if claims and all(_HEDGE_RE.match(c["claim"]) for c in claims):
        issues.append({"code": "hedge_only", "detail": "通篇是材料转述，没有给出直接结论"})
    uncited = [c for c in claims if not c["citations"]]
    if claims and len(uncited) == len(claims):
        issues.append({"code": "uncited", "detail": "结论句均未附带可核对的引用"})

    retry_codes = {"only_citation", "no_claim"}
    return {"ok": not issues, "issues": issues,
            "retry": any(x["code"] in retry_codes for x in issues),
            "detail": "；".join(x["detail"] for x in issues) or "答案直接回应了问题"}


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
        "concise": "Keep it to one tight paragraph that still reads as an explanation, not a quotation.",
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
_STYLE_GATE_MAX = 0.96


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
                  preference="", rich=True):
    history = history or []
    parts = []
    if history:
        parts.append("Conversation context (use only to resolve references; the retrieved material remains the authority):\n"
                     + _history_text(history))
    parts.append("Current question: " + question)
    # 目标是"像人讲课那样连贯地讲清楚"，同时每句仍可溯源。
    # 旧规则里的 "State the answer before explaining evidence" 会被模型执行成
    # 「Answer: …／Evidence: …」两段标签，读起来像检索结果而不是回答，故改写。
    rules = ["Answer the current question directly, in the same language as the current question."]
    if rich:
        rules += [
            "Write flowing explanatory prose, the way a teacher would explain it. "
            "Open with what it is, then why it matters or how it works, then a concrete example "
            "or application — but only using what the material actually supports.",
            "Do NOT use section headers or labels such as 'Answer:', 'Evidence:', '证据：'. "
            "Do not output a citation tag on its own line.",
        ]
    else:
        # 证据看着不足时退回严格模式：这一档实测能稳定输出 [NO REFERENCE FOUND]。
        rules.append("State only what the material supports, in as few sentences as it takes. "
                     "Do not elaborate, do not add background.")
    rules += [
        "Every factual claim must come from the supplied material; put its exact source tag "
        "inline right after the sentence it supports.",
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
        rules.append("This is a verification retry: correct any unsupported, missing, or fabricated citation; do not merely repeat the first answer.")
    parts.append("Response requirements:\n- " + "\n- ".join(rules))
    tags = list(dict.fromkeys(_web_cite_tag(metas[i]) for i in packed_idx if i < len(metas)))[:2]
    tag_example = " or ".join("[%s]" % x for x in tags) if tags else "[p.112]"
    return M.PROMPT.format(context=context, question="\n\n".join(parts), tag_example=tag_example)


def _pack_agent(docs, metas, question, budget):
    """多库场景给每本书至少一个上下文席位；单库保持既有全量评测口径。"""
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
    sep_cost = max(0, len(first_by_library) - 1) * len("\n---\n")
    quota = max(220, (budget - sep_cost) // len(first_by_library))
    packed = [docs[i][:quota] for i in first_by_library]
    return packed, first_by_library


def _run_agent_once(docs, metas, question, history, budget, verification=False,
                    preference="", style="standard", rich=True):
    """返回 packed 而不只是 packed_idx：接地率必须对着**模型实际看到的文本**算，
       用未截断的原块会高估支持度（多库配额与相关度裁剪都会截断）。"""
    packed, packed_idx = _pack_agent(docs, metas, question, budget)
    context = _labeled_context(packed, packed_idx, metas)
    out = M._generate(
        M.LLM_MODEL,
        _agent_prompt(context, question, packed_idx, metas, history, verification,
                      preference, rich),
        options={"temperature": M.TEMPERATURE, "num_predict": _web_num_predict(style)},
    )
    toks = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
    return out["response"].strip(), toks, packed_idx, packed


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
    covered = "" if abstained else (
        "The textbook section already stated the following; do not simply repeat it, "
        "add the background and context it left out:\n%s\n\n" % _snippet(grounded, 600))
    return (
        "You are answering from your own general knowledge. No source document is available "
        "to you for this part.\n\n"
        "%sQuestion: %s\n\n"
        "Write a clear, self-contained explanation for a student reader.\n"
        "Rules:\n"
        "- Reply in the same language as the question.\n"
        "- Do NOT output any bracketed source tags such as [p.12]; you have nothing to cite here.\n"
        "- Give the definition, why it matters, and a concrete example where useful.\n"
        "- If you are not confident about something, say so plainly instead of inventing detail.\n"
        % (covered, question))


def _supplement_answer(question, grounded, abstained):
    """生成"教材之外"的补充说明。失败不影响主答案，返回 None 即前端不展示。"""
    try:
        out = M._generate(
            M.LLM_MODEL, _supplement_prompt(question, grounded, abstained),
            options={"temperature": 0.3, "num_predict": _SUPPLEMENT_TOKENS})
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


def _merge_retrieval(first, second):
    """原检索 top-3 锁位，补充检索只能补位，避免第二轮把正确首块挤掉。"""
    docs1, metas1, dists1 = [list(x) for x in first]
    d2, m2, s2 = second
    locked = min(3, len(docs1), M.TOP_K)
    docs = docs1[:locked]
    metas = metas1[:locked]
    dists = dists1[:locked]
    seen = set(docs)
    for doc, meta, dist in zip(d2, m2, s2):
        if doc in seen:
            continue
        docs.append(doc); metas.append(meta); dists.append(dist); seen.add(doc)
        if len(docs) >= M.TOP_K:
            break
    # 补充检索不足时，再用首轮剩余候选补齐，确保返回长度稳定。
    for doc, meta, dist in zip(docs1[locked:], metas1[locked:], dists1[locked:]):
        if len(docs) >= M.TOP_K:
            break
        if doc in seen:
            continue
        docs.append(doc); metas.append(meta); dists.append(dist); seen.add(doc)
    return docs[:M.TOP_K], metas[:M.TOP_K], dists[:M.TOP_K]


def _should_agent_continue(answer, cite_check, docs, dists, mode="auto", round_no=1,
                           directness=None):
    if round_no >= 3 or mode == "fast":
        return False
    if mode == "deep" and round_no == 1:
        return True
    if M.is_abstain(answer):
        return M.should_escalate(answer, docs, dists, M.DYNAMIC_BUDGET)
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
    independent = len({(x.get("library") or "", x.get("label")) for x in sources if x.get("label")})
    cross_library = len({x.get("library") or "" for x in sources if x.get("label")}) > 1
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
        {"name": "多来源印证", "ok": independent >= 2,
         "detail": "%d 个独立来源%s" % (independent, "（跨知识库）" if cross_library else "")},
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

    strong = ((coverage is None or coverage >= 0.8) and independent >= 2
              and (relevance_ok or not usable)
              and (supported_ratio is None or supported_ratio >= 0.8))
    if strong:
        return {"level": "高", "state": "supported", "signals": signals,
                "reason": "引用全部命中、结论均有据可查，且有 %d 个独立来源相互印证。" % independent}
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
            relations.append({
                "type": "互相印证", "claim": claim["claim"],
                "detail": "%d 个来源同时支持该结论：%s" %
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


def _agent_payload(answer, cite_check, sources, rounds, mode, history_used,
                   intent=None, libraries=None, claims=None, dists=None,
                   directness=None, support_audit=None):
    claims = claims or []
    confidence = _confidence_payload(answer, cite_check, sources, claims, dists,
                                     rounds, libraries, directness, support_audit)
    if M.is_abstain(answer):
        stop = "补充检索后证据仍不足，停止生成结论" if rounds > 1 else "首轮未找到可支持结论的证据"
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
    """非流式 Agent 问答：简单题快速返回，复杂/低证据题补充检索并校准。"""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    history = _normalize_history(payload.get("history"))
    mode = str(payload.get("mode") or "auto").lower()
    if mode not in ("auto", "fast", "deep"):
        mode = "auto"
    requested_libraries = _normalize_library_ids(payload.get("libraries"))
    style = str(payload.get("style") or "standard").lower()
    preference = _response_preference(style, payload.get("instruction"))

    t0 = time.time()
    retrieval_q = _retrieval_question(question, history)
    docs, metas, dists, libraries = _retrieve_selected(retrieval_q, requested_libraries)
    intent = _detect_intent(question, history, len(libraries))
    rich = _evidence_looks_present(dists)
    answer, toks, packed_idx, packed = _run_agent_once(
        docs, metas, question, history, M.CONTEXT_BUDGET, verification=False,
        preference=preference, style=style, rich=rich)
    cite_check = _verify_citations(answer, packed_idx, metas)
    claims = _claim_evidence_map(answer, packed_idx, metas, packed)
    directness = _answer_directness(question, answer, claims)
    rounds = 1
    if intent["route"] == "agent_loop" and mode == "auto":
        mode = "deep"
    if _should_agent_continue(answer, cite_check, docs, dists, mode, rounds, directness):
        second_docs, second_metas, second_dists, _ = _retrieve_selected(
            _followup_query(retrieval_q, 2), requested_libraries)
        second = (second_docs, second_metas, second_dists)
        docs, metas, dists = _merge_retrieval((docs, metas, dists), second)
        ans2, toks2, idx2, packed2 = _run_agent_once(
            docs, metas, question, history, M.BUDGET_ESCALATED, verification=True,
            preference=preference, style=style, rich=rich)
        toks += toks2; rounds = 2
        cc2 = _verify_citations(ans2, idx2, metas)
        # 第二轮只要非空就采用；若仍拒答，保留诚实拒答而不是第一轮风险答案。
        if ans2.strip():
            answer, packed_idx, packed, cite_check = ans2, idx2, packed2, cc2
            claims = _claim_evidence_map(answer, packed_idx, metas, packed)
            directness = _answer_directness(question, answer, claims)
    if _should_agent_continue(answer, cite_check, docs, dists, mode, rounds, directness):
        third_docs, third_metas, third_dists, _ = _retrieve_selected(
            _followup_query(retrieval_q, 3), requested_libraries)
        docs, metas, dists = _merge_retrieval((docs, metas, dists),
                                              (third_docs, third_metas, third_dists))
        ans3, toks3, idx3, packed3 = _run_agent_once(
            docs, metas, question, history, M.BUDGET_ESCALATED, verification=True,
            preference=preference, style=style, rich=rich)
        toks += toks3; rounds = 3
        cc3 = _verify_citations(ans3, idx3, metas)
        if ans3.strip():
            answer, packed_idx, packed, cite_check = ans3, idx3, packed3, cc3
            claims = _claim_evidence_map(answer, packed_idx, metas, packed)
            directness = _answer_directness(question, answer, claims)

    answer = _clean_answer_echo(question, answer)
    answer, cite_check, claims, support_audit, guard_tokens = _finalize_agent_answer(
        answer, packed_idx, metas, packed)
    toks += guard_tokens
    directness = _answer_directness(question, answer, claims)
    sources = _sources_from(metas, packed_idx, docs)

    # 第二部分在**所有溯源判定都算完之后**才生成，确保它不可能影响任何指标。
    supplement = None
    if str(payload.get("extend", "0")).lower() not in ("0", "false", "off", "none", ""):
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
        "supplement": supplement,
        "agent": _agent_payload(answer, cite_check, sources, rounds, mode, bool(history),
                                intent, libraries, claims, dists, directness, support_audit),
    }


@app.post("/api/brief")
def api_brief(payload: dict):
    """非流式简报生成：围绕 topic 综合检索资料，产出带出处的结构化 brief 文档。
       复用 main.brief 的生成逻辑（_run_once_brief），返回结构与 /api/ask 一致，前端可复用渲染。"""
    topic = (payload.get("topic") or payload.get("question") or "").strip()
    if not topic:
        return JSONResponse({"error": "主题为空"}, status_code=400)

    t0 = time.time()
    docs, metas, dists, libraries = _retrieve_selected(
        topic, _normalize_library_ids(payload.get("libraries")))
    packed, packed_idx = _pack_agent(docs, metas, topic, M.BUDGET_ESCALATED)
    context = _labeled_context(packed, packed_idx, metas)
    answer, toks = M._gen_brief_raw(M.BRIEF_PROMPT.format(context=context, topic=topic))
    answer = _clean_answer_echo(topic, answer)
    answer, cite_check, claims, support_audit, guard_tokens = _finalize_agent_answer(
        answer, packed_idx, metas, packed)
    toks += guard_tokens
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
                                intent, libraries, claims, dists, directness, support_audit),
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
    """从所选知识库生成可核对的演示问题、预期答案和原文出处。"""
    topic = str(payload.get("topic") or "").strip()
    count = min(5, max(2, int(payload.get("count") or 3)))
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
                      options={"temperature": 0.1, "num_predict": min(900, M.NUM_PREDICT)})
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
                continue          # 单行损坏不该让整份反馈读不出来
    return rows


@app.post("/api/feedback")
def api_feedback(payload: dict):
    """记录一条用户反馈。失败样本随后可由 /api/feedback/regression 导出成回归集。"""
    kind = str(payload.get("kind") or "").strip()
    question = str(payload.get("question") or "").strip()
    if kind not in _FEEDBACK_KINDS:
        return JSONResponse({"error": "未知的反馈类型：%s" % kind}, status_code=400)
    if not question:
        return JSONResponse({"error": "缺少问题内容"}, status_code=400)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind, "kind_label": _FEEDBACK_KINDS[kind],
        "question": question[:1000],
        "answer": str(payload.get("answer") or "")[:4000],
        "libraries": _normalize_library_ids(payload.get("libraries")),
        "sources": [str(x)[:120] for x in (payload.get("sources") or [])][:12],
        "confidence": str(payload.get("confidence") or "")[:20],
        "abstained": bool(payload.get("abstained")),
        "rounds": int(payload.get("rounds") or 1),
        # 只有非"有用"的反馈才是回归集素材
        "is_failure": kind != "useful",
    }
    with _feedback_lock:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
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


@app.get("/api/feedback/regression")
def api_feedback_regression():
    """把失败样本导出成与 eval_*.jsonl 同构的回归集，可直接喂 run_eval_batch.py。

    字段对齐现有题集：``book / question / keywords / type / expect``。
    ``keywords`` 取答案里的内容词——**只作为占位起点，需人工订正**，
    因为被标记为失败的那条答案本身可能就是错的，拿它当标准答案会把错误固化下来。
    """
    rows = [x for x in _read_feedback() if x.get("is_failure")]
    seen, out = set(), []
    for row in rows:
        question = (row.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        expect_answer = not row.get("abstained")
        words = sorted(_latin_words(_strip_tags(row.get("answer"))))[:3]
        out.append({
            "book": ", ".join(row.get("libraries") or []) or "webui",
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


@app.post("/api/retrieve")
def api_retrieve_only(payload: dict):
    """只返回检索证据，不调用生成模型；用于调试召回与快速查原文。"""
    question = str(payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "检索词为空"}, status_code=400)
    limit = min(12, max(1, int(payload.get("limit") or M.TOP_K)))
    requested = _normalize_library_ids(payload.get("libraries"))
    t0 = time.time()
    docs, metas, dists, libraries = _retrieve_selected(question, requested)
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
            "elapsed_ms": int((time.time() - t0) * 1000), "llm_called": False}


@app.get("/api/libraries/{library_id}/chunks")
def api_library_chunks(library_id: str, q: str = "", limit: int = 12, offset: int = 0):
    """浏览或搜索分块内容；只读，不修改向量库。"""
    limit = min(30, max(1, int(limit)))
    offset = max(0, int(offset))
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
                         style: str = "standard", instruction: str = "", extend: str = "0"):
    """流式 Agent 问答。首次证据充分则快速返回，否则补充检索并校准一次。"""
    question = (q or "").strip()
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    history = _normalize_history(h)
    mode = (mode or "auto").lower()
    if mode not in ("auto", "fast", "deep"):
        mode = "auto"
    requested_libraries = _normalize_library_ids(libs)
    preference = _response_preference(style, instruction)

    async def gen():
        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            retrieval_q = _retrieval_question(question, history)
            docs, metas, dists, libraries = await loop.run_in_executor(
                None, _retrieve_selected, retrieval_q, requested_libraries)
        except Exception as e:
            msg = str(e)[:200]
            if "10061" in msg or "refused" in msg.lower() or "urlopen" in msg.lower():
                msg = ("连不上 Ollama（%s）。请确认：① Ollama 已启动（ollama list 能列出模型）；"
                       "② 若设过 OLLAMA_HOST 环境变量，需与 Ollama 实际监听地址一致。"
                       % M._ollama_host())
            yield _sse("error", {"msg": msg})
            return
        intent = _detect_intent(question, history, len(libraries))
        effective_mode = "deep" if mode == "auto" and intent["route"] == "agent_loop" else mode
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
                                 # 交互式问答用 Web 端自己的篇幅，不动评测口径的 M.NUM_PREDICT
                                 "num_predict": _web_num_predict(style)},
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
            yield ("__done__", (full, packed_idx, toks, packed))

        # ---- 第一档 ----
        answer = ""
        packed_idx, toks, packed = [], 0, []
        async for ev, data in run(M.CONTEXT_BUDGET, "first", False):
            if ev == "__done__":
                answer, packed_idx, toks, packed = data
            else:
                yield _sse(ev, data)

        # ---- Agent 第二轮：复杂题、拒答或引用校验失败时补充检索与重答 ----
        cite_check = _verify_citations(answer, packed_idx, metas)
        claims = _claim_evidence_map(answer, packed_idx, metas, packed)
        directness = _answer_directness(question, answer, claims)
        rounds = 1
        if _should_agent_continue(answer, cite_check, docs, dists, effective_mode, rounds,
                                  directness):
            rounds = 2
            yield _sse("escalate", {"from": M.CONTEXT_BUDGET, "to": M.BUDGET_ESCALATED,
                                    "reason": "进入 Agent 校准：补充检索、扩大证据并重新核验"})
            yield _sse("agent", {"round": 2, "phase": "retrieve",
                                  "label": "补充检索与证据校准"})
            d2, m2, s2, _ = await loop.run_in_executor(
                None, _retrieve_selected, _followup_query(retrieval_q, 2), requested_libraries)
            second = (d2, m2, s2)
            docs, metas, dists = _merge_retrieval((docs, metas, dists), second)
            a2, idx2, tk2, pk2 = "", [], 0, []
            async for ev, data in run(M.BUDGET_ESCALATED, "escalated", True):
                if ev == "__done__":
                    a2, idx2, tk2, pk2 = data
                else:
                    yield _sse(ev, data)
            toks += tk2
            if a2.strip():
                answer, packed_idx, packed = a2, idx2, pk2
                cite_check = _verify_citations(answer, packed_idx, metas)
                claims = _claim_evidence_map(answer, packed_idx, metas, packed)
                directness = _answer_directness(question, answer, claims)

        # ---- 第三轮仅用于二轮后仍拒答或引用失败；避免所有复杂题无条件变慢 ----
        if _should_agent_continue(answer, cite_check, docs, dists, effective_mode, rounds,
                                  directness):
            rounds = 3
            yield _sse("escalate", {"from": M.BUDGET_ESCALATED, "to": M.BUDGET_ESCALATED,
                                    "reason": "最终校准：查找反例、限制条件和缺失证据"})
            yield _sse("agent", {"round": 3, "phase": "verify", "label": "最终证据校准"})
            d3, m3, s3, _ = await loop.run_in_executor(
                None, _retrieve_selected, _followup_query(retrieval_q, 3), requested_libraries)
            docs, metas, dists = _merge_retrieval((docs, metas, dists), (d3, m3, s3))
            a3, idx3, tk3, pk3 = "", [], 0, []
            async for ev, data in run(M.BUDGET_ESCALATED, "final", True):
                if ev == "__done__":
                    a3, idx3, tk3, pk3 = data
                else:
                    yield _sse(ev, data)
            toks += tk3
            if a3.strip():
                answer, packed_idx, packed = a3, idx3, pk3
                cite_check = _verify_citations(answer, packed_idx, metas)
                claims = _claim_evidence_map(answer, packed_idx, metas, packed)
                directness = _answer_directness(question, answer, claims)

        answer = _clean_answer_echo(question, answer)
        answer, cite_check, claims, support_audit, guard_tokens = await loop.run_in_executor(
            None, _finalize_agent_answer, answer, packed_idx, metas, packed)
        toks += guard_tokens
        directness = _answer_directness(question, answer, claims)
        sources = _sources_from(metas, packed_idx, docs)

        # 第二部分在全部溯源判定之后才生成，不参与任何指标；失败则安静跳过。
        supplement = None
        if str(extend or "0").lower() not in ("0", "false", "off"):
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
            "supplement": supplement,
            "agent": _agent_payload(answer, cite_check, sources, rounds, effective_mode,
                                    bool(history), intent, libraries, claims, dists,
                                    directness, support_audit),
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
    question = str(payload.get("question") or payload.get("q") or "")
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
        extend="1" if extend is True or str(extend).lower() in ("1", "true", "on") else "0",
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
                               bool(payload.get("use_vl", True)),
                               max(1, int(payload.get("vl_from") or 1)))
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
