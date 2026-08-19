"""进程内桌面适配层。

原生界面不启动 FastAPI/Uvicorn，也不占用端口。本模块把已经过全量验证的
``code/webui.py`` 业务函数直接包装为普通 Python 方法，并统一处理原本面向 HTTP 的
``JSONResponse``。这样桌面版和 Beta WebUI 共用同一条检索、生成、引用与拒答链。
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
from html import escape as html_escape
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable
import uuid


DEFAULT_LLM_MODEL = "qwen3:8b"
VISION_MODEL = "qwen3-vl:8b"
EMBEDDING_MODEL = "bge-m3"
LLM_PRESETS = (
    {"model": "qwen3:4b", "label": "Qwen3 4B · 轻量版",
     "description": "资源占用较低，适合普通办公电脑；准确度需按实际教材复核。"},
    {"model": DEFAULT_LLM_MODEL, "label": "Qwen3 8B · 推荐版",
     "description": "本项目全量评测使用的默认档，准确度与拒答口径证据最完整。"},
    {"model": "qwen3:14b", "label": "Qwen3 14B · 高质量版",
     "description": "能力更强但速度更慢、显存和内存需求更高；切换后建议跑回归集。"},
)
REQUIRED_MODELS = (DEFAULT_LLM_MODEL, VISION_MODEL, EMBEDDING_MODEL)
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,126}(?::[A-Za-z0-9._-]{1,64})?$")


class BackendError(RuntimeError):
    """可直接展示给用户的后端错误。"""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = int(status_code)


def default_data_root() -> Path:
    """返回冻结包的默认可写数据根；源码运行则返回项目根。"""
    override = os.environ.get("AITIC_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parents[1]
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return (Path(local) / "AITIC Desktop").resolve()


def bundled_root() -> Path:
    """返回源码/冻结资源所在根目录。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    return Path(__file__).resolve().parents[1]


def _epub_text_pdf(source: Path, target: Path) -> None:
    """Convert an EPUB's reading-order text to a searchable paginated PDF.

    The validated builder intentionally remains PDF-only.  Desktop EPUB import therefore
    normalizes the book once, then sends the generated PDF through exactly the same build,
    citation and source-page pipeline as a normal PDF.  The EPUB is never extracted to disk,
    which also avoids zip-slip paths from untrusted archives.
    """
    try:
        import fitz
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub
    except Exception as exc:  # pragma: no cover - packaging contract covers the frozen app
        raise BackendError("EPUB 支持组件不可用，请重新安装完整版：%s" % exc) from exc

    try:
        book = epub.read_epub(str(source), options={"ignore_ncx": True})
    except Exception as exc:
        raise BackendError("EPUB 文件无法读取或已损坏：%s" % exc, 400) from exc

    title_values = book.get_metadata("DC", "title") or []
    title = str(title_values[0][0] if title_values else source.stem).strip() or source.stem
    ordered = []
    seen: set[str] = set()
    for idref, _linear in list(book.spine or []):
        item = book.get_item_with_id(str(idref))
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            ordered.append(item)
            seen.add(str(item.get_id()))
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        if str(item.get_id()) not in seen:
            ordered.append(item)

    sections: list[tuple[str, list[str]]] = []
    total_chars = 0
    for item in ordered[:2000]:
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
        except Exception:
            continue
        for node in soup(["script", "style", "svg", "noscript"]):
            node.decompose()
        body = soup.body or soup
        lines = []
        for value in body.get_text("\n", strip=True).splitlines():
            value = " ".join(value.split())
            if value and (not lines or value != lines[-1]):
                lines.append(value)
        if not lines:
            continue
        heading = ""
        heading_node = body.find(["h1", "h2", "h3", "title"])
        if heading_node:
            heading = " ".join(heading_node.get_text(" ", strip=True).split())
        heading = heading or Path(str(item.get_name() or "")).stem
        total_chars += sum(len(value) for value in lines)
        if total_chars > 12_000_000:
            raise BackendError("EPUB 解压后的正文过大，已停止导入。", 400)
        sections.append((heading, lines))
    if total_chars < 20:
        raise BackendError("EPUB 中没有可建库的正文文本。", 400)

    parts = ["<h1>%s</h1>" % html_escape(title)]
    for heading, lines in sections:
        if heading and (not lines or heading != lines[0]):
            parts.append("<h2>%s</h2>" % html_escape(heading))
        parts.extend("<p>%s</p>" % html_escape(value) for value in lines)
    css = """
        @page { size: a4; }
        body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #172b46; }
        h1 { font-size: 20pt; color: #102a4b; margin: 0 0 18pt 0; }
        h2 { font-size: 14pt; color: #245f96; margin: 14pt 0 7pt 0; }
        p { margin: 0 0 6pt 0; }
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = fitz.DocumentWriter(str(target))
    story = fitz.Story(html="\n".join(parts), user_css=css)
    page_rect = fitz.paper_rect("a4")
    content_rect = fitz.Rect(54, 54, page_rect.width - 54, page_rect.height - 54)
    try:
        more = True
        pages = 0
        while more:
            if pages >= 20_000:
                raise BackendError("EPUB 转换页数异常，已停止导入。", 400)
            device = writer.begin_page(page_rect)
            more, _filled = story.place(content_rect)
            story.draw(device)
            writer.end_page()
            pages += 1
    finally:
        writer.close()
    try:
        with fitz.open(str(target)) as document:
            if document.page_count < 1 or sum(len(page.get_text()) for page in document) < 20:
                raise BackendError("EPUB 转换后没有可检索正文。", 400)
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError("EPUB 转换结果无法读取：%s" % exc, 400) from exc


class DesktopBackend:
    """Qt 前端使用的稳定同步接口；耗时方法必须放到工作线程调用。"""

    def __init__(self, project_root: str | os.PathLike[str] | None = None):
        self.project_root = Path(project_root or default_data_root()).expanduser().resolve()
        self._ollama_process: subprocess.Popen | None = None
        self._ollama_log_handle = None
        self._import_jobs: dict[str, dict[str, str]] = {}
        self._prepare_paths()
        self._config = self._read_desktop_config()
        self._apply_process_config()
        self.webui = self._load_backend()
        self._apply_model_config()
        self.ensure_ollama_running(wait_seconds=1.5)
        atexit.register(self.close)

    def _prepare_paths(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / "data" / "vectordb").mkdir(parents=True, exist_ok=True)
        (self.project_root / "data" / "webui_knowledge_bases").mkdir(parents=True, exist_ok=True)
        (self.project_root / "books").mkdir(parents=True, exist_ok=True)
        os.environ["AITIC_PROJECT_ROOT"] = str(self.project_root)
        os.environ.setdefault("DISTILL_DB", str(self.project_root / "data" / "vectordb"))

        root = bundled_root()
        code_dir = root / "code"
        if code_dir.is_dir() and str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

    @property
    def config_path(self) -> Path:
        return self.project_root / "data" / "desktop_settings.json"

    @property
    def sessions_path(self) -> Path:
        """桌面会话文件；只保存本地文本、选择的教材和引用元数据。"""
        return self.project_root / "data" / "desktop_sessions.json"

    @classmethod
    def _session_json_value(cls, value: Any, depth: int = 0) -> Any:
        """Convert an untrusted result payload to bounded JSON-safe data."""
        if depth > 8:
            return None
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            return value[:120_000]
        if isinstance(value, (list, tuple)):
            return [cls._session_json_value(item, depth + 1) for item in value[:100]]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in list(value.items())[:150]:
                result[str(key)[:120]] = cls._session_json_value(item, depth + 1)
            return result
        return None

    @classmethod
    def _normalise_session(cls, value: Any) -> dict[str, Any] | None:
        """Apply the same schema and size limits on both save and load."""
        if not isinstance(value, dict) or not str(value.get("id") or "").strip():
            return None
        messages: list[dict[str, Any]] = []
        for raw_message in (value.get("messages") or [])[-100:]:
            if not isinstance(raw_message, dict):
                continue
            role = str(raw_message.get("role") or "")
            if role not in ("user", "assistant", "system"):
                continue
            sources = []
            for source in (raw_message.get("sources") or [])[:20]:
                if isinstance(source, dict):
                    safe_source = cls._session_json_value(source)
                    if isinstance(safe_source, dict):
                        sources.append(safe_source)
            message: dict[str, Any] = {
                "role": role,
                "content": str(raw_message.get("content") or "")[:50_000],
                "meta": str(raw_message.get("meta") or "")[:1_000],
                "sources": sources,
                "favorite": bool(raw_message.get("favorite")),
            }
            payload = raw_message.get("payload")
            if isinstance(payload, dict):
                safe_payload = cls._session_json_value(payload)
                if isinstance(safe_payload, dict):
                    # Rich Agent data is needed to reconstruct the verified result
                    # after restart, but a malformed model field must not grow the
                    # local session file without bound.
                    encoded = json.dumps(
                        safe_payload, ensure_ascii=False, allow_nan=False,
                        separators=(",", ":"))
                    if len(encoded.encode("utf-8")) <= 500_000:
                        message["payload"] = safe_payload
            messages.append(message)
        return {
            "id": str(value.get("id"))[:120],
            "title": str(value.get("title") or "新对话")[:80],
            "updated_at": str(value.get("updated_at") or "")[:64],
            "pinned": bool(value.get("pinned")),
            "library_ids": [str(item)[:160] for item in
                            (value.get("library_ids") or [])[:4]],
            "messages": messages,
        }

    def load_sessions(self, limit: int = 80) -> dict[str, Any]:
        """读取经过收敛处理的本地会话，损坏文件按空列表处理。"""
        try:
            raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            raw = []
        if isinstance(raw, dict):
            raw = raw.get("sessions") or []
        sessions: list[dict[str, Any]] = []
        for value in raw if isinstance(raw, list) else []:
            session = self._normalise_session(value)
            if session is not None:
                sessions.append(session)
        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return {"sessions": sessions[:max(1, min(int(limit), 200))]}

    def save_sessions(self, sessions: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """原子保存桌面会话；复用读取端的格式校验避免无限增长。"""
        values = []
        for value in list(sessions)[:80]:
            session = self._normalise_session(value)
            if session is not None:
                values.append(session)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.sessions_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.sessions_path)
        return {"saved": len(values), "path": str(self.sessions_path)}

    def _read_desktop_config(self) -> dict[str, Any]:
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_desktop_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.config_path)

    def _apply_process_config(self) -> None:
        model_dir = str(self._config.get("models_directory") or "").strip()
        if model_dir:
            os.environ["OLLAMA_MODELS"] = str(Path(model_dir).expanduser().resolve())

    def _apply_model_config(self) -> None:
        selected = str(self._config.get("llm_model") or DEFAULT_LLM_MODEL).strip()
        if not _MODEL_NAME_RE.fullmatch(selected):
            selected = DEFAULT_LLM_MODEL
        self.webui.M.LLM_MODEL = selected
        if hasattr(self.webui.M, "_ENV_FP_CACHE"):
            self.webui.M._ENV_FP_CACHE.clear()

    @staticmethod
    def _load_backend():
        try:
            return importlib.import_module("webui")
        except BaseException as exc:  # main.py 某些缺依赖路径会抛 SystemExit
            raise BackendError("桌面后端加载失败：%s" % str(exc)) from exc

    @staticmethod
    def _json_body(response: Any) -> Any:
        body = getattr(response, "body", b"")
        if isinstance(body, memoryview):
            body = body.tobytes()
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="replace")
        else:
            text = str(body or "")
        try:
            return json.loads(text) if text else {}
        except ValueError:
            return {"error": text or "后端返回了不可解析的数据"}

    def _unwrap(self, result: Any) -> Any:
        status = int(getattr(result, "status_code", 200) or 200)
        payload = self._json_body(result) if hasattr(result, "body") else result
        if status >= 400:
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("detail") or str(payload)
            else:
                message = str(payload)
            raise BackendError(message, status)
        return payload

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return self._unwrap(fn(*args, **kwargs))
        except BackendError:
            raise
        except BaseException as exc:
            raise BackendError(str(exc) or exc.__class__.__name__) from exc

    # ---------- 状态与知识库 ----------
    def status(self) -> dict[str, Any]:
        return self._call(self.webui.status)

    def libraries(self) -> dict[str, Any]:
        payload = self._call(self.webui.api_libraries)
        # 全新安装的数据目录里只有为 Chroma 预留的空 vectordb。WebUI 的兼容
        # 列表只按“目录存在”会把它标成 ready，原生侧因此出现一条可勾选但无法
        # 使用的“原有知识库”。仅对桌面默认空目录隐藏该兼容项；带 manifest 的
        # 老库仍保留，并直接显示建库时记录的块数。
        default_legacy = (self.project_root / "data" / "vectordb").resolve()
        registry = self.webui._read_registry()
        raw_path = registry.get("legacy_db_path") or str(default_legacy)
        legacy_path = Path(raw_path)
        if not legacy_path.is_absolute():
            legacy_path = self.project_root / legacy_path
        legacy_path = legacy_path.resolve()
        manifest_path = legacy_path / "build_manifest.json"
        manifest = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        libraries = list(payload.get("libraries") or [])
        if legacy_path == default_legacy and not manifest:
            libraries = [item for item in libraries if str(item.get("id")) != "legacy"]
        else:
            count = int(manifest.get("n_chunks_total") or 0) if manifest else 0
            if count > 0:
                for item in libraries:
                    if str(item.get("id")) == "legacy":
                        item["chunks"] = count
                        break
        aliases = self._config.get("library_aliases") or {}
        if isinstance(aliases, dict):
            for item in libraries:
                alias = aliases.get(str(item.get("id") or ""))
                if not isinstance(alias, dict):
                    continue
                if str(alias.get("name") or "").strip():
                    item["name"] = str(alias["name"])
                if str(alias.get("source") or "").strip():
                    item["source"] = str(alias["source"])
                item["format"] = str(alias.get("format") or item.get("format") or "")
        return dict(payload, libraries=libraries)

    def activate_library(self, library_id: str) -> dict[str, Any]:
        return self._call(self.webui.api_activate_library, str(library_id))

    def library_health(self, library_id: str, sample: int = 800) -> dict[str, Any]:
        return self._call(self.webui.api_library_health, str(library_id), int(sample))

    def library_chunks(self, library_id: str, query: str = "", limit: int = 20,
                       offset: int = 0) -> dict[str, Any]:
        return self._call(
            self.webui.api_library_chunks, str(library_id), str(query), int(limit), int(offset))

    def start_build(self, path: str, *, max_pages: int = 0, use_vl: bool = True,
                    vl_limit: int = 15, vl_from: int = 1) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise BackendError("文件不存在：%s" % source, 400)
        source_suffix = source.suffix.lower()
        if source_suffix not in (".pdf", ".epub"):
            raise BackendError("请选择 PDF 或 EPUB 文件", 400)
        if source.stat().st_size > int(self.webui.MAX_UPLOAD_BYTES):
            raise BackendError("教材超过导入上限（%d MB）" %
                               (int(self.webui.MAX_UPLOAD_BYTES) // 1024 // 1024), 400)
        original_source = source
        if source_suffix == ".epub":
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            cache_dir = self.project_root / "data" / "epub_imports"
            safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "_", source.stem).strip("_") or "book"
            converted = cache_dir / ("%s_%s.pdf" % (safe_stem[:60], digest.hexdigest()[:16]))
            if not converted.is_file() or converted.stat().st_size < 100:
                temporary = converted.with_name(converted.stem + ".tmp.pdf")
                try:
                    _epub_text_pdf(source, temporary)
                    temporary.replace(converted)
                finally:
                    temporary.unlink(missing_ok=True)
            source = converted
            # EPUB 正文已被规范化为可检索文本 PDF；没有原 PDF 图表页可交给视觉模型。
            use_vl = False
        else:
            with source.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise BackendError("文件内容不是有效 PDF", 400)
        upload_dir = (self.project_root / "data" / "webui_knowledge_bases" / "uploads" /
                      uuid.uuid4().hex)
        upload_dir.mkdir(parents=True, exist_ok=False)
        copied = upload_dir / self.webui._safe_filename(source.name)
        shutil.copy2(source, copied)
        payload = {
            "kind": "pdf", "path": str(copied), "max_pages": int(max_pages),
            "use_vl": bool(use_vl), "vl_limit": int(vl_limit), "vl_from": int(vl_from),
        }
        try:
            result = asyncio.run(self.webui.api_build(payload))
            value = self._unwrap(result)
            job = value.get("job") or {} if isinstance(value, dict) else {}
            job_id = str(job.get("id") or job.get("job_id") or "")
            if job_id:
                self._import_jobs[job_id] = {
                    "name": original_source.stem,
                    "source": original_source.name,
                    "format": source_suffix.lstrip("."),
                }
            return value
        except BaseException as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            if isinstance(exc, BackendError):
                raise
            raise BackendError("无法启动建库任务：%s" % exc) from exc

    def build_status(self, job_id: str) -> dict[str, Any]:
        job_id = str(job_id)
        value = self._call(self.webui.api_build_job, job_id)
        status = str(value.get("status") or "") if isinstance(value, dict) else ""
        tracked = self._import_jobs.get(job_id)
        if tracked and status in ("completed", "ready"):
            library_id = str(value.get("library_id") or "")
            if library_id:
                aliases = self._config.setdefault("library_aliases", {})
                aliases[library_id] = dict(tracked)
                self._write_desktop_config()
            self._import_jobs.pop(job_id, None)
        elif tracked and status == "failed":
            self._import_jobs.pop(job_id, None)
        return value

    def active_build(self) -> dict[str, Any]:
        """Return a build already running in this process so the native UI can resume polling."""
        return self._call(self.webui.api_active_build)

    # ---------- 问答与学习工具 ----------
    def ask(self, question: str, *, libraries: Iterable[str] = (), history: list | None = None,
            mode: str = "auto", style: str = "standard", instruction: str = "",
            hybrid: bool = False, extend: bool = False,
            page_scope: dict[str, int | None] | None = None) -> dict[str, Any]:
        return self._call(self.webui.api_ask, {
            "question": question, "libraries": list(libraries), "history": history or [],
            "mode": mode, "style": style, "instruction": instruction,
            "hybrid": bool(hybrid), "extend": bool(extend),
            "page_scope": page_scope,
        })

    def ask_stream(self, question: str, *, libraries: Iterable[str] = (),
                   history: list | None = None, mode: str = "auto",
                   style: str = "standard", instruction: str = "",
                   hybrid: bool = False, extend: bool = False,
                   page_scope: dict[str, int | None] | None = None,
                   cancel_event: Any = None,
                   on_event: Callable[[str, dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Consume the validated SSE generator in-process and expose its safe UI events."""
        payload = {
            "question": question, "libraries": list(libraries), "history": history or [],
            "mode": mode, "style": style, "instruction": instruction,
            "hybrid": bool(hybrid), "extend": bool(extend), "page_scope": page_scope,
        }

        async def consume() -> dict[str, Any]:
            response = await self.webui.api_ask_stream_post(payload)
            if hasattr(response, "body"):
                return self._unwrap(response)
            iterator = getattr(response, "body_iterator", None)
            if iterator is None:
                raise BackendError("流式问答没有返回事件流。")
            buffer = ""
            final: dict[str, Any] | None = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    close = getattr(iterator, "aclose", None)
                    if callable(close):
                        await close()
                    raise BackendError("本轮已停止")
                next_chunk = asyncio.create_task(iterator.__anext__())
                while not next_chunk.done():
                    await asyncio.sleep(0.05)
                    if cancel_event is not None and cancel_event.is_set():
                        next_chunk.cancel()
                        await asyncio.gather(next_chunk, return_exceptions=True)
                        close = getattr(iterator, "aclose", None)
                        if callable(close):
                            await close()
                        raise BackendError("本轮已停止")
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="replace")
                buffer += str(chunk)
                buffer = buffer.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event = "message"
                    data_parts = []
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data_parts.append(line[5:].strip())
                    if not data_parts:
                        continue
                    try:
                        data = json.loads("\n".join(data_parts))
                    except ValueError:
                        data = {"text": "\n".join(data_parts)}
                    if event == "error":
                        raise BackendError(str(data.get("msg") or data.get("error") or "问答失败"))
                    if on_event:
                        on_event(event, data if isinstance(data, dict) else {"value": data})
                    if event == "done" and isinstance(data, dict):
                        final = data
            if final is None:
                raise BackendError("流式问答提前结束，未收到已核验答案。")
            return final

        try:
            return asyncio.run(consume())
        except BackendError:
            raise
        except BaseException as exc:
            raise BackendError(str(exc) or exc.__class__.__name__) from exc

    def brief(self, topic: str, *, libraries: Iterable[str] = ()) -> dict[str, Any]:
        return self._call(self.webui.api_brief, {
            "topic": topic, "libraries": list(libraries),
        })

    def questions(self, topic: str, *, libraries: Iterable[str] = (),
                  count: int = 3) -> dict[str, Any]:
        return self._call(self.webui.api_questions, {
            "topic": topic, "libraries": list(libraries), "count": int(count),
        })

    def concept(self, concept: str, *, libraries: Iterable[str] = (),
                style: str = "standard") -> dict[str, Any]:
        return self._call(self.webui.api_concept, {
            "concept": concept, "libraries": list(libraries), "style": style,
        })

    def compare(self, question: str, *, libraries: Iterable[str] = ()) -> dict[str, Any]:
        return self._call(self.webui.api_compare, {
            "question": question, "libraries": list(libraries), "mode": "auto",
            "style": "standard", "variants": [
                {"label": "向量检索", "hybrid": False},
                {"label": "混合检索", "hybrid": True},
            ],
        })

    def batch(self, questions: Iterable[str], *, libraries: Iterable[str] = ()) -> dict[str, Any]:
        items = [{"question": str(q), "answerable": True} for q in questions]
        return self._call(self.webui.api_batch, {
            "items": items, "libraries": list(libraries), "style": "standard",
        })

    def retrieve(self, question: str, *, libraries: Iterable[str] = (),
                 hybrid: bool = False, limit: int = 8) -> dict[str, Any]:
        return self._call(self.webui.api_retrieve_only, {
            "question": question, "libraries": list(libraries),
            "hybrid": bool(hybrid), "limit": int(limit),
        })

    # ---------- 反馈闭环 ----------
    def feedback(self, kind: str, question: str, answer: str,
                 libraries: Iterable[str] = ()) -> dict[str, Any]:
        return self._call(self.webui.api_feedback, {
            "kind": kind, "question": question, "answer": answer,
            "libraries": list(libraries),
        })

    def feedback_list(self, limit: int = 100) -> dict[str, Any]:
        return self._call(self.webui.api_feedback_list, int(limit))

    def rerun_feedback(self, limit: int = 10) -> dict[str, Any]:
        return self._call(self.webui.api_feedback_rerun, {"limit": int(limit)})

    def export_regression(self) -> dict[str, Any]:
        return self._call(self.webui.api_feedback_regression)

    # ---------- 原文与 PDF 页 ----------
    def source_blocks(self, source: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            self.webui.api_source,
            str(source.get("library_id") or ""),
            int(source.get("page") or 0),
            str(source.get("loc") or ""),
            str(source.get("source") or ""),
            str(source.get("snippet") or ""),
        )

    def source_page_png(self, source: dict[str, Any], dpi: int = 120) -> bytes:
        result = self.webui.api_source_page(
            str(source.get("library_id") or ""), int(source.get("page") or 0),
            str(source.get("source") or ""), str(source.get("snippet") or ""), int(dpi))
        status = int(getattr(result, "status_code", 200) or 200)
        if status >= 400:
            self._unwrap(result)
        return bytes(getattr(result, "body", b""))

    # ---------- 首次设置与模型管理 ----------
    def ollama_path(self) -> str | None:
        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / "runtime" / "ollama" / "ollama.exe")
        else:
            runtime_root = bundled_root() / "packaging" / "runtime"
            candidates.append(runtime_root / "ollama" / "ollama.exe")
            # The build cache is versioned so a newer verified runtime never
            # overwrites an older one in place. Source-mode diagnostics should
            # be able to use that same cache before the frozen package exists.
            candidates.extend(sorted(
                runtime_root.glob("ollama-v*/ollama.exe"), reverse=True))
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return shutil.which("ollama")

    def vc_redist_path(self) -> str | None:
        """返回随 Ollama 发行包提供的 VC++ 运行库安装器。"""
        candidates = []
        executable = self.ollama_path()
        if executable:
            candidates.append(Path(executable).resolve().parent / "vc_redist.x64.exe")
        if not getattr(sys, "frozen", False):
            candidates.append(bundled_root() / "packaging" / ".downloads" / "vc_redist.x64.exe")
        return next((str(path) for path in candidates if path.is_file()), None)

    def repair_vc_runtime(self) -> dict[str, Any]:
        """在用户明确点击后，以系统确认窗口启动微软运行库修复。"""
        installer = self.vc_redist_path()
        if not installer:
            raise BackendError("安装包中未找到 VC++ 运行库安装器。")
        if os.name != "nt":
            raise BackendError("VC++ 运行库修复仅适用于 Windows。", 400)
        import ctypes

        result = int(ctypes.windll.shell32.ShellExecuteW(
            None, "runas", installer, "/install /quiet /norestart",
            str(Path(installer).parent), 1))
        if result <= 32:
            raise BackendError("未能启动 VC++ 运行库安装器（系统返回 %d）。" % result)
        return {"started": True, "path": installer}

    def _ollama_api_ready(self, timeout: float = 0.7) -> bool:
        import urllib.request

        try:
            host = self.webui.M._ollama_host().rstrip("/")
            with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as response:
                return int(getattr(response, "status", 200)) == 200
        except Exception:
            return False

    def ensure_ollama_running(self, wait_seconds: float = 8.0) -> dict[str, Any]:
        """使用现有服务；若未运行则无窗口启动随包/本机 Ollama。"""
        if self._ollama_api_ready():
            return {"connected": True, "started": False, "path": self.ollama_path() or ""}
        executable = self.ollama_path()
        if not executable:
            return {"connected": False, "started": False, "path": "",
                    "error": "安装包中未找到 Ollama 运行时"}
        if self._ollama_process is None or self._ollama_process.poll() is not None:
            if self._ollama_log_handle is not None:
                try:
                    self._ollama_log_handle.close()
                except OSError:
                    pass
                self._ollama_log_handle = None
            log_dir = self.project_root / "data" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._ollama_log_handle = (log_dir / "ollama-serve.log").open(
                "a", encoding="utf-8", errors="replace")
            env = os.environ.copy()
            env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
            self._ollama_process = subprocess.Popen(
                [executable, "serve"], cwd=str(Path(executable).parent), env=env,
                stdout=self._ollama_log_handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while time.monotonic() < deadline:
            if self._ollama_api_ready():
                return {"connected": True, "started": True, "path": executable}
            if self._ollama_process.poll() is not None:
                code = self._ollama_process.returncode
                return {
                    "connected": False, "started": True, "path": executable,
                    "error": ("Ollama 启动失败（退出码 %s）。请在模型管理中执行“修复 VC++ 运行库”，"
                              "然后重新打开程序。" % code),
                }
            time.sleep(0.15)
        return {"connected": self._ollama_api_ready(), "started": True, "path": executable,
                "error": "Ollama 正在启动，请稍候后刷新"}

    def model_catalog(self) -> dict[str, Any]:
        inventory = self.model_inventory()
        available = set(inventory.get("available_normalized") or [])
        presets = []
        for preset in LLM_PRESETS:
            model = str(preset["model"])
            presets.append(dict(preset, installed=(model in available),
                                active=(model == self.webui.M.LLM_MODEL)))
        return dict(inventory, presets=presets, active=self.webui.M.LLM_MODEL,
                    vision_model=VISION_MODEL, embedding_model=EMBEDDING_MODEL,
                    models_directory=os.environ.get("OLLAMA_MODELS", ""))

    def model_inventory(self) -> dict[str, Any]:
        import urllib.request

        host = self.webui.M._ollama_host().rstrip("/")
        try:
            with urllib.request.urlopen(host + "/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {"connected": False, "available": [], "available_normalized": [],
                    "missing": [self.webui.M.LLM_MODEL, VISION_MODEL, EMBEDDING_MODEL],
                    "error": str(exc)}
        available = sorted(str(x.get("name") or "") for x in payload.get("models", []))
        normalized = set(available)
        normalized.update(x.removesuffix(":latest") for x in available)
        required = (self.webui.M.LLM_MODEL, VISION_MODEL, EMBEDDING_MODEL)
        missing = [model for model in required
                   if model not in normalized and model.removesuffix(":latest") not in normalized]
        return {"connected": True, "available": available,
                "available_normalized": sorted(normalized), "missing": missing}

    def set_llm_model(self, model: str, *, require_installed: bool = True) -> dict[str, Any]:
        selected = str(model or "").strip()
        if not _MODEL_NAME_RE.fullmatch(selected):
            raise BackendError("模型名称无效，只能使用字母、数字、点、横线、斜线和标签。", 400)
        if require_installed:
            inventory = self.model_inventory()
            normalized = set(inventory.get("available_normalized") or [])
            if selected not in normalized and selected.removesuffix(":latest") not in normalized:
                raise BackendError("模型尚未下载或导入：%s" % selected, 400)
        self._config["llm_model"] = selected
        self._write_desktop_config()
        self._apply_model_config()
        return self.model_catalog()

    def set_model_storage(self, path: str) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        self._config["models_directory"] = str(target)
        self._write_desktop_config()
        os.environ["OLLAMA_MODELS"] = str(target)
        restarted = False
        if self._ollama_process is not None and self._ollama_process.poll() is None:
            self._stop_owned_ollama(timeout_seconds=5)
            restarted = bool(self.ensure_ollama_running(wait_seconds=5).get("connected"))
        return {"path": str(target), "restarted": restarted,
                "restart_external_ollama": not restarted}

    def mark_model_setup_complete(self) -> None:
        self._config["model_setup_complete"] = True
        self._write_desktop_config()

    def needs_model_setup(self) -> bool:
        if bool(self._config.get("model_setup_complete")):
            return False
        inventory = self.model_inventory()
        return bool(inventory.get("missing"))

    def pull_models(self, models: Iterable[str] | None = None,
                    on_line: Callable[[str], None] | None = None) -> dict[str, Any]:
        executable = self.ollama_path()
        if not executable:
            raise BackendError("未找到 Ollama 运行时，请重新安装完整版 AITIC Desktop。")
        if not self._ollama_api_ready():
            started = self.ensure_ollama_running(wait_seconds=8)
            if not started.get("connected"):
                raise BackendError(str(started.get("error") or "Ollama 无法启动"))
        targets = list(models or REQUIRED_MODELS)
        for model in targets:
            process = subprocess.Popen(
                [executable, "pull", str(model)], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            assert process.stdout is not None
            for line in process.stdout:
                if on_line:
                    on_line("%s · %s" % (model, line.rstrip()))
            code = process.wait()
            if code:
                raise BackendError("模型 %s 下载失败（退出码 %d）" % (model, code))
        return self.model_inventory()

    def import_model(self, source: str, model_name: str,
                     on_line: Callable[[str], None] | None = None) -> dict[str, Any]:
        path = Path(source).expanduser().resolve()
        name = str(model_name or "").strip()
        if not path.is_file():
            raise BackendError("模型文件不存在：%s" % path, 400)
        if not _MODEL_NAME_RE.fullmatch(name):
            raise BackendError("模型名称无效。示例：my-course-model:latest", 400)
        executable = self.ollama_path()
        if not executable:
            raise BackendError("未找到 Ollama 运行时。")
        if not self.ensure_ollama_running(wait_seconds=8).get("connected"):
            raise BackendError("Ollama 无法启动，不能导入模型。")
        if path.suffix.lower() == ".gguf":
            import_dir = self.project_root / "data" / "model_imports" / uuid.uuid4().hex
            import_dir.mkdir(parents=True, exist_ok=False)
            modelfile = import_dir / "Modelfile"
            escaped_path = str(path).replace("\\", "/").replace('"', '\\"')
            modelfile.write_text('FROM "%s"\n' % escaped_path, encoding="utf-8")
        elif path.name.lower() == "modelfile" or path.suffix.lower() in (".modelfile", ".txt"):
            modelfile = path
        else:
            raise BackendError("请选择 .gguf、Modelfile 或 .modelfile 文件。", 400)
        process = subprocess.Popen(
            [executable, "create", name, "-f", str(modelfile)],
            cwd=str(modelfile.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert process.stdout is not None
        for line in process.stdout:
            if on_line:
                on_line(line.rstrip())
        code = process.wait()
        if code:
            raise BackendError("模型导入失败（退出码 %d）" % code)
        self.set_llm_model(name, require_installed=False)
        return self.model_catalog()

    def _stop_owned_ollama(self, timeout_seconds: float = 4.0) -> None:
        """Stop only the Ollama service started by this backend, including children.

        On Windows, ``ollama serve`` launches one or more ``llama-server.exe``
        children.  Terminating only the parent leaves those model runners alive and
        repeated desktop/evaluation launches eventually exhaust system memory.  ``/T``
        is required to terminate the owned process tree.  When an already-running
        external Ollama service was reused, ``_ollama_process`` is ``None`` and this
        method deliberately does nothing.
        """
        process, self._ollama_process = self._ollama_process, None
        if process is None or process.poll() is not None:
            return

        timeout_seconds = max(0.1, float(timeout_seconds))
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(5.0, timeout_seconds + 1.0),
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode == 0:
                    try:
                        process.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        pass
                    return
            except (OSError, subprocess.SubprocessError):
                # Fall back to the portable parent-only stop.  A taskkill failure is
                # still preferable to keeping the owned parent alive indefinitely.
                pass

        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        self._stop_owned_ollama(timeout_seconds=4)
        if self._ollama_log_handle is not None:
            try:
                self._ollama_log_handle.close()
            except OSError:
                pass
            self._ollama_log_handle = None

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "bundle_root": str(bundled_root()),
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "ollama_executable": self.ollama_path() or "",
            "vc_redist_executable": self.vc_redist_path() or "",
            "llm_model": self.webui.M.LLM_MODEL,
            "models_directory": os.environ.get("OLLAMA_MODELS", ""),
        }
