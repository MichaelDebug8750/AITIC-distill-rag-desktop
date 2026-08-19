"""Create an EPUB fixture and prove it reaches the validated desktop build pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from ebooklib import epub


def make_epub(path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("aitic-epub-smoke-v1")
    book.set_title("AITIC EPUB 导入验证教材")
    book.set_language("zh-CN")
    book.add_author("AITIC QA")
    chapter = epub.EpubHtml(title="第一章", file_name="chapter.xhtml", lang="zh-CN")
    chapter.content = """<html><body><h1>第一章 可追溯问答</h1>
    <p>蓝杉协议是一种用于验证 EPUB 多选导入的测试协议。</p>
    <p>协议要求每条回答都必须引用教材证据，缺少依据时明确拒答。</p>
    <p>The Blue Cedar protocol verifies searchable EPUB ingestion.</p>
    </body></html>"""
    book.add_item(chapter)
    book.toc = (epub.Link("chapter.xhtml", "第一章", "chapter"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    sys.path[:0] = [str(source_root), str(source_root / "code")]
    os.environ["AITIC_PROJECT_ROOT"] = str(Path(args.project_root).resolve())
    from desktop_app.backend import DesktopBackend

    profile = Path(args.project_root).resolve()
    fixture = profile / "books" / "AITIC_EPUB_import_smoke.epub"
    make_epub(fixture)
    backend = DesktopBackend(profile)
    try:
        started = backend.start_build(str(fixture), max_pages=0, use_vl=True, vl_limit=15)
        job = started.get("job") or {}
        job_id = str(job.get("id") or "")
        deadline = time.monotonic() + 900
        while job_id and time.monotonic() < deadline:
            job = backend.build_status(job_id)
            if job.get("status") in ("ready", "completed", "failed"):
                break
            time.sleep(0.25)
        library_id = str(job.get("library_id") or "")
        libraries = backend.libraries().get("libraries") or []
        library = next((item for item in libraries if str(item.get("id")) == library_id), {})
        chunks = backend.library_chunks(library_id, "蓝杉协议", 10, 0) if library_id else {}
        texts = "\n".join(str(item.get("text") or "") for item in chunks.get("chunks") or [])
        report = {
            "ok": job.get("status") in ("ready", "completed") and "蓝杉协议" in texts,
            "fixture": str(fixture),
            "job": job,
            "library": library,
            "retrieved_chunks": len(chunks.get("chunks") or []),
            "text_found": "蓝杉协议" in texts,
            "epub_alias_preserved": str(library.get("source") or "").lower().endswith(".epub"),
        }
        if not report["ok"] or not report["epub_alias_preserved"]:
            raise RuntimeError("EPUB end-to-end smoke failed: %s" % report)
    finally:
        backend.close()
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
