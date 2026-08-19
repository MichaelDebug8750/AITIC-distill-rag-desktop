"""Native backend functional smoke without HTTP, browser, or Uvicorn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.backend import DesktopBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--library", default="")
    parser.add_argument("--query", default="GNU make 中的变量")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    os.environ["AITIC_PROJECT_ROOT"] = str(Path(args.project_root).resolve())
    backend = DesktopBackend(args.project_root)
    libraries = backend.libraries()
    ready = [item for item in libraries.get("libraries", []) if item.get("status") == "ready"]
    library_id = args.library or str(libraries.get("active_id") or "")
    if not library_id or not any(str(item.get("id")) == library_id for item in ready):
        raise RuntimeError("No ready library is available for the desktop functional smoke")

    status = backend.status()
    chunks = backend.library_chunks(library_id, limit=5)
    health = backend.library_health(library_id, sample=40)
    retrieval = backend.retrieve(args.query, libraries=[library_id], limit=3)
    sources = retrieval.get("sources") or []
    if not sources:
        raise RuntimeError("Retrieval returned no source")
    source = sources[0]
    source_blocks = backend.source_blocks(source)
    page_png = backend.source_page_png(source, dpi=96)
    inventory = backend.model_inventory()

    report = {
        "ok": True,
        "status": {"ready": status.get("ready"), "chunks": status.get("chunks")},
        "active_library": library_id,
        "ready_libraries": len(ready),
        "browse_total": chunks.get("total"),
        "browse_returned": len(chunks.get("chunks") or []),
        "health_total": health.get("chunks_total"),
        "health_sampled": health.get("sampled"),
        "retrieval": retrieval.get("retrieval"),
        "retrieval_sources": len(sources),
        "retrieval_llm_called": retrieval.get("llm_called"),
        "source_blocks": len(source_blocks.get("blocks") or []),
        "source_page_png": page_png.startswith(b"\x89PNG\r\n\x1a\n"),
        "source_page_bytes": len(page_png),
        "models_connected": inventory.get("connected"),
        "models_missing": inventory.get("missing"),
        "uvicorn_imported": "uvicorn" in sys.modules,
    }
    checks = (
        report["status"]["ready"], report["browse_total"] and report["browse_total"] > 0,
        report["browse_returned"] == 5, report["health_total"] == report["browse_total"],
        report["retrieval_sources"] > 0, report["retrieval_llm_called"] is False,
        report["source_blocks"] > 0, report["source_page_png"],
        report["models_connected"], not report["uvicorn_imported"],
    )
    report["ok"] = all(checks)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        target = Path(args.report).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
