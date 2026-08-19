"""Import real textbooks through the desktop backend in a clean user profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.backend import DesktopBackend


def wait_for_build(backend: DesktopBackend, job_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + max(10, timeout)
    job = {}
    while time.monotonic() < deadline:
        job = backend.build_status(job_id)
        if job.get("status") in ("ready", "failed"):
            return job
        time.sleep(0.25)
    raise TimeoutError("建库超时：%s" % job_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pdf", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--simulate-frozen-bundle", default="")
    args = parser.parse_args()

    profile = Path(args.profile).resolve()
    if profile.exists():
        raise RuntimeError("测试用户目录必须在运行前不存在：%s" % profile)
    sources = [Path(value).resolve() for value in args.pdf]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    os.environ["AITIC_PROJECT_ROOT"] = str(profile)
    backend = DesktopBackend(profile)
    if args.simulate_frozen_bundle:
        bundle = Path(args.simulate_frozen_bundle).resolve()
        backend.webui.M.__file__ = str(bundle / "virtual-pyz" / "main.py")
        backend.webui.sys._MEIPASS = str(bundle)

    records = []
    try:
        for source in sources:
            started = time.monotonic()
            response = backend.start_build(
                str(source), max_pages=max(0, args.max_pages), use_vl=False)
            initial = response.get("job") or {}
            job_id = str(initial.get("id") or "")
            job = wait_for_build(backend, job_id, args.timeout)
            library_id = str(job.get("library_id") or "")
            registry = backend.webui._read_registry()
            item = next((row for row in registry["libraries"]
                         if str(row.get("id")) == library_id), {})
            copied = Path(backend.webui._resolve_db_ref(item.get("source_path")))
            chunks = backend.library_chunks(library_id, limit=2)
            health = backend.library_health(library_id, sample=50)
            health_chunks = int(health.get("chunks_total") or 0)
            row = {
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "job_status": job.get("status"),
                "job_error": job.get("error"),
                "chunks": int(job.get("chunks") or 0),
                "library_id": library_id,
                "source_copied": copied.is_file() and copied != source,
                "browsed_chunks": len(chunks.get("chunks") or []),
                "health_ok": health_chunks == int(job.get("chunks") or 0),
                "health_sampled": int(health.get("sampled") or 0),
                "health_warnings": list(health.get("warnings") or []),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            row["ok"] = all((
                row["job_status"] == "ready", row["chunks"] > 0,
                row["source_copied"], row["browsed_chunks"] > 0, row["health_ok"],
            ))
            records.append(row)
            if not row["ok"]:
                break
    finally:
        backend.close()

    report = {
        "ok": len(records) == len(sources) and all(row["ok"] for row in records),
        "profile": str(profile),
        "max_pages": max(0, args.max_pages),
        "simulated_frozen_bundle": str(args.simulate_frozen_bundle or ""),
        "books": records,
    }
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
