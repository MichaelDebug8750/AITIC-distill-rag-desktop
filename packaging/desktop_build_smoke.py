"""Exercise a clean desktop profile: import PDF, build, activate, retrieve, feedback."""

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


def make_fixture(path: Path) -> None:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    text = (
        "AITIC Desktop Build Fixture\n\n"
        "A dependency graph represents prerequisites as directed edges. "
        "A topological ordering places every prerequisite before the task that depends on it.\n\n"
        "Incremental builds compare targets with their prerequisites. A target is rebuilt when a "
        "prerequisite is newer, or when the target does not exist. This fixture is deliberately "
        "self-contained so retrieval has a deterministic concept to find.\n\n"
        "Phony targets name actions rather than files. Declaring a target phony prevents a file "
        "with the same name from suppressing the action."
    )
    page.insert_textbox(pymupdf.Rect(55, 60, 540, 780), text, fontsize=13, lineheight=1.45)
    document.save(path)
    document.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    profile = Path(args.profile).resolve()
    fixture = Path(args.fixture).resolve()
    if profile.exists() or fixture.exists():
        raise RuntimeError("Smoke profile and fixture must not exist before the run")
    profile.parent.mkdir(parents=True, exist_ok=True)
    make_fixture(fixture)

    os.environ["AITIC_PROJECT_ROOT"] = str(profile)
    backend = DesktopBackend(profile)
    first_libraries = backend.libraries()
    response = backend.start_build(str(fixture), max_pages=1, use_vl=False)
    job = response.get("job") or {}
    job_id = str(job.get("id") or "")
    deadline = time.monotonic() + max(10, args.timeout)
    while time.monotonic() < deadline:
        job = backend.build_status(job_id)
        if job.get("status") in ("ready", "failed"):
            break
        time.sleep(0.25)
    if job.get("status") != "ready":
        raise RuntimeError("Desktop build did not complete: %s" % job)

    library_id = str(job.get("library_id") or "")
    libraries = backend.libraries()
    active = str(libraries.get("active_id") or "")
    registry = backend.webui._read_registry()
    record = next(item for item in registry["libraries"] if str(item.get("id")) == library_id)
    copied = Path(backend.webui._resolve_db_ref(record.get("source_path")))
    retrieval = backend.retrieve("What is a phony target?", libraries=[library_id], limit=3)
    feedback = backend.feedback("useful", "What is a phony target?", "Validated fixture answer",
                                libraries=[library_id])
    feedback_list = backend.feedback_list(10)
    status = backend.status()

    report = {
        "ok": True,
        "initial_libraries": len(first_libraries.get("libraries") or []),
        "job_status": job.get("status"),
        "job_chunks": job.get("chunks"),
        "library_id": library_id,
        "active_library": active,
        "source_was_copied": copied.is_file() and copied != fixture,
        "source_copy_inside_profile": str(copied).lower().startswith(str(profile).lower()),
        "retrieval_sources": len(retrieval.get("sources") or []),
        "retrieval_llm_called": retrieval.get("llm_called"),
        "feedback_ok": feedback.get("ok"),
        "feedback_count": len(feedback_list.get("recent") or []),
        "status_ready": status.get("ready"),
        "status_chunks": status.get("chunks"),
        "uvicorn_imported": "uvicorn" in sys.modules,
    }
    report["ok"] = all((
        report["initial_libraries"] == 0,
        report["job_status"] == "ready", int(report["job_chunks"] or 0) > 0,
        report["active_library"] == library_id, report["source_was_copied"],
        report["source_copy_inside_profile"], report["retrieval_sources"] > 0,
        report["retrieval_llm_called"] is False, report["feedback_ok"],
        report["feedback_count"] == 1, report["status_ready"],
        report["status_chunks"] == report["job_chunks"], not report["uvicorn_imported"],
    ))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
