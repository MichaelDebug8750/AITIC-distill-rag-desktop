"""Regression contract for bounded, restart-safe rich desktop sessions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.backend import DesktopBackend


def backend_for(root: Path) -> DesktopBackend:
    backend = DesktopBackend.__new__(DesktopBackend)
    backend.project_root = root
    return backend


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aitic-session-contract-") as temp:
        root = Path(temp)
        backend = backend_for(root)
        payload = {
            "answer": "Python 是编程语言。[p.1]",
            "abstained": False,
            "sources": [{"label": "Book · p1", "snippet": "Python language"}],
            "agent": {
                "rounds": 2,
                "stop_reason": "补充检索与引用校准完成",
                "trace": [{"step": "证据检索", "detail": "召回 1 条"}],
                "confidence": {"level": "高", "signals": []},
            },
        }
        sessions = [{
            "id": "session-1", "title": "Python", "updated_at": "2026-08-18T00:00:00",
            "pinned": True, "library_ids": ["book-1"],
            "messages": [
                {"role": "user", "content": "什么是 Python？", "meta": "", "sources": []},
                {"role": "assistant", "content": payload["answer"], "meta": "123 ms",
                 "sources": payload["sources"], "payload": payload, "favorite": True},
            ],
        }]
        saved = backend.save_sessions(sessions)
        assert saved["saved"] == 1
        restored = backend.load_sessions()["sessions"]
        assistant = restored[0]["messages"][-1]
        assert assistant["favorite"] is True
        assert assistant["payload"]["agent"]["rounds"] == 2
        assert assistant["payload"]["sources"][0]["label"] == "Book · p1"

        # Non-JSON values and runaway data must not make all session saving fail.
        sessions[0]["messages"][-1]["payload"]["unsupported"] = object()
        sessions[0]["messages"][-1]["payload"]["huge"] = "x" * 700_000
        backend.save_sessions(sessions)
        disk = json.loads(backend.sessions_path.read_text(encoding="utf-8"))
        json.dumps(disk, ensure_ascii=False, allow_nan=False)
        restored = backend.load_sessions()["sessions"]
        assert len(restored) == 1 and len(restored[0]["messages"]) == 2

        backend.sessions_path.write_text("{broken", encoding="utf-8")
        assert backend.load_sessions() == {"sessions": []}

    print("RICH_SESSION_RESTART=PASS")
    print("SESSION_JSON_BOUNDS=PASS")
    print("CORRUPTED_SESSION_RECOVERY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
