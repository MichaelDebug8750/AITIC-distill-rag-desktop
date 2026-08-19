"""Reproduce an installed-profile desktop Agent answer through the native backend.

This is a diagnostic runner, not a synthetic unit test: it exercises the same
in-process SSE path used by ``AITIC Desktop.exe`` and preserves the full final
payload so UI omissions can be distinguished from pipeline omissions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.backend import DesktopBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--library", action="append", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    profile = Path(args.profile).expanduser().resolve()
    if not profile.is_dir():
        raise FileNotFoundError(profile)
    os.environ["AITIC_PROJECT_ROOT"] = str(profile)
    events: list[dict] = []
    backend = DesktopBackend(profile)
    try:
        payload = backend.ask_stream(
            args.question,
            libraries=args.library,
            mode="auto",
            style="standard",
            on_event=lambda event, data: events.append({"event": event, "data": data}),
        )
    finally:
        backend.close()

    report = {"question": args.question, "libraries": args.library,
              "events": events, "result": payload}
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "answer": payload.get("answer"),
        "abstained": payload.get("abstained"),
        "cite_check": payload.get("cite_check"),
        "support_audit": (payload.get("agent") or {}).get("support_audit"),
        "confidence": (payload.get("agent") or {}).get("confidence"),
        "rounds": (payload.get("agent") or {}).get("rounds"),
        "claims": ((payload.get("agent") or {}).get("evidence_chain") or {}).get("basis"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
