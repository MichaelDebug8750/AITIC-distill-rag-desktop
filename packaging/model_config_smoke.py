"""Validate model presets, persistence, input validation, and local GGUF import."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.backend import BackendError, DEFAULT_LLM_MODEL, DesktopBackend, LLM_PRESETS


class FakeProcess:
    last_command: list[str] = []

    def __init__(self, command: list[str], **_kwargs):
        self.command = command
        type(self).last_command = list(command)
        self.stdout = iter(("creating model\n", "success\n"))

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class FailedServeProcess:
    returncode = 17

    def __init__(self, _command: list[str], **_kwargs):
        pass

    def poll(self):
        return self.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    old_root = os.environ.get("AITIC_PROJECT_ROOT")
    report: dict[str, object] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="aitic-model-smoke-") as temp:
            profile = Path(temp) / "profile"
            os.environ["AITIC_PROJECT_ROOT"] = str(profile)
            backend = DesktopBackend(profile)
            calls = {"inventory": 0}

            def inventory():
                calls["inventory"] += 1
                return {
                    "connected": True,
                    "available": ["qwen3:4b", "qwen3:8b", "qwen3:14b"],
                    "available_normalized": ["qwen3:4b", "qwen3:8b", "qwen3:14b"],
                    "missing": [],
                }

            backend.model_inventory = inventory
            catalog = backend.set_llm_model("qwen3:4b", require_installed=False)
            saved = json.loads(backend.config_path.read_text(encoding="utf-8"))
            invalid_rejected = False
            try:
                backend.set_llm_model("bad model & command", require_installed=False)
            except BackendError:
                invalid_rejected = True

            gguf = Path(temp) / "course model.gguf"
            gguf.write_bytes(b"GGUF-test-placeholder")
            imported_inventory = {
                "connected": True,
                "available": ["course-model:latest"],
                "available_normalized": ["course-model:latest", "course-model"],
                "missing": [],
            }
            backend.ollama_path = lambda: str(Path(temp) / "ollama.exe")
            backend.ensure_ollama_running = lambda wait_seconds=8: {"connected": True}
            backend.model_inventory = lambda: imported_inventory
            lines: list[str] = []
            with patch("desktop_app.backend.subprocess.Popen", FakeProcess):
                imported = backend.import_model(str(gguf), "course-model:latest", lines.append)
            command = list(FakeProcess.last_command)
            modelfile = Path(command[command.index("-f") + 1])
            modelfile_text = modelfile.read_text(encoding="utf-8")

            backend._ollama_process = None
            backend.ensure_ollama_running = DesktopBackend.ensure_ollama_running.__get__(
                backend, DesktopBackend)
            backend._ollama_api_ready = lambda timeout=0.7: False
            with patch("desktop_app.backend.subprocess.Popen", FailedServeProcess):
                startup_failure = backend.ensure_ollama_running(wait_seconds=0.2)

            report = {
                "ok": True,
                "preset_count": len(LLM_PRESETS),
                "preset_models": [item["model"] for item in LLM_PRESETS],
                "default_model": DEFAULT_LLM_MODEL,
                "active_after_switch": catalog.get("active"),
                "saved_model": saved.get("llm_model"),
                "inventory_calls_for_unchecked_switch": calls["inventory"],
                "invalid_name_rejected": invalid_rejected,
                "import_command": command[1:3],
                "import_active": imported.get("active"),
                "gguf_modelfile_quoted": 'FROM "' in modelfile_text and "course model.gguf" in modelfile_text,
                "import_progress_lines": len(lines),
                "startup_failure_actionable": (
                    startup_failure.get("connected") is False
                    and "17" in str(startup_failure.get("error"))
                    and "VC++" in str(startup_failure.get("error"))),
                "startup_failure": startup_failure,
            }
            report["ok"] = all((
                report["preset_count"] == 3,
                report["active_after_switch"] == "qwen3:4b",
                report["saved_model"] == "qwen3:4b",
                report["inventory_calls_for_unchecked_switch"] == 1,
                report["invalid_name_rejected"],
                report["import_command"] == ["create", "course-model:latest"],
                report["import_active"] == "course-model:latest",
                report["gguf_modelfile_quoted"],
                report["import_progress_lines"] == 2,
                report["startup_failure_actionable"],
            ))
            backend.close()
    finally:
        if old_root is None:
            os.environ.pop("AITIC_PROJECT_ROOT", None)
        else:
            os.environ["AITIC_PROJECT_ROOT"] = old_root

    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
