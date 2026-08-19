"""Drive the native one-click model-import action against the real local Ollama."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    sys.path[:0] = [str(source_root), str(source_root / "code")]
    profile = Path(args.project_root).resolve()
    os.environ["AITIC_PROJECT_ROOT"] = str(profile)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
    from desktop_app.backend import DesktopBackend
    from desktop_app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("AITIC")
    app.setApplicationName("AITIC Live Model Import Smoke")
    backend = DesktopBackend(profile)
    previous = str(backend.model_catalog().get("active") or "qwen3:8b")
    test_name = "aitic-import-smoke:latest"
    executable = backend.ollama_path()
    if not executable:
        raise RuntimeError("Ollama runtime not found")
    fixture_dir = profile / "data" / "model_import_live_smoke"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    modelfile = fixture_dir / "Modelfile"
    modelfile.write_text("FROM %s\nPARAMETER temperature 0\n" % previous, encoding="utf-8")
    subprocess.run([executable, "rm", test_name], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    window = MainWindow(backend)
    original_file = QFileDialog.getOpenFileName
    original_name = QInputDialog.getText
    original_info = QMessageBox.information
    original_critical = QMessageBox.critical
    dialogs: list[str] = []
    QFileDialog.getOpenFileName = staticmethod(
        lambda *_args, **_kwargs: (str(modelfile), "Modelfile (Modelfile)"))
    QInputDialog.getText = staticmethod(lambda *_args, **_kwargs: (test_name, True))
    QMessageBox.information = staticmethod(
        lambda _parent, title, message, *_args, **_kwargs:
        (dialogs.append("%s: %s" % (title, message)), QMessageBox.Ok)[1])
    QMessageBox.critical = staticmethod(
        lambda _parent, title, message, *_args, **_kwargs:
        (dialogs.append("ERROR %s: %s" % (title, message)), QMessageBox.Ok)[1])
    state = {"started": time.monotonic(), "done": False, "error": ""}
    window._import_local_model(window.model_import_button)

    def poll() -> None:
        if time.monotonic() - state["started"] > 300:
            state["error"] = "native model import timed out"
            app.quit()
            return
        available = set(backend.model_inventory().get("available_normalized") or [])
        if test_name in available and not window._tasks:
            state["done"] = True
            app.quit()
            return
        QTimer.singleShot(200, poll)

    QTimer.singleShot(200, poll)
    app.exec()
    QFileDialog.getOpenFileName = original_file
    QInputDialog.getText = original_name
    QMessageBox.information = original_info
    QMessageBox.critical = original_critical
    created = test_name in set(backend.model_inventory().get("available_normalized") or [])
    active_after_import = str(backend.model_catalog().get("active") or "")
    shown = subprocess.run(
        [executable, "show", test_name], capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0
    cleanup_error = ""
    try:
        backend.set_llm_model(previous, require_installed=True)
        removed = subprocess.run(
            [executable, "rm", test_name], capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if removed.returncode:
            cleanup_error = (removed.stdout or "model cleanup failed")[-500:]
    finally:
        remaining = test_name in set(backend.model_inventory().get("available_normalized") or [])
        restored = str(backend.model_catalog().get("active") or "") == previous
        backend.close()
        window.close()
    report = {
        "ok": bool(state["done"] and created and shown and active_after_import == test_name
                   and restored and not remaining and not cleanup_error and not state["error"]),
        "previous_model": previous,
        "test_model": test_name,
        "created": created,
        "ollama_show": shown,
        "active_after_import": active_after_import,
        "restored": restored,
        "removed": not remaining,
        "dialogs": dialogs,
        "ui_log_tail": window.model_log.toPlainText()[-1000:],
        "error": state["error"],
        "cleanup_error": cleanup_error,
    }
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise RuntimeError("live model import smoke failed: %s" % report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
