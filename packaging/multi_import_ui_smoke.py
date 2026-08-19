"""Prove that the native file picker queues every selected PDF/EPUB."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "packaging")]

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from desktop_app.main_window import MainWindow
from desktop_ui_contract_smoke import FakeBackend


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("AITIC")
    app.setApplicationName("AITIC Multi Import Smoke")
    backend = FakeBackend()
    calls: list[str] = []
    jobs: dict[str, str] = {}

    def start_build(path: str, **_options):
        job_id = "job-%d" % (len(calls) + 1)
        calls.append(path)
        jobs[job_id] = path
        return {"job": {"id": job_id, "status": "queued"}}

    def build_status(job_id: str):
        return {"id": job_id, "status": "ready", "phase": "完成", "progress": 100,
                "chunks": 1, "library_id": "lib-" + job_id}

    backend.start_build = start_build
    backend.build_status = build_status
    backend.active_build = lambda: {"job": None}
    window = MainWindow(backend)
    original_picker = QFileDialog.getOpenFileNames
    original_information = QMessageBox.information
    original_critical = QMessageBox.critical
    captured = {"filter": "", "ticks": 0, "error": ""}

    def picker(_parent, _title, _directory, file_filter):
        captured["filter"] = file_filter
        return ([r"C:\Books\first.pdf", r"C:\Books\second.epub"], file_filter)

    QFileDialog.getOpenFileNames = staticmethod(picker)
    QMessageBox.information = staticmethod(lambda *_args, **_kwargs: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *_args, **_kwargs: QMessageBox.Ok)
    button = window.import_button
    window._choose_and_build(button)

    def poll() -> None:
        captured["ticks"] += 1
        if captured["ticks"] > 1000:
            captured["error"] = "multi import UI smoke timed out"
            app.quit()
            return
        if window._build_job_id and not window._build_poll_pending:
            window._poll_build()
        if len(calls) == 2 and not window._build_job_id and not window._build_queue and button.isEnabled():
            app.quit()
            return
        QTimer.singleShot(10, poll)

    QTimer.singleShot(10, poll)
    app.exec()
    QFileDialog.getOpenFileNames = original_picker
    QMessageBox.information = original_information
    QMessageBox.critical = original_critical
    if captured["error"]:
        raise AssertionError(captured["error"])
    assert calls == [r"C:\Books\first.pdf", r"C:\Books\second.epub"]
    assert "*.pdf *.epub" in captured["filter"]
    assert "成功 2 本，失败 0 本" in window.build_progress.format()
    assert window.nav.height() < 332 and window.more_button.geometry().top() <= window.nav.geometry().bottom() + 18
    print("MULTI_SELECT_IMPORT=PASS")
    print("PDF_EPUB_FILTER=PASS")
    print("SERIAL_BUILD_QUEUE=PASS")
    print("MORE_BUTTON_POSITION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
