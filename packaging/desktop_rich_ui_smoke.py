"""用真实知识库穿过完整原生桌面 UI 的关键交互链路。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AITIC_DESKTOP_SMOKE_REPORT", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from desktop_app.backend import DesktopBackend
from desktop_app.main_window import MainWindow
from desktop_app.style import APP_STYLE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--query", default="什么是AIGC")
    parser.add_argument("--study-topic", default="")
    parser.add_argument("--report", required=True)
    parser.add_argument("--screenshot", default="")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    os.environ["AITIC_PROJECT_ROOT"] = str(Path(args.project_root).resolve())
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("AITIC")
    app.setApplicationName("AITIC Desktop Rich UI Smoke")
    app.setStyle("Fusion")
    # Match the production entry point.  Without an explicit system UI font the
    # offscreen Qt plugin may fall back to a Latin-only bitmap face, making every
    # CJK glyph a square and invalidating visual QA even though the installed app
    # is fine.
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    cjk_fallbacks = ("Microsoft YaHei UI", "Microsoft YaHei", "SimSun")
    fallback = next((name for name in cjk_fallbacks if QFontDatabase.hasFamily(name)), "")
    if fallback and fallback.casefold() != font.family().casefold():
        font.setFamilies([font.family(), fallback])
    font.setPointSizeF(10.0)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferDefault)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE)
    backend = DesktopBackend(args.project_root)
    saved_sessions: list[dict] = []
    backend.load_sessions = lambda _limit=80: {"sessions": []}

    def save_sessions(values):
        saved_sessions[:] = list(values)
        return {"saved": len(saved_sessions)}

    backend.save_sessions = save_sessions
    original_information = QMessageBox.information
    original_critical = QMessageBox.critical
    dialogs: list[str] = []
    QMessageBox.information = staticmethod(
        lambda _parent, title, message, *_args, **_kwargs:
        (dialogs.append("%s: %s" % (title, message)), QMessageBox.Ok)[1])
    QMessageBox.critical = staticmethod(
        lambda _parent, title, message, *_args, **_kwargs:
        (dialogs.append("ERROR %s: %s" % (title, message)), QMessageBox.Ok)[1])

    window = MainWindow(backend)
    window.resize(1440, 900)
    window.show()
    state = {"stage": "startup", "started": time.monotonic(), "error": ""}
    report: dict[str, object] = {
        "ok": False, "navigation": window.nav.count(), "pages": window.stack.count(),
        "page_keys": list(window._page_indexes),
    }

    def fail(message: str) -> None:
        state["error"] = message
        app.quit()

    def poll() -> None:
        if time.monotonic() - state["started"] > args.timeout:
            fail("rich UI smoke timed out at %s" % state["stage"])
            return
        if window._tasks or window._chat_busy:
            QTimer.singleShot(80, poll)
            return
        try:
            stage = state["stage"]
            if stage == "startup":
                if window.library_picker.count() < 1:
                    QTimer.singleShot(80, poll)
                    return
                window._set_selected_libraries([args.library])
                for key in window._page_indexes:
                    window._show_page(key)
                    if window.stack.currentIndex() != window._page_indexes[key]:
                        raise AssertionError("navigation failed: %s" % key)
                window._show_page("evidence")
                window.retrieve_input.setText(args.query)
                state["stage"] = "retrieve"
                window._run_retrieve(QPushButton())
            elif stage == "retrieve":
                report["evidence_rows"] = window.retrieve_table.rowCount()
                if window.retrieve_table.rowCount() < 1:
                    raise AssertionError("evidence review returned no rows")
                index = window.chunk_library_combo.findData(args.library)
                window.chunk_library_combo.setCurrentIndex(index)
                window.chunk_limit.setValue(5)
                window.chunk_query.clear()
                state["stage"] = "chunks"
                window._load_chunks(QPushButton())
            elif stage == "chunks":
                report["chunk_rows"] = window.chunk_table.rowCount()
                if window.chunk_table.rowCount() != 5:
                    raise AssertionError("chunk browser did not return 5 rows")
                index = window.health_library_combo.findData(args.library)
                window.health_library_combo.setCurrentIndex(index)
                window.health_sample.setValue(50)
                state["stage"] = "health"
                window._run_health(QPushButton())
            elif stage == "health":
                health_text = window.health_output.toPlainText()
                report["health_rendered"] = args.library in health_text or "诊断结果" in health_text
                if not report["health_rendered"]:
                    raise AssertionError("health report was not rendered")
                window._show_page("chat")
                window.question_edit.setPlainText(args.query)
                state["stage"] = "ask"
                window._send_chat()
            elif stage == "ask":
                report["answer"] = window._latest_answer
                report["answer_has_citation"] = "[p." in window._latest_answer
                report["source_rows"] = window.source_tree.topLevelItemCount()
                report["session_rows"] = window.session_list.count()
                rendered = window.chat_view.toPlainText()
                required_rich_sections = (
                    "Agent 运行步骤", "逐句语义核验", "本轮检索证据", "可信度：")
                report["rich_sections"] = {
                    label: label in rendered for label in required_rich_sections}
                if not window._latest_answer or not report["answer_has_citation"]:
                    raise AssertionError("answer or inline citation is missing")
                if window.source_tree.topLevelItemCount() < 1 or not saved_sessions:
                    raise AssertionError("source drawer or local session was not populated")
                missing = [label for label in required_rich_sections if label not in rendered]
                if missing:
                    raise AssertionError("rich Agent result sections are missing: %s" % missing)
                assistant_messages = [message for message in
                                      (saved_sessions[0].get("messages") or [])
                                      if message.get("role") == "assistant"]
                if not assistant_messages or not isinstance(
                        assistant_messages[-1].get("payload"), dict):
                    raise AssertionError("rich Agent payload was not saved in the session")
                report["session_payload_saved"] = True
                study_topic = args.study_topic.strip() or args.query.strip()
                window.brief_topic.setText(study_topic)
                state["stage"] = "brief"
                window._run_brief(QPushButton())
            elif stage == "brief":
                report["brief_rendered"] = len(window.brief_output.toPlainText().strip()) > 20
                if not report["brief_rendered"]:
                    raise AssertionError("brief output is empty")
                window.quiz_topic.setText(args.study_topic.strip() or args.query.strip())
                window.quiz_count.setValue(2)
                state["stage"] = "quiz"
                window._run_quiz(QPushButton())
            elif stage == "quiz":
                report["quiz_rows"] = window.quiz_tree.topLevelItemCount()
                if window.quiz_tree.topLevelItemCount() < 2:
                    raise AssertionError("quiz did not render two questions")
                if args.screenshot:
                    target = Path(args.screenshot).resolve()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    window._show_page("chat")
                    report["screenshot_saved"] = window.grab().save(str(target), "PNG")
                report["saved_session_messages"] = len(saved_sessions[0].get("messages") or [])
                report["dialogs"] = dialogs
                report["ok"] = True
                app.quit()
                return
        except Exception as exc:
            fail("%s: %s" % (state["stage"], exc))
            return
        QTimer.singleShot(80, poll)

    QTimer.singleShot(100, poll)
    app.exec()
    QMessageBox.information = original_information
    QMessageBox.critical = original_critical
    backend.close()
    if state["error"]:
        report["error"] = state["error"]
    report["elapsed_seconds"] = round(time.monotonic() - state["started"], 2)
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
