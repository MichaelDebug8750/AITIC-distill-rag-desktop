"""验证原生诊断页的反馈列表与回归集导出数据契约。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from desktop_app.main_window import MainWindow


class FakeBackend:
    project_root = ROOT

    def __init__(self):
        self.saved_sessions = []

    def libraries(self):
        return {"libraries": [], "active_id": ""}

    def status(self):
        return {"ready": False, "chunks": 0, "ollama": "测试"}

    def runtime_summary(self):
        return {"project_root": str(ROOT), "frozen": False}

    def model_catalog(self):
        return {
            "connected": False, "available": [], "active": "qwen3:8b",
            "vision_model": "qwen3-vl:8b", "embedding_model": "bge-m3",
            "models_directory": "", "presets": [
                {"model": "qwen3:4b", "label": "Qwen3 4B · 轻量版",
                 "description": "轻量", "installed": False, "active": False},
                {"model": "qwen3:8b", "label": "Qwen3 8B · 推荐版",
                 "description": "推荐", "installed": True, "active": True},
                {"model": "qwen3:14b", "label": "Qwen3 14B · 高质量版",
                 "description": "高质量", "installed": False, "active": False},
            ],
        }

    def needs_model_setup(self):
        return False

    def load_sessions(self, _limit: int = 80):
        return {"sessions": []}

    def save_sessions(self, sessions):
        self.saved_sessions = list(sessions)
        return {"saved": len(self.saved_sessions)}

    def feedback_list(self, _limit: int = 100):
        # This is the real /api/feedback contract; the key is intentionally `recent`.
        return {"total": 1, "failures": 0, "recent": [{
            "time": "2026-08-17 11:00:00", "kind": "helpful", "kind_label": "有用",
            "question": "测试问题", "libraries": ["lib-a"], "answer": "测试回答",
        }]}

    def export_regression(self):
        return {"count": 1, "jsonl": '{"question":"测试问题"}\n'}

    def ask_stream(self, _question, **kwargs):
        callback = kwargs.get("on_event")
        if callback:
            callback("retrieved", {"n": 1})
            callback("agent", {"round": 2, "label": "正在补充检索并核验…"})
        return {
            "answer": "**Python** 是编程语言。[p.1]", "abstained": False,
            "elapsed_ms": 123, "tokens": 45, "escalated": True,
            "cite_check": {"ok": True, "total": 1, "hit": ["p.1"]},
            "sources": [{"label": "Book · p1", "type": "text",
                         "snippet": "Python is a programming language."}],
            "agent": {
                "rounds": 2, "stop_reason": "证据已完整",
                "support_audit": {"triggered": True, "state": "verified",
                                  "checked": 1, "pruned": 0, "unknown": 0,
                                  "reason": "逐句核验通过"},
                "confidence": {"level": "高", "reason": "引用与原文一致",
                               "signals": [{"name": "引用命中", "ok": True,
                                            "detail": "1/1"}]},
                "trace": [{"step": "证据检索", "detail": "召回 1 条"}],
                "evidence_chain": {"basis": [{"claim": "Python 是编程语言。",
                                                "supported": True,
                                                "citations": ["p.1"]}],
                                   "uncertainty": []},
            },
        }


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("AITIC")
    app.setApplicationName("AITIC Desktop UI Contract Smoke")
    backend = FakeBackend()
    window = MainWindow(backend)
    button = QPushButton()

    with tempfile.TemporaryDirectory(prefix="aitic-ui-contract-") as temp:
        export_path = Path(temp) / "feedback.jsonl"
        original_dialog = QFileDialog.getSaveFileName
        original_information = QMessageBox.information
        QFileDialog.getSaveFileName = staticmethod(
            lambda *_args, **_kwargs: (str(export_path), "JSON Lines (*.jsonl)"))
        QMessageBox.information = staticmethod(lambda *_args, **_kwargs: QMessageBox.Ok)
        state = {"export_started": False, "ticks": 0, "error": ""}

        def poll() -> None:
            state["ticks"] += 1
            if state["ticks"] > 500:
                state["error"] = "feedback UI contract smoke timed out"
                app.quit()
                return
            if window.feedback_table.rowCount() == 1 and not state["export_started"]:
                state["export_started"] = True
                window._export_feedback(button)
            model_ready = "qwen3:8b" in window.model_status.toPlainText()
            if (state["export_started"] and export_path.is_file()
                    and model_ready and not window._tasks):
                app.quit()
                return
            QTimer.singleShot(10, poll)

        window._refresh_feedback(button)
        QTimer.singleShot(10, poll)
        app.exec()
        QFileDialog.getSaveFileName = original_dialog
        QMessageBox.information = original_information

        if state["error"]:
            raise AssertionError(state["error"])
        assert window.feedback_table.rowCount() == 1
        assert window.feedback_table.item(0, 1).text() == "有用"
        assert window.feedback_table.item(0, 2).text() == "测试问题"
        assert window.feedback_table.item(0, 3).text() == "lib-a"
        assert export_path.read_text(encoding="utf-8") == '{"question":"测试问题"}\n'
        assert [window.model_combo.itemData(i) for i in range(window.model_combo.count())] == [
            "qwen3:4b", "qwen3:8b", "qwen3:14b"]
        assert "qwen3:8b" in window.model_status.toPlainText()
        button_labels = {item.text() for item in window.findChildren(QPushButton)}
        assert {"一键配置推荐环境", "一键导入本地模型", "模型存储目录",
                "配置教程", "修复 VC++ 运行库", "批量导入 PDF / EPUB",
                "检索设置", "专注模式", "复制回答", "收藏", "重新生成"}.issubset(button_labels)
        assert window.nav.count() == 6
        assert window.stack.count() == 12
        assert set(window._page_indexes) == {
            "chat", "library", "evidence", "chunks", "health", "settings",
            "status", "learning", "batch", "compare", "concept", "feedback"}
        window._fit_navigation()
        assert window.nav.height() < 332
        assert hasattr(window, "_open_retrieval_settings")
        assert hasattr(window, "_show_session_metrics")
        app_source = (ROOT / "desktop_app" / "app.py").read_text(encoding="utf-8")
        style_source = (ROOT / "desktop_app" / "style.py").read_text(encoding="utf-8")
        assert "PreferFullHinting" not in app_source
        assert "PreferVerticalHinting" in app_source
        assert "setDevicePixelRatio" in app_source
        assert "devicePixelRatio" in app_source
        assert 'font-family: "Microsoft YaHei UI"' not in style_source
        assert window.mode_combo.objectName() == "CrispComboBox"
        assert "<h2>" not in window._welcome_html
        window._toggle_focus_mode(True)
        assert not window.sidebar.isVisible()
        window._toggle_focus_mode(False)
        window._clear_chat()
        window._append_chat("user", "什么是 AIGC？")
        window._append_chat("assistant", "AIGC 是由 AI 生成内容的系统。", "12 ms · 20 token")
        window._show_sources([{"label": "p.8", "snippet": "AI-generated contents"}])
        assert window.session_list.count() == 1
        assert backend.saved_sessions[0]["title"] == "什么是 AIGC？"
        assert backend.saved_sessions[0]["messages"][-1]["sources"][0]["label"] == "p.8"
        # Exercise the actual production result handler.  Calling
        # _append_rich_answer directly previously let an override silently flatten
        # the payload to answer-only text without this contract noticing.
        window._clear_chat()
        window.question_edit.setPlainText("什么是 Python？")
        window._send_chat()
        send_state = {"ticks": 0}

        def wait_for_answer() -> None:
            send_state["ticks"] += 1
            if not window._tasks and not window._chat_busy:
                app.quit()
                return
            if send_state["ticks"] > 500:
                app.quit()
                return
            QTimer.singleShot(10, wait_for_answer)

        QTimer.singleShot(10, wait_for_answer)
        app.exec()
        assert not window._tasks and not window._chat_busy
        rendered = window.chat_view.toPlainText()
        assert "Agent 运行步骤" in rendered
        assert "逐句语义核验" in rendered
        assert "本轮检索证据" in rendered
        assert "可信度：高" in rendered
        assert "**Python**" not in rendered and "Python" in rendered
        latest = backend.saved_sessions[0]["messages"][-1]
        assert latest["payload"]["agent"]["rounds"] == 2
        assert latest["sources"][0]["label"] == "Book · p1"
        assert window.copy_answer_button.isEnabled()
        assert window.favorite_button.isEnabled()
        assert window.regenerate_button.isEnabled()
        window._chat_busy = True
        window.send_button.setEnabled(False)
        window._clear_chat()
        assert not window._chat_busy and window.send_button.isEnabled()

    print("FEEDBACK_TABLE_ROWS=1")
    print("FEEDBACK_KIND_LABEL=有用")
    print("FEEDBACK_EXPORT=PASS")
    print("MODEL_MANAGEMENT_UI=PASS")
    print("RICH_DESKTOP_NAVIGATION=PASS")
    print("WEB_PARITY_CONTROLS=PASS")
    print("HIGH_DPI_SIDEBAR_LAYOUT=PASS")
    print("NATIVE_FONT_RASTERIZATION=PASS")
    print("LOCAL_SESSION_PERSISTENCE=PASS")
    print("RICH_RESULT_PAYLOAD_PARITY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
