"""AITIC Desktop 主窗口。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
from datetime import datetime
from typing import Any, Callable
import uuid

from PySide6.QtCore import QSettings, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .backend import (
    DEFAULT_LLM_MODEL, EMBEDDING_MODEL, LLM_PRESETS, VISION_MODEL,
    DesktopBackend, REQUIRED_MODELS,
)
from .ui_common import (
    CrispComboBox, QuestionEdit, SourceDialog, answer_html, card_layout, escaped,
    page_header, show_error, text_html,
)
from .workers import Task


class _CoreMainWindow(QMainWindow):
    def __init__(self, backend: DesktopBackend):
        super().__init__()
        self.backend = backend
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(3)
        self._tasks: set[Task] = set()
        self._busy_count = 0
        self._libraries: list[dict[str, Any]] = []
        self._last_status: dict[str, Any] = {}
        self._active_library = ""
        self._updating_picker = False
        self._chat_generation = 0
        self._chat_busy = False
        self._chat_parts: list[str] = []
        self._chat_sources: list[dict[str, Any]] = []
        self._history: list[dict[str, str]] = []
        self._latest_question = ""
        self._latest_answer = ""
        self._latest_payload: dict[str, Any] = {}
        self._latest_libraries: list[str] = []
        self._build_job_id = ""
        self._selected_pdf = ""
        self.setWindowTitle("AITIC Desktop · 本地教材智能体")
        self.setMinimumSize(1120, 720)
        self.resize(1420, 900)
        self._build_shell()
        self._restore_window()
        QTimer.singleShot(0, self.refresh_all)
        if not os.environ.get("AITIC_DESKTOP_SMOKE_REPORT"):
            QTimer.singleShot(1400, self._maybe_first_model_setup)

    # ----------------------------- 壳层与异步 -----------------------------
    def _build_shell(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(17, 20, 17, 16)
        side.setSpacing(10)
        brand = QLabel("AITIC Desktop")
        brand.setObjectName("Brand")
        brand_sub = QLabel("离线教材智能体 · 原生版")
        brand_sub.setObjectName("BrandSub")
        side.addWidget(brand)
        side.addWidget(brand_sub)
        side.addSpacing(14)

        self.nav = QListWidget()
        self.nav.setObjectName("Navigation")
        self.nav.setFocusPolicy(Qt.NoFocus)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        for label in ("对话问答", "学习工具", "知识库", "批量评测", "诊断与设置"):
            self.nav.addItem(QListWidgetItem(label))
        self.nav.setFixedHeight(235)
        side.addWidget(self.nav)

        picker_title = QLabel("本轮使用的教材（最多 4 本）")
        picker_title.setObjectName("BrandSub")
        side.addWidget(picker_title)
        self.library_picker = QListWidget()
        self.library_picker.setObjectName("LibraryPicker")
        self.library_picker.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.library_picker.setTextElideMode(Qt.ElideRight)
        self.library_picker.itemChanged.connect(self._picker_changed)
        side.addWidget(self.library_picker, 1)
        self.picker_hint = QLabel("未勾选时使用当前活动库")
        self.picker_hint.setObjectName("BrandSub")
        self.picker_hint.setWordWrap(True)
        side.addWidget(self.picker_hint)
        self.side_status = QLabel("正在检查本地环境…")
        self.side_status.setObjectName("BrandSub")
        self.side_status.setWordWrap(True)
        side.addWidget(self.side_status)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 18, 24, 12)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._make_chat_page())
        self.stack.addWidget(self._make_study_page())
        self.stack.addWidget(self._make_library_page())
        self.stack.addWidget(self._make_batch_page())
        self.stack.addWidget(self._make_diagnostics_page())
        content_layout.addWidget(self.stack)

        layout.addWidget(sidebar)
        layout.addWidget(content, 1)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.busy_bar = QProgressBar()
        self.busy_bar.setMaximumWidth(145)
        self.busy_bar.hide()
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.busy_bar)

    def run_task(self, fn: Callable[[], Any], on_result: Callable[[Any], None] | None = None,
                 *, label: str = "正在处理…", button: QPushButton | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_finished: Callable[[], None] | None = None) -> Task:
        task = Task(fn)
        self._tasks.add(task)
        self._busy_count += 1
        self.busy_bar.setRange(0, 0)
        self.busy_bar.show()
        self.status_label.setText(label)
        if button:
            button.setEnabled(False)

        if on_result:
            task.signals.result.connect(on_result)

        def error(message: str) -> None:
            if on_error:
                on_error(message)
            else:
                show_error(self, "操作失败", message)

        def finished() -> None:
            self._tasks.discard(task)
            self._busy_count = max(0, self._busy_count - 1)
            if button:
                button.setEnabled(True)
            if on_finished:
                on_finished()
            if not self._busy_count:
                self.busy_bar.hide()
                self.status_label.setText("就绪")

        task.signals.error.connect(error)
        task.signals.finished.connect(finished)
        self.pool.start(task)
        return task

    def refresh_all(self) -> None:
        self.run_task(self.backend.libraries, self._apply_libraries, label="正在读取知识库…")
        self.run_task(self.backend.status, self._apply_status, label="正在检查 Ollama…")

    # ----------------------------- 共用状态 -----------------------------
    def _apply_libraries(self, payload: dict[str, Any]) -> None:
        self._libraries = list(payload.get("libraries") or [])
        self._active_library = str(payload.get("active_id") or "")
        previous = set(self.selected_libraries())
        self._updating_picker = True
        self.library_picker.clear()
        for library in self._libraries:
            if library.get("status") != "ready":
                continue
            item = QListWidgetItem(str(library.get("name") or library.get("id")))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = str(library.get("id")) in previous
            if not previous and library.get("active"):
                checked = True
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setData(Qt.UserRole, dict(library))
            item.setToolTip("%s\n%s 块" % (library.get("source", ""), library.get("chunks", "?")))
            self.library_picker.addItem(item)
        self._updating_picker = False
        self._update_picker_hint()
        self._fill_library_table()

    def _apply_status(self, status: dict[str, Any]) -> None:
        self._last_status = dict(status)
        ready = bool(status.get("ready"))
        chunks = int(status.get("chunks") or 0)
        ollama = status.get("ollama", "未知")
        self.side_status.setText("● %s · %s 个知识块" % (ollama, chunks))
        self.side_status.setStyleSheet("color:%s" % ("#78d5a8" if ready else "#f2b9a7"))
        self._render_system_status(status)

    def selected_libraries(self) -> list[str]:
        selected = []
        for index in range(self.library_picker.count()):
            item = self.library_picker.item(index)
            if item.checkState() == Qt.Checked:
                library = item.data(Qt.UserRole) or {}
                selected.append(str(library.get("id") or ""))
        return [value for value in selected if value]

    def _picker_changed(self, changed: QListWidgetItem) -> None:
        if self._updating_picker:
            return
        selected = self.selected_libraries()
        if len(selected) > 4:
            self._updating_picker = True
            changed.setCheckState(Qt.Unchecked)
            self._updating_picker = False
            QMessageBox.information(self, "最多四本", "单次最多选择 4 本教材。")
        self._update_picker_hint()

    def _update_picker_hint(self) -> None:
        count = len(self.selected_libraries())
        self.picker_hint.setText("已选择 %d 本" % count if count else "未勾选时使用当前活动库")

    # ----------------------------- 对话 -----------------------------
    def _make_chat_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(page_header("对话问答", "直接查询本地教材；回答、拒答、引用和可信度均沿用已验证链路。"))

        controls = QHBoxLayout()
        self.mode_combo = CrispComboBox()
        self.mode_combo.addItem("自动", "auto")
        self.mode_combo.addItem("快速", "fast")
        self.mode_combo.addItem("深入", "deep")
        self.style_combo = CrispComboBox()
        self.style_combo.addItem("标准回答", "standard")
        self.style_combo.addItem("简洁", "concise")
        self.style_combo.addItem("详细", "detailed")
        self.hybrid_check = QCheckBox("混合检索")
        self.hybrid_check.setToolTip("默认关闭；只有明确需要关键词召回时再启用。")
        self.extend_check = QCheckBox("教材外补充")
        clear = QPushButton("新对话")
        clear.setProperty("secondary", True)
        clear.clicked.connect(self._clear_chat)
        controls.addWidget(QLabel("模式"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("篇幅"))
        controls.addWidget(self.style_combo)
        controls.addWidget(self.hybrid_check)
        controls.addWidget(self.extend_check)
        controls.addStretch(1)
        controls.addWidget(clear)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenLinks(False)
        self.chat_view.setHtml(
            "<h3>从教材开始提问</h3><p style='color:#697b91'>按 Ctrl+Enter 发送。没有依据时，系统会明确拒答。</p>")
        splitter.addWidget(self.chat_view)
        self.source_tree = QTreeWidget()
        self.source_tree.setHeaderLabels(["引用来源", "证据片段"])
        self.source_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.source_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.source_tree.itemDoubleClicked.connect(self._open_source_item)
        splitter.addWidget(self.source_tree)
        splitter.setSizes([820, 360])
        layout.addWidget(splitter, 1)

        feedback = QHBoxLayout()
        feedback.addWidget(QLabel("本轮反馈"))
        self.feedback_buttons = []
        for label, kind in (("有用", "useful"), ("待改进", "needs-improvement"),
                            ("没回答问题", "no-answer"), ("引用不正确", "bad-citation"),
                            ("证据不足", "insufficient"), ("太慢", "slow")):
            button = QPushButton(label)
            button.setProperty("secondary", True)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, value=kind: self._send_feedback(value))
            feedback.addWidget(button)
            self.feedback_buttons.append(button)
        feedback.addStretch(1)
        layout.addLayout(feedback)

        input_row = QHBoxLayout()
        self.question_edit = QuestionEdit()
        self.question_edit.setPlaceholderText("输入问题，例如：什么是递归？（Ctrl+Enter 发送）")
        self.question_edit.setMaximumHeight(110)
        self.question_edit.submit.connect(self._send_chat)
        buttons = QVBoxLayout()
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self._send_chat)
        self.stop_button = QPushButton("停止显示")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("停止等待并忽略本轮结果；底层本地推理会安全收尾。")
        self.stop_button.clicked.connect(self._stop_chat)
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.stop_button)
        input_row.addWidget(self.question_edit, 1)
        input_row.addLayout(buttons)
        layout.addLayout(input_row)
        return page

    def _append_chat(self, role: str, text: str, meta: str = "") -> None:
        if role == "user":
            title, bg, border = "你", "#e9f2ff", "#bdd6f8"
        elif role == "assistant":
            title, bg, border = "AITIC", "#ffffff", "#dce5ef"
        else:
            title, bg, border = "系统", "#fff7e6", "#f2d49a"
        display_body = answer_html(text) if role == "assistant" else text_html(text)
        block = (
            '<div style="margin:9px 0;padding:12px 14px;background:%s;border:1px solid %s;'
            'border-radius:9px"><b>%s</b><div style="margin-top:6px;line-height:1.55">%s</div>%s</div>'
            % (bg, border, title, display_body,
               ('<div style="color:#738399;font-size:11px;margin-top:7px">%s</div>' % escaped(meta))
               if meta else ""))
        self._chat_parts.append(block)
        self.chat_view.setHtml("".join(self._chat_parts))
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send_chat(self) -> None:
        if self._chat_busy:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            return
        if len(question) > 4000:
            show_error(self, "问题过长", "单次最多 4000 个字符，请拆分后重试。")
            return
        libraries = self.selected_libraries()
        history = list(self._history)
        mode = str(self.mode_combo.currentData())
        style = str(self.style_combo.currentData())
        hybrid = self.hybrid_check.isChecked()
        extend = self.extend_check.isChecked()
        self.question_edit.clear()
        self._append_chat("user", question)
        self._chat_generation += 1
        generation = self._chat_generation
        self._chat_busy = True
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for button in self.feedback_buttons:
            button.setEnabled(False)

        def work():
            return self.backend.ask(
                question, libraries=libraries, history=history,
                mode=mode, style=style, hybrid=hybrid, extend=extend)

        def result(payload: dict[str, Any]) -> None:
            if generation != self._chat_generation:
                return
            answer = str(payload.get("answer") or "")
            agent = payload.get("agent") or {}
            confidence = (agent.get("confidence") or {}).get("level") or "未评级"
            meta = "%d ms · %s token · %s · 可信度 %s" % (
                int(payload.get("elapsed_ms") or 0), payload.get("tokens", 0),
                "补充检索" if payload.get("escalated") else "单轮", confidence)
            self._append_rich_answer(payload, meta)
            supplement = payload.get("supplement") or {}
            if supplement.get("answer"):
                self._append_chat("system", "教材外补充：\n" + supplement["answer"])
            self._history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            self._history = self._history[-12:]
            self._latest_question = question
            self._latest_answer = answer
            self._latest_libraries = list(libraries)
            self._show_sources(payload.get("sources") or [])
            for button in self.feedback_buttons:
                button.setEnabled(True)
            for button in (self.copy_answer_button, self.favorite_button,
                           self.regenerate_button):
                button.setEnabled(True)
            self.favorite_button.setChecked(False)
            self.favorite_button.setText("收藏")

        def error(message: str) -> None:
            if generation == self._chat_generation:
                self._append_chat("system", "本轮失败：" + message)

        def finished() -> None:
            if generation == self._chat_generation:
                self._chat_busy = False
                self.send_button.setEnabled(True)
                self.stop_button.setEnabled(False)

        self.run_task(work, result, label="正在检索并核验答案…", on_error=error,
                      on_finished=finished)

    def _stop_chat(self) -> None:
        if not self._chat_busy:
            return
        self._chat_generation += 1
        self._chat_busy = False
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_chat("system", "已停止等待并忽略本轮结果；本地推理会在后台安全收尾。")

    def _clear_chat(self) -> None:
        self._chat_generation += 1
        self._history.clear()
        self._chat_parts.clear()
        self._chat_sources.clear()
        self._latest_question = self._latest_answer = ""
        self.chat_view.setHtml("<h3>新对话</h3><p>请选择教材后开始提问。</p>")
        self.source_tree.clear()
        for button in self.feedback_buttons:
            button.setEnabled(False)

    def _show_sources(self, sources: list[dict[str, Any]]) -> None:
        self._chat_sources = [dict(value) for value in sources]
        self.source_tree.clear()
        for source in self._chat_sources:
            item = QTreeWidgetItem([
                str(source.get("label") or "来源"), str(source.get("snippet") or "")])
            item.setData(0, Qt.UserRole, source)
            self.source_tree.addTopLevelItem(item)

    def _open_source_item(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        source = item.data(0, Qt.UserRole) or {}
        dialog = SourceDialog(self.backend, source, self)
        dialog.exec()

    def _send_feedback(self, kind: str) -> None:
        if not self._latest_question:
            return
        question = self._latest_question
        answer = self._latest_answer
        libraries = list(self._latest_libraries)
        for button in self.feedback_buttons:
            button.setEnabled(False)

        def done(_payload):
            self.statusBar().showMessage("反馈已保存到本地回归闭环", 4000)

        self.run_task(
            lambda: self.backend.feedback(kind, question, answer, libraries),
            done, label="正在保存反馈…")

    # ----------------------------- 学习工具 -----------------------------
    def _make_study_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("学习工具", "简报、测试题、跨教材概念对照和受控 A/B 均在本机完成。"))
        tabs = QTabWidget()
        tabs.addTab(self._brief_tab(), "教材简报")
        tabs.addTab(self._quiz_tab(), "自测题")
        tabs.addTab(self._concept_tab(), "概念对照")
        tabs.addTab(self._compare_tab(), "检索 A/B")
        layout.addWidget(tabs, 1)
        return page

    def _brief_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.brief_topic = QLineEdit()
        self.brief_topic.setPlaceholderText("输入简报主题")
        button = QPushButton("生成带出处简报")
        button.clicked.connect(lambda: self._run_brief(button))
        row.addWidget(self.brief_topic, 1)
        row.addWidget(button)
        self.brief_output = QTextBrowser()
        layout.addLayout(row)
        layout.addWidget(self.brief_output, 1)
        return tab

    def _run_brief(self, button: QPushButton) -> None:
        topic = self.brief_topic.text().strip()
        if not topic:
            return
        libraries = self.selected_libraries()
        self.run_task(
            lambda: self.backend.brief(topic, libraries=libraries),
            lambda data: self.brief_output.setHtml(
                "<h3>%s</h3><p>%s</p><hr><small>%s ms · %s token</small>" %
                (escaped(topic), text_html(data.get("answer")), data.get("elapsed_ms", 0),
                 data.get("tokens", 0))),
            label="正在生成教材简报…", button=button)

    def _quiz_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.quiz_topic = QLineEdit()
        self.quiz_topic.setPlaceholderText("可选主题；留空则覆盖核心内容")
        self.quiz_count = QSpinBox()
        self.quiz_count.setRange(2, 5)
        self.quiz_count.setValue(3)
        self.quiz_hide = QCheckBox("自测模式（隐藏答案）")
        self.quiz_hide.setChecked(True)
        self.quiz_hide.toggled.connect(self._toggle_quiz_answers)
        button = QPushButton("生成测试题")
        button.clicked.connect(lambda: self._run_quiz(button))
        row.addWidget(self.quiz_topic, 1)
        row.addWidget(QLabel("题数"))
        row.addWidget(self.quiz_count)
        row.addWidget(self.quiz_hide)
        row.addWidget(button)
        self.quiz_tree = QTreeWidget()
        self.quiz_tree.setHeaderLabels(["难度", "问题", "参考答案", "出处"])
        self.quiz_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.quiz_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.quiz_tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.quiz_tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addLayout(row)
        layout.addWidget(self.quiz_tree, 1)
        return tab

    def _run_quiz(self, button: QPushButton) -> None:
        topic = self.quiz_topic.text().strip()
        libraries = self.selected_libraries()
        count = self.quiz_count.value()

        def result(data: dict[str, Any]) -> None:
            self.quiz_tree.clear()
            for question in data.get("questions", []):
                answer = str(question.get("expected_answer") or "")
                item = QTreeWidgetItem([
                    str(question.get("difficulty") or "basic"),
                    str(question.get("question") or ""),
                    "点击或关闭自测模式后查看" if self.quiz_hide.isChecked() else answer,
                    str(question.get("source") or question.get("probe_basis") or ""),
                ])
                item.setData(2, Qt.UserRole, answer)
                self.quiz_tree.addTopLevelItem(item)

        self.run_task(
            lambda: self.backend.questions(topic, libraries=libraries, count=count),
            result, label="正在生成并核对测试题…", button=button)

    def _toggle_quiz_answers(self, hidden: bool) -> None:
        for index in range(self.quiz_tree.topLevelItemCount()):
            item = self.quiz_tree.topLevelItem(index)
            answer = str(item.data(2, Qt.UserRole) or "")
            item.setText(2, "点击或关闭自测模式后查看" if hidden else answer)

    def _concept_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.concept_input = QLineEdit()
        self.concept_input.setPlaceholderText("输入要在 2–4 本教材中对照的概念")
        button = QPushButton("跨教材对照")
        button.clicked.connect(lambda: self._run_concept(button))
        row.addWidget(self.concept_input, 1)
        row.addWidget(button)
        self.concept_output = QTextBrowser()
        layout.addLayout(row)
        layout.addWidget(self.concept_output, 1)
        return tab

    def _run_concept(self, button: QPushButton) -> None:
        concept = self.concept_input.text().strip()
        libraries = self.selected_libraries()
        if not concept:
            return
        if len(libraries) < 2:
            QMessageBox.information(self, "请选择教材", "概念对照需要在左侧勾选至少两本教材。")
            return

        def result(data: dict[str, Any]) -> None:
            parts = ["<h3>%s</h3>" % escaped(data.get("concept"))]
            for book in data.get("books", []):
                parts.append(
                    '<div style="margin:9px 0;padding:10px;border:1px solid #dde6ef;border-radius:7px">'
                    '<b>%s</b><p>%s</p><small>%s</small></div>' %
                    (escaped(book.get("library")), text_html(book.get("answer") or book.get("note")),
                     escaped(" · ".join(book.get("sources") or []))))
            parts.append("<p><small>%s</small></p>" % text_html(data.get("note")))
            self.concept_output.setHtml("".join(parts))

        self.run_task(lambda: self.backend.concept(concept, libraries=libraries), result,
                      label="正在逐本检索并对照…", button=button)

    def _compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.compare_input = QLineEdit()
        self.compare_input.setPlaceholderText("输入同一问题，对照向量检索与混合检索")
        button = QPushButton("运行 A/B")
        button.clicked.connect(lambda: self._run_compare(button))
        row.addWidget(self.compare_input, 1)
        row.addWidget(button)
        split = QSplitter(Qt.Horizontal)
        self.compare_a = QTextBrowser()
        self.compare_b = QTextBrowser()
        split.addWidget(self.compare_a)
        split.addWidget(self.compare_b)
        layout.addLayout(row)
        layout.addWidget(split, 1)
        return tab

    def _run_compare(self, button: QPushButton) -> None:
        question = self.compare_input.text().strip()
        if not question:
            return
        libraries = self.selected_libraries()

        def result(data: dict[str, Any]) -> None:
            views = (self.compare_a, self.compare_b)
            for view, arm in zip(views, data.get("arms", [])):
                metrics = arm.get("metrics") or {}
                view.setHtml("<h3>%s</h3><p>%s</p><hr><small>%s</small>" % (
                    escaped(arm.get("label")), text_html(arm.get("answer")),
                    escaped(json.dumps(metrics, ensure_ascii=False))))

        self.run_task(lambda: self.backend.compare(question, libraries=libraries),
                      result, label="正在执行受控 A/B…", button=button)

    # ----------------------------- 知识库 -----------------------------
    def _make_library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("知识库", "导入 PDF、跟踪建库、切换教材，并直接浏览向量库中的原文分块。"))
        toolbar = QHBoxLayout()
        self.max_pages = QSpinBox()
        self.max_pages.setRange(0, 10000)
        self.max_pages.setSpecialValueText("全部页")
        self.use_vl = QCheckBox("识别图表页")
        self.use_vl.setChecked(True)
        self.vl_limit = QSpinBox()
        self.vl_limit.setRange(0, 100)
        self.vl_limit.setValue(15)
        import_button = QPushButton("导入 PDF 并建库")
        import_button.clicked.connect(lambda: self._choose_and_build(import_button))
        activate = QPushButton("设为当前库")
        activate.setProperty("secondary", True)
        activate.clicked.connect(lambda: self._activate_selected(activate))
        refresh = QPushButton("刷新")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.refresh_all)
        health = QPushButton("检查库健康")
        health.setProperty("secondary", True)
        health.clicked.connect(lambda: self._health_selected(health))
        toolbar.addWidget(import_button)
        toolbar.addWidget(QLabel("页数"))
        toolbar.addWidget(self.max_pages)
        toolbar.addWidget(self.use_vl)
        toolbar.addWidget(QLabel("图表页上限"))
        toolbar.addWidget(self.vl_limit)
        toolbar.addStretch(1)
        toolbar.addWidget(health)
        toolbar.addWidget(activate)
        toolbar.addWidget(refresh)
        layout.addLayout(toolbar)

        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 100)
        self.build_progress.setValue(0)
        self.build_progress.setFormat("没有正在运行的建库任务")
        layout.addWidget(self.build_progress)

        split = QSplitter(Qt.Vertical)
        self.library_table = QTableWidget(0, 6)
        self.library_table.setHorizontalHeaderLabels(["状态", "教材", "学科", "块数", "来源", "建成时间"])
        self.library_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.library_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.library_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.library_table.setAlternatingRowColors(True)
        self.library_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.library_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        split.addWidget(self.library_table)

        chunks = QWidget()
        chunk_layout = QVBoxLayout(chunks)
        chunk_row = QHBoxLayout()
        self.chunk_query = QLineEdit()
        self.chunk_query.setPlaceholderText("浏览全部块，或输入检索词")
        chunk_button = QPushButton("读取分块")
        chunk_button.setProperty("secondary", True)
        chunk_button.clicked.connect(lambda: self._load_chunks(chunk_button))
        chunk_row.addWidget(self.chunk_query, 1)
        chunk_row.addWidget(chunk_button)
        self.chunk_table = QTableWidget(0, 4)
        self.chunk_table.setHorizontalHeaderLabels(["#", "来源", "距离", "正文"])
        self.chunk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.chunk_table.setAlternatingRowColors(True)
        self.chunk_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        chunk_layout.addLayout(chunk_row)
        chunk_layout.addWidget(self.chunk_table)
        split.addWidget(chunks)
        split.setSizes([390, 260])
        layout.addWidget(split, 1)

        self.build_timer = QTimer(self)
        self.build_timer.setInterval(1300)
        self.build_timer.timeout.connect(self._poll_build)
        return page

    def _fill_library_table(self) -> None:
        if not hasattr(self, "library_table"):
            return
        self.library_table.setRowCount(len(self._libraries))
        for row, library in enumerate(self._libraries):
            values = [
                "当前" if library.get("active") else str(library.get("status") or ""),
                str(library.get("name") or ""), str(library.get("subject") or ""),
                "—" if library.get("chunks") is None else str(library.get("chunks")),
                str(library.get("source") or ""), str(library.get("built_at") or ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, dict(library))
                self.library_table.setItem(row, column, item)
        self.library_table.resizeRowsToContents()

    def _selected_library_row(self) -> dict[str, Any] | None:
        row = self.library_table.currentRow()
        if row < 0:
            return None
        item = self.library_table.item(row, 0)
        return dict(item.data(Qt.UserRole) or {}) if item else None

    def _choose_and_build(self, button: QPushButton) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择要建库的 PDF", "", "PDF 文件 (*.pdf)")
        if not path:
            return
        self._selected_pdf = path
        max_pages = self.max_pages.value()
        use_vl = self.use_vl.isChecked()
        vl_limit = self.vl_limit.value()

        def result(data: dict[str, Any]) -> None:
            job = data.get("job") or {}
            self._build_job_id = str(job.get("id") or job.get("job_id") or "")
            if not self._build_job_id:
                show_error(self, "建库失败", "后端没有返回任务编号。")
                return
            self.build_progress.setRange(0, 0)
            self.build_progress.setFormat("正在建库：%s" % Path(path).name)
            self.build_timer.start()

        self.run_task(
            lambda: self.backend.start_build(
                path, max_pages=max_pages, use_vl=use_vl, vl_limit=vl_limit),
            result, label="正在创建建库任务…", button=button)

    def _poll_build(self) -> None:
        if not self._build_job_id:
            self.build_timer.stop()
            return
        try:
            job = self.backend.build_status(self._build_job_id)
        except Exception as exc:
            self.build_timer.stop()
            self.build_progress.setRange(0, 100)
            self.build_progress.setFormat("建库状态读取失败：%s" % exc)
            return
        status = str(job.get("status") or "")
        message = str(job.get("phase") or job.get("message") or job.get("stage") or status)
        progress = job.get("progress")
        if isinstance(progress, (int, float)):
            value = int(progress * 100 if progress <= 1 else progress)
            self.build_progress.setRange(0, 100)
            self.build_progress.setValue(max(0, min(100, value)))
        else:
            self.build_progress.setRange(0, 0)
        self.build_progress.setFormat(message)
        if status in ("completed", "ready", "failed"):
            self.build_timer.stop()
            self.build_progress.setRange(0, 100)
            self.build_progress.setValue(100 if status in ("completed", "ready") else 0)
            if status == "failed":
                show_error(self, "建库失败", str(job.get("error") or message))
            else:
                QMessageBox.information(self, "建库完成", "新教材已经可用并已切换。")
            self._build_job_id = ""
            self.refresh_all()

    def _activate_selected(self, button: QPushButton) -> None:
        library = self._selected_library_row()
        if not library:
            QMessageBox.information(self, "请选择知识库", "先在表格中选择一行。")
            return
        self.run_task(
            lambda: self.backend.activate_library(str(library.get("id"))),
            lambda _data: self.refresh_all(), label="正在切换知识库…", button=button)

    def _health_selected(self, button: QPushButton) -> None:
        library = self._selected_library_row()
        if not library:
            return
        self.run_task(
            lambda: self.backend.library_health(str(library.get("id"))),
            lambda data: QMessageBox.information(
                self, "知识库健康报告", json.dumps(data, ensure_ascii=False, indent=2)),
            label="正在抽样检查知识库…", button=button)

    def _load_chunks(self, button: QPushButton) -> None:
        library = self._selected_library_row()
        if not library:
            QMessageBox.information(self, "请选择知识库", "先在上方表格选择一本教材。")
            return
        library_id = str(library.get("id"))
        query = self.chunk_query.text().strip()

        def result(data: dict[str, Any]) -> None:
            chunks = data.get("chunks") or []
            self.chunk_table.setRowCount(len(chunks))
            for row, chunk in enumerate(chunks):
                values = [chunk.get("index"), chunk.get("label"), chunk.get("distance"), chunk.get("text")]
                for column, value in enumerate(values):
                    self.chunk_table.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))
            self.chunk_table.resizeRowsToContents()

        self.run_task(
            lambda: self.backend.library_chunks(
                library_id, query, 20, 0),
            result, label="正在读取原文分块…", button=button)

    # ----------------------------- 批量 -----------------------------
    def _make_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("批量评测", "每行一个问题，最多 20 题；结果严格区分命中、漏答与拒答。"))
        split = QSplitter(Qt.Vertical)
        top, top_layout = card_layout()
        self.batch_input = QPlainTextEdit()
        self.batch_input.setPlaceholderText("什么是递归？\nWhat is a tuple?\n……")
        self.batch_button = QPushButton("运行批量评测")
        self.batch_button.clicked.connect(self._run_batch)
        top_layout.addWidget(self.batch_input)
        top_layout.addWidget(self.batch_button)
        split.addWidget(top)
        bottom, bottom_layout = card_layout()
        self.batch_summary = QLabel("尚未运行")
        self.batch_summary.setObjectName("SectionTitle")
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(["结果", "问题", "回答", "理由", "可信度"])
        self.batch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        bottom_layout.addWidget(self.batch_summary)
        bottom_layout.addWidget(self.batch_table)
        split.addWidget(bottom)
        split.setSizes([240, 430])
        layout.addWidget(split, 1)
        return page

    def _run_batch(self) -> None:
        questions = [line.strip() for line in self.batch_input.toPlainText().splitlines() if line.strip()]
        if not questions:
            return
        if len(questions) > 20:
            show_error(self, "题目过多", "单次最多 20 题，请分批运行。")
            return
        libraries = self.selected_libraries()

        def result(data: dict[str, Any]) -> None:
            summary = data.get("summary") or {}
            self.batch_summary.setText(
                "共 {total} 题 · 命中 {hit} · 未命中 {miss} · 过度拒答 {over_refused} · "
                "引用通过率 {cite_ok_rate}".format(**{
                    "total": summary.get("total", 0), "hit": summary.get("hit", 0),
                    "miss": summary.get("miss", 0), "over_refused": summary.get("over_refused", 0),
                    "cite_ok_rate": summary.get("cite_ok_rate", "—"),
                }))
            rows = data.get("rows") or []
            self.batch_table.setRowCount(len(rows))
            for row, value in enumerate(rows):
                columns = [value.get("verdict"), value.get("question"), value.get("answer"),
                           value.get("reason"), value.get("confidence")]
                for column, cell in enumerate(columns):
                    self.batch_table.setItem(row, column, QTableWidgetItem(str(cell or "")))
            self.batch_table.resizeRowsToContents()

        self.run_task(lambda: self.backend.batch(questions, libraries=libraries),
                      result, label="正在逐题评测…", button=self.batch_button)

    # ----------------------------- 诊断与设置 -----------------------------
    def _make_diagnostics_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("诊断与设置", "检查 Ollama、模型、运行时和证据召回；管理本地反馈闭环。"))
        tabs = QTabWidget()
        self.diagnostic_tabs = tabs
        tabs.addTab(self._model_tab(), "模型管理")
        tabs.addTab(self._system_tab(), "系统状态")
        tabs.addTab(self._retrieve_tab(), "仅检索诊断")
        tabs.addTab(self._feedback_tab(), "反馈闭环")
        layout.addWidget(tabs, 1)
        return page

    def _model_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("回答模型"))
        self.model_combo = CrispComboBox()
        self.model_combo.setMinimumWidth(290)
        for preset in LLM_PRESETS:
            self.model_combo.addItem(str(preset["label"]), str(preset["model"]))
        self.model_refresh_button = QPushButton("刷新")
        self.model_refresh_button.setProperty("secondary", True)
        self.model_refresh_button.clicked.connect(self._refresh_model_catalog)
        self.model_download_button = QPushButton("下载所选")
        self.model_download_button.setProperty("secondary", True)
        self.model_download_button.clicked.connect(self._download_selected_model)
        self.model_switch_button = QPushButton("切换到所选")
        self.model_switch_button.clicked.connect(self._switch_selected_model)
        row.addWidget(self.model_combo)
        row.addWidget(self.model_refresh_button)
        row.addWidget(self.model_download_button)
        row.addWidget(self.model_switch_button)
        row.addStretch(1)

        actions = QHBoxLayout()
        self.recommended_models_button = QPushButton("一键配置推荐环境")
        self.recommended_models_button.clicked.connect(self._pull_recommended_models)
        import_button = QPushButton("一键导入本地模型")
        import_button.setProperty("secondary", True)
        import_button.clicked.connect(lambda: self._import_local_model(import_button))
        self.model_import_button = import_button
        storage_button = QPushButton("模型存储目录")
        storage_button.setProperty("secondary", True)
        storage_button.clicked.connect(lambda: self._choose_model_storage(storage_button))
        guide_button = QPushButton("配置教程")
        guide_button.setProperty("secondary", True)
        guide_button.clicked.connect(self._show_model_guide)
        repair_button = QPushButton("修复 VC++ 运行库")
        repair_button.setProperty("secondary", True)
        repair_button.clicked.connect(lambda: self._repair_vc_runtime(repair_button))
        actions.addWidget(self.recommended_models_button)
        actions.addWidget(import_button)
        actions.addWidget(storage_button)
        actions.addWidget(guide_button)
        actions.addWidget(repair_button)
        actions.addStretch(1)

        self.model_status = QTextBrowser()
        self.model_status.setMaximumHeight(245)
        self.model_log = QPlainTextEdit()
        self.model_log.setReadOnly(True)
        self.model_log.setPlaceholderText("模型下载、导入和 Ollama 启动日志会显示在这里。")
        layout.addLayout(row)
        layout.addLayout(actions)
        layout.addWidget(self.model_status)
        layout.addWidget(self.model_log, 1)
        QTimer.singleShot(250, self._refresh_model_catalog)
        return tab

    def _selected_model(self) -> str:
        return str(self.model_combo.currentData() or self.model_combo.currentText()).strip()

    def _refresh_model_catalog(self) -> None:
        self.run_task(self.backend.model_catalog, self._apply_model_catalog,
                      label="正在读取本机模型…", button=self.model_refresh_button)

    def _apply_model_catalog(self, data: dict[str, Any]) -> None:
        active = str(data.get("active") or DEFAULT_LLM_MODEL)
        installed = list(data.get("available") or [])
        known = {str(self.model_combo.itemData(i)) for i in range(self.model_combo.count())}
        for model in installed:
            if model not in known:
                self.model_combo.addItem("%s · 本地导入" % model, model)
                known.add(model)
        index = self.model_combo.findData(active)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        preset_rows = []
        for preset in data.get("presets") or []:
            state = "已安装" if preset.get("installed") else "未安装"
            if preset.get("active"):
                state += " / 当前使用"
            preset_rows.append("<li><b>%s</b>：%s<br>%s</li>" % (
                escaped(preset.get("label")), escaped(state), escaped(preset.get("description"))))
        connection = "已连接" if data.get("connected") else "未连接"
        self.model_status.setHtml(
            "<h3>模型环境</h3><p><b>Ollama：</b>%s　<b>当前回答模型：</b>%s</p>"
            "<p><b>视觉模型：</b>%s　<b>嵌入模型：</b>%s</p>"
            "<p><b>模型目录：</b>%s</p><ul>%s</ul>%s" % (
                escaped(connection), escaped(active), escaped(data.get("vision_model")),
                escaped(data.get("embedding_model")),
                escaped(data.get("models_directory") or "Ollama 默认目录"),
                "".join(preset_rows),
                ("<p style='color:#a33a30'>%s</p>" % escaped(data.get("error")))
                if data.get("error") else ""))

    def _start_model_download(self, models: list[str], *, activate: str = "",
                              button: QPushButton | None = None) -> None:
        self.model_log.clear()
        if button:
            button.setEnabled(False)
        self._busy_count += 1
        self.busy_bar.setRange(0, 0)
        self.busy_bar.show()
        self.status_label.setText("正在下载模型…")
        def download() -> dict[str, Any]:
            data = self.backend.pull_models(models, task.signals.progress.emit)
            if activate:
                data = self.backend.set_llm_model(activate)
            self.backend.mark_model_setup_complete()
            return data

        task = Task(download)
        self._tasks.add(task)
        task.signals.progress.connect(self.model_log.appendPlainText)

        def result(data: dict[str, Any]) -> None:
            self._apply_model_catalog(data)
            QMessageBox.information(self, "模型配置完成", "所选模型已经可用。")

        task.signals.result.connect(result)
        task.signals.error.connect(lambda msg: show_error(self, "模型下载失败", msg))

        def finished() -> None:
            self._tasks.discard(task)
            self._busy_count = max(0, self._busy_count - 1)
            if button:
                button.setEnabled(True)
            if not self._busy_count:
                self.busy_bar.hide()
                self.status_label.setText("就绪")
            self._refresh_model_catalog()
            self.refresh_all()

        task.signals.finished.connect(finished)
        self.pool.start(task)

    def _pull_recommended_models(self) -> None:
        self._start_model_download(
            [DEFAULT_LLM_MODEL, VISION_MODEL, EMBEDDING_MODEL],
            activate=DEFAULT_LLM_MODEL, button=self.recommended_models_button)

    def _download_selected_model(self) -> None:
        model = self._selected_model()
        if model:
            self._start_model_download([model], button=self.model_download_button)

    def _switch_selected_model(self) -> None:
        model = self._selected_model()
        if not model:
            return
        self.run_task(
            lambda: self.backend.set_llm_model(model),
            lambda data: (self._apply_model_catalog(data), self.refresh_all(),
                          QMessageBox.information(self, "切换完成", "当前回答模型：%s" % model)),
            label="正在切换模型…", button=self.model_switch_button)

    def _import_local_model(self, button: QPushButton) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "选择本地模型", "", "Ollama 模型 (*.gguf *.modelfile *.txt);;Modelfile (Modelfile);;所有文件 (*)")
        if not path:
            return
        suggested = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).stem).strip("-.") or "local-model"
        name, ok = QInputDialog.getText(self, "模型名称", "导入后的 Ollama 模型名称：", text=suggested)
        if not ok or not name.strip():
            return
        self.model_log.clear()
        task = Task(lambda: self.backend.import_model(path, name.strip(), task.signals.progress.emit))
        self._tasks.add(task)
        self._busy_count += 1
        button.setEnabled(False)
        self.busy_bar.setRange(0, 0)
        self.busy_bar.show()
        self.status_label.setText("正在导入本地模型…")
        task.signals.progress.connect(self.model_log.appendPlainText)
        task.signals.result.connect(lambda data: (
            self._apply_model_catalog(data),
            QMessageBox.information(self, "导入完成", "模型已导入并切换为当前回答模型。")))
        task.signals.error.connect(lambda msg: show_error(self, "模型导入失败", msg))

        def finished() -> None:
            self._tasks.discard(task)
            self._busy_count = max(0, self._busy_count - 1)
            button.setEnabled(True)
            if not self._busy_count:
                self.busy_bar.hide()
                self.status_label.setText("就绪")
            self.refresh_all()

        task.signals.finished.connect(finished)
        self.pool.start(task)

    def _choose_model_storage(self, button: QPushButton) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择模型存储目录")
        if not path:
            return
        self.run_task(
            lambda: self.backend.set_model_storage(path),
            lambda data: (self._refresh_model_catalog(), QMessageBox.information(
                self, "模型目录已保存",
                "模型将存放在：\n%s\n\n如果正在使用外部 Ollama，请重启 Ollama 后生效。" % data.get("path"))),
            label="正在更新模型目录…", button=button)

    def _show_model_guide(self) -> None:
        QMessageBox.information(
            self, "AITIC 模型配置教程",
            "1. 安装完整版已包含 Ollama 运行环境，无需再安装 Python 或 Web 服务。\n\n"
            "2. 第一次使用点击“一键配置推荐环境”，程序会下载 qwen3:8b、"
            "qwen3-vl:8b 和 bge-m3。模型权重较大，请保持网络连接。\n\n"
            "3. 回答模型可在 Qwen3 4B / 8B / 14B 之间切换。8B 是本项目经过全量评测的默认档；"
            "换模型后建议在“批量评测”中复核准确度。\n\n"
            "4. 已有 GGUF 时点击“一键导入本地模型”，选择文件并填写名称；Modelfile 也可直接导入。\n\n"
            "5. 模型默认由 Ollama 保存在用户目录。空间不足时先点击“模型存储目录”改到其他磁盘，"
            "再开始下载。知识库和程序目录与模型目录互不影响。")

    def _repair_vc_runtime(self, button: QPushButton) -> None:
        answer = QMessageBox.question(
            self, "安装/修复 VC++ 运行库",
            "即将打开 Windows 权限确认并静默安装微软 VC++ 运行库。\n"
            "只在 Ollama 无法启动时需要执行。是否继续？")
        if answer != QMessageBox.Yes:
            return
        self.run_task(
            self.backend.repair_vc_runtime,
            lambda _data: QMessageBox.information(
                self, "修复已启动", "安装程序已经启动。完成后请重新打开 AITIC Desktop。"),
            label="正在启动运行库修复…", button=button)

    def _maybe_first_model_setup(self) -> None:
        if not hasattr(self.backend, "needs_model_setup"):
            return
        try:
            if not self.backend.needs_model_setup():
                return
        except Exception:
            return
        box = QMessageBox(self)
        box.setWindowTitle("首次模型配置")
        box.setIcon(QMessageBox.Information)
        box.setText("AITIC Desktop 主程序已安装完成")
        box.setInformativeText(
            "模型权重不包含在安装包内。建议现在一键下载经过验证的推荐模型；"
            "也可以进入模型管理选择轻量版、增强版或导入本地 GGUF。")
        configure = box.addButton("一键配置推荐模型", QMessageBox.AcceptRole)
        manage = box.addButton("打开模型管理", QMessageBox.ActionRole)
        later = box.addButton("稍后", QMessageBox.RejectRole)
        box.setDefaultButton(configure)
        box.exec()
        self.nav.setCurrentRow(4)
        self.diagnostic_tabs.setCurrentIndex(0)
        if box.clickedButton() is configure:
            self._pull_recommended_models()
        elif box.clickedButton() is later:
            self.backend.mark_model_setup_complete()
        elif box.clickedButton() is manage:
            self._refresh_model_catalog()

    def _system_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        actions = QHBoxLayout()
        refresh = QPushButton("重新检测")
        refresh.clicked.connect(lambda: self.run_task(self.backend.status, self._apply_status,
                                                       label="正在重新检测…", button=refresh))
        install = QPushButton("Ollama 帮助")
        install.setProperty("secondary", True)
        install.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://ollama.com/download/windows")))
        self.pull_button = QPushButton("下载缺失模型")
        self.pull_button.setProperty("secondary", True)
        self.pull_button.clicked.connect(self._pull_models)
        open_data = QPushButton("打开数据目录")
        open_data.setProperty("secondary", True)
        open_data.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.backend.project_root))))
        actions.addWidget(refresh)
        actions.addWidget(install)
        actions.addWidget(self.pull_button)
        actions.addWidget(open_data)
        actions.addStretch(1)
        self.system_status = QTextBrowser()
        layout.addLayout(actions)
        layout.addWidget(self.system_status, 1)
        return tab

    def _render_system_status(self, status: dict[str, Any]) -> None:
        if not hasattr(self, "system_status"):
            return
        runtime = status.get("runtime") or {}
        packages = runtime.get("packages") or {}
        root = self.backend.runtime_summary()
        rows = [
            ("Ollama", status.get("ollama")), ("LLM", status.get("llm_model")),
            ("视觉模型", status.get("vl_model")), ("嵌入模型", status.get("embed_model")),
            ("当前知识块", status.get("chunks")), ("活动知识库", status.get("active_library_id")),
            ("Python", runtime.get("python")), ("数据目录", root.get("project_root")),
            ("冻结程序", "是" if root.get("frozen") else "否（源码开发模式）"),
        ]
        body = ["<h3>运行状态</h3><table cellspacing='7'>"]
        for key, value in rows:
            body.append("<tr><td><b>%s</b></td><td>%s</td></tr>" % (escaped(key), escaped(value)))
        body.append("</table><h3>依赖版本</h3><pre>%s</pre>" %
                    escaped(json.dumps(packages, ensure_ascii=False, indent=2)))
        if status.get("ollama_error"):
            body.append("<p style='color:#a33a30'>%s</p>" % escaped(status["ollama_error"]))
        self.system_status.setHtml("".join(body))

    def _pull_models(self) -> None:
        self.model_log.clear()
        inventory = self.backend.model_inventory()
        missing = inventory.get("missing") or []
        if not inventory.get("connected"):
            show_error(self, "Ollama 未连接", "请先安装并启动 Ollama。")
            return
        if not missing:
            QMessageBox.information(self, "模型齐全", "三个必需模型均已安装。")
            return

        task = Task(lambda: self.backend.pull_models(missing, task.signals.progress.emit))
        self._tasks.add(task)
        self._busy_count += 1
        self.pull_button.setEnabled(False)
        self.busy_bar.setRange(0, 0)
        self.busy_bar.show()
        self.status_label.setText("正在下载模型…")
        task.signals.progress.connect(self.model_log.appendPlainText)
        task.signals.result.connect(lambda _data: QMessageBox.information(self, "完成", "缺失模型已下载。"))
        task.signals.error.connect(lambda msg: show_error(self, "模型下载失败", msg))

        def finished():
            self._tasks.discard(task)
            self._busy_count = max(0, self._busy_count - 1)
            self.pull_button.setEnabled(True)
            if not self._busy_count:
                self.busy_bar.hide()
                self.status_label.setText("就绪")
            self.refresh_all()

        task.signals.finished.connect(finished)
        self.pool.start(task)

    def _retrieve_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.retrieve_input = QLineEdit()
        self.retrieve_input.setPlaceholderText("输入查询，只看召回证据，不调用生成模型")
        self.retrieve_hybrid = QCheckBox("混合检索")
        button = QPushButton("检查召回")
        button.clicked.connect(lambda: self._run_retrieve(button))
        row.addWidget(self.retrieve_input, 1)
        row.addWidget(self.retrieve_hybrid)
        row.addWidget(button)
        self.retrieve_table = QTableWidget(0, 4)
        self.retrieve_table.setHorizontalHeaderLabels(["来源", "距离", "教材", "片段"])
        self.retrieve_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.retrieve_table.setAlternatingRowColors(True)
        self.retrieve_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addLayout(row)
        layout.addWidget(self.retrieve_table, 1)
        return tab

    def _run_retrieve(self, button: QPushButton) -> None:
        query = self.retrieve_input.text().strip()
        if not query:
            return
        libraries = self.selected_libraries()
        hybrid = self.retrieve_hybrid.isChecked()

        def result(data: dict[str, Any]) -> None:
            sources = data.get("sources") or []
            self.retrieve_table.setRowCount(len(sources))
            for row, source in enumerate(sources):
                values = [source.get("label"), source.get("distance"), source.get("library"),
                          source.get("snippet")]
                for column, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    if column == 0:
                        item.setData(Qt.UserRole, source)
                    self.retrieve_table.setItem(row, column, item)
            self.retrieve_table.resizeRowsToContents()

        self.run_task(
            lambda: self.backend.retrieve(query, libraries=libraries, hybrid=hybrid),
            result, label="正在检查召回…", button=button)

    def _feedback_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        actions = QHBoxLayout()
        refresh = QPushButton("刷新反馈")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(lambda: self._refresh_feedback(refresh))
        rerun = QPushButton("复跑最近失败")
        rerun.clicked.connect(lambda: self._rerun_feedback(rerun))
        export = QPushButton("导出回归集")
        export.setProperty("secondary", True)
        export.clicked.connect(lambda: self._export_feedback(export))
        actions.addWidget(refresh)
        actions.addWidget(rerun)
        actions.addWidget(export)
        actions.addStretch(1)
        self.feedback_table = QTableWidget(0, 5)
        self.feedback_table.setHorizontalHeaderLabels(["时间", "类型", "问题", "教材", "回答"])
        self.feedback_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.feedback_table.setAlternatingRowColors(True)
        self.feedback_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.feedback_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addLayout(actions)
        layout.addWidget(self.feedback_table, 1)
        return tab

    def _refresh_feedback(self, button: QPushButton) -> None:
        def result(data: dict[str, Any]) -> None:
            rows = data.get("recent") or data.get("feedback") or data.get("rows") or []
            self.feedback_table.setRowCount(len(rows))
            for row, value in enumerate(rows):
                columns = [value.get("created_at") or value.get("time"),
                           value.get("kind_label") or value.get("label") or value.get("kind"),
                           value.get("question"), ", ".join(value.get("libraries") or []), value.get("answer")]
                for column, cell in enumerate(columns):
                    self.feedback_table.setItem(row, column, QTableWidgetItem(str(cell or "")))
            self.feedback_table.resizeRowsToContents()

        self.run_task(lambda: self.backend.feedback_list(100), result,
                      label="正在读取反馈…", button=button)

    def _rerun_feedback(self, button: QPushButton) -> None:
        self.run_task(
            lambda: self.backend.rerun_feedback(10),
            lambda data: QMessageBox.information(
                self, "复跑完成", json.dumps(data, ensure_ascii=False, indent=2)[:6000]),
            label="正在复跑失败样本…", button=button)

    def _export_feedback(self, button: QPushButton) -> None:
        def result(data: dict[str, Any]) -> None:
            count = int(data.get("count") or 0)
            if count < 1:
                QMessageBox.information(self, "没有可导出的样本", "目前没有被标记为失败的反馈。")
                return
            path, _selected = QFileDialog.getSaveFileName(
                self, "保存反馈回归集", str(Path.home() / "AITIC-feedback-regression.jsonl"),
                "JSON Lines (*.jsonl);;所有文件 (*)")
            if not path:
                return
            try:
                Path(path).write_text(str(data.get("jsonl") or ""), encoding="utf-8")
            except OSError as exc:
                show_error(self, "导出失败", str(exc))
                return
            QMessageBox.information(
                self, "导出完成", "已保存 %d 条待人工订正的失败样本：\n%s" % (count, path))

        self.run_task(
            self.backend.export_regression,
            result,
            label="正在导出回归集…", button=button)

    # ----------------------------- 窗口状态 -----------------------------
    def _restore_window(self) -> None:
        settings = QSettings()
        geometry = settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        QSettings().setValue("window/geometry", self.saveGeometry())
        event.accept()


class MainWindow(_CoreMainWindow):
    """完整原生桌面壳层。

    业务调用继续由 ``_CoreMainWindow`` 和 ``DesktopBackend`` 承担；这里仅重组网页端已经
    验证过的信息架构，并补上原生会话、独立证据页和快捷操作。
    """

    MAIN_PAGES = ("chat", "library", "evidence", "chunks", "health", "settings")

    def __init__(self, backend: DesktopBackend):
        self._sessions: list[dict[str, Any]] = []
        self._current_session_id = ""
        self._restoring_session = False
        self._pending_library_ids: list[str] = []
        self._page_indexes: dict[str, int] = {}
        self._build_queue: list[dict[str, Any]] = []
        self._build_total = 0
        self._build_completed = 0
        self._build_failures: list[str] = []
        self._build_poll_pending = False
        self._build_button: QPushButton | None = None
        self._chat_cancel_event: threading.Event | None = None
        settings = QSettings()
        self._workspace_instruction = str(settings.value("chat/workspace_instruction", "") or "")[:300]
        scope_from = int(settings.value("chat/page_from", 0) or 0)
        scope_to = int(settings.value("chat/page_to", 0) or 0)
        self._page_scope = ({"from": scope_from or None, "to": scope_to or None}
                            if scope_from or scope_to else None)
        self._session_metrics = {"queries": 0, "tokens": 0, "total_ms": 0,
                                 "refusals": 0, "retrievals": 0}
        super().__init__(backend)
        self._load_sessions()
        QTimer.singleShot(250, self._resume_active_build)

    # ----------------------------- 完整桌面壳层 -----------------------------
    def _build_shell(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(286)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(18, 20, 18, 16)
        side.setSpacing(9)

        brand_row = QHBoxLayout()
        mark = QLabel("知")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(38, 38)
        brand = QLabel("知识蒸馏")
        brand.setObjectName("Brand")
        brand_row.addWidget(mark)
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        side.addLayout(brand_row)
        privacy = QLabel("本地运行 · 数据不出机")
        privacy.setObjectName("PrivacyPill")
        privacy.setAlignment(Qt.AlignCenter)
        side.addWidget(privacy, 0, Qt.AlignLeft)
        side.addSpacing(5)

        new_chat = QPushButton("新对话")
        new_chat.setObjectName("NewChatButton")
        new_chat.clicked.connect(self._new_chat)
        side.addWidget(new_chat)
        side.addSpacing(6)

        self.nav = QListWidget()
        self.nav.setObjectName("Navigation")
        self.nav.setFocusPolicy(Qt.NoFocus)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for label in ("问答", "资料库", "证据审查", "分块浏览", "知识库诊断", "设置"):
            self.nav.addItem(QListWidgetItem(label))
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setFixedHeight(306)
        side.addWidget(self.nav)

        self.more_button = QToolButton()
        self.more_button.setObjectName("MoreButton")
        self.more_button.setText("更多功能")
        self.more_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more = QMenu(self.more_button)
        for label, key in (("运行状态", "status"), ("学习工具", "learning"),
                           ("批量评测", "batch"), ("A/B 对比", "compare"),
                           ("概念对照", "concept"), ("反馈与回归集", "feedback")):
            action = more.addAction(label)
            action.triggered.connect(lambda _checked=False, value=key: self._show_page(value))
        more.addSeparator()
        more.addAction("本轮运行统计").triggered.connect(self._show_session_metrics)
        more.addAction("导出当前对话").triggered.connect(self._export_conversation)
        more.addAction("复制当前对话").triggered.connect(self._copy_conversation)
        self.more_button.setMenu(more)
        side.addWidget(self.more_button)

        recent_title = QLabel("最近")
        recent_title.setObjectName("SidebarSection")
        side.addWidget(recent_title)
        self.session_search = QLineEdit()
        self.session_search.setObjectName("SessionSearch")
        self.session_search.setPlaceholderText("搜索本地对话")
        self.session_search.textChanged.connect(self._refresh_session_list)
        side.addWidget(self.session_search)
        self.session_list = QListWidget()
        self.session_list.setObjectName("SessionList")
        self.session_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.session_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.session_list.itemClicked.connect(self._open_session_item)
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._session_context_menu)
        side.addWidget(self.session_list, 1)

        self.selection_summary = QLabel("提问前请选择资料")
        self.selection_summary.setObjectName("SelectionSummary")
        self.selection_summary.setWordWrap(True)
        side.addWidget(self.selection_summary)
        select_material = QPushButton("选择资料")
        select_material.setProperty("secondary", True)
        select_material.clicked.connect(self._choose_materials)
        side.addWidget(select_material)
        self.side_status = QLabel("正在检查本地环境…")
        self.side_status.setObjectName("RuntimeState")
        self.side_status.setWordWrap(True)
        side.addWidget(self.side_status)

        # 核心方法仍通过这个隐藏选择器读取最多四本教材，避免另造一条问答路径。
        self.library_picker = QListWidget()
        self.library_picker.hide()
        self.library_picker.itemChanged.connect(self._picker_changed)
        self.picker_hint = QLabel()
        self.picker_hint.hide()

        content = QWidget()
        content.setObjectName("ContentRoot")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(34, 24, 34, 12)
        self.stack = QStackedWidget()
        pages = (
            ("chat", self._make_chat_page()),
            ("library", self._make_library_page()),
            ("evidence", self._page("证据审查", "只查看召回证据，不调用回答模型。双击结果可打开原文。",
                                    self._retrieve_tab())),
            ("chunks", self._make_chunks_page()),
            ("health", self._make_health_page()),
            ("settings", self._make_settings_page()),
            ("status", self._page("运行状态", "检查本机服务、模型、数据目录和依赖版本。",
                                  self._system_tab())),
            ("learning", self._make_learning_page()),
            ("batch", self._make_batch_page()),
            ("compare", self._page("A/B 对比", "同一问题受控比较向量检索与混合检索。",
                                   self._compare_tab())),
            ("concept", self._page("概念对照", "在 2–4 本教材之间逐本检索同一概念。",
                                   self._concept_tab())),
            ("feedback", self._page("反馈与回归集", "保存本地反馈、复跑失败样本并导出回归集。",
                                    self._feedback_tab())),
        )
        for key, widget in pages:
            self._page_indexes[key] = self.stack.addWidget(widget)
        content_layout.addWidget(self.stack)
        layout.addWidget(self.sidebar)
        layout.addWidget(content, 1)
        self.nav.currentRowChanged.connect(self._navigate_main)
        self.nav.setCurrentRow(0)
        QTimer.singleShot(0, self._fit_navigation)

        self.busy_bar = QProgressBar()
        self.busy_bar.setMaximumWidth(170)
        self.busy_bar.hide()
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.busy_bar)

    def _fit_navigation(self) -> None:
        """Keep More directly below navigation at every Windows DPI scale."""
        heights = [self.nav.sizeHintForRow(index) for index in range(self.nav.count())]
        height = sum(value if value > 0 else 48 for value in heights) + 4
        self.nav.setFixedHeight(height)

    def _toggle_focus_mode(self, enabled: bool) -> None:
        self.sidebar.setVisible(not enabled)
        self.focus_badge.setText("退出专注" if enabled else "专注模式")
        self.focus_badge.setToolTip("恢复侧栏" if enabled else "隐藏侧栏，扩大问答区域")

    def _show_session_metrics(self) -> None:
        values = self._session_metrics
        average = (int(values["total_ms"]) / int(values["queries"])) if values["queries"] else 0
        QMessageBox.information(
            self, "本轮运行统计",
            "生成请求：%d\n仅检索请求：%d\n累计 Token：%d\n平均耗时：%.1f 秒\n"
            "无依据拒答：%d\n本次查询资料：%d\n\n统计只保存在当前程序进程，不上传。" % (
                values["queries"], values["retrievals"], values["tokens"], average / 1000,
                values["refusals"], len(self.selected_libraries()) or 1))

    def _page(self, title: str, description: str, body: QWidget) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(page_header(title, description))
        layout.addWidget(body, 1)
        return page

    def _navigate_main(self, row: int) -> None:
        if 0 <= row < len(self.MAIN_PAGES):
            self.stack.setCurrentIndex(self._page_indexes[self.MAIN_PAGES[row]])

    def _show_page(self, key: str) -> None:
        index = self._page_indexes.get(key)
        if index is None:
            return
        self.stack.setCurrentIndex(index)
        if key in self.MAIN_PAGES:
            row = self.MAIN_PAGES.index(key)
            if self.nav.currentRow() != row:
                self.nav.setCurrentRow(row)
        else:
            self.nav.blockSignals(True)
            self.nav.clearSelection()
            self.nav.setCurrentRow(-1)
            self.nav.blockSignals(False)

    # ----------------------------- 网页式问答页 -----------------------------
    def _make_chat_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(page_header("问答", "每句话可溯源到原文页码 · 无依据即拒答"), 1)
        self.round_badge = QLabel("本轮 0 次")
        self.round_badge.setObjectName("TopBadge")
        self.focus_badge = QPushButton("专注模式")
        self.focus_badge.setObjectName("TopBadge")
        self.focus_badge.setCheckable(True)
        self.focus_badge.clicked.connect(self._toggle_focus_mode)
        self.library_badge = QLabel("尚未选择资料")
        self.library_badge.setObjectName("TopBadge")
        header.addWidget(self.round_badge)
        header.addWidget(self.focus_badge)
        header.addWidget(self.library_badge)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.chat_view = QTextBrowser()
        self.chat_view.setObjectName("ChatView")
        self.chat_view.setOpenLinks(False)
        self.chat_view.anchorClicked.connect(self._open_chat_link)
        self._welcome_html = (
            "<div style='margin:46px 42px;padding:0;line-height:1.65'>"
            "<p style='margin:0 0 14px;color:#276397;font-size:12px;font-weight:600'>当前资料演示</p>"
            "<p style='margin:0 0 16px;color:#17324f;font-size:17px;font-weight:600'>"
            "这个系统只依据知识库回答，没有依据就明确说不知道。</p>"
            "<p style='margin:0 0 12px;color:#718197;font-size:12px'>"
            "问题意图 → 检索证据 → 引用校验 → 可信回答</p>"
            "<p style='margin:0;color:#8794a5;font-size:12px'>"
            "选择资料后，输入与教材内容直接相关的问题。</p></div>")
        self.chat_view.setHtml(self._welcome_html)
        splitter.addWidget(self.chat_view)
        self.source_panel = QFrame()
        self.source_panel.setObjectName("EvidencePanel")
        source_layout = QVBoxLayout(self.source_panel)
        source_title = QHBoxLayout()
        source_title.addWidget(QLabel("引用来源与证据片段"))
        close_sources = QPushButton("收起")
        close_sources.setProperty("secondary", True)
        close_sources.clicked.connect(lambda: self.source_panel.hide())
        source_title.addStretch(1)
        source_title.addWidget(close_sources)
        source_layout.addLayout(source_title)
        self.source_tree = QTreeWidget()
        self.source_tree.setHeaderLabels(["引用与证据片段"])
        self.source_tree.setWordWrap(True)
        self.source_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.source_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.source_tree.itemDoubleClicked.connect(self._open_source_item)
        source_layout.addWidget(self.source_tree, 1)
        self.source_panel.setMinimumWidth(340)
        self.source_panel.hide()
        splitter.addWidget(self.source_panel)
        splitter.setSizes([950, 390])
        layout.addWidget(splitter, 1)

        feedback = QHBoxLayout()
        feedback.addWidget(QLabel("本轮反馈"))
        self.feedback_buttons = []
        for label, kind in (("有用", "useful"), ("待改进", "needs-improvement"),
                            ("没回答问题", "no-answer"), ("引用不正确", "bad-citation"),
                            ("证据不足", "insufficient"), ("太慢", "slow")):
            button = QPushButton(label)
            button.setProperty("chip", True)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, value=kind: self._send_feedback(value))
            feedback.addWidget(button)
            self.feedback_buttons.append(button)
        feedback.addStretch(1)
        self.source_toggle = QPushButton("查看证据")
        self.source_toggle.setProperty("secondary", True)
        self.source_toggle.clicked.connect(lambda: self.source_panel.setVisible(not self.source_panel.isVisible()))
        self.source_toggle.hide()
        feedback.addWidget(self.source_toggle)
        layout.addLayout(feedback)

        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        self.copy_answer_button = QPushButton("复制回答")
        self.copy_answer_button.setProperty("secondary", True)
        self.copy_answer_button.clicked.connect(self._copy_latest_answer)
        self.favorite_button = QPushButton("收藏")
        self.favorite_button.setProperty("secondary", True)
        self.favorite_button.setCheckable(True)
        self.favorite_button.clicked.connect(self._toggle_latest_favorite)
        self.regenerate_button = QPushButton("重新生成")
        self.regenerate_button.setProperty("secondary", True)
        self.regenerate_button.clicked.connect(self._regenerate_latest)
        for button in (self.copy_answer_button, self.favorite_button, self.regenerate_button):
            button.setEnabled(False)
            result_actions.addWidget(button)
        layout.addLayout(result_actions)

        material_bar = QFrame()
        material_bar.setObjectName("MaterialBar")
        material_layout = QHBoxLayout(material_bar)
        material_layout.setContentsMargins(14, 8, 10, 8)
        self.chat_material_summary = QLabel("📚 提问前请先选择资料")
        self.chat_material_summary.setWordWrap(True)
        material_button = QPushButton("选择资料")
        material_button.setProperty("secondary", True)
        material_button.clicked.connect(self._choose_materials)
        material_layout.addWidget(self.chat_material_summary, 1)
        material_layout.addWidget(material_button)
        layout.addWidget(material_bar)

        self.question_edit = QuestionEdit()
        self.question_edit.setObjectName("Composer")
        self.question_edit.setPlaceholderText("输入与当前教材相关的问题…")
        self.question_edit.setMaximumHeight(105)
        self.question_edit.submit.connect(self._send_chat)
        layout.addWidget(self.question_edit)

        controls = QHBoxLayout()
        self.mode_combo = CrispComboBox()
        self.mode_combo.addItem("Agent 校验", "auto")
        self.mode_combo.addItem("快速回答", "fast")
        self.mode_combo.addItem("深入回答", "deep")
        self.style_combo = CrispComboBox()
        self.style_combo.addItem("标准", "standard")
        self.style_combo.addItem("简洁", "concise")
        self.style_combo.addItem("详细", "detailed")
        self.hybrid_check = QCheckBox("混合检索")
        self.hybrid_check.setToolTip("默认关闭；明确需要关键词补召回时再启用。")
        self.extend_check = QCheckBox("教材外补充")
        settings_button = QPushButton("检索设置")
        settings_button.setProperty("secondary", True)
        settings_button.clicked.connect(self._open_retrieval_settings)
        quiz = QPushButton("测试题")
        quiz.setProperty("secondary", True)
        quiz.clicked.connect(lambda: self._open_learning_tab(1))
        brief = QPushButton("简报")
        brief.setProperty("secondary", True)
        brief.clicked.connect(lambda: self._open_learning_tab(0))
        self.send_button = QPushButton("提问")
        self.send_button.setObjectName("AskButton")
        self.send_button.clicked.connect(self._send_chat)
        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_chat)
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.style_combo)
        controls.addWidget(self.hybrid_check)
        controls.addWidget(self.extend_check)
        controls.addWidget(settings_button)
        controls.addStretch(1)
        controls.addWidget(quiz)
        controls.addWidget(brief)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.send_button)
        layout.addLayout(controls)
        return page

    def _open_learning_tab(self, index: int) -> None:
        self._show_page("learning")
        self.learning_tabs.setCurrentIndex(index)

    def _open_retrieval_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("检索与表达设置")
        dialog.resize(560, 390)
        layout = QVBoxLayout(dialog)
        heading = QLabel("限定检索页范围")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        note = QLabel("留空即检索全书。范围内没有依据时会拒答，不会偷偷使用范围外内容。")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        scope_row = QHBoxLayout()
        scope_from = QSpinBox()
        scope_from.setRange(0, 100000)
        scope_from.setSpecialValueText("不限")
        scope_to = QSpinBox()
        scope_to.setRange(0, 100000)
        scope_to.setSpecialValueText("不限")
        scope_from.setValue(int((self._page_scope or {}).get("from") or 0))
        scope_to.setValue(int((self._page_scope or {}).get("to") or 0))
        scope_row.addWidget(QLabel("从第"))
        scope_row.addWidget(scope_from)
        scope_row.addWidget(QLabel("页，到第"))
        scope_row.addWidget(scope_to)
        scope_row.addWidget(QLabel("页"))
        scope_row.addStretch(1)
        layout.addLayout(scope_row)
        layout.addSpacing(8)
        instruction_label = QLabel("工作区角色说明")
        instruction_label.setObjectName("SectionTitle")
        layout.addWidget(instruction_label)
        instruction = QPlainTextEdit()
        instruction.setPlaceholderText("例如：面向本科生，用通俗中文解释专业术语。（最多 300 字）")
        instruction.setPlainText(self._workspace_instruction)
        instruction.setMaximumHeight(130)
        layout.addWidget(instruction)
        safety = QLabel("角色说明只影响表达方式，不能绕过只依据资料、引用校验和无依据拒答。")
        safety.setObjectName("Muted")
        safety.setWordWrap(True)
        layout.addWidget(safety)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存到本机")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        start, end = scope_from.value(), scope_to.value()
        if start and end and start > end:
            show_error(self, "页范围无效", "起始页不能大于结束页。")
            return
        self._page_scope = ({"from": start or None, "to": end or None}
                            if start or end else None)
        self._workspace_instruction = instruction.toPlainText().strip()[:300]
        settings = QSettings()
        settings.setValue("chat/page_from", start)
        settings.setValue("chat/page_to", end)
        settings.setValue("chat/workspace_instruction", self._workspace_instruction)
        if self._page_scope:
            lo = self._page_scope.get("from") or ""
            hi = self._page_scope.get("to") or ""
            self.statusBar().showMessage("已限定检索页范围：%s–%s" % (lo or "开头", hi or "末尾"), 5000)
        else:
            self.statusBar().showMessage("已恢复全书检索", 4000)

    def _open_chat_link(self, url: QUrl) -> None:
        """Open evidence links inside the native source/PDF dialog."""
        if url.scheme() != "aitic-source":
            return
        token = url.host() or url.path().strip("/")
        try:
            index = int(token)
        except (TypeError, ValueError):
            return
        sources = getattr(self, "_chat_sources", [])
        if 0 <= index < len(sources):
            SourceDialog(self.backend, sources[index], self).exec()

    @staticmethod
    def _result_styles() -> str:
        return (
            "<style>"
            "p{margin:0 0 10px 0;line-height:1.62}"
            "h2,h3,h4{margin:12px 0 8px;color:#183a5e}"
            "ul,ol{margin:6px 0 10px 24px}li{margin:3px 0}"
            ".gap{height:6px}.cite{color:#245f96;font-weight:600}"
            "code{background:#eef3f8;color:#173a5e;padding:1px 4px}"
            "blockquote{margin:8px 0;padding:7px 11px;background:#f4f7fa;color:#53677d}"
            "a{color:#245f96;text-decoration:none}"
            "</style>"
        )

    def _rich_answer_block(self, payload: dict[str, Any]) -> str:
        """Render the complete verified response payload, not only answer text."""
        answer = str(payload.get("answer") or "")
        agent = payload.get("agent") or {}
        confidence = agent.get("confidence") or {}
        audit = agent.get("support_audit") or {}
        chain = agent.get("evidence_chain") or {}
        cite = payload.get("cite_check") or {}
        sources = [dict(value) for value in (payload.get("sources") or [])]
        self._chat_sources = sources
        parts = [self._result_styles(),
                 '<div style="margin:10px 0 16px;padding:16px 18px;background:#ffffff;'
                 'border:1px solid #d9e3ed;border-radius:11px">',
                 '<div style="font-size:13px;font-weight:600;color:#173a5e;margin-bottom:10px">AITIC</div>']

        if payload.get("escalated"):
            parts.append(
                '<div style="margin:0 0 12px;padding:9px 11px;background:#fff7e8;'
                'border-left:3px solid #df8a10;color:#9b5a05;font-weight:600">'
                '已进入 Agent 补充检索：扩大证据范围并重新核验</div>')

        supplement = payload.get("supplement") or {}
        supplement_text = str(supplement.get("text") or "")
        if supplement_text:
            parts.append(
                '<div style="margin:0 0 14px;padding:11px 13px;background:#fffcf5;'
                'border:1px dashed #e0b45a;border-left:3px solid #d98a3a;border-radius:8px">'
                '<div style="font-weight:700;color:#8c5a00">完整解答　'
                '<span style="font-size:11px;border:1px solid #e0b45a;background:#fff3d9;'
                'padding:2px 7px">含模型常识 · 未经溯源</span></div>'
                '<div style="margin-top:5px;color:#9a6b1e;font-size:11px">本段包含教材之外的内容，'
                '不计入下方的引用校验与可信度；严格溯源请只看“教材依据”。</div>'
                '<div style="margin-top:8px;color:#3a3020">%s</div></div>'
                '<div style="margin:10px 0 8px;font-weight:700;color:#173a5e">教材依据　'
                '<span style="font-size:11px;color:#2c6e49">仅含可回查原文的结论</span></div>'
                % answer_html(supplement_text))
        if payload.get("abstained"):
            parts.append(
                '<div style="padding:12px;background:#fff8ed;border:1px solid #f0d7aa;'
                'border-radius:8px;color:#7a4b08"><b>证据不足，已诚实拒答</b><br>'
                '当前所选资料不足以支持可交付结论。</div>')
        else:
            parts.append('<div style="font-size:14px;color:#142a44">%s</div>' % answer_html(answer))

        hit_count = len(cite.get("hit") or [])
        total = int(cite.get("total") or 0)
        if total:
            ok_color = "#168153" if cite.get("ok") else "#b7483f"
            parts.append(
                '<div style="margin-top:13px;padding:7px 10px;background:#f2faf5;'
                'border-radius:7px;color:%s;font-weight:600">引用校验 %d/%d 命中检索来源</div>'
                % (ok_color, hit_count, total))

        if sources:
            parts.append(
                '<div style="margin-top:15px;border-top:1px solid #dde5ee;padding-top:11px">'
                '<b>本轮检索证据</b><span style="color:#7b899b">（点击打开入库原文和 PDF 原页）</span>')
            for index, source in enumerate(sources[:12]):
                label = escaped(source.get("label") or source.get("source") or "来源")
                kind = escaped(source.get("type") or "text")
                snippet = escaped(source.get("snippet") or "暂无片段")
                distance = source.get("distance")
                distance_text = (" · 距离 %.4f" % float(distance)
                                 if isinstance(distance, (int, float)) else "")
                parts.append(
                    '<div style="margin:7px 0;padding:9px 11px;background:#f6f9fc;'
                    'border-left:3px solid #4b86c2">'
                    '<b>%s</b> <span style="color:#708197">%s%s</span><br>'
                    '<span style="color:#52677d">%s</span><br>'
                    '<a href="aitic-source://%d">查看原文 / 原页</a></div>'
                    % (label, kind, distance_text, snippet, index))
            parts.append("</div>")

        if audit.get("triggered"):
            state = str(audit.get("state") or "")
            tint = "#f2faf5" if state in {"verified", "pruned"} else "#fff8ed"
            orphaned = int(audit.get("orphaned") or 0)
            audit_detail = escaped(audit.get("reason") or "")
            if orphaned:
                audit_detail += "；另有 %d 条因先行词被移除而失去指代对象，一并删去" % orphaned
            parts.append(
                '<div style="margin-top:13px;padding:9px 11px;background:%s;border-radius:7px">'
                '<b>逐句语义核验</b> · 检查 %d 条，移除 %d 条，未判定 %d 条<br>'
                '<span style="color:#61748a">%s</span></div>'
                % (tint, int(audit.get("checked") or 0), int(audit.get("pruned") or 0),
                   int(audit.get("unknown") or 0), audit_detail))

        basis = chain.get("basis") or []
        if basis:
            parts.append(
                '<div style="margin-top:13px;padding-top:10px;border-top:1px solid #dde5ee">'
                '<b>证据链判断</b>')
            for item in basis:
                verified = item.get("measured") is True and item.get("supported") is True
                supported = bool(item.get("supported"))
                marker = "原文匹配" if verified else ("支持" if supported else "需复核")
                color = "#168153" if verified or supported else "#aa6a0a"
                citations = " ".join("[%s]" % escaped(value) for value in (item.get("citations") or []))
                grounding = item.get("grounding")
                grounding_text = (" · 接地率 %.2f" % float(grounding)
                                  if isinstance(grounding, (int, float)) else "")
                parts.append(
                    '<div style="margin-top:7px;padding:7px 9px;background:#f8fafc">'
                    '<span style="color:%s;font-weight:600">%s</span>%s　%s %s</div>'
                    % (color, marker, grounding_text, escaped(item.get("claim") or ""), citations))
            for relation in chain.get("relations") or []:
                if isinstance(relation, dict):
                    detail = relation.get("detail") or relation.get("reason") or relation.get("type") or ""
                    parts.append('<div style="margin-top:5px;color:#53677d">证据关系：%s</div>' %
                                 escaped(detail))
            for warning in chain.get("uncertainty") or []:
                parts.append('<div style="margin-top:5px;color:#8a650c">注意：%s</div>' % escaped(warning))
            parts.append("</div>")

        signals = confidence.get("signals") or []
        if confidence:
            level = escaped(confidence.get("level") or "未评级")
            parts.append(
                '<div style="margin-top:13px;padding:10px 11px;background:#f5f8fb;'
                'border-radius:7px"><b>可信度：%s</b><br>'
                '<span style="color:#60738a">%s</span>'
                % (level, escaped(confidence.get("reason") or "")))
            for signal in signals:
                marker = "通过" if signal.get("ok") else "注意"
                color = "#168153" if signal.get("ok") else "#aa6a0a"
                parts.append('<div style="margin-top:5px"><span style="color:%s">%s</span> · '
                             '<b>%s</b>：%s</div>' % (
                                 color, marker, escaped(signal.get("name") or ""),
                                 escaped(signal.get("detail") or "")))
            parts.append("</div>")

        trace = agent.get("trace") or []
        if trace:
            parts.append(
                '<div style="margin-top:13px;padding-top:10px;border-top:1px solid #dde5ee">'
                '<b>Agent 运行步骤</b>')
            intent = agent.get("intent") or {}
            library_names = [str(value.get("name") or "") for value in
                             (agent.get("libraries") or []) if isinstance(value, dict)]
            parts.append('<div style="margin-top:6px;color:#53677d">意图：%s（%s）%s</div>' % (
                escaped(intent.get("name") or "事实查询"),
                escaped(intent.get("complexity") or "简单"),
                (" · 知识库：" + escaped(" + ".join(filter(None, library_names))))
                if any(library_names) else ""))
            for step in trace:
                parts.append('<div style="margin-top:5px;color:#53677d"><b>%s</b> · %s</div>' % (
                    escaped(step.get("step") or ""), escaped(step.get("detail") or "")))
            parts.append("</div>")

        diagnosis = agent.get("diagnosis")
        if diagnosis:
            if isinstance(diagnosis, dict):
                parts.append('<div style="margin-top:11px;padding:9px 11px;background:#f7fbff;'
                             'border-left:3px solid #4f81bd"><b>这次没答好，原因是：</b>%s' %
                             escaped(diagnosis.get("cause") or "证据链尚不完整"))
                for action in diagnosis.get("actions") or []:
                    if isinstance(action, dict):
                        parts.append('<div style="margin-top:5px;color:#3e5771">· <b>%s</b>：%s</div>' % (
                            escaped(action.get("label") or "建议"),
                            escaped(action.get("why") or "")))
                if diagnosis.get("note"):
                    parts.append('<div style="margin-top:6px;color:#718197">%s</div>' %
                                 escaped(diagnosis.get("note")))
                parts.append('</div>')
            else:
                parts.append('<div style="margin-top:10px;color:#8a650c"><b>诊断：</b>%s</div>' %
                             escaped(diagnosis))
        dropped = payload.get("dropped_libraries") or []
        if dropped:
            names = [str(value.get("name") or value.get("id") or value)
                     if isinstance(value, dict) else str(value) for value in dropped]
            parts.append('<div style="margin-top:9px;color:#a34d42">未能读取的资料：%s</div>' %
                         escaped("、".join(names)))

        footer = "%d ms · %s token · %d 轮 Agent · %s" % (
            int(payload.get("elapsed_ms") or 0), escaped(payload.get("tokens") or 0),
            int(agent.get("rounds") or payload.get("rounds") or 0),
            escaped(agent.get("stop_reason") or "已完成"))
        parts.append('<div style="margin-top:13px;color:#718197;font-size:11px">%s</div></div>' % footer)
        return "".join(parts)

    def _append_rich_answer(self, payload: dict[str, Any], meta: str = "") -> None:
        answer = str(payload.get("answer") or "")
        block = self._rich_answer_block(payload)
        self._chat_parts.append(block)
        self.chat_view.setHtml("".join(self._chat_parts))
        self._latest_payload = dict(payload)
        if hasattr(self, "round_badge"):
            turns = sum(1 for value in self._history if value.get("role") == "assistant") + 1
            self.round_badge.setText("本轮 %d 次" % turns)
        if not self._restoring_session:
            self._store_session_message("assistant", answer, meta, payload)
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _copy_latest_answer(self) -> None:
        if not self._latest_answer:
            return
        QApplication.clipboard().setText(self._latest_answer)
        self.statusBar().showMessage("回答已复制到剪贴板", 3000)

    def _toggle_latest_favorite(self, checked: bool) -> None:
        session = self._current_session()
        if not session:
            self.favorite_button.setChecked(False)
            return
        for message in reversed(session.get("messages") or []):
            if message.get("role") == "assistant":
                message["favorite"] = bool(checked)
                break
        self.favorite_button.setText("已收藏" if checked else "收藏")
        self._save_sessions()

    def _regenerate_latest(self) -> None:
        if self._chat_busy or not self._latest_question:
            return
        self.question_edit.setPlainText(self._latest_question)
        QTimer.singleShot(0, self._send_chat)

    def _render_stream_preview(self, text: str, note: str = "已核验，正在显示…") -> None:
        preview = (
            '<div style="margin:9px 0;padding:12px 14px;background:#ffffff;border:1px solid #dce5ef;'
            'border-radius:9px"><b>AITIC</b><div style="margin-top:6px;line-height:1.55">%s'
            '<span style="color:#477eae"> ▍</span></div><div style="color:#738399;font-size:11px;'
            'margin-top:7px">%s</div></div>' % (text_html(text), escaped(note)))
        self.chat_view.setHtml("".join(self._chat_parts) + preview)
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send_chat(self) -> None:
        if self._chat_busy:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            return
        if len(question) > 4000:
            show_error(self, "问题过长", "单次最多 4000 个字符，请拆分后重试。")
            return
        libraries = self.selected_libraries()
        history = list(self._history)
        mode = str(self.mode_combo.currentData())
        style = str(self.style_combo.currentData())
        hybrid = self.hybrid_check.isChecked()
        extend = self.extend_check.isChecked()
        instruction = self._workspace_instruction
        page_scope = dict(self._page_scope) if self._page_scope else None
        self.question_edit.clear()
        self._append_chat("user", question)
        self._chat_generation += 1
        generation = self._chat_generation
        self._chat_busy = True
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for button in self.feedback_buttons:
            button.setEnabled(False)

        stream_text = {"value": ""}
        cancel_event = threading.Event()
        self._chat_cancel_event = cancel_event
        task: Task

        def work() -> dict[str, Any]:
            if hasattr(self.backend, "ask_stream"):
                return self.backend.ask_stream(
                    question, libraries=libraries, history=history, mode=mode, style=style,
                    instruction=instruction, hybrid=hybrid, extend=extend,
                    page_scope=page_scope,
                    cancel_event=cancel_event,
                    on_event=lambda event, data: task.signals.progress.emit(
                        json.dumps({"event": event, "data": data}, ensure_ascii=False)))
            return self.backend.ask(
                question, libraries=libraries, history=history, mode=mode, style=style,
                instruction=instruction, hybrid=hybrid, extend=extend,
                page_scope=page_scope)

        task = Task(work)
        self._tasks.add(task)
        self._busy_count += 1
        self.busy_bar.setRange(0, 0)
        self.busy_bar.show()
        self.status_label.setText("正在检索并核验答案…")

        def progress(raw: str) -> None:
            if generation != self._chat_generation:
                return
            try:
                event = json.loads(raw)
                kind, data = str(event.get("event") or ""), event.get("data") or {}
            except (ValueError, TypeError):
                return
            if kind == "verified_delta":
                stream_text["value"] += str(data.get("text") or "")
                self._render_stream_preview(stream_text["value"])
            elif kind == "agent":
                self.status_label.setText(str(data.get("label") or "正在校验证据…"))
            elif kind == "retrieved":
                self.status_label.setText("已召回 %d 条证据，正在生成并校验…" % int(data.get("n") or 0))
            elif kind == "escalate":
                self.status_label.setText(str(data.get("reason") or "正在补充检索…"))
            elif kind == "supplement_start":
                self.status_label.setText(str(data.get("label") or "正在生成教材外补充…"))

        def result(payload: dict[str, Any]) -> None:
            if generation != self._chat_generation:
                return
            self.chat_view.setHtml("".join(self._chat_parts))
            answer = str(payload.get("answer") or "")
            agent = payload.get("agent") or {}
            confidence = (agent.get("confidence") or {}).get("level") or "未评级"
            meta = "%d ms · %s token · %s · 可信度 %s" % (
                int(payload.get("elapsed_ms") or 0), payload.get("tokens", 0),
                "补充检索" if payload.get("escalated") else "单轮", confidence)
            # Keep the complete Agent/evidence payload.  Flattening this to
            # `_append_chat` made the desktop answer look as if Agent validation
            # had never run and also made restart restoration impossible.
            self._append_rich_answer(payload, meta)
            self._history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            self._history = self._history[-12:]
            self._latest_question = question
            self._latest_answer = answer
            self._latest_libraries = list(libraries)
            self._show_sources(payload.get("sources") or [])
            self._session_metrics["queries"] += 1
            self._session_metrics["tokens"] += int(payload.get("tokens") or 0)
            self._session_metrics["total_ms"] += int(payload.get("elapsed_ms") or 0)
            self._session_metrics["refusals"] += int(bool(payload.get("abstained")))
            for button in self.feedback_buttons:
                button.setEnabled(True)
            for button in (self.copy_answer_button, self.favorite_button,
                           self.regenerate_button):
                button.setEnabled(True)
            self.favorite_button.setChecked(False)
            self.favorite_button.setText("收藏")

        def error(message: str) -> None:
            if generation == self._chat_generation:
                self.chat_view.setHtml("".join(self._chat_parts))
                self._append_chat("system", "本轮失败：" + message)

        def finished() -> None:
            self._tasks.discard(task)
            self._busy_count = max(0, self._busy_count - 1)
            if generation == self._chat_generation:
                self._chat_busy = False
                self.send_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self._chat_cancel_event = None
            if not self._busy_count:
                self.busy_bar.hide()
                self.status_label.setText("就绪")

        task.signals.progress.connect(progress)
        task.signals.result.connect(result)
        task.signals.error.connect(error)
        task.signals.finished.connect(finished)
        self.pool.start(task)

    def _stop_chat(self) -> None:
        if self._chat_cancel_event is not None:
            self._chat_cancel_event.set()
        super()._stop_chat()

    def _append_chat(self, role: str, text: str, meta: str = "") -> None:
        super()._append_chat(role, text, meta)
        if role == "assistant" and hasattr(self, "round_badge"):
            turns = sum(1 for value in self._history if value.get("role") == "assistant") + 1
            self.round_badge.setText("本轮 %d 次" % turns)
        if not self._restoring_session:
            self._store_session_message(role, text, meta)

    def _show_sources(self, sources: list[dict[str, Any]]) -> None:
        self._chat_sources = [dict(value) for value in sources]
        self.source_tree.clear()
        for source in self._chat_sources:
            text = "%s\n%s" % (str(source.get("label") or "来源"),
                                str(source.get("snippet") or ""))
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.UserRole, source)
            item.setToolTip(0, text)
            self.source_tree.addTopLevelItem(item)
        self.source_toggle.setVisible(bool(sources))
        if sources:
            self.source_panel.show()
        session = self._current_session()
        if session and session.get("messages"):
            for message in reversed(session["messages"]):
                if message.get("role") == "assistant":
                    message["sources"] = [dict(source) for source in sources]
                    break
            self._save_sessions()

    def _clear_chat(self) -> None:
        super()._clear_chat()
        self._chat_busy = False
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._current_session_id = ""
        self._pending_library_ids = []
        self._latest_payload = {}
        self.chat_view.setHtml(self._welcome_html)
        self.source_panel.hide()
        self.source_toggle.hide()
        self.round_badge.setText("本轮 0 次")
        self._session_metrics = {"queries": 0, "tokens": 0, "total_ms": 0,
                                 "refusals": 0, "retrievals": 0}
        for button in (self.copy_answer_button, self.favorite_button,
                       self.regenerate_button):
            button.setEnabled(False)
        self.favorite_button.setChecked(False)
        self.favorite_button.setText("收藏")
        self._refresh_session_list()

    def _new_chat(self) -> None:
        self._clear_chat()
        self._show_page("chat")
        self.question_edit.setFocus()

    # ----------------------------- 本地会话 -----------------------------
    def _load_sessions(self) -> None:
        try:
            payload = self.backend.load_sessions(80) if hasattr(self.backend, "load_sessions") else {}
            self._sessions = list(payload.get("sessions") or [])
        except Exception:
            self._sessions = []
        self._refresh_session_list()

    def _current_session(self) -> dict[str, Any] | None:
        for session in self._sessions:
            if session.get("id") == self._current_session_id:
                return session
        return None

    def _ensure_session(self, first_text: str = "") -> dict[str, Any]:
        session = self._current_session()
        if session:
            return session
        title = re.sub(r"\s+", " ", first_text).strip()[:34] or "新对话"
        session = {
            "id": uuid.uuid4().hex,
            "title": title,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "pinned": False,
            "library_ids": self.selected_libraries(),
            "messages": [],
        }
        self._current_session_id = session["id"]
        self._sessions.insert(0, session)
        return session

    def _store_session_message(self, role: str, text: str, meta: str = "",
                               payload: dict[str, Any] | None = None) -> None:
        session = self._ensure_session(text if role == "user" else "")
        message = {"role": role, "content": text, "meta": meta, "sources": []}
        if payload:
            message["payload"] = dict(payload)
        session["messages"].append(message)
        session["messages"] = session["messages"][-100:]
        session["library_ids"] = self.selected_libraries()
        session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if session.get("title") == "新对话" and role == "user":
            session["title"] = re.sub(r"\s+", " ", text).strip()[:34] or "新对话"
        self._sessions.sort(key=lambda value: value.get("updated_at", ""), reverse=True)
        self._save_sessions()
        self._refresh_session_list()

    def _save_sessions(self) -> None:
        if not hasattr(self.backend, "save_sessions"):
            return
        try:
            self.backend.save_sessions(self._sessions[:80])
        except Exception as exc:
            self.statusBar().showMessage("会话保存失败：%s" % exc, 5000)

    def _refresh_session_list(self) -> None:
        if not hasattr(self, "session_list"):
            return
        query = self.session_search.text().strip().casefold() if hasattr(self, "session_search") else ""
        self.session_list.blockSignals(True)
        self.session_list.clear()
        ordered = sorted(self._sessions, key=lambda value: (
            bool(value.get("pinned")), str(value.get("updated_at") or "")), reverse=True)
        pinned = [value for value in ordered if value.get("pinned")]
        normal = sorted((value for value in ordered if not value.get("pinned")),
                        key=lambda value: value.get("updated_at", ""), reverse=True)
        for session in pinned + normal:
            title = str(session.get("title") or "新对话")
            if query and query not in title.casefold():
                continue
            item = QListWidgetItem(("📌 " if session.get("pinned") else "") + title)
            item.setData(Qt.UserRole, str(session.get("id") or ""))
            item.setToolTip(str(session.get("updated_at") or ""))
            self.session_list.addItem(item)
            if session.get("id") == self._current_session_id:
                item.setSelected(True)
        self.session_list.blockSignals(False)

    def _session_context_menu(self, point) -> None:
        item = self.session_list.itemAt(point)
        if not item:
            return
        session_id = str(item.data(Qt.UserRole) or "")
        session = next((value for value in self._sessions if value.get("id") == session_id), None)
        if not session:
            return
        menu = QMenu(self)
        pin = menu.addAction("取消置顶" if session.get("pinned") else "置顶对话")
        rename = menu.addAction("重命名")
        menu.addSeparator()
        remove = menu.addAction("删除对话")
        chosen = menu.exec(self.session_list.viewport().mapToGlobal(point))
        if chosen is pin:
            session["pinned"] = not bool(session.get("pinned"))
            session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        elif chosen is rename:
            title, ok = QInputDialog.getText(
                self, "重命名对话", "对话名称：", text=str(session.get("title") or "新对话"))
            if not ok or not title.strip():
                return
            session["title"] = title.strip()[:80]
            session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        elif chosen is remove:
            answer = QMessageBox.question(
                self, "删除对话", "确定删除“%s”吗？此操作只删除本地会话记录。" %
                str(session.get("title") or "新对话"))
            if answer != QMessageBox.Yes:
                return
            self._sessions = [value for value in self._sessions if value.get("id") != session_id]
            if self._current_session_id == session_id:
                self._clear_chat()
        else:
            return
        self._save_sessions()
        self._refresh_session_list()

    def _open_session_item(self, item: QListWidgetItem) -> None:
        session_id = str(item.data(Qt.UserRole) or "")
        session = next((value for value in self._sessions if value.get("id") == session_id), None)
        if not session:
            return
        self._restoring_session = True
        try:
            self._chat_generation += 1
            self._chat_busy = False
            self.send_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self._current_session_id = session_id
            self._chat_parts.clear()
            self._history.clear()
            self.source_tree.clear()
            latest_sources: list[dict[str, Any]] = []
            latest_favorite = False
            latest_question = latest_answer = ""
            for message in session.get("messages") or []:
                role = str(message.get("role") or "system")
                content = str(message.get("content") or "")
                rich_payload = message.get("payload")
                if role == "assistant" and isinstance(rich_payload, dict):
                    self._append_rich_answer(rich_payload, str(message.get("meta") or ""))
                else:
                    _CoreMainWindow._append_chat(
                        self, role, content, str(message.get("meta") or ""))
                if role in ("user", "assistant"):
                    self._history.append({"role": role, "content": content})
                if role == "user":
                    latest_question = content
                elif role == "assistant":
                    latest_answer = content
                    latest_sources = [dict(source) for source in (message.get("sources") or [])]
                    latest_favorite = bool(message.get("favorite"))
            self._history = self._history[-12:]
            self._latest_question = latest_question
            self._latest_answer = latest_answer
            self._latest_libraries = list(session.get("library_ids") or [])
            self._pending_library_ids = list(self._latest_libraries)
            self._set_selected_libraries(self._pending_library_ids)
            self._show_sources(latest_sources)
            self.source_toggle.setVisible(bool(latest_sources))
            self.source_panel.setVisible(bool(latest_sources))
            turns = sum(1 for value in session.get("messages") or [] if value.get("role") == "assistant")
            self.round_badge.setText("本轮 %d 次" % turns)
            for button in self.feedback_buttons:
                button.setEnabled(bool(latest_answer))
            for button in (self.copy_answer_button, self.favorite_button,
                           self.regenerate_button):
                button.setEnabled(bool(latest_answer))
            self.favorite_button.setChecked(latest_favorite)
            self.favorite_button.setText("已收藏" if latest_favorite else "收藏")
        finally:
            self._restoring_session = False
        self._show_page("chat")
        self._update_picker_hint()

    def _conversation_text(self) -> str:
        session = self._current_session()
        messages = (session or {}).get("messages") or []
        labels = {"user": "你", "assistant": "AITIC", "system": "系统"}
        parts = ["# %s" % str((session or {}).get("title") or "AITIC 对话")]
        for message in messages:
            parts.append("\n## %s\n\n%s" % (
                labels.get(str(message.get("role")), "系统"), str(message.get("content") or "")))
            if message.get("meta"):
                parts.append("\n_%s_" % message.get("meta"))
        return "\n".join(parts).strip() + "\n"

    def _copy_conversation(self) -> None:
        text = self._conversation_text()
        if text.strip() == "# AITIC 对话":
            QMessageBox.information(self, "没有对话", "当前还没有可复制的对话。")
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("当前对话已复制到剪贴板", 3500)

    def _export_conversation(self) -> None:
        session = self._current_session()
        if not session or not session.get("messages"):
            QMessageBox.information(self, "没有对话", "当前还没有可导出的对话。")
            return
        suggested = re.sub(r'[<>:"/\\|?*]+', "-", str(session.get("title") or "AITIC 对话"))
        path, _selected = QFileDialog.getSaveFileName(
            self, "导出当前对话", str(Path.home() / (suggested + ".md")),
            "Markdown (*.md);;文本文件 (*.txt);;所有文件 (*)")
        if not path:
            return
        try:
            Path(path).write_text(self._conversation_text(), encoding="utf-8")
        except OSError as exc:
            show_error(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", "当前对话已保存：\n%s" % path)

    # ----------------------------- 资料选择与状态 -----------------------------
    def _choose_materials(self) -> None:
        ready = [value for value in self._libraries if value.get("status") == "ready"]
        if not ready:
            QMessageBox.information(self, "尚无资料", "请先进入“资料库”导入 PDF 或 EPUB 并完成建库。")
            self._show_page("library")
            return
        selected = set(self.selected_libraries())
        dialog = QDialog(self)
        dialog.setWindowTitle("选择本轮资料（最多 4 本）")
        dialog.resize(560, 430)
        box = QVBoxLayout(dialog)
        note = QLabel("回答只依据勾选的资料；未勾选时使用当前活动库。")
        note.setObjectName("Muted")
        box.addWidget(note)
        search = QLineEdit()
        search.setPlaceholderText("搜索教材名称")
        box.addWidget(search)
        choices = QListWidget()
        for library in ready:
            item = QListWidgetItem("%s  ·  %s 块" % (
                library.get("name") or library.get("id"), library.get("chunks") or 0))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if str(library.get("id")) in selected else Qt.Unchecked)
            item.setData(Qt.UserRole, str(library.get("id") or ""))
            choices.addItem(item)
        search.textChanged.connect(lambda value: [
            choices.item(index).setHidden(value.strip().casefold() not in choices.item(index).text().casefold())
            for index in range(choices.count())])
        box.addWidget(choices, 1)
        quick = QHBoxLayout()
        select_visible = QPushButton("选择当前可见（最多 4 本）")
        select_visible.setProperty("secondary", True)
        clear = QPushButton("清除选择")
        clear.setProperty("secondary", True)

        def choose_visible() -> None:
            chosen = 0
            for index in range(choices.count()):
                item = choices.item(index)
                checked = not item.isHidden() and chosen < 4
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                chosen += int(checked)

        select_visible.clicked.connect(choose_visible)
        clear.clicked.connect(lambda: [choices.item(index).setCheckState(Qt.Unchecked)
                                       for index in range(choices.count())])
        quick.addWidget(select_visible)
        quick.addWidget(clear)
        quick.addStretch(1)
        box.addLayout(quick)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        box.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        wanted = [str(choices.item(i).data(Qt.UserRole)) for i in range(choices.count())
                  if choices.item(i).checkState() == Qt.Checked]
        if len(wanted) > 4:
            QMessageBox.information(self, "最多四本", "单次最多选择 4 本教材，请减少选择。")
            return
        self._pending_library_ids = wanted
        self._set_selected_libraries(wanted)
        session = self._current_session()
        if session:
            session["library_ids"] = wanted
            session["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._save_sessions()

    def _set_selected_libraries(self, wanted: list[str]) -> None:
        ids = set(wanted)
        self._updating_picker = True
        for index in range(self.library_picker.count()):
            item = self.library_picker.item(index)
            library = item.data(Qt.UserRole) or {}
            item.setCheckState(Qt.Checked if str(library.get("id")) in ids else Qt.Unchecked)
        self._updating_picker = False
        self._update_picker_hint()

    def _update_picker_hint(self) -> None:
        if not hasattr(self, "library_picker"):
            return
        selected = self.selected_libraries()
        names = []
        for library in self._libraries:
            if str(library.get("id")) in selected:
                names.append(str(library.get("name") or library.get("id")))
        if names:
            short = "、".join(names[:2]) + (" 等 %d 本" % len(names) if len(names) > 2 else "")
            side = "已选 %d 本：%s" % (len(names), short)
            chat = "📚 当前已选 %d 本：%s。回答只依据所选资料。" % (len(names), short)
            badge = "正在检索：%s" % short
        else:
            active = next((str(value.get("name") or value.get("id")) for value in self._libraries
                           if value.get("active")), "尚未选择资料")
            side = "未勾选时使用当前活动库"
            chat = "📚 当前资料：%s。回答只依据当前资料。" % active
            badge = "正在检索：%s" % active
        if hasattr(self, "picker_hint"):
            self.picker_hint.setText(side)
        if hasattr(self, "selection_summary"):
            self.selection_summary.setText(side)
        if hasattr(self, "chat_material_summary"):
            self.chat_material_summary.setText(chat)
        if hasattr(self, "library_badge"):
            self.library_badge.setText(badge)

    def _apply_libraries(self, payload: dict[str, Any]) -> None:
        wanted = list(self._pending_library_ids)
        super()._apply_libraries(payload)
        if wanted:
            self._set_selected_libraries(wanted)
        for combo_name in ("chunk_library_combo", "health_library_combo"):
            combo = getattr(self, combo_name, None)
            if combo is None:
                continue
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for library in self._libraries:
                if library.get("status") == "ready":
                    combo.addItem(str(library.get("name") or library.get("id")), str(library.get("id")))
            index = combo.findData(current or self._active_library)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._update_picker_hint()

    def _apply_status(self, status: dict[str, Any]) -> None:
        super()._apply_status(status)
        ready = bool(status.get("ready"))
        self.side_status.setText("● %s · %s 个知识块" % (
            status.get("ollama", "未知"), int(status.get("chunks") or 0)))
        self.side_status.setProperty("ready", ready)
        self.side_status.style().unpolish(self.side_status)
        self.side_status.style().polish(self.side_status)

    # ----------------------------- 独立功能页 -----------------------------
    def _make_library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(page_header(
            "资料库", "批量导入 PDF / EPUB、跟踪建库、切换当前教材并管理本地知识库。"))
        toolbar = QHBoxLayout()
        import_button = QPushButton("批量导入 PDF / EPUB")
        import_button.clicked.connect(lambda: self._choose_and_build(import_button))
        self.import_button = import_button
        self.max_pages = QSpinBox()
        self.max_pages.setRange(0, 10000)
        self.max_pages.setSpecialValueText("全部页")
        self.use_vl = QCheckBox("识别图表页")
        self.use_vl.setChecked(True)
        self.use_vl.setToolTip("仅对原始 PDF 生效；EPUB 会转换为可检索文本后建库。")
        self.vl_limit = QSpinBox()
        self.vl_limit.setRange(0, 100)
        self.vl_limit.setValue(15)
        activate = QPushButton("设为当前库")
        activate.setProperty("secondary", True)
        activate.clicked.connect(lambda: self._activate_selected(activate))
        health = QPushButton("查看诊断")
        health.setProperty("secondary", True)
        health.clicked.connect(self._open_selected_health)
        refresh = QPushButton("刷新")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.refresh_all)
        toolbar.addWidget(import_button)
        toolbar.addWidget(QLabel("页数"))
        toolbar.addWidget(self.max_pages)
        toolbar.addWidget(self.use_vl)
        toolbar.addWidget(QLabel("图表页上限"))
        toolbar.addWidget(self.vl_limit)
        toolbar.addStretch(1)
        toolbar.addWidget(health)
        toolbar.addWidget(activate)
        toolbar.addWidget(refresh)
        layout.addLayout(toolbar)
        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 100)
        self.build_progress.setValue(0)
        self.build_progress.setFormat("没有正在运行的建库任务")
        layout.addWidget(self.build_progress)
        self.library_table = QTableWidget(0, 6)
        self.library_table.setHorizontalHeaderLabels(["状态", "教材", "学科", "块数", "来源", "建成时间"])
        self.library_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.library_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.library_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.library_table.setAlternatingRowColors(True)
        self.library_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.library_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.library_table, 1)
        self.build_timer = QTimer(self)
        self.build_timer.setInterval(1300)
        self.build_timer.timeout.connect(self._poll_build)
        return page

    def _choose_and_build(self, button: QPushButton) -> None:
        if self._build_job_id or self._build_queue:
            QMessageBox.information(self, "建库进行中", "请等待当前批次完成后再导入新教材。")
            return
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "选择要建库的教材（可多选）", "", "教材文件 (*.pdf *.epub);;PDF (*.pdf);;EPUB (*.epub)")
        unique = []
        seen = set()
        for value in paths:
            path = str(Path(value).resolve())
            if path.casefold() in seen:
                continue
            seen.add(path.casefold())
            unique.append(path)
        if not unique:
            return
        self._build_total = len(unique)
        self._build_completed = 0
        self._build_failures = []
        self._build_button = button
        self._build_button.setEnabled(False)
        self._build_queue = [{
            "path": path,
            "max_pages": self.max_pages.value(),
            "use_vl": self.use_vl.isChecked(),
            "vl_limit": self.vl_limit.value(),
        } for path in unique]
        self._start_next_build()

    def _start_next_build(self) -> None:
        if self._build_job_id:
            return
        if not self._build_queue:
            self._finish_build_batch()
            return
        item = self._build_queue.pop(0)
        path = str(item["path"])
        self._selected_pdf = path
        position = self._build_completed + 1
        self.build_progress.setRange(0, 0)
        self.build_progress.setFormat(
            "第 %d/%d 本 · 正在准备：%s" % (position, self._build_total, Path(path).name))

        def result(data: dict[str, Any]) -> None:
            job = data.get("job") or {}
            self._build_job_id = str(job.get("id") or job.get("job_id") or "")
            if not self._build_job_id:
                self._build_failures.append("%s：后端没有返回任务编号" % Path(path).name)
                self._build_completed += 1
                QTimer.singleShot(0, self._start_next_build)
                return
            self.build_timer.start()

        def failed(message: str) -> None:
            self._build_failures.append("%s：%s" % (Path(path).name, message))
            self._build_completed += 1
            QTimer.singleShot(0, self._start_next_build)

        self.run_task(
            lambda: self.backend.start_build(
                path, max_pages=int(item["max_pages"]), use_vl=bool(item["use_vl"]),
                vl_limit=int(item["vl_limit"])),
            result, label="正在创建第 %d/%d 个建库任务…" % (position, self._build_total),
            on_error=failed)

    def _poll_build(self) -> None:
        if not self._build_job_id:
            self.build_timer.stop()
            return
        if self._build_poll_pending:
            return
        self._build_poll_pending = True
        job_id = self._build_job_id
        task = Task(lambda: self.backend.build_status(job_id))
        self._tasks.add(task)

        def result(job: dict[str, Any]) -> None:
            if job_id != self._build_job_id:
                return
            status = str(job.get("status") or "")
            message = str(job.get("phase") or job.get("message") or job.get("stage") or status)
            progress = job.get("progress")
            fraction = None
            if isinstance(progress, (int, float)):
                fraction = float(progress if progress <= 1 else progress / 100)
                overall = int(100 * (self._build_completed + max(0, min(1, fraction))) /
                              max(1, self._build_total))
                self.build_progress.setRange(0, 100)
                self.build_progress.setValue(overall)
            else:
                self.build_progress.setRange(0, 0)
            self.build_progress.setFormat(
                "第 %d/%d 本 · %s · %s" % (
                    min(self._build_completed + 1, self._build_total), self._build_total,
                    Path(self._selected_pdf).name, message))
            if status not in ("completed", "ready", "failed"):
                return
            self.build_timer.stop()
            if status == "failed":
                self._build_failures.append("%s：%s" % (
                    Path(self._selected_pdf).name, str(job.get("error") or message)))
            self._build_completed += 1
            self._build_job_id = ""
            self.refresh_all()
            QTimer.singleShot(0, self._start_next_build)

        def failed(message: str) -> None:
            if job_id != self._build_job_id:
                return
            self.build_timer.stop()
            self._build_failures.append("%s：状态读取失败：%s" % (
                Path(self._selected_pdf).name, message))
            self._build_completed += 1
            self._build_job_id = ""
            QTimer.singleShot(0, self._start_next_build)

        def finished() -> None:
            self._tasks.discard(task)
            self._build_poll_pending = False

        task.signals.result.connect(result)
        task.signals.error.connect(failed)
        task.signals.finished.connect(finished)
        self.pool.start(task)

    def _finish_build_batch(self) -> None:
        if self._build_button:
            self._build_button.setEnabled(True)
        self._build_button = None
        self.build_timer.stop()
        self.build_progress.setRange(0, 100)
        succeeded = max(0, self._build_completed - len(self._build_failures))
        self.build_progress.setValue(100 if not self._build_failures else int(
            100 * succeeded / max(1, self._build_total)))
        self.build_progress.setFormat(
            "批量建库完成：成功 %d 本，失败 %d 本" % (succeeded, len(self._build_failures)))
        if self._build_total:
            if self._build_failures:
                show_error(self, "批量建库完成但有失败",
                           "成功 %d 本，失败 %d 本：\n\n%s" % (
                               succeeded, len(self._build_failures), "\n".join(self._build_failures)))
            else:
                QMessageBox.information(self, "批量建库完成", "%d 本教材均已成功导入。" % succeeded)
        self._build_queue = []
        self._build_total = self._build_completed = 0
        self._build_failures = []
        self.refresh_all()

    def _resume_active_build(self) -> None:
        if self._build_job_id or self._build_queue or not hasattr(self.backend, "active_build"):
            return

        def result(data: dict[str, Any]) -> None:
            job = data.get("job") or {}
            if str(job.get("status") or "") not in ("queued", "running"):
                return
            self._build_job_id = str(job.get("id") or "")
            if not self._build_job_id:
                return
            self._selected_pdf = str(job.get("filename") or job.get("source") or "正在建库的教材")
            self._build_total = 1
            self._build_completed = 0
            self._build_failures = []
            self._build_button = getattr(self, "import_button", None)
            if self._build_button:
                self._build_button.setEnabled(False)
            self.build_progress.setRange(0, 0)
            self.build_progress.setFormat("已恢复正在运行的建库任务")
            self.build_timer.start()

        self.run_task(self.backend.active_build, result, label="正在检查未完成的建库任务…")

    def _make_chunks_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.chunk_library_combo = CrispComboBox()
        self.chunk_library_combo.setMinimumWidth(260)
        self.chunk_query = QLineEdit()
        self.chunk_query.setPlaceholderText("留空浏览全部块，或输入检索词")
        self.chunk_limit = QSpinBox()
        self.chunk_limit.setRange(5, 100)
        self.chunk_limit.setValue(30)
        self.chunk_offset = QSpinBox()
        self.chunk_offset.setRange(0, 100000)
        button = QPushButton("读取分块")
        button.clicked.connect(lambda: self._load_chunks(button))
        controls.addWidget(self.chunk_library_combo)
        controls.addWidget(self.chunk_query, 1)
        controls.addWidget(QLabel("数量"))
        controls.addWidget(self.chunk_limit)
        controls.addWidget(QLabel("起始"))
        controls.addWidget(self.chunk_offset)
        controls.addWidget(button)
        self.chunk_table = QTableWidget(0, 4)
        self.chunk_table.setHorizontalHeaderLabels(["#", "来源", "距离", "正文"])
        self.chunk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.chunk_table.setAlternatingRowColors(True)
        self.chunk_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addLayout(controls)
        layout.addWidget(self.chunk_table, 1)
        return self._page("分块浏览", "直接浏览向量库中的原文块，或用关键词定位入库内容。", body)

    def _load_chunks(self, button: QPushButton) -> None:
        library_id = str(self.chunk_library_combo.currentData() or "")
        if not library_id:
            QMessageBox.information(self, "请选择知识库", "请先导入并选择一本教材。")
            return
        query = self.chunk_query.text().strip()
        limit = self.chunk_limit.value()
        offset = self.chunk_offset.value()

        def result(data: dict[str, Any]) -> None:
            chunks = data.get("chunks") or []
            self.chunk_table.setRowCount(len(chunks))
            for row, chunk in enumerate(chunks):
                values = [chunk.get("index"), chunk.get("label"), chunk.get("distance"), chunk.get("text")]
                for column, value in enumerate(values):
                    self.chunk_table.setItem(row, column, QTableWidgetItem(
                        "" if value is None else str(value)))
            self.chunk_table.resizeRowsToContents()

        self.run_task(lambda: self.backend.library_chunks(library_id, query, limit, offset), result,
                      label="正在读取原文分块…", button=button)

    def _make_health_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.health_library_combo = CrispComboBox()
        self.health_library_combo.setMinimumWidth(280)
        self.health_sample = QSpinBox()
        self.health_sample.setRange(50, 5000)
        self.health_sample.setValue(800)
        run = QPushButton("运行知识库诊断")
        run.clicked.connect(lambda: self._run_health(run))
        row.addWidget(self.health_library_combo)
        row.addWidget(QLabel("抽样块数"))
        row.addWidget(self.health_sample)
        row.addWidget(run)
        row.addStretch(1)
        self.health_output = QTextBrowser()
        self.health_output.setHtml(
            "<h3>尚未运行诊断</h3><p>诊断会检查文本质量、引用标签、空块和抽样统计，不调用回答模型。</p>")
        layout.addLayout(row)
        layout.addWidget(self.health_output, 1)
        return self._page("知识库诊断", "检查入库文本质量、分块结构与引用标签是否可用。", body)

    def _run_health(self, button: QPushButton) -> None:
        library_id = str(self.health_library_combo.currentData() or "")
        if not library_id:
            QMessageBox.information(self, "请选择知识库", "请先导入并选择一本教材。")
            return
        sample = self.health_sample.value()

        def result(data: dict[str, Any]) -> None:
            state = data.get("status") or data.get("health") or "已完成"
            self.health_output.setHtml(
                "<h2>诊断结果：%s</h2><p>知识库：%s</p><pre>%s</pre>" % (
                    escaped(state), escaped(self.health_library_combo.currentText()),
                    escaped(json.dumps(data, ensure_ascii=False, indent=2))))

        self.run_task(lambda: self.backend.library_health(library_id, sample), result,
                      label="正在检查知识库…", button=button)

    def _open_selected_health(self) -> None:
        library = self._selected_library_row()
        self._show_page("health")
        if library:
            index = self.health_library_combo.findData(str(library.get("id") or ""))
            if index >= 0:
                self.health_library_combo.setCurrentIndex(index)

    def _make_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("设置", "配置回答模型、模型存储目录和本地运行环境。"))
        tabs = QTabWidget()
        self.settings_tabs = tabs
        tabs.addTab(self._model_tab(), "模型管理")
        guide = QTextBrowser()
        guide.setHtml(
            "<h2>本地运行说明</h2><p>AITIC Desktop 的教材、会话、反馈与向量库均保存在本机。"
            "模型权重不包含在安装包中，可在模型管理页一键下载或导入 GGUF。</p>"
            "<h3>准确度口径</h3><p>默认回答模型为经过项目全量评测的 Qwen3 8B。切换到 4B 或 14B 后，"
            "建议使用批量评测和反馈回归集重新验证。</p>"
            "<h3>快捷键</h3><p>Ctrl+Enter 提问；双击引用或证据行打开原文；更多菜单可复制或导出对话。</p>")
        tabs.addTab(guide, "使用说明")
        layout.addWidget(tabs, 1)
        return page

    def _make_learning_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header("学习工具", "从所选教材生成带出处简报和可核验测试题。"))
        self.learning_tabs = QTabWidget()
        self.learning_tabs.addTab(self._brief_tab(), "教材简报")
        self.learning_tabs.addTab(self._quiz_tab(), "测试题")
        layout.addWidget(self.learning_tabs, 1)
        return page

    def _retrieve_tab(self) -> QWidget:
        tab = super()._retrieve_tab()
        self.retrieve_table.itemDoubleClicked.connect(self._open_retrieve_item)
        return tab

    def _open_retrieve_item(self, item: QTableWidgetItem) -> None:
        source_item = self.retrieve_table.item(item.row(), 0)
        source = source_item.data(Qt.UserRole) if source_item else {}
        if source:
            SourceDialog(self.backend, dict(source), self).exec()

    def _maybe_first_model_setup(self) -> None:
        if not hasattr(self.backend, "needs_model_setup"):
            return
        try:
            if not self.backend.needs_model_setup():
                return
        except Exception:
            return
        box = QMessageBox(self)
        box.setWindowTitle("首次模型配置")
        box.setIcon(QMessageBox.Information)
        box.setText("AITIC Desktop 主程序已安装完成")
        box.setInformativeText(
            "模型权重不包含在安装包内。建议一键下载经过验证的推荐模型；"
            "也可以进入设置选择轻量版、增强版或导入本地 GGUF。")
        configure = box.addButton("一键配置推荐模型", QMessageBox.AcceptRole)
        manage = box.addButton("打开设置", QMessageBox.ActionRole)
        later = box.addButton("稍后", QMessageBox.RejectRole)
        box.setDefaultButton(configure)
        box.exec()
        self._show_page("settings")
        self.settings_tabs.setCurrentIndex(0)
        if box.clickedButton() is configure:
            self._pull_recommended_models()
        elif box.clickedButton() is later:
            self.backend.mark_model_setup_complete()
        elif box.clickedButton() is manage:
            self._refresh_model_catalog()
