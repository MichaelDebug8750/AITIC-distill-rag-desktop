"""原生界面的复用控件。"""

from __future__ import annotations

import html
import re
from typing import Any

from PySide6.QtCore import QPointF, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPolygonF, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QScrollArea, QTabWidget,
    QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from .workers import Task


def escaped(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def text_html(value: Any) -> str:
    text = escaped(value)
    return text.replace("\r\n", "\n").replace("\n", "<br>")


_CITE_RE = re.compile(r"\[([Kk]\d+:)?(?:p\.?\s*\d+|ch\d+(?::\d+)?)[^\]]*\]", re.I)


def _inline_markdown(value: str) -> str:
    """Render the small, safe markdown subset produced by the local models."""
    out = escaped(value)
    out = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", out)
    out = _CITE_RE.sub(lambda match: '<span class="cite">%s</span>' % match.group(0), out)
    return out


def answer_html(value: Any) -> str:
    """Convert model text to safe, readable HTML without exposing raw markup.

    QTextBrowser does not understand Markdown.  The previous desktop renderer
    therefore showed literal ``**bold**`` markers and flattened lists while the
    web client rendered the same answer correctly.  This deliberately supports
    only headings, lists, quotes, code, bold and citations; arbitrary HTML stays
    escaped.
    """
    lines = str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rendered: list[str] = []
    list_kind = ""

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            rendered.append("</%s>" % list_kind)
            list_kind = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            close_list()
            rendered.append('<div class="gap"></div>')
            continue
        bullet = re.match(r"^[-*•]\s+(.+)$", line)
        numbered = re.match(r"^\d+[.)、]\s+(.+)$", line)
        if bullet or numbered:
            wanted = "ul" if bullet else "ol"
            if list_kind != wanted:
                close_list()
                rendered.append("<%s>" % wanted)
                list_kind = wanted
            rendered.append("<li>%s</li>" % _inline_markdown((bullet or numbered).group(1)))
            continue
        close_list()
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            level = min(4, len(heading.group(1)) + 1)
            rendered.append("<h%d>%s</h%d>" % (
                level, _inline_markdown(heading.group(2)), level))
        elif line.startswith("> "):
            rendered.append("<blockquote>%s</blockquote>" % _inline_markdown(line[2:]))
        else:
            rendered.append("<p>%s</p>" % _inline_markdown(line))
    close_list()
    return "".join(rendered)


class CrispComboBox(QComboBox):
    """QComboBox with a device-independent vector arrow.

    The native Windows/Fusion arrow was bitmap-like at 175–200% DPI and left a
    visible separator inside the Agent controls.  The stylesheet hides that
    sub-control; this widget paints a small vector chevron in logical pixels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("CrispComboBox")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#5d6d80" if self.isEnabled() else "#a8b2bd"))
        x = float(self.width() - 17)
        y = float(self.height()) / 2.0
        painter.drawPolygon(QPolygonF([
            QPointF(x - 4.5, y - 2.0), QPointF(x + 4.5, y - 2.0), QPointF(x, y + 3.0)
        ]))
        painter.end()


def card_layout(parent: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(parent)
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 15, 16, 15)
    layout.setSpacing(10)
    return frame, layout


def page_header(title: str, description: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 8)
    layout.setSpacing(3)
    heading = QLabel(title)
    heading.setObjectName("PageTitle")
    subtitle = QLabel(description)
    subtitle.setObjectName("PageDescription")
    subtitle.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(subtitle)
    return widget


class QuestionEdit(QTextEdit):
    submit = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.submit.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SourceDialog(QDialog):
    """原生原文查看器：文本块与 PDF 页在 Qt 控件内展示。"""

    def __init__(self, backend, source: dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.backend = backend
        self.source = dict(source)
        self.setWindowTitle("查看原文 · %s" % source.get("label", "来源"))
        self.resize(960, 760)
        layout = QVBoxLayout(self)
        title = QLabel("%s\n%s" % (source.get("label", "来源"), source.get("source", "")))
        title.setObjectName("SectionTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        self.tabs = QTabWidget()
        self.text = QTextBrowser()
        self.text.setHtml("<p>正在读取原文…</p>")
        self.tabs.addTab(self.text, "入库原文")
        self.image_label = QLabel("正在渲染 PDF 页…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        if source.get("page"):
            self.tabs.addTab(self.image_scroll, "PDF 原页")
        layout.addWidget(self.tabs, 1)
        self._load()

    def _load(self) -> None:
        def work():
            blocks = self.backend.source_blocks(self.source)
            page = None
            page_error = ""
            if self.source.get("page"):
                try:
                    page = self.backend.source_page_png(self.source)
                except Exception as exc:
                    page_error = str(exc)
            return blocks, page, page_error

        task = Task(work)
        self._task = task
        task.signals.result.connect(self._loaded)
        task.signals.error.connect(self._failed)
        task.signals.finished.connect(lambda: setattr(self, "_task", None))
        QThreadPool.globalInstance().start(task)

    def _loaded(self, value: object) -> None:
        blocks, page, page_error = value
        chunks = []
        for block in blocks.get("blocks", []):
            marker = "命中证据" if block.get("matched") else "同页上下文"
            chunks.append(
                '<div style="margin:8px 0;padding:10px;border:1px solid #dde6ef;'
                'border-radius:7px"><b>%s · %s</b><p>%s</p></div>'
                % (escaped(block.get("label")), marker, text_html(block.get("text"))))
        self.text.setHtml("".join(chunks) or "<p>该位置没有可显示的文本块。</p>")
        if page:
            pixmap = QPixmap()
            if pixmap.loadFromData(page):
                self.image_label.setPixmap(pixmap)
                self.image_label.adjustSize()
            else:
                self.image_label.setText("PDF 页图像无法解码。")
        elif self.source.get("page"):
            self.image_label.setText("无法显示 PDF 原页：%s" % (page_error or "源文件不可用"))

    def _failed(self, message: str) -> None:
        self.text.setHtml("<p style='color:#a33a30'>%s</p>" % escaped(message))
        if self.source.get("page"):
            self.image_label.setText("加载失败")


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, str(message))
