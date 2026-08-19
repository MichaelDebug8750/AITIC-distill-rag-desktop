"""AITIC Desktop 程序入口。"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import json
import sys
import time
import traceback

from PySide6.QtCore import QDir, QLockFile, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from .backend import BackendError, DesktopBackend, bundled_root, default_data_root
from .main_window import MainWindow
from .style import APP_STYLE


_NULL_STREAMS = []


def _ensure_standard_streams() -> None:
    """为 Windows 无控制台冻结包提供可写日志流。

    PyInstaller 的 ``console=False`` 会把 stdout/stderr 设为 ``None``；核心管线会把
    库指纹告警写到 stderr。原生 GUI 不应因为一条诊断日志而无法打开知识库。
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            stream = open(os.devnull, "w", encoding="utf-8")
            _NULL_STREAMS.append(stream)
            setattr(sys, name, stream)


def _splash() -> QSplashScreen:
    # A 600x300 pixmap was previously enlarged by Windows on the target 200%
    # display, which bitmap-scaled every glyph.  Paint at the monitor's physical
    # pixel density while keeping all coordinates in device-independent pixels.
    screen = QGuiApplication.primaryScreen()
    dpr = max(1.0, float(screen.devicePixelRatio()) if screen else 1.0)
    pixmap = QPixmap(max(1, round(600 * dpr)), max(1, round(300 * dpr)))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor("#f7f8fa"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setBrush(QColor("#ffffff"))
    painter.setPen(QColor("#cbd2da"))
    painter.drawRoundedRect(42, 52, 54, 54, 13, 13)
    painter.setPen(QColor("#101922"))
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    font.setPointSize(20)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(59, 89, "知")
    painter.setPen(QColor("#102b4d"))
    font.setPointSize(25)
    painter.setFont(font)
    painter.drawText(116, 88, "知识蒸馏")
    font.setPointSize(12)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor("#64758c"))
    painter.drawText(117, 115, "AITIC Desktop · 本地教材智能体")
    painter.setPen(QColor("#245f96"))
    painter.drawText(44, 177, "正在连接本地模型并载入知识库…")
    painter.setPen(QColor("#137542"))
    # Draw the lock as vectors.  The emoji glyph used before came from a colour
    # bitmap font and remained visibly soft even after the text became HiDPI.
    lock_pen = QPen(QColor("#137542"), 2.0)
    lock_pen.setCapStyle(Qt.RoundCap)
    painter.setPen(lock_pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawArc(45, 205, 14, 16, 0, 180 * 16)
    painter.setBrush(QColor("#edf9f1"))
    painter.drawRoundedRect(43, 212, 18, 15, 3, 3)
    painter.setPen(QColor("#137542"))
    painter.drawText(70, 225, "全本地运行 · 教材与会话数据不出机")
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint)
    return splash


def _instance_lock() -> QLockFile:
    temp = QStandardPaths.writableLocation(QStandardPaths.TempLocation)
    QDir().mkpath(temp)
    lock = QLockFile(str(Path(temp) / "aitic-desktop.lock"))
    lock.setStaleLockTime(30_000)
    return lock


def main() -> int:
    _ensure_standard_streams()
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    if sys.platform == "win32":
        try:
            # Per-monitor-v2 awareness prevents Windows from bitmap-scaling the whole Qt window
            # when it moves between monitors with different DPI settings.
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except (AttributeError, RuntimeError):
        pass
    app = QApplication(sys.argv)
    app.setOrganizationName("AITIC")
    app.setOrganizationDomain("local.aitic")
    app.setApplicationName("AITIC Desktop")
    app.setApplicationDisplayName("AITIC Desktop")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")
    # Use Windows' own DirectWrite-tuned UI font and its native sub-pixel strategy.
    # Forcing Microsoft YaHei UI + full hinting looked acceptable at 100%, but at
    # the 200% DPI used by the target machine it produced visibly harsh / fuzzy
    # CJK edges and synthetic Latin bold glyphs.  Vertical hinting preserves the
    # glyph's horizontal outline while still aligning baselines to device pixels.
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPointSizeF(10.0)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferDefault)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE)

    icon_path = bundled_root() / "desktop_app" / "resources" / "aitic.svg"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    lock = _instance_lock()
    if not lock.tryLock(100):
        QMessageBox.information(None, "AITIC Desktop 已在运行", "请切换到已经打开的窗口。")
        return 0

    splash = _splash()
    splash.show()
    app.processEvents()
    try:
        backend = DesktopBackend(default_data_root())
        window = MainWindow(backend)
    except (BackendError, Exception) as exc:
        splash.close()
        QMessageBox.critical(None, "AITIC Desktop 启动失败", str(exc))
        lock.unlock()
        return 1

    window.show()
    splash.finish(window)
    _arm_packaged_smoke(app, window, backend)
    code = app.exec()
    backend.close()
    lock.unlock()
    return int(code)


def _arm_packaged_smoke(app: QApplication, window: MainWindow,
                        backend: DesktopBackend) -> None:
    """可自动化验证冻结包；普通启动没有相关环境变量时完全不生效。"""
    report_path = os.environ.get("AITIC_DESKTOP_SMOKE_REPORT", "").strip()
    if not report_path:
        return
    screenshot_path = os.environ.get("AITIC_DESKTOP_SMOKE_SCREENSHOT", "").strip()
    question = os.environ.get("AITIC_DESKTOP_SMOKE_ASK", "").strip()
    library = os.environ.get("AITIC_DESKTOP_SMOKE_LIBRARY", "").strip()
    libraries = [item.strip() for item in
                 os.environ.get("AITIC_DESKTOP_SMOKE_LIBRARIES", "").split(",")
                 if item.strip()]
    if not libraries and library:
        libraries = [library]
    build_pdf = os.environ.get("AITIC_DESKTOP_SMOKE_BUILD_PDF", "").strip()
    try:
        delay = max(100, int(os.environ.get("AITIC_DESKTOP_SMOKE_MS", "3000")))
    except ValueError:
        delay = 3000
    state = {"finished": False}

    def finish(extra=None, error=""):
        if state["finished"]:
            return
        state["finished"] = True
        report = {
            "ok": not bool(error), "error": error, "frozen": bool(getattr(sys, "frozen", False)),
            "pages": window.stack.count(), "navigation": window.nav.count(),
            "libraries": window.library_picker.count(),
            "library_rows": window.library_table.rowCount(),
            "font": {
                "family": app.font().family(), "point_size": app.font().pointSizeF(),
                "hinting": app.font().hintingPreference().name,
            },
            "runtime": backend.runtime_summary(), "uvicorn_imported": "uvicorn" in sys.modules,
            "status": dict(window._last_status),
        }
        if (window._last_status or {}).get("db_error"):
            try:
                backend.webui._collection().count()
            except BaseException:
                report["database_traceback"] = traceback.format_exc()
        if isinstance(extra, dict) and "_smoke_build" in extra:
            report["build"] = extra["_smoke_build"]
        elif isinstance(extra, dict):
            report["ask"] = {
                "answer": extra.get("answer"), "abstained": extra.get("abstained"),
                "cite_ok": (extra.get("cite_check") or {}).get("ok"),
                "sources": len(extra.get("sources") or []),
            }
        if screenshot_path:
            target = Path(screenshot_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            report["screenshot_saved"] = bool(window.grab().save(str(target), "PNG"))
            report["screenshot"] = str(target)
        target = Path(report_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        app.quit()

    if build_pdf:
        def build_book():
            response = backend.start_build(build_pdf, max_pages=0, use_vl=False)
            job = response.get("job") or {}
            job_id = str(job.get("id") or "")
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                job = backend.build_status(job_id)
                if job.get("status") in ("ready", "failed"):
                    break
                time.sleep(0.25)
            if job.get("status") != "ready":
                raise RuntimeError("冻结版真实建库失败：%s" % job)
            library_id = str(job.get("library_id") or "")
            chunks = backend.library_chunks(library_id, limit=2)
            health = backend.library_health(library_id, sample=50)
            expected = int(job.get("chunks") or 0)
            browsed = len(chunks.get("chunks") or [])
            healthy = int(health.get("chunks_total") or 0)
            if expected < 1 or browsed < 1 or healthy != expected:
                raise RuntimeError(
                    "冻结版建库产物校验失败：chunks=%d, browsed=%d, health=%d"
                    % (expected, browsed, healthy))
            return {"_smoke_build": {
                "source": build_pdf,
                "job_status": job.get("status"),
                "job_error": job.get("error"),
                "chunks": expected,
                "library_id": library_id,
                "browsed_chunks": browsed,
                "health_chunks": healthy,
            }}

        def start_build():
            window.run_task(
                build_book, lambda value: finish(value), label="Packaged PDF build smoke…",
                on_error=lambda message: finish(error=message))

        QTimer.singleShot(delay, start_build)
        QTimer.singleShot(930_000, lambda: finish(error="packaged PDF build timed out"))
    elif question:
        def start_ask():
            window.run_task(
                lambda: backend.ask(question, libraries=libraries),
                lambda value: finish(value), label="Packaged smoke test…",
                on_error=lambda message: finish(error=message))
        QTimer.singleShot(delay, start_ask)
        QTimer.singleShot(180_000, lambda: finish(error="packaged smoke timed out"))
    else:
        def finish_when_ready():
            if window._last_status:
                finish()
            else:
                QTimer.singleShot(100, finish_when_ready)

        QTimer.singleShot(delay, finish_when_ready)
        QTimer.singleShot(30_000, lambda: finish(error="desktop status check timed out"))


if __name__ == "__main__":
    raise SystemExit(main())
