"""无浏览器的原生桌面冒烟；可在 CI 用 offscreen Qt 运行。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from desktop_app.backend import DesktopBackend
from desktop_app.main_window import MainWindow
from desktop_app.style import APP_STYLE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default="")
    parser.add_argument("--wait-ms", type=int, default=5000)
    args = parser.parse_args()
    app = QApplication([])
    app.setOrganizationName("AITIC")
    app.setApplicationName("AITIC Desktop Smoke")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    backend = DesktopBackend()
    window = MainWindow(backend)
    window.resize(1440, 900)
    window.show()
    result = {"saved": False}

    def finish() -> None:
        if args.screenshot:
            target = Path(args.screenshot).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            result["saved"] = window.grab().save(str(target), "PNG")
            result["screenshot"] = str(target)
        result.update({
            "pages": window.stack.count(), "navigation": window.nav.count(),
            "libraries": window.library_picker.count(),
            "library_rows": window.library_table.rowCount(),
            "title": window.windowTitle(),
        })
        print(json.dumps(result, ensure_ascii=False))
        app.quit()

    QTimer.singleShot(max(100, args.wait_ms), finish)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
