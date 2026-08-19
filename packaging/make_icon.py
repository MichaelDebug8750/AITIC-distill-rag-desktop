"""把源码 SVG 渲染成 PyInstaller/WiX 使用的 Windows ICO。"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "desktop_app" / "resources" / "aitic.svg"
    target = root / "packaging" / "aitic.ico"
    app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise SystemExit("invalid SVG: %s" % source)
    image = QImage(256, 256, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, 256, 256))
    painter.end()
    if not image.save(str(target), "ICO"):
        raise SystemExit("failed to write ICO: %s" % target)
    print(target)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

