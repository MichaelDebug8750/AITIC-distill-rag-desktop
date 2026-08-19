"""Render the existing AITIC SVG as high-DPI installer artwork."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def render(source: Path, target: Path, size: int) -> None:
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise RuntimeError(f"Invalid installer SVG: {source}")
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    margin = max(4, round(size * 0.08))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(
        painter,
        QRectF(float(margin), float(margin), float(size - margin * 2), float(size - margin * 2)),
    )
    painter.end()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(target), "PNG"):
        raise RuntimeError(f"Failed to save installer image: {target}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "desktop_app" / "resources" / "aitic.svg"
    output = root / "packaging" / "generated"
    # Inno Setup selects the closest size for the current DPI instead of
    # stretching one small bitmap, keeping the header crisp at 100-250% DPI.
    app = QGuiApplication.instance() or QGuiApplication([])
    for size in (64, 96, 128, 160, 192):
        render(source, output / f"aitic_setup_{size}.png", size)
    print(output)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
