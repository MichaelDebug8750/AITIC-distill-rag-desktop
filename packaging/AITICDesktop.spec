# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata,
)

root = Path(SPECPATH).resolve().parent
datas = [
    (str(root / "desktop_app" / "resources"), "desktop_app/resources"),
    # 建库线程必须重新加载隔离的 main 模块，不能复用在线问答的全局模块。
    # PyInstaller 的 PYZ 只有逻辑 __file__，因此保留一份只读源码供该线程加载。
    (str(root / "code" / "main.py"), "code"),
]
binaries = []
hiddenimports = ["webui", "main"]

# Chroma 通过配置字符串动态导入实现类，普通静态分析看不到；完整收集避免安装后
# 在首次打开某个库时才报 hidden import。Qt 自带官方 PyInstaller hooks。
datas += collect_data_files("chromadb")
binaries += collect_dynamic_libs("chromadb")
hiddenimports += collect_submodules(
    "chromadb",
    filter=lambda name: not (
        name.startswith("chromadb.server") or name.startswith("chromadb.cli")
        or name.startswith("chromadb.test")
    ),
    on_error="ignore",
)

for distribution in (
    "chromadb", "ollama", "fastapi", "starlette", "pydantic", "pydantic-settings",
    "pymupdf", "python-multipart", "onnxruntime", "opentelemetry-api",
    "opentelemetry-sdk", "tokenizers",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

a = Analysis(
    [str(root / "run_desktop.py")],
    pathex=[str(root), str(root / "code")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "IPython", "jupyter", "notebook", "matplotlib", "tkinter",
        "torch", "tensorflow", "modelscope", "faster_whisper",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AITIC Desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "packaging" / "aitic.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AITIC Desktop",
)
