@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"
REM ============================================================
REM  知识蒸馏管线 - 一键环境脚本 (Windows)
REM  作用: 建虚拟环境 -> 装依赖 -> 拉模型 -> 构建自定义模型
REM  用法: 双击本文件, 或在 code 目录下运行  setup.bat
REM ============================================================

echo.
echo === [1/5] 检查 Python 和 Ollama ===
set "PYTHON_CMD="
where py >nul 2>nul && py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
    where python >nul 2>nul && python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [错误] 未找到 Python 3.11。本项目固定使用 3.11，避免 Chroma/ASR 在 3.14 缺少兼容轮子。
    goto :fail
)
where ollama >nul 2>nul || (echo [错误] 未找到 ollama, 请先从 ollama.com 安装 & goto :fail)
%PYTHON_CMD% --version || goto :fail
ollama --version || goto :fail

echo.
echo === [2/5] 创建虚拟环境 .venv ===
if not exist .venv\Scripts\python.exe (
    %PYTHON_CMD% -m venv .venv || goto :fail
) else (
    .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [错误] 已有 .venv 不是 Python 3.11。请先重命名或移走该目录后重跑。
        goto :fail
    )
    echo 已存在兼容的 Python 3.11 .venv, 复用
)
call .venv\Scripts\activate.bat || goto :fail

echo.
echo === [3/5] 安装 Python 依赖 ===
python -m pip install --upgrade pip || goto :fail
python -m pip install -r requirements.txt || goto :fail

echo.
echo === [4/5] 拉取模型 (首次较慢, 约 12GB) ===
ollama pull qwen3:8b || goto :fail
ollama pull qwen3-vl:8b || goto :fail
ollama pull bge-m3 || goto :fail

echo.
echo === [5/5] 构建自定义助手模型 (可选) ===
if exist Modelfile (
    ollama create distill-assistant -f Modelfile || goto :fail
    echo 已创建 distill-assistant
) else (
    echo 未找到 Modelfile, 跳过
)

echo.
echo ============================================================
echo  环境就绪! 接下来:
echo    .venv\Scripts\activate.bat
echo    python main.py build --pdf ..\data\med.pdf --vl-limit 15
echo    python main.py ask "What are the parts of a bacteriophage?"
echo ============================================================
popd
pause
exit /b 0

:fail
echo.
echo [失败] 环境未完成，请根据上方第一条错误处理后重跑 setup.bat。
popd
pause
exit /b 1
