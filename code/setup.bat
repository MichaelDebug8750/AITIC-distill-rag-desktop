@echo off
chcp 65001 >nul
setlocal
REM ============================================================
REM  知识蒸馏管线 - 一键环境脚本 (Windows)
REM  作用: 建虚拟环境 -> 装依赖 -> 拉模型 -> 构建自定义模型
REM  用法: 双击本文件, 或在 code 目录下运行  setup.bat
REM ============================================================

echo.
echo === [1/5] 检查 Python 和 Ollama ===
where python >nul 2>nul || (echo [错误] 未找到 python, 请先安装 Python 3.11+ 并加入 PATH & pause & exit /b 1)
where ollama >nul 2>nul || (echo [错误] 未找到 ollama, 请先从 ollama.com 安装 & pause & exit /b 1)
python --version
ollama --version

echo.
echo === [2/5] 创建虚拟环境 .venv ===
if not exist .venv (
    python -m venv .venv
) else (
    echo 已存在 .venv, 跳过
)
call .venv\Scripts\activate.bat

echo.
echo === [3/5] 安装 Python 依赖 ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === [4/5] 拉取模型 (首次较慢, 约 12GB) ===
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
ollama pull bge-m3

echo.
echo === [5/5] 构建自定义助手模型 (可选) ===
if exist Modelfile (
    ollama create distill-assistant -f Modelfile
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
pause
