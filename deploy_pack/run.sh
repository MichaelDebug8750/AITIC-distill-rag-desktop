#!/usr/bin/env bash
# run.sh — 从预编译 GGUF 权重一键创建可运行模型
# 用法：在 deploy_pack 目录下  bash run.sh
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> 1/3 检查 Ollama..."
ollama --version >/dev/null

echo "==> 2/3 检查 GGUF 权重..."
[ -f ./qwen3-8b-Q4_K_M.gguf ] || { echo "[错误] 找不到 qwen3-8b-Q4_K_M.gguf" >&2; exit 1; }

echo "==> 3/3 从 GGUF 创建模型 distill-assistant..."
ollama create distill-assistant -f ./Modelfile

echo ""
echo "完成。测试："
echo '  ollama run distill-assistant "What is a process in an operating system?"'
