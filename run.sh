#!/usr/bin/env bash
# run.sh — 知识蒸馏管线 Docker 一键部署（路线A：连宿主 Ollama）
# 用法：项目根目录下  bash run.sh
set -euo pipefail

echo "==> 1/4 检查 Docker 引擎..."
docker version >/dev/null

echo "==> 2/4 检查宿主 Ollama (127.0.0.1:11434)..."
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "    Ollama 在线 OK"
else
  echo "    [错误] 宿主 Ollama 没响应，先 ollama serve 再重跑" >&2
  exit 1
fi

echo "==> 3/4 构建镜像..."
docker compose build

echo "==> 4/4 冒烟测试 (--help)..."
docker compose run --rm pipeline --help

cat <<'EOF'

部署完成。常用命令：
  建库(PDF):  docker compose run --rm pipeline build --pdf med.pdf
  建库(音频): docker compose run --rm pipeline build --audio Starmer.mp3 --max-seconds 300
  提问:       docker compose run --rm pipeline ask "What is a process in an operating system?"
  生成智能体: docker compose run --rm pipeline agent --pdf med.pdf
EOF
