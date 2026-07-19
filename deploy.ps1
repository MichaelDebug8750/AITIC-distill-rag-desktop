# deploy.ps1 — 知识蒸馏管线 Docker 一键部署（路线A：连宿主 Ollama）
# 用法：项目根目录 E:\Ollama_test 下右键"用 PowerShell 运行"，或  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> 1/4 检查 Docker 引擎..." -ForegroundColor Cyan
docker version | Out-Null

Write-Host "==> 2/4 检查宿主 Ollama (127.0.0.1:11434)..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Host "    Ollama 在线 OK" -ForegroundColor Green
} catch {
    Write-Warning "宿主 Ollama 没响应。先确保 Ollama 在跑（Ollama Desktop 或 ollama serve），再重跑本脚本。"
    exit 1
}

Write-Host "==> 3/4 构建镜像..." -ForegroundColor Cyan
docker compose build

Write-Host "==> 4/4 冒烟测试 (--help)..." -ForegroundColor Cyan
docker compose run --rm pipeline --help

Write-Host ""
Write-Host "部署完成。常用命令：" -ForegroundColor Green
Write-Host '  建库(PDF):  docker compose run --rm pipeline build --pdf med.pdf'
Write-Host '  建库(音频): docker compose run --rm pipeline build --audio Starmer.mp3 --max-seconds 300'
Write-Host '  提问:       docker compose run --rm pipeline ask "What is a process in an operating system?"'
Write-Host '  生成智能体: docker compose run --rm pipeline agent --pdf med.pdf'
