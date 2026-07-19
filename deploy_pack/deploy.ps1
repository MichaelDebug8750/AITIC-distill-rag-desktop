# deploy.ps1 - 从预编译 GGUF 权重一键创建可运行模型（UTF-8 with BOM）
# 用法: 在 deploy_pack 目录下  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
$ErrorActionPreference = 'Stop'
$gguf  = 'qwen3-8b-Q4_K_M.gguf'
$model = 'distill-assistant'

Write-Host '==> 1/3 Checking Ollama...' -ForegroundColor Cyan
$null = ollama --version

Write-Host '==> 2/3 Checking GGUF weight...' -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $PSScriptRoot $gguf))) {
    Write-Error ('Missing ' + $gguf + ' (must sit next to this script).')
    exit 1
}

Write-Host ('==> 3/3 Creating model ' + $model + ' from GGUF...') -ForegroundColor Cyan
ollama create $model -f (Join-Path $PSScriptRoot 'Modelfile')

Write-Host ''
Write-Host 'Done. Test command:' -ForegroundColor Green
Write-Host ('  ollama run ' + $model + ' ' + [char]34 + 'What is a process in an operating system?' + [char]34)
