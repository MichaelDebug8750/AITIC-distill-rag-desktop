# deploy.ps1 - one-click Docker deployment (host Ollama route)
# Usage: powershell -ExecutionPolicy Bypass -File .\deploy.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> 1/4 Checking Docker engine..." -ForegroundColor Cyan
docker version | Out-Null

Write-Host "==> 2/4 Checking host Ollama (127.0.0.1:11434)..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Host "    Ollama is online." -ForegroundColor Green
} catch {
    Write-Warning "Host Ollama did not respond. Start Ollama Desktop (or ollama serve), then rerun this script."
    exit 1
}

Write-Host "==> 3/4 Building the pipeline image..." -ForegroundColor Cyan
docker compose build

Write-Host "==> 4/4 Running the CLI smoke test (--help)..." -ForegroundColor Cyan
docker compose run --rm pipeline --help

Write-Host ""
Write-Host "Deployment completed. Common commands:" -ForegroundColor Green
Write-Host '  Build PDF:   docker compose run --rm pipeline build --pdf med.pdf'
Write-Host '  Build audio: docker compose run --rm pipeline build --audio Starmer.mp3 --max-seconds 300'
Write-Host '  Ask:         docker compose run --rm pipeline ask "What is a process in an operating system?"'
Write-Host '  Make agent:  docker compose run --rm pipeline agent --pdf med.pdf'
