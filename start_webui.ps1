param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = Join-Path $ProjectRoot "code"
$DbDir = Join-Path $ProjectRoot "data\vectordb"
$Url = "http://127.0.0.1:$Port"

Write-Host "AITIC WebUI preflight" -ForegroundColor Cyan

$candidates = @()
if ($env:VIRTUAL_ENV) {
    $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}
$candidates += @(
    (Join-Path $CodeDir ".venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE "distill\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
)

$PythonExe = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    & $candidate -c "import fastapi,uvicorn,fitz,chromadb,ollama,multipart" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = $candidate
        break
    }
    Write-Host "  Skipping Python with missing dependencies: $candidate" -ForegroundColor DarkYellow
}

if (-not $PythonExe) {
    throw "No Python environment has all WebUI dependencies. Run code\setup.bat first."
}
Write-Host "  Python: $PythonExe" -ForegroundColor Green

try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
} catch {
    throw "Ollama is not responding on port 11434. Start Ollama first."
}
$available = @($tags.models | ForEach-Object { $_.name })
$required = @("qwen3:8b", "qwen3-vl:8b", "bge-m3:latest")
$missing = @($required | Where-Object { $_ -notin $available })
if ($missing.Count) {
    throw "Missing demo models: $($missing -join ', ')"
}
Write-Host "  Ollama and required models: ready" -ForegroundColor Green

if (-not (Test-Path -LiteralPath (Join-Path $DbDir "chroma.sqlite3") -PathType Leaf)) {
    throw "Demo vector database was not found: $DbDir"
}
$env:DISTILL_DB = $DbDir
Write-Host "  Vector database: $DbDir" -ForegroundColor Green

if ($CheckOnly) {
    Write-Host "Preflight passed. The demo is ready to start." -ForegroundColor Green
    exit 0
}

try {
    $existing = Invoke-RestMethod -Uri "$Url/api/status" -TimeoutSec 2
    if ($existing.ready) {
        Write-Host "WebUI is already running: $Url" -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $Url }
        exit 0
    }
} catch {
    # No service on this port yet; continue with startup.
}

if (-not $NoBrowser) {
    $openCommand = "Start-Sleep -Seconds 2; Start-Process '$Url'"
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden `
        -ArgumentList @("-NoProfile", "-WindowStyle", "Hidden", "-Command", $openCommand) | Out-Null
}

Write-Host "Starting WebUI: $Url" -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this window after the demo to stop the server." -ForegroundColor DarkGray
Push-Location $CodeDir
try {
    & $PythonExe -m uvicorn webui:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
