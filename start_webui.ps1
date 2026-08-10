param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = Join-Path $ProjectRoot "code"
$DbDir = Join-Path $ProjectRoot "data\vectordb"
$DataDir = Join-Path $ProjectRoot "data"
$Url = "http://127.0.0.1:$Port"

function Get-NormalizedPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    try {
        return [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]"\/")
    } catch {
        return $null
    }
}

function Test-LocalPortOpen {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$TargetPort,
        [int]$TimeoutMs = 500
    )
    $client = New-Object System.Net.Sockets.TcpClient
    $waitHandle = $null
    try {
        $attempt = $client.BeginConnect($HostName, $TargetPort, $null, $null)
        $waitHandle = $attempt.AsyncWaitHandle
        if (-not $waitHandle.WaitOne($TimeoutMs, $false)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch {
        return $false
    } finally {
        if ($waitHandle) { $waitHandle.Close() }
        $client.Close()
    }
}

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
    Write-Host "  No initial vector database found. WebUI will start; choose a PDF in the left panel to build one." -ForegroundColor DarkYellow
} else {
    Write-Host "  Initial vector database: $DbDir" -ForegroundColor Green
}
$env:DISTILL_DB = $DbDir

if ($CheckOnly) {
    Write-Host "Preflight passed. The demo is ready to start." -ForegroundColor Green
    exit 0
}

if (Test-LocalPortOpen -TargetPort $Port) {
    try {
        $existing = Invoke-RestMethod -Uri "$Url/api/status" -TimeoutSec 2
    } catch {
        throw ("Port {0} is already occupied, but the service is not this AITIC beta WebUI. " +
               "It will not be stopped or replaced. Details: {1}" -f $Port, $_.Exception.Message)
    }

    $expectedCwd = Get-NormalizedPath $CodeDir
    $expectedDb = Get-NormalizedPath $DbDir
    $expectedData = Get-NormalizedPath $DataDir
    $actualCwd = Get-NormalizedPath ([string]$existing.cwd)
    $actualDb = Get-NormalizedPath ([string]$existing.db_path)
    $sameCwd = $false
    if ($actualCwd) {
        $sameCwd = $actualCwd.Equals(
            $expectedCwd, [System.StringComparison]::OrdinalIgnoreCase)
    }
    # 当前知识库可能是初始 data\vectordb，也可能是用户在前端新建后切换到
    # data\webui_knowledge_bases\...\vectordb。只要求它仍位于本 beta 的 data 根下，
    # 避免把稳定版或其他副本误认作当前服务。
    $isInitialDb = $false
    if ($actualDb) {
        $isInitialDb = $actualDb.Equals(
            $expectedDb, [System.StringComparison]::OrdinalIgnoreCase)
    }
    $isBetaDataDb = $false
    if ($actualDb -and $expectedData) {
        $dataPrefix = $expectedData + [System.IO.Path]::DirectorySeparatorChar
        $isBetaDataDb = $actualDb.StartsWith(
            $dataPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    }
    $sameDb = $isInitialDb -or $isBetaDataDb

    if (-not ($sameCwd -and $sameDb)) {
        throw ("Port {0} is already used by another AITIC/Web service. " +
               "Expected beta cwd='{1}', db_path='{2}', but received cwd='{3}', db_path='{4}'. " +
               "The existing process will not be stopped or replaced." -f
               $Port, $expectedCwd, $expectedDb, $actualCwd, $actualDb)
    }
    if (-not $existing.ready) {
        throw ("This beta WebUI is already listening on port {0}, but it is not ready. " +
               "Check Ollama and the active knowledge base; the process will not be restarted automatically." -f $Port)
    }

    Write-Host "This beta WebUI is already running: $Url" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $Url }
    exit 0
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
