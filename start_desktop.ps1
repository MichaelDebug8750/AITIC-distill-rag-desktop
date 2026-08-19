param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Candidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "code\.venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE "distill\Scripts\python.exe")
)
$Python = $null
foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
    & $Candidate -c "import PySide6,fitz,chromadb,ollama" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = $Candidate; break }
}
if (-not $Python) {
    throw "No Python environment has the desktop dependencies. Run build_desktop.ps1 -InstallOnly first."
}
Write-Host "AITIC Desktop runtime: $Python" -ForegroundColor Cyan
& $Python -c "import PySide6,fitz,chromadb,ollama; print('Desktop dependencies: ready')"
if ($CheckOnly) { exit $LASTEXITCODE }
$env:AITIC_PROJECT_ROOT = $Root
Push-Location $Root
try {
    & $Python .\run_desktop.py
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
