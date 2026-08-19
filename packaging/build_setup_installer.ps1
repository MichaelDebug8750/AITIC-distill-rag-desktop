param(
    [string]$Version = "1.0.0",
    [string]$InnoVersion = "7.1.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Packaging = Join-Path $Root "packaging"
$Dist = Join-Path $Root "dist"
$AppDist = Join-Path $Dist "AITIC Desktop"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Packaging "AITICDesktop.iss"

if (-not (Test-Path -LiteralPath (Join-Path $AppDist "AITIC Desktop.exe") -PathType Leaf)) {
    throw "The frozen application is missing. Run build_desktop.ps1 first: $AppDist"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "The desktop build environment is missing: $Python"
}

& $Python (Join-Path $Packaging "make_setup_assets.py")
if ($LASTEXITCODE -ne 0) { throw "Installer artwork generation failed." }

$IsccOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    (Join-Path $Packaging "fetch_inno_setup.ps1") -Version $InnoVersion
if ($LASTEXITCODE -ne 0) { throw "Inno Setup acquisition failed." }
$Iscc = [string]($IsccOutput | Select-Object -Last 1)

& $Iscc --quiet-progress "--define=SourceDir=$AppDist" "--define=OutputDir=$Dist" $Script
if ($LASTEXITCODE -ne 0) { throw "Setup EXE build failed." }

$SetupExe = Join-Path $Dist "AITIC-Desktop-$Version-Setup-x64.exe"
if (-not (Test-Path -LiteralPath $SetupExe -PathType Leaf)) {
    throw "Setup EXE was not created: $SetupExe"
}

$Artifacts = Get-ChildItem -LiteralPath $Dist -File | Where-Object {
    $_.Extension -in '.exe','.msi','.zip' -and $_.Name -like 'AITIC-Desktop-*'
}
$Lines = foreach ($Artifact in $Artifacts) {
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact.FullName).Hash
    "$Hash  $($Artifact.Name)"
}
$Lines | Set-Content -LiteralPath (Join-Path $Dist "SHA256SUMS.txt") -Encoding ascii
Get-Item -LiteralPath $SetupExe | Select-Object FullName,Length,LastWriteTime
