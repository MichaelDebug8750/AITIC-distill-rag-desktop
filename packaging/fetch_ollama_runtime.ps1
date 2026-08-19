param(
    [string]$Version = "0.32.4",
    [string]$Sha256 = "4CE7E765DC2BF1BB424A76B96D6631CC0462F5C7507E85F0DC2ABF30C564953B",
    [string]$RocmSha256 = "8A665BB883CF7A4E46A75F211D1BF621F695A0F0544925082CD15AD7D7D4C6EE"
)

$ErrorActionPreference = "Stop"
$Packaging = Split-Path -Parent $MyInvocation.MyCommand.Path
$Downloads = Join-Path $Packaging ".downloads"
$RuntimeBase = Join-Path $Packaging "runtime"
$Target = Join-Path $RuntimeBase ("ollama-v" + $Version)
$Archive = Join-Path $Downloads ("ollama-windows-amd64-v" + $Version + ".zip")
$RocmArchive = Join-Path $Downloads ("ollama-windows-amd64-rocm-v" + $Version + ".zip")
$Expected = $Sha256.Trim().ToUpperInvariant()
$RocmExpected = $RocmSha256.Trim().ToUpperInvariant()
$RocmMarker = Join-Path $Target "AITIC_ROCM_VERSION.txt"
$BaseReady = Test-Path -LiteralPath (Join-Path $Target "ollama.exe") -PathType Leaf

if ((Test-Path -LiteralPath $Target) -and -not $BaseReady) {
    throw "Incomplete Ollama runtime directory already exists: $Target"
}
New-Item -ItemType Directory -Path $Downloads -Force | Out-Null
New-Item -ItemType Directory -Path $RuntimeBase -Force | Out-Null

if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    $Url = "https://github.com/ollama/ollama/releases/download/v$Version/ollama-windows-amd64.zip"
    Write-Host "Downloading verified Ollama runtime $Version ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Url -OutFile $Archive
}
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToUpperInvariant()
if ($Actual -ne $Expected) {
    throw "Ollama archive checksum mismatch. Expected $Expected, got $Actual."
}

if (-not $BaseReady) {
    $Stage = Join-Path $RuntimeBase (".ollama-stage-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $Stage | Out-Null
    try {
        Expand-Archive -LiteralPath $Archive -DestinationPath $Stage
        if (-not (Test-Path -LiteralPath (Join-Path $Stage "ollama.exe") -PathType Leaf)) {
            throw "The verified Ollama archive did not contain ollama.exe."
        }
        ("Ollama $Version" + [Environment]::NewLine + "SHA256 $Expected") | Set-Content -LiteralPath (Join-Path $Stage "AITIC_RUNTIME_VERSION.txt") -Encoding ascii
        Move-Item -LiteralPath $Stage -Destination $Target
    } catch {
        if (Test-Path -LiteralPath $Stage) {
            Write-Warning "Incomplete extraction retained for diagnosis: $Stage"
        }
        throw
    }
}

if (-not (Test-Path -LiteralPath $RocmMarker -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $RocmArchive -PathType Leaf)) {
        $RocmUrl = "https://github.com/ollama/ollama/releases/download/v$Version/ollama-windows-amd64-rocm.zip"
        Write-Host "Downloading verified Ollama AMD ROCm runtime $Version ..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $RocmUrl -OutFile $RocmArchive
    }
    $RocmActual = (Get-FileHash -Algorithm SHA256 -LiteralPath $RocmArchive).Hash.ToUpperInvariant()
    if ($RocmActual -ne $RocmExpected) {
        throw "Ollama ROCm archive checksum mismatch. Expected $RocmExpected, got $RocmActual."
    }
    $RocmStage = Join-Path $RuntimeBase (".ollama-rocm-stage-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $RocmStage | Out-Null
    try {
        Expand-Archive -LiteralPath $RocmArchive -DestinationPath $RocmStage
        if (-not (Get-ChildItem -LiteralPath $RocmStage -File -Recurse | Select-Object -First 1)) {
            throw "The verified Ollama ROCm archive was empty."
        }
        Get-ChildItem -LiteralPath $RocmStage -Force | Copy-Item -Destination $Target -Recurse -Force
        ("Ollama ROCm $Version" + [Environment]::NewLine + "SHA256 $RocmExpected") | Set-Content -LiteralPath $RocmMarker -Encoding ascii
    } catch {
        Write-Warning "Incomplete ROCm staging retained for diagnosis: $RocmStage"
        throw
    }
}

Write-Output $Target
