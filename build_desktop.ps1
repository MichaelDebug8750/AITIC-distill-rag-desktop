param(
    [switch]$InstallOnly,
    [switch]$SkipMsi,
    [switch]$SkipSetupExe,
    [switch]$SkipPortableZip,
    [switch]$SkipOllamaRuntime,
    [string]$OllamaRuntimeVersion = "0.32.4"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Dist = Join-Path $Root "dist"
$AppDist = Join-Path $Dist "AITIC Desktop"
$Work = Join-Path $Root "build\pyinstaller"
$Tools = Join-Path $Root "packaging\.tools"
$Wix = Join-Path $Tools "wix.exe"
$LocalDotnetRoot = Join-Path $Root "packaging\.dotnet"
$LocalDotnet = Join-Path $LocalDotnetRoot "dotnet.exe"
$Version = "1.0.0"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$env:DOTNET_NOLOGO = "1"
if (Test-Path -LiteralPath $LocalDotnet -PathType Leaf) {
    # WiX is a framework-dependent local .NET tool. Keep later builds reproducible
    # even when the machine has only a global runtime and no SDK.
    $env:DOTNET_ROOT = $LocalDotnetRoot
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $Py) { throw "Python was not found. The build machine needs Python 3.11 or newer." }
    & $Py.Source -3.14 -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the isolated build environment." }
}

Write-Host "[1/8] Installing pinned dependencies" -ForegroundColor Cyan
& $Python -m pip install --quiet --disable-pip-version-check -r (Join-Path $Root "desktop-runtime-requirements.txt") `
    -r (Join-Path $Root "desktop-requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
if ($InstallOnly) { Write-Host "Desktop build environment is ready." -ForegroundColor Green; exit 0 }

Write-Host "[2/8] Generating the Windows icon" -ForegroundColor Cyan
& $Python (Join-Path $Root "packaging\make_icon.py")
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed." }

Write-Host "[3/8] Freezing the native EXE (onedir)" -ForegroundColor Cyan
& $Python -m PyInstaller --noconfirm --clean --distpath $Dist --workpath $Work `
    (Join-Path $Root "packaging\AITICDesktop.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
$Exe = Join-Path $AppDist "AITIC Desktop.exe"
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "Main executable was not created: $Exe" }

Write-Host "[4/8] Adding the local AI runtime and user documentation" -ForegroundColor Cyan
if (-not $SkipOllamaRuntime) {
    $RuntimeOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $Root "packaging\fetch_ollama_runtime.ps1") -Version $OllamaRuntimeVersion
    if ($LASTEXITCODE -ne 0) { throw "Ollama runtime acquisition failed." }
    $RuntimeSource = [string]($RuntimeOutput | Select-Object -Last 1)
    if (-not (Test-Path -LiteralPath (Join-Path $RuntimeSource "ollama.exe") -PathType Leaf)) {
        throw "The verified Ollama runtime is incomplete: $RuntimeSource"
    }
    $RuntimeTarget = Join-Path $AppDist "runtime\ollama"
    New-Item -ItemType Directory -Path $RuntimeTarget -Force | Out-Null
    Get-ChildItem -LiteralPath $RuntimeSource -Force | Copy-Item `
        -Destination $RuntimeTarget -Recurse -Force
    $VcOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $Root "packaging\fetch_vc_runtime.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Microsoft VC++ runtime acquisition failed." }
    $VcSource = [string]($VcOutput | Select-Object -Last 1)
    Copy-Item -LiteralPath $VcSource -Destination `
        (Join-Path $RuntimeTarget "vc_redist.x64.exe") -Force
}
# Keep end-user documentation beside the EXE instead of hiding it in PyInstaller's
# _internal directory. WiX and the portable archive both consume AppDist afterwards.
Copy-Item -LiteralPath (Join-Path $Root "DESKTOP_README.md") `
    -Destination (Join-Path $AppDist "DESKTOP_README.zh-CN.md") -Force
Copy-Item -LiteralPath (Join-Path $Root "packaging\THIRD_PARTY_NOTICES.md") `
    -Destination (Join-Path $AppDist "THIRD_PARTY_NOTICES.md") -Force
Copy-Item -LiteralPath (Join-Path $Root "MODEL_SETUP_GUIDE.md") `
    -Destination (Join-Path $AppDist "MODEL_SETUP_GUIDE.zh-CN.md") -Force

Write-Host "[5/8] Checking the frozen output" -ForegroundColor Cyan
& $Python -c "import pathlib; p=pathlib.Path(r'$Exe'); print('EXE_BYTES=' + str(p.stat().st_size))"

if (-not $SkipMsi) {
    Write-Host "[6/8] Building the per-user MSI" -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $Wix -PathType Leaf)) {
        New-Item -ItemType Directory -Path $Tools -Force | Out-Null
        $Dotnet = (Get-Command dotnet -ErrorAction SilentlyContinue).Source
        $HasSdk = $false
        if ($Dotnet) {
            $HasSdk = [bool]((& $Dotnet --list-sdks 2>$null) | Select-Object -First 1)
        }
        if (-not $HasSdk) {
            $InstallScript = Join-Path $Tools "dotnet-install.ps1"
            if (-not (Test-Path -LiteralPath $InstallScript -PathType Leaf)) {
                Invoke-WebRequest "https://dot.net/v1/dotnet-install.ps1" -OutFile $InstallScript
            }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallScript `
                -Channel 8.0 -Quality GA -InstallDir $LocalDotnetRoot -NoPath
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $LocalDotnet -PathType Leaf)) {
                throw "Local .NET SDK installation failed. Use -SkipMsi to build only the EXE."
            }
            $Dotnet = $LocalDotnet
            $env:DOTNET_ROOT = $LocalDotnetRoot
        }
        & $Dotnet tool install wix --version 4.0.6 --tool-path $Tools
        if ($LASTEXITCODE -ne 0) { throw "WiX installation failed. Use -SkipMsi to build only the EXE." }
    }
    $Wxs = Join-Path $Root "packaging\generated\AITICDesktop.wxs"
    $WixUiExtension = "WixToolset.UI.wixext/4.0.6"
    $InstalledExtensions = (& $Wix extension list 2>$null) -join "`n"
    if ($InstalledExtensions -notmatch "WixToolset\.UI\.wixext") {
        & $Wix extension add $WixUiExtension
        if ($LASTEXITCODE -ne 0) { throw "WiX UI extension installation failed." }
    }
    & $Python (Join-Path $Root "packaging\make_wix.py") --dist $AppDist --output $Wxs `
        --icon (Join-Path $Root "packaging\aitic.ico") `
        --license (Join-Path $Root "packaging\AITIC_EULA.rtf") --version $Version
    if ($LASTEXITCODE -ne 0) { throw "WiX source generation failed." }
    $Msi = Join-Path $Dist "AITIC-Desktop-$Version-x64.msi"
    & $Wix build $Wxs -arch x64 -ext WixToolset.UI.wixext -culture zh-CN -o $Msi
    if ($LASTEXITCODE -ne 0) { throw "MSI build failed." }
    # ICE91 only warns that per-user files would not be available to every user in
    # a hypothetical per-machine install. This package is explicitly Scope=perUser;
    # suppress that inapplicable warning while retaining every other ICE check.
    & $Wix msi validate -sice ICE91 $Msi
    if ($LASTEXITCODE -ne 0) { throw "MSI validation failed." }
}

if (-not $SkipSetupExe) {
    Write-Host "[7/8] Building the bilingual public Setup EXE" -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $Root "packaging\build_setup_installer.ps1") -Version $Version
    if ($LASTEXITCODE -ne 0) { throw "Setup EXE build failed." }
}

Write-Host "[8/8] Creating the portable archive and hashes" -ForegroundColor Cyan
if (-not $SkipPortableZip) {
    $Zip = Join-Path $Dist "AITIC-Desktop-$Version-Portable.zip"
    if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
    Compress-Archive -LiteralPath $AppDist -DestinationPath $Zip -CompressionLevel Optimal
}
$Artifacts = Get-ChildItem -LiteralPath $Dist -File | Where-Object {
    $_.Extension -in '.exe','.msi','.zip' -and $_.Name -like 'AITIC-Desktop-*'
}
$Lines = @()
foreach ($Artifact in $Artifacts) {
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact.FullName).Hash
    $Lines += "$Hash  $($Artifact.Name)"
}
$Lines | Set-Content -LiteralPath (Join-Path $Dist "SHA256SUMS.txt") -Encoding ascii
Write-Host "Build completed: $Dist" -ForegroundColor Green
Get-ChildItem -LiteralPath $Dist | Select-Object Name,Length,LastWriteTime
