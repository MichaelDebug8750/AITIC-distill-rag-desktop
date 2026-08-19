param(
    [string]$Version = "7.1.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DownloadDir = Join-Path $Root ".downloads"
$ToolDir = Join-Path $Root ".inno7"
$Installer = Join-Path $DownloadDir "innosetup-$Version-x64.exe"
$Iscc = Join-Path $ToolDir "ISCC.exe"
$UriVersion = $Version.Replace('.', '_')
$Uri = "https://github.com/jrsoftware/issrc/releases/download/is-$UriVersion/innosetup-$Version-x64.exe"

New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    Invoke-WebRequest -Uri $Uri -OutFile $Installer
}

$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($Signature.Status -ne "Valid" -or
    $Signature.SignerCertificate.Subject -notlike "CN=Pyrsys B.V.*") {
    throw "The Inno Setup compiler signature is invalid or has an unexpected publisher."
}

if (-not (Test-Path -LiteralPath $Iscc -PathType Leaf)) {
    if (Test-Path -LiteralPath $ToolDir) {
        throw "Refusing to overwrite an unexpected Inno Setup tools directory: $ToolDir"
    }
    $Process = Start-Process -FilePath $Installer -ArgumentList @(
        "/PORTABLE=1", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/CURRENTUSER", ('/DIR="' + $ToolDir + '"')
    ) -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "The portable Inno Setup compiler installation failed: $($Process.ExitCode)"
    }
}

if (-not (Test-Path -LiteralPath $Iscc -PathType Leaf)) {
    throw "ISCC.exe was not created: $Iscc"
}
Write-Output $Iscc
