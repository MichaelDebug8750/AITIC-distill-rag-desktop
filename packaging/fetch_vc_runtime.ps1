param(
    [string]$Sha256 = "843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C"
)

$ErrorActionPreference = "Stop"
$Packaging = Split-Path -Parent $MyInvocation.MyCommand.Path
$Downloads = Join-Path $Packaging ".downloads"
$Target = Join-Path $Downloads "vc_redist.x64.exe"
$Expected = $Sha256.Trim().ToUpperInvariant()
New-Item -ItemType Directory -Path $Downloads -Force | Out-Null

if (-not (Test-Path -LiteralPath $Target -PathType Leaf)) {
    Write-Host "Downloading Microsoft Visual C++ x64 Redistributable ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://aka.ms/vc14/vc_redist.x64.exe" -OutFile $Target
}
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToUpperInvariant()
if ($Actual -ne $Expected) {
    throw "VC++ Redistributable checksum mismatch. Expected $Expected, got $Actual."
}
$Signature = Get-AuthenticodeSignature -LiteralPath $Target
$Subject = [string]$Signature.SignerCertificate.Subject
if ($Signature.Status -ne "Valid" -or $Subject -notmatch "(^|, )O=Microsoft Corporation(,|$)") {
    throw "VC++ Redistributable is not validly signed by Microsoft Corporation. Status=$($Signature.Status); Subject=$Subject"
}
Write-Output $Target
