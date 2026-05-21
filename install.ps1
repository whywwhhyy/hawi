$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiDir = Join-Path $scriptDir "hawi_gui"

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npm) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
}

if ($null -eq $npm) {
    [Console]::Error.WriteLine("npm is required to install Hawi GUI.")
    exit 1
}

$requiresUv = $true
foreach ($arg in $args) {
    if ($arg -eq "--skip-build" -or $arg -eq "-h" -or $arg -eq "--help") {
        $requiresUv = $false
    }
}

if ($requiresUv) {
    $uv = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($null -eq $uv) {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
    }
    if ($null -eq $uv) {
        [Console]::Error.WriteLine("uv is required to build the bundled Hawi engine. Use --skip-build only if hawi_gui/release already exists.")
        exit 1
    }
}

function Invoke-Npm {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $npm.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location -LiteralPath $guiDir
try {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        Write-Host "Installing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    if ([string]::IsNullOrEmpty($env:HAWI_RELEASE_COMMAND)) {
        $env:HAWI_RELEASE_COMMAND = Join-Path $scriptDir "install.ps1"
    }

    Invoke-Npm run release:local '--' @args
} finally {
    Pop-Location
}
