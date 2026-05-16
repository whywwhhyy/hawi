$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npm) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
}

if ($null -eq $npm) {
    [Console]::Error.WriteLine("npm is required to package Hawi GUI.")
    exit 1
}

$uv = Get-Command uv.exe -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
}

if ($null -eq $uv) {
    [Console]::Error.WriteLine("uv is required to package Hawi GUI.")
    exit 1
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location -LiteralPath $repoRoot
try {
    Write-Host "Syncing Hawi Python dependencies..."
    Invoke-Checked $uv.Source sync
} finally {
    Pop-Location
}

Push-Location -LiteralPath $scriptDir
try {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        Write-Host "Installing Hawi GUI dependencies..."
        Invoke-Checked $npm.Source install
    }

    Write-Host "Packaging Hawi GUI..."
    $packArgs = @($args)
    if ($packArgs.Count -gt 0) {
        Invoke-Checked $npm.Source run dist '--' @packArgs
    } else {
        Invoke-Checked $npm.Source run dist
    }
} finally {
    Pop-Location
}
