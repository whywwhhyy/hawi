$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npm) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
}

if ($null -eq $npm) {
    [Console]::Error.WriteLine("npm is required to launch Hawi GUI.")
    exit 1
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

Push-Location -LiteralPath $scriptDir
try {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        Write-Host "Installing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    $startArgs = @($args)
    if ($startArgs.Count -eq 1 -and -not $startArgs[0].StartsWith("--")) {
        $startArgs = @("--model", $startArgs[0])
    }

    if ($startArgs.Count -gt 0) {
        Invoke-Npm run start '--' @startArgs
    } else {
        Invoke-Npm run start
    }
} finally {
    Pop-Location
}
