$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$shell = if ($env:HAWI_GUI_SHELL) { $env:HAWI_GUI_SHELL } else { "tauri" }
$forwardArgs = New-Object System.Collections.Generic.List[string]

for ($index = 0; $index -lt $args.Count; $index++) {
    $arg = $args[$index]
    if ($arg -eq "--shell" -or $arg -eq "--runtime" -or $arg -eq "--gui") {
        if ($index + 1 -ge $args.Count) {
            [Console]::Error.WriteLine("$arg requires a value")
            exit 1
        }
        $index++
        $shell = $args[$index]
    } elseif ($arg.StartsWith("--shell=") -or $arg.StartsWith("--runtime=") -or $arg.StartsWith("--gui=")) {
        $shell = $arg.Substring($arg.IndexOf("=") + 1)
    } else {
        $forwardArgs.Add($arg)
    }
}

$shell = $shell.ToLowerInvariant()
if ($shell -ne "tauri" -and $shell -ne "electron") {
    [Console]::Error.WriteLine("--shell must be one of: tauri, electron")
    exit 1
}

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

$cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
if ($null -eq $cargo) {
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
}

if ($shell -eq "tauri" -and $null -eq $cargo) {
    [Console]::Error.WriteLine("cargo is required to package Hawi GUI with Tauri.")
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

    Write-Host "Packaging Hawi GUI with $shell..."
    $packArgs = @($forwardArgs)
    if ($shell -eq "tauri") {
        Invoke-Checked $npm.Source run tauri:build '--' @packArgs
    } else {
        Invoke-Checked $npm.Source run dist:electron '--' @packArgs
    }
} finally {
    Pop-Location
}
