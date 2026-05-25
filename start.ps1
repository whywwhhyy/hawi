$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiDir = Join-Path $scriptDir "hawi_gui"
$launchCwd = (Get-Location).Path

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

Push-Location -LiteralPath $guiDir
try {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        Write-Host "Installing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    $desktopShell = if ([string]::IsNullOrWhiteSpace($env:HAWI_GUI_SHELL)) { "tauri" } else { $env:HAWI_GUI_SHELL }
    for ($i = 0; $i -lt $args.Count; $i++) {
        $arg = $args[$i]
        if ($arg -eq "--shell" -or $arg -eq "--runtime" -or $arg -eq "--gui") {
            if ($i + 1 -ge $args.Count) {
                [Console]::Error.WriteLine("$arg requires a value.")
                exit 1
            }
            $desktopShell = $args[$i + 1]
            $i++
        } elseif ($arg.StartsWith("--shell=")) {
            $desktopShell = $arg.Substring("--shell=".Length)
        } elseif ($arg.StartsWith("--runtime=")) {
            $desktopShell = $arg.Substring("--runtime=".Length)
        } elseif ($arg.StartsWith("--gui=")) {
            $desktopShell = $arg.Substring("--gui=".Length)
        }
    }
    $desktopShell = $desktopShell.Trim().ToLowerInvariant()
    if ($desktopShell -ne "tauri" -and $desktopShell -ne "electron") {
        [Console]::Error.WriteLine("--shell must be one of: tauri, electron")
        exit 1
    }

    if ($desktopShell -eq "tauri") {
        $cargo = Get-Command cargo -ErrorAction SilentlyContinue
        if ($null -eq $cargo) {
            [Console]::Error.WriteLine("cargo is required to launch the Tauri Hawi GUI.")
            exit 1
        }
        if (-not (Test-Path -LiteralPath "src-tauri/tauri.conf.json" -PathType Leaf)) {
            [Console]::Error.WriteLine("Tauri project files are missing under $guiDir/src-tauri.")
            exit 1
        }
    } elseif (
        -not (Test-Path -LiteralPath "dist/index.html" -PathType Leaf) -or
        -not (Test-Path -LiteralPath "dist-electron/main/main.js" -PathType Leaf)
    ) {
        [Console]::Error.WriteLine("Electron GUI build output is missing. Run '$scriptDir\install.ps1 --shell electron' once, or run 'npm run build' in $guiDir.")
        exit 1
    }

    $startArgs = @($args)
    if ($startArgs.Count -eq 1 -and -not $startArgs[0].StartsWith("--")) {
        $startArgs = @("--model", $startArgs[0])
    }

    if ([string]::IsNullOrEmpty($env:HAWI_GUI_CWD)) {
        $env:HAWI_GUI_CWD = $launchCwd
    }
    Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue

    if ($startArgs.Count -gt 0) {
        Invoke-Npm run start '--' @startArgs
    } else {
        Invoke-Npm run start
    }
} finally {
    Pop-Location
}
