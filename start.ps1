$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiDirName = "hawi_gui"
$guiDir = Join-Path $scriptDir $guiDirName
$launchCwd = (Get-Location).Path
$rawArgs = @($args)

if (-not (Test-Path -LiteralPath (Join-Path $guiDir "package.json") -PathType Leaf)) {
    [Console]::Error.WriteLine("Current Hawi GUI directory is missing: $guiDir")
    exit 1
}

function Add-DirectoryToProcessPath {
    param([string]$Directory)

    if ([string]::IsNullOrWhiteSpace($Directory)) {
        return
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Directory)
    if (-not (Test-Path -LiteralPath $expanded -PathType Container)) {
        return
    }

    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    $pathParts = @()
    if (-not [string]::IsNullOrWhiteSpace($currentPath)) {
        $pathParts = @($currentPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    foreach ($part in $pathParts) {
        if ([string]::Equals($part.TrimEnd('\'), $expanded.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }

    $env:Path = if ([string]::IsNullOrWhiteSpace($currentPath)) { $expanded } else { "$expanded;$currentPath" }
}

function Add-CommonToolDirsToProcessPath {
    $userProfile = [Environment]::GetEnvironmentVariable("USERPROFILE")
    if (-not [string]::IsNullOrWhiteSpace($userProfile)) {
        Add-DirectoryToProcessPath (Join-Path $userProfile ".local\bin")
        Add-DirectoryToProcessPath (Join-Path $userProfile ".cargo\bin")
    }

    $programFiles = [Environment]::GetEnvironmentVariable("ProgramFiles")
    if (-not [string]::IsNullOrWhiteSpace($programFiles)) {
        Add-DirectoryToProcessPath (Join-Path $programFiles "nodejs")
    }
}

function Find-HawiCommand {
    param([string[]]$Names)

    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command
        }
    }
    return $null
}

function Invoke-WingetInstall {
    param(
        [string]$PackageId,
        [string]$DisplayName
    )

    $winget = Find-HawiCommand @("winget.exe", "winget")
    if ($null -eq $winget) {
        [Console]::Error.WriteLine("$DisplayName is required, and winget is not available for automatic installation.")
        exit 1
    }

    Write-Host "$DisplayName is missing; installing with winget..."
    & $winget.Source install --id $PackageId -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Ensure-Npm {
    $script:npm = Find-HawiCommand @("npm.cmd", "npm.exe", "npm")
    if ($null -ne $script:npm) {
        return
    }

    Invoke-WingetInstall "OpenJS.NodeJS.LTS" "Node.js/npm"
    Add-CommonToolDirsToProcessPath
    $script:npm = Find-HawiCommand @("npm.cmd", "npm.exe", "npm")
    if ($null -eq $script:npm) {
        [Console]::Error.WriteLine("Node.js/npm was installed, but npm is still not on PATH. Restart PowerShell and run start.ps1 again.")
        exit 1
    }
}

function Ensure-Uv {
    $script:uv = Find-HawiCommand @("uv.exe", "uv")
    if ($null -ne $script:uv) {
        return
    }

    Invoke-WingetInstall "astral-sh.uv" "uv"
    Add-CommonToolDirsToProcessPath
    $script:uv = Find-HawiCommand @("uv.exe", "uv")
    if ($null -eq $script:uv) {
        [Console]::Error.WriteLine("uv was installed, but uv is still not on PATH. Restart PowerShell and run start.ps1 again.")
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

function Invoke-Uv {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $uv.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Test-NpmDependencies {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        return $false
    }

    & $npm.Source ls --depth=0 --silent *> $null
    return $LASTEXITCODE -eq 0
}

function Test-PythonDependencies {
    if (-not (Test-Path -LiteralPath (Join-Path $scriptDir ".venv") -PathType Container)) {
        return $false
    }

    & $uv.Source run --project $scriptDir python -c "import hawi" *> $null
    return $LASTEXITCODE -eq 0
}

function Sync-PythonDependencies {
    Write-Host "Syncing Hawi Python dependencies..."
    Invoke-Uv sync --all-extras --all-groups
}

$desktopShell = if ([string]::IsNullOrWhiteSpace($env:HAWI_GUI_SHELL)) { "tauri" } else { $env:HAWI_GUI_SHELL }
$helpRequested = $false
for ($i = 0; $i -lt $rawArgs.Count; $i++) {
    $arg = $rawArgs[$i]
    if ($arg -eq "--help" -or $arg -eq "-h") {
        $helpRequested = $true
    } elseif ($arg -eq "--shell" -or $arg -eq "--runtime" -or $arg -eq "--gui") {
        if ($i + 1 -ge $rawArgs.Count) {
            [Console]::Error.WriteLine("$arg requires a value.")
            exit 1
        }
        $desktopShell = $rawArgs[$i + 1]
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

Add-CommonToolDirsToProcessPath
Ensure-Npm

Push-Location -LiteralPath $guiDir
try {
    if ($helpRequested) {
        Invoke-Npm run start '--' @rawArgs
        exit 0
    }

    Ensure-Uv
    if ([string]::IsNullOrEmpty($env:HAWI_GUI_UV_COMMAND)) {
        $env:HAWI_GUI_UV_COMMAND = $uv.Source
    }
    if (-not (Test-PythonDependencies)) {
        Sync-PythonDependencies
    }

    if (-not (Test-NpmDependencies)) {
        Write-Host "Installing or repairing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    if ($desktopShell -eq "tauri") {
        Invoke-Npm run install:preflight '--' --skip-build
        Add-CommonToolDirsToProcessPath
        $cargo = Find-HawiCommand @("cargo.exe", "cargo")
        if ($null -eq $cargo) {
            [Console]::Error.WriteLine("cargo is required to launch the Tauri Hawi GUI, and automatic installation did not make it available on PATH.")
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

    $startArgs = @($rawArgs)
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
