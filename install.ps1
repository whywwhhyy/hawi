$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiDirName = "hawi_gui"
$guiDir = Join-Path $scriptDir $guiDirName

if (-not (Test-Path -LiteralPath (Join-Path $guiDir "package.json") -PathType Leaf)) {
    [Console]::Error.WriteLine("Current Hawi GUI directory is missing: $guiDir")
    [Console]::Error.WriteLine("Install only targets $guiDirName; hawi_legacy_gui is archived and not used by this script.")
    exit 1
}

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
        [Console]::Error.WriteLine("uv is required to build the bundled Hawi engine. Use --skip-build only if the selected GUI shell already has build output.")
        exit 1
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

function Add-HawiBinToUserPath {
    if ($env:OS -ne "Windows_NT") {
        return
    }

    $binDir = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path -LiteralPath $binDir -PathType Container)) {
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathParts = @()
    if (-not [string]::IsNullOrWhiteSpace($userPath)) {
        $pathParts = @($userPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    $alreadyPresent = $false
    foreach ($part in $pathParts) {
        if ([string]::Equals($part.TrimEnd('\'), $binDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
            $alreadyPresent = $true
            break
        }
    }

    if (-not $alreadyPresent) {
        $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $binDir } else { "$userPath;$binDir" }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "Added $binDir to the user PATH."
    }

    $processParts = @($env:Path -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $processHasBin = $false
    foreach ($part in $processParts) {
        if ([string]::Equals($part.TrimEnd('\'), $binDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
            $processHasBin = $true
            break
        }
    }
    if (-not $processHasBin) {
        $env:Path = "$binDir;$env:Path"
    }
}

if ($requiresUv) {
    Write-Host "Syncing Hawi Python dependencies..."
    Invoke-Uv sync --all-extras --all-groups
}

Push-Location -LiteralPath $guiDir
try {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        Write-Host "Installing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    if ($env:HAWI_INSTALL_SKIP_PREFLIGHT -ne "1") {
        Invoke-Npm run install:preflight '--' @args
    }

    if ([string]::IsNullOrEmpty($env:HAWI_RELEASE_COMMAND)) {
        $env:HAWI_RELEASE_COMMAND = Join-Path $scriptDir "install.ps1"
    }

    Invoke-Npm run release:local '--' @args
    Add-HawiBinToUserPath
} finally {
    Pop-Location
}
