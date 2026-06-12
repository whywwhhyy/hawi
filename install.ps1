$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiDirName = "hawi_gui"
$guiDir = Join-Path $scriptDir $guiDirName
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
        [Console]::Error.WriteLine("Node.js/npm was installed, but npm is still not on PATH. Restart PowerShell and run install.ps1 again.")
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
        [Console]::Error.WriteLine("uv was installed, but uv is still not on PATH. Restart PowerShell and run install.ps1 again.")
        exit 1
    }
}

$requiresUv = $true
foreach ($arg in $rawArgs) {
    if ($arg -eq "--skip-build" -or $arg -eq "-h" -or $arg -eq "--help") {
        $requiresUv = $false
    }
}

Add-CommonToolDirsToProcessPath
Ensure-Npm
if ($requiresUv) {
    Ensure-Uv
    if ([string]::IsNullOrEmpty($env:HAWI_GUI_UV_COMMAND)) {
        $env:HAWI_GUI_UV_COMMAND = $uv.Source
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

function Test-NpmDependencies {
    if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
        return $false
    }

    & $npm.Source ls --depth=0 --silent *> $null
    return $LASTEXITCODE -eq 0
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
    if (-not (Test-NpmDependencies)) {
        Write-Host "Installing or repairing Hawi GUI dependencies..."
        Invoke-Npm install
    }

    if ($env:HAWI_INSTALL_SKIP_PREFLIGHT -ne "1") {
        Invoke-Npm run install:preflight '--' @rawArgs
        Add-CommonToolDirsToProcessPath
    }

    if ([string]::IsNullOrEmpty($env:HAWI_RELEASE_COMMAND)) {
        $env:HAWI_RELEASE_COMMAND = Join-Path $scriptDir "install.ps1"
    }

    Invoke-Npm run release:local '--' @rawArgs
    Add-HawiBinToUserPath
} finally {
    Pop-Location
}
