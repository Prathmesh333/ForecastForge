[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$DataDir = "./data",

    [Parameter(Position = 1)]
    [string]$ModelPath = "./pickle/model.pkl",

    [Parameter(Position = 2)]
    [string]$OutputPath = "./output/predictions.csv"
)

$ErrorActionPreference = "Stop"
$invocationRoot = (Get-Location).Path
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-RunnerPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $invocationRoot $Value))
}

$dataResolved = Resolve-RunnerPath $DataDir
$modelResolved = Resolve-RunnerPath $ModelPath
$outputResolved = Resolve-RunnerPath $OutputPath
$outputParent = Split-Path -Parent $outputResolved
if (-not (Test-Path -LiteralPath $outputParent)) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}

$pythonCommand = Get-Command python -ErrorAction Stop
$predictArgs = @(
    "-m", "src.predict",
    "--data-dir", $dataResolved,
    "--model", $modelResolved,
    "--output", $outputResolved
)
Push-Location $scriptRoot
try {
    & $pythonCommand.Source @predictArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Prediction command failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $outputResolved)) {
    throw "Prediction output was not created: $outputResolved"
}
if ((Get-Item -LiteralPath $outputResolved).Length -le 0) {
    throw "Prediction output is empty: $outputResolved"
}

Write-Output "Predictions written to $outputResolved"
