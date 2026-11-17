param(
    [string]$Dataset = "fsl-dataset.json",
    [string]$OutputDir = "",
    [int]$Epochs = 80,
    [int]$BatchSize = 32,
    [switch]$TestMode,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$toolkitDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $toolkitDir

if (-not $OutputDir) {
    $OutputDir = if ($TestMode) { "test-model" } else { "trained-model" }
}
if ($Epochs -lt 1) {
    throw "Epochs must be at least 1."
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be at least 1."
}
if (-not (Test-Path -LiteralPath $Dataset -PathType Leaf)) {
    throw "Dataset not found: $Dataset. Pass the exported JSON with -Dataset."
}

$python = Join-Path $toolkitDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if ($SkipInstall) {
        throw "Virtual environment not found. Run without -SkipInstall to create it and install dependencies."
    }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw "Python Launcher (py) is required to create .venv. Install Python 3.10-3.12, then retry."
    }
    $pythonVersion = $null
    foreach ($version in @("3.11", "3.10", "3.12")) {
        & $launcher.Source "-$version" --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonVersion = $version
            break
        }
    }
    if (-not $pythonVersion) {
        throw "Python 3.10, 3.11, or 3.12 was not found. Install one and retry."
    }
    & $launcher.Source "-$pythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment."
    }
}

$report = Join-Path $OutputDir "dataset-report.json"
if (-not $TestMode) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    & $python training\validate_dataset.py --dataset $Dataset --report $report
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the dataset. Check the dataset path and JSON format."
    }
    Write-Host "Readiness report saved to $report. Continuing with training; readiness affects deployment, not whether training can run."
} else {
    Write-Host "TEST MODE: output is for pipeline testing only and must not be deployed." -ForegroundColor Yellow
}

if (-not $SkipInstall) {
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
    if ($env:OS -eq "Windows_NT") {
        & $python -m pip install --no-deps tensorflowjs==4.22.0
        if ($LASTEXITCODE -ne 0) {
            throw "TensorFlow.js converter installation failed."
        }
    }
}

$arguments = @(
    "training\train_model.py",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize"
)
if ($TestMode) {
    $arguments += "--test-mode"
}
& $python @arguments
exit $LASTEXITCODE
