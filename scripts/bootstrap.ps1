$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".\backend[dev]"
& ".\.venv\Scripts\python.exe" -m pytest

Write-Host "GeoVision development environment is ready."
Write-Host "Start the API with:"
Write-Host ".\.venv\Scripts\python.exe -m uvicorn geovision.main:app --app-dir backend\src --reload"

