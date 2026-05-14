# Run the FastAPI app from the project root (Windows).
# Usage: .\scripts\run_api.ps1
#        .\scripts\run_api.ps1 --reload
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$env:PYTHONPATH = "."
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}
& $py -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8080 @args
