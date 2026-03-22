$env_name = "MLOps"
$fallback_python = "C:\Users\Hossam\anaconda3\envs\MLOps\python.exe"

Write-Host "--- MLOps Pipeline Runner ---" -ForegroundColor Cyan

# Check if conda is available
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Activating environment: $env_name via Conda..." -ForegroundColor Green
    conda activate $env_name
    $python_cmd = "python"
} elseif (Test-Path $fallback_python) {
    Write-Host "Conda not in PATH. Using direct environment path: $fallback_python" -ForegroundColor Yellow
    $python_cmd = $fallback_python
} else {
    Write-Error "Could not find Conda or the MLOps environment python at $fallback_python"
    exit
}

Write-Host "Running Preprocessing Pipeline..." -ForegroundColor Yellow
& $python_cmd src/pipelines/preprocessing_pipeline.py

Write-Host "Pipeline execution finished. Check MLflow for results." -ForegroundColor Green
