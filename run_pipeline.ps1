param (
    [string]$Stage = "all"
)

$env_name = "MLOps"
$fallback_python = "C:\Users\Hossam\anaconda3\envs\MLOps\python.exe"

Write-Host "--- DeepClassifier MLOps Runner ---" -ForegroundColor Cyan

# Check Environment
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Using Conda environment: $env_name" -ForegroundColor Green
    $python_cmd = "python"
} elseif (Test-Path $fallback_python) {
    Write-Host "Using direct environment path: $fallback_python" -ForegroundColor Yellow
    $python_cmd = $fallback_python
} else {
    Write-Error "Could not find Conda or the MLOps environment python at $fallback_python"
    exit
}

function Run-Stage($name, $script) {
    Write-Host "`n>>> Running $name Stage..." -ForegroundColor Magenta
    & $python_cmd $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$name Stage Failed!"
        exit $LASTEXITCODE
    }
}

if ($Stage -eq "all" -or $Stage -eq "preprocess") {
    Run-Stage "Preprocessing" "src/pipelines/preprocessing_pipeline.py"
}

if ($Stage -eq "all" -or $Stage -eq "train") {
    Run-Stage "Training" "src/pipelines/training_pipeline.py"
}

if ($Stage -eq "all" -or $Stage -eq "eval") {
    Run-Stage "Evaluation" "src/pipelines/evaluation_pipeline.py"
}

Write-Host "`nAll requested stages finished successfully!" -ForegroundColor Green
