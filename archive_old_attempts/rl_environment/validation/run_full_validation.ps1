# Run full validation process for CARLA environment wrapper
# This script executes the complete validation workflow

# Stop on any error
$ErrorActionPreference = "Stop"

# Set up environment variables for Python module path
$env:PYTHONPATH = "$($pwd | Split-Path -Parent);$env:PYTHONPATH"
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"

Write-Host "===== CARLA Environment Validation Workflow =====" -ForegroundColor Cyan

# Step 1: Set up dependencies
Write-Host "`n[1/6] Setting up dependencies..." -ForegroundColor Yellow
python setup_validation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to set up dependencies. Exiting." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 2: Check if CARLA is running or start it
Write-Host "`n[2/6] Checking CARLA server..." -ForegroundColor Yellow
python prepare_carla_for_validation_coursera.py --check-only
$carlaRunning = $LASTEXITCODE
if ($carlaRunning -ne 0) {
    Write-Host "CARLA server not running. Starting CARLA..." -ForegroundColor Yellow
    Start-Process python -ArgumentList "prepare_carla_for_validation_coursera.py --map /Game/Maps/Course4 --fps 30" -NoNewWindow -Wait
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start CARLA server. Exiting." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Step 3: Run advanced validation tests
Write-Host "`n[3/6] Running validation tests..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsPath = "validation_results_$timestamp.json"
python run_advanced_validation.py --tests all --iterations 3 --generate-report
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some validation tests failed." -ForegroundColor Yellow
}

# Step 4: View results summary
Write-Host "`n[4/6] Viewing validation results summary..." -ForegroundColor Yellow
$latestResult = Get-ChildItem -Path . -Filter "validation_results_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python view_validation_results.py --results $latestResult.FullName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to view results. Continuing anyway..." -ForegroundColor Yellow
}

# Step 5: Generate detailed report
Write-Host "`n[5/6] Generating detailed validation report..." -ForegroundColor Yellow
python generate_validation_report.py --results $latestResult.FullName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to generate report. Continuing anyway..." -ForegroundColor Yellow
}

# Step 6: Compare with previous runs if available
Write-Host "`n[6/6] Comparing with previous validation runs..." -ForegroundColor Yellow
$previousResults = Get-ChildItem -Path . -Filter "validation_results_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 2
if ($previousResults.Count -ge 2) {
    $comparisonPath = "validation_comparison_$timestamp.md"
    python compare_validation_results.py --results $previousResults[1].FullName $previousResults[0].FullName --output $comparisonPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to compare results. Continuing anyway..." -ForegroundColor Yellow
    }
    else {
        Write-Host "Comparison report generated: $comparisonPath" -ForegroundColor Green
    }
}
else {
    Write-Host "Not enough previous results for comparison. Skipping comparison step." -ForegroundColor Yellow
}

# Final summary
Write-Host "`n===== Validation Complete =====" -ForegroundColor Cyan
$latestReport = Get-ChildItem -Path . -Filter "validation_report_*.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "`nResults available at:"
Write-Host "- Results JSON: $($latestResult.FullName)" -ForegroundColor Green
if ($latestReport) {
    Write-Host "- Detailed Report: $($latestReport.FullName)" -ForegroundColor Green
}

Write-Host "`nNext steps:"
Write-Host "1. Review the detailed report to identify any issues"
Write-Host "2. Fix any identified issues in the environment wrapper"
Write-Host "3. Re-run validation to verify fixes"

if ($carlaRunning -ne 0) {
    Write-Host "`nNote: CARLA server was started by this script. You might want to close it if it's no longer needed." -ForegroundColor Yellow
}

exit 0
