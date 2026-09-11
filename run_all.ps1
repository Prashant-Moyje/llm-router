# Full pipeline, Windows PowerShell.
# Activate the venv FIRST, then run this script from the repo root.
#   py -3.12 -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install -r requirements.txt
#   .\run_all.ps1 -Provider mock

param(
    [ValidateSet("mock", "groq", "anthropic", "ollama")] [string]$Provider = "mock",
    [string]$Config = "configs/default.yaml",
    [int]$Limit = 600
)

$ErrorActionPreference = "Stop"

if (-not $env:VIRTUAL_ENV) {
    Write-Host "No virtualenv active. Run .\.venv\Scripts\Activate.ps1 first." -ForegroundColor Yellow
    exit 1
}
if ($Provider -eq "groq" -and -not $env:GROQ_API_KEY) {
    Write-Host "GROQ_API_KEY is not set. Free key: https://console.groq.com" -ForegroundColor Red
    exit 1
}
if ($Provider -eq "anthropic" -and -not $env:ANTHROPIC_API_KEY) {
    Write-Host "ANTHROPIC_API_KEY is not set." -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"

Write-Host "`n[1/5] build dataset"      -ForegroundColor Cyan
python scripts/01_build_dataset.py --config $Config --sources synthetic --limit $Limit
Write-Host "`n[2/5] offline eval"       -ForegroundColor Cyan
python scripts/02_run_offline_eval.py --config $Config --provider $Provider --workers 8
Write-Host "`n[3/5] train router"       -ForegroundColor Cyan
python scripts/03_train_router.py --config $Config
Write-Host "`n[4/5] simulate + sweep"   -ForegroundColor Cyan
python scripts/04_simulate.py --config $Config --quality-floor-drop 0.02
Write-Host "`n[5/5] plot + bench"       -ForegroundColor Cyan
python scripts/05_plot.py --config $Config
python scripts/06_bench_latency.py --config $Config

Write-Host "`nDone. See reports\headline.json and reports\pareto.png" -ForegroundColor Green
