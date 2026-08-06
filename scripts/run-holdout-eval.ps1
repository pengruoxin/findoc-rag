param([int]$TopK = 5)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
& "$PSScriptRoot/validate-artifacts.ps1"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv is required to execute retrieval evaluation. Install uv and run this script again."
}
uv run findoc-rag evaluate-holdout data/diagnostics/holdout-eval-v2.json --top-k $TopK --adaptive-candidate-budget --output reports/ranking/holdout-eval-v2-runtime.json
uv run findoc-rag analyze-holdout-failures reports/ranking/holdout-eval-v2-runtime.json --output reports/ranking/holdout-eval-v2-failures.json
