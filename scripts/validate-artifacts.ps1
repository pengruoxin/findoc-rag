param(
  [string]$IndexRoot = "data/indexes/corpus"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Read-Json($path) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Missing artifact: $path" }
  try { return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json }
  catch { throw "Invalid JSON: $path`n$($_.Exception.Message)" }
}

$eval = Read-Json "data/diagnostics/holdout-eval-v2.json"
if ($eval.items.Count -ne 16) { throw "Expected 16 holdout items, got $($eval.items.Count)" }
if ($eval.independent_gold -ne $false) { throw "Holdout must remain marked independent_gold=false" }

$registry = Read-Json "reports/ranking/experiment-registry-v1.json"
if ($registry.runs.Count -lt 1) { throw "Experiment registry is empty" }

$generation = Read-Json "data/evaluation/generation-eval-v1.json"
$generationReport = Read-Json "reports/generation/dataset-validation-v1.json"
if ($generation.item_count -ne $generation.items.Count) {
  throw "Generation dataset item_count does not match items"
}
if ($generation.item_count -ne 48) {
  throw "Expected 48 generation items, got $($generation.item_count)"
}
if ($generation.dataset_id -ne $generationReport.dataset_id) {
  throw "Generation dataset and validation report IDs differ"
}
if ($generationReport.warning_count -ne 0) {
  throw "Generation dataset has $($generationReport.warning_count) validation warnings"
}
if ($generationReport.robustness_split_counts.frozen_test -ne 12) {
  throw "Frozen generation split must have complete robustness coverage"
}

$ids = @($eval.items | % gold_chunk_ids | % { $_ })
$chunkFiles = Get-ChildItem -LiteralPath $IndexRoot -Recurse -Filter "*.jsonl" -ErrorAction SilentlyContinue
$text = ($chunkFiles | Get-Content -Raw -ErrorAction SilentlyContinue) -join "`n"
$missing = @($ids | ? { $text -notmatch [regex]::Escape($_) } | Select -Unique)
if ($missing.Count) { throw "Missing gold chunk IDs: $($missing -join ', ')" }

Write-Host "Artifacts valid: $($eval.items.Count) retrieval holdout items, $($generation.item_count) generation items, $($registry.runs.Count) experiment runs."
