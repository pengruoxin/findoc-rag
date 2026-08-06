param(
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $env:FINDOC_RAG_INDEX_DIR) {
  $env:FINDOC_RAG_INDEX_DIR = "data/indexes/corpus"
}

& (Join-Path $PSScriptRoot "validate-artifacts.ps1")

if (Get-Command uv -ErrorAction SilentlyContinue) {
  uv run uvicorn findoc_rag.api:create_app --factory --host $BindHost --port $Port
} else {
  Write-Error "uv is required. Install it, then run: uv run uvicorn findoc_rag.api:create_app --factory --host $BindHost --port $Port"
}
