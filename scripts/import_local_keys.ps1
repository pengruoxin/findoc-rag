$keyFile = Join-Path (Split-Path $PSScriptRoot -Parent) "local-keys.env"
if (-not (Test-Path -LiteralPath $keyFile -PathType Leaf)) {
    throw "local-keys.env was not found at the workspace root"
}

$line = (Get-Content -LiteralPath $keyFile -Raw).Trim()
if ($line -notmatch '^DEEPSEEK_API_KEY=(.+)$') {
    throw "local-keys.env must contain exactly DEEPSEEK_API_KEY=<value>"
}

$env:DEEPSEEK_API_KEY = $Matches[1]
Write-Output "Loaded DEEPSEEK_API_KEY into the current PowerShell process (value hidden)."
