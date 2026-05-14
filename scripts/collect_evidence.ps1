# Save API JSON/text responses for a submission evidence pack (no GUI screenshots).
# Requires the server: .\scripts\run_api.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$out = Join-Path $root "artifacts\reports\evidence"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$base = "http://127.0.0.1:8080"

function Save-Json($rel, $uri) {
    $path = Join-Path $out $rel
    Invoke-RestMethod -Uri $uri -Method Get | ConvertTo-Json -Depth 8 | Set-Content -Path $path -Encoding utf8
}

Save-Json "health.json" "$base/health"
Save-Json "history_page1.json" "$base/history?limit=5&offset=0"
Save-Json "retrieve_debug.json" "$base/retrieve_debug?q=attention&k=3"

$gen = @{ prompt = "Explain attention in one sentence."; max_new_tokens = 32 } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/generate" -Method Post -ContentType "application/json" -Body $gen | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out "generate.json") -Encoding utf8

$hyb = @{ prompt = "Explain attention in one sentence."; retrieval_top_k = 3; max_new_tokens = 32 } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/hybrid_generate" -Method Post -ContentType "application/json" -Body $hyb | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $out "hybrid_generate.json") -Encoding utf8

Write-Host "Wrote evidence files to $out"
