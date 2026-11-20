# Runs the whole follow-up experiment on the laptop: power logger + both models.
# Run from this folder in PowerShell (an Administrator window if nvidia-smi reports a permission error):
#     powershell -ExecutionPolicy Bypass -File .\run_experiment.ps1
# Keep the laptop plugged in, in Performance mode, with the dGPU enabled (Hybrid or dGPU mode).
# The harness is resumable: re-running this script continues where it stopped.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# 0. Pre-flight: the NVIDIA GPU must be visible, and Ollama must run on it.
nvidia-smi --query-gpu=name,power.draw.instant --format=csv
if ($LASTEXITCODE -ne 0) { "nvidia-smi failed" | Tee-Object -Append -FilePath progress.log; throw "nvidia-smi failed: enable the NVIDIA GPU (Legion Space: Hybrid/dGPU mode) and retry." }

# 1. Start the power logger (every 100 ms, appends across restarts).
$logger = Start-Process -FilePath "nvidia-smi" -ArgumentList @(
  "--query-gpu=timestamp,power.draw.instant", "--format=csv,noheader,nounits", "-lms", "100", "-f", "power_log_$(Get-Date -Format yyyyMMdd_HHmmss).csv"
) -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 20   # idle baseline before any request

try {
  python harness.py --model qwen2.5-coder:7b   --tag 7b   --out runs_7b.jsonl   2>&1 | Tee-Object -Append -FilePath progress.log
  python harness.py --model qwen2.5-coder:1.5b --tag 1.5b --out runs_1.5b.jsonl --seeds 11 2>&1 | Tee-Object -Append -FilePath progress.log
  Start-Sleep -Seconds 20 # idle baseline after
}
finally {
  Stop-Process -Id $logger.Id -ErrorAction SilentlyContinue
}
"done" | Tee-Object -Append -FilePath progress.log
