# Temperature follow-up: full-history repair at T=0.8 on every seed-11 failed first answer.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
nvidia-smi --query-gpu=name,power.draw.instant --format=csv
$logger = Start-Process -FilePath "nvidia-smi" -ArgumentList @(
  "--query-gpu=timestamp,power.draw.instant", "--format=csv,noheader,nounits", "-lms", "100", "-f", "power_log_temp_$(Get-Date -Format yyyyMMdd_HHmmss).csv"
) -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 15
try {
  python temp_experiment.py --model qwen2.5-coder:1.5b --tag 1.5b --runs runs_1.5b.jsonl --temp 0.8 --out runs_temp_1.5b.jsonl 2>&1 | Tee-Object -Append -FilePath progress_temp.log
  python temp_experiment.py --model qwen2.5-coder:7b --tag 7b --runs runs_7b.jsonl --temp 0.8 --out runs_temp_7b.jsonl 2>&1 | Tee-Object -Append -FilePath progress_temp.log
  Start-Sleep -Seconds 15
} finally { Stop-Process -Id $logger.Id -ErrorAction SilentlyContinue }
"done" | Tee-Object -Append -FilePath progress_temp.log
