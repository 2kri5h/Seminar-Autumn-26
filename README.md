# When Is Another AI Step Worth It? Accuracy vs. GPU energy of agentic code repair

Companion repository for the seminar report by Krish Tandel (B.S., Centre for Multidisciplinary Education, IIT Bombay; roll no. 25B3004).

The report asks when one more repair attempt by an AI coding agent is worth its GPU energy. It measures three policies (one attempt, repair with full history, repair with short history) on 24 held-out HumanEval problems with EvalPlus scoring, using Qwen2.5-Coder 7B (Q4_K_M) on an RTX 5070 Laptop GPU, three seeds, 216 runs.

## What is in this repository

| Path | Status | Contents |
|---|---|---|
| `report/` | included | LaTeX source (`main.tex`, `refs.bib`), figures, and the compiled `main.pdf` |
| `data/reported_results.json` | included | Every number printed in the report |
| `analysis/recompute_reported_numbers.py` | included | Recomputes all derived numbers in the report from the reported ones and checks they agree (J per correct, ratios, request counts, break-even test, emissions, SWE-bench 2x2 interval) |
| `analysis/make_figures.py` | included | Regenerates the results figure |
| `analysis/bootstrap_ratios.py` | included | Task-level bootstrap (Section 5.3), runs on `data/runs.csv` once that file is added |
| `tests/test_bootstrap.py` | included | Unit tests for the bootstrap code on synthetic data |
| `experiment/` | **to add** | Original runner, power logger, scoring wrapper, cache probe, and environment versions |
| `data/runs.csv`, `requests.csv`, `power/`, `cache_probe.csv` | **to add** | Raw per-run, per-request and power-sample data. Schemas in `data/README.md` |

## Reproduce what is currently reproducible

```bash
pip install -r requirements.txt
make check test figures report     # consistency checks, tests, figure, PDF
```

The report compiles with XeLaTeX and uses the free Linux Libertine O / Linux Biolinum O fonts (Ubuntu: `apt install fonts-linuxlibertine`; also on Overleaf). On Overleaf, set the compiler to XeLaTeX.

```bash
```

## Scope and limits

- The checks here verify that the report's numbers are internally consistent. They do **not** re-run the experiment, and they cannot confirm the raw measurements until the raw logs are added.
- The report states these limits itself: one model, one GPU, 24 problems, three seeds, gross board energy only.
- The SWE-bench comparison (Section 6) uses counts derived from two public submissions in the SWE-bench experiments repository; see the report for what could and could not be verified.
- Emissions use the CEA all-India weighted average for FY 2025-26 (0.675 kg CO2/kWh), a national annual figure, not the local grid at run time.

## Follow-up experiment (Case study 2) and frontier-scale models (Section 7)

- `experiment2/harness.py` — branched repair on local Ollama models (full history, short history, fresh retry from the same failed first program).
- `experiment2/openrouter_harness.py` — the same harness against any OpenAI-compatible endpoint (`--api-url`, `--key-env`, `--extra`). Used for Kimi K3.1 Pro (`moonshotai/kimi-k3.1-pro`, reasoning effort low, provider sorted by throughput) and Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`) on OpenRouter, 140 seed-11 problems each.
- `experiment2/gemini_harness.py` — Gemini API backend. The Gemini (11 problems) and Mistral Large 4 (16 problems) runs were stopped by quotas and endpoint speed; their logs are kept in `data/followup/` but are not used in the report.
- `experiment2/energy.py`, `score_hidden.py`, `analyze.py`, `crossmodel.py`, `make_followup_figure.py` — energy integration, hidden EvalPlus scoring, policy analysis, cross-model comparison, figure.
- `data/followup/` — run logs (`runs_*.jsonl`), merged GPU power log, per-request energy, hidden-test scores and the results files used in the report.

Reproduce: `python experiment2/analyze.py --runs data/followup/runs_7b.jsonl data/followup/runs_1.5b.jsonl --hidden data/followup/hidden_scores.json --energy data/followup/energy.json --out results.json`

Four-model comparison (Table 4): `python experiment2/crossmodel.py --runs data/followup/runs_1.5b.jsonl data/followup/runs_7b.jsonl data/followup/runs_kimi.jsonl data/followup/runs_nemotron.jsonl --hidden data/followup/hidden_scores_all.json --energy data/followup/energy.json --out crossmodel_all.json`
