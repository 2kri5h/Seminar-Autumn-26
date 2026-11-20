# Data

| File | Status | What it is |
|---|---|---|
| `reported_results.json` | included | Every number printed in the report (tables, figure values, SWE-bench counts). Used by `analysis/recompute_reported_numbers.py` to check internal consistency. |
| `runs.csv` | **to add (original data)** | One row per run, 216 rows. Columns below. Enables the numeric bootstrap intervals. |
| `requests.csv` | **to add (original data)** | One row per inference request, 259 rows. |
| `power/` | **to add (original data)** | Raw power samples, one file per request (`<problem>_<seed>_<policy>_<attempt>.csv`, columns `t_s,power_W`). Needed to regenerate Fig. 2 and to recompute request energies with Eq. (3). |
| `cache_probe.csv` | **to add (original data)** | Prefix-reuse probe behind the table in Section 2.1 (5 requests per condition). |

## `runs.csv` schema
`problem_id` (HumanEval number), `seed` (11/29/47), `policy` (`one_attempt`, `full_history`, `short_history`), `attempts` (1-4), `visible_pass` (0/1), `correct` (0/1, passes visible and hidden suites), `energy_J` (sum of request energies), `n_requests`.

## `requests.csv` schema
`problem_id`, `seed`, `policy`, `attempt`, `prompt_tokens`, `output_tokens`, `energy_J`, `power_window_s`, `model_load_s`.
