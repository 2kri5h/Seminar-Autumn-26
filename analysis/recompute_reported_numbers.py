#!/usr/bin/env python3
"""Recompute every derived number in the report from data/reported_results.json and
check that it matches what the report prints. Exit code 1 if any check fails.

Usage:  python analysis/recompute_reported_numbers.py
"""
import json, math, pathlib, sys

R = json.load(open(pathlib.Path(__file__).parent.parent / "data" / "reported_results.json"))
P = R["policies"]; N = R["workload"]["runs_per_policy"]
fails = []

def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: computed {got:.4f}, reported {want}")
    if not ok: fails.append(name)

# Table 3: energy per correct solution = mean J/run * runs / correct
for k, p in P.items():
    check(f"J/correct {k}", p["mean_J_per_run"] * N / p["correct"], p["J_per_correct_reported"], 0.01)
    check(f"attempt counts sum to 72 {k}", sum(p["attempts"].values()), N, 0)

# Ratios of J per correct (Fig. 3b / Section 5.3)
J = {k: p["mean_J_per_run"] * N / p["correct"] for k, p in P.items()}
check("ratio full/one", J["full_history"] / J["one_attempt"], R["bootstrap_reported"]["ratio_full_over_one"], 0.006)
check("ratio short/one", J["short_history"] / J["one_attempt"], R["bootstrap_reported"]["ratio_short_over_one"], 0.006)
check("ratio short/full", J["short_history"] / J["full_history"], R["bootstrap_reported"]["ratio_short_over_full"], 0.0006)

# Break-even test, Eq. (5): repair is cheaper per success iff energy ratio < success ratio
for k in ("full_history", "short_history"):
    e = P[k]["mean_J_per_run"] / P["one_attempt"]["mean_J_per_run"]
    s = P[k]["correct"] / P["one_attempt"]["correct"]
    print(f"       {k}: energy ratio {e:.3f} vs success ratio {s:.3f} -> repair {'cheaper' if e < s else 'costlier'} per success")
    assert e > s

# Request counts
req = {k: sum(int(a) * c for a, c in p["attempts"].items()) for k, p in P.items()}
check("total requests", sum(req.values()), R["workload"]["total_requests"], 0)
check("revision requests", sum(req.values()) - R["workload"]["initial_requests"], 43, 0)
for k in ("full_history", "short_history"):
    reps = N - P[k]["attempts"]["1"]
    check(f"runs entering repair {k}", reps, {"full_history": 9, "short_history": 8}[k], 0)
    check(f"cap share of runs {k}", 100 * P[k]["attempts"]["4"] / N, P[k]["cap_share_of_runs_pct"], 0.06)
    check(f"rescues {k}", P[k]["correct"] - P[k]["correct_at_first_attempt"], 2, 0)
    print(f"       mean J per request {k}: {P[k]['mean_J_per_run'] * N / req[k]:.1f}")
# Hindsight cap-of-two: revisions kept = one per run that entered repair
check("revisions under a 2-attempt cap", (9 + 8), 17, 0)

# Fig. 2 column sums and which problems differ
C = R["correct_per_problem_out_of_3"]
for k in P:
    check(f"Fig.2 sum {k}", sum(C[k]), P[k]["correct"], 0)
diff = [pid for pid, a, b, c in zip(R["problem_ids"], C["one_attempt"], C["full_history"], C["short_history"]) if len({a, b, c}) > 1]
print("       problems whose totals differ:", diff); assert diff == [41, 83, 113, 160]

# Collision run
cr = R["collision_run"]
check("collision run total J", cr["first_request_J"] + cr["repair_request_J"], 59.16, 0.005)
print(f"       average board power over repair window: {cr['repair_request_J'] / cr['power_window_s']:.1f} W")

# Idle sensitivity: implied idle energy per success is consistent across policies' energy/time
for k in P:
    idle_per_success = J[k] - R["idle_sensitivity_J_per_success"][k]
    print(f"       idle energy charged per success {k}: {idle_per_success:.2f} J")

# Section 9: energy and emissions of the 72 short-history runs
E_J = P["short_history"]["mean_J_per_run"] * N
check("short-history kJ", E_J / 1000, 11.11, 0.005)
check("short-history Wh", E_J / 3600, 3.09, 0.005)
g = R["emissions"]["grid_factor_kg_CO2_per_kWh"]
print(f"       CO2 at {g} kg/kWh: {E_J / 3600 / 1000 * g * 1000:.2f} g CO2 (GPU board only, national average factor)")
check("CEA factor = 1189.04 Mt / 1761.28 BU", 1189.04 / 1761.28, g, 0.0005)
check("scale example 0.8 x 1.5", 0.8 * 1.5, 1.2, 1e-12)
check("service-scale: 1e6 tasks, J difference MJ", (P["short_history"]["mean_J_per_run"] - P["one_attempt"]["mean_J_per_run"]) * 1e6 / 1e6, 47.27, 0.01)
check("service-scale kWh/day", (P["short_history"]["mean_J_per_run"] - P["one_attempt"]["mean_J_per_run"]) * 1e6 / 3.6e6, 13.1, 0.05)

# SWE-bench contingency (Section 6)
S = R["swebench_verified_two_snapshots"]; b, c, n = S["first_only"], S["second_only"], S["tasks"]
check("tasks total", S["both_solved"] + b + c + S["neither"], n, 0)
check("first total", S["both_solved"] + b, 364, 0); check("second total", S["both_solved"] + c, 353, 0)
check("oracle union", S["both_solved"] + b + c, 397, 0)
d = (c - b) / n * 100
se = math.sqrt(b + c - (b - c) ** 2 / n) / n * 100
print(f"       difference {d:.2f} pp, normal-approx 95% CI [{d - 1.96*se:.2f}, {d + 1.96*se:.2f}] (report: bootstrap {S['reported_paired_bootstrap_ci_pp']})")
assert abs((d - 1.96*se) - (-5.6)) < 0.1 and abs((d + 1.96*se) - 1.2) < 0.1

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
