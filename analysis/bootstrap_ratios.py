#!/usr/bin/env python3
"""Task-level (cluster) bootstrap of energy-per-correct-solution ratios between policies.

Input: data/runs.csv with one row per run (schema in data/README.md). Resamples the
problems with replacement, keeping every seed and every policy of a chosen problem
together, then recomputes J = sum(E) / sum(correct) per policy and the ratios.

NOTE: this is a reference implementation of the procedure described in Section 5.3 of the
report. data/runs.csv must come from the original experiment; it is not included until the
raw per-run logs are added. The unit tests in tests/ use tiny SYNTHETIC data to check the code only.

Usage:  python analysis/bootstrap_ratios.py data/runs.csv [--resamples 10000] [--seed 0]
"""
import argparse, csv, random, collections

POLICIES = ("one_attempt", "full_history", "short_history")
PAIRS = (("full_history", "one_attempt"), ("short_history", "one_attempt"), ("short_history", "full_history"))

def load(path):
    by_problem = collections.defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_problem[r["problem_id"]].append((r["policy"], int(r["correct"]), float(r["energy_J"])))
    return by_problem

def j_per_policy(problems, by_problem):
    e = dict.fromkeys(POLICIES, 0.0); c = dict.fromkeys(POLICIES, 0)
    for pid in problems:
        for pol, y, en in by_problem[pid]:
            e[pol] += en; c[pol] += y
    return {p: (e[p] / c[p] if c[p] > 0 else None) for p in POLICIES}

def bootstrap(by_problem, resamples=10000, seed=0):
    rng = random.Random(seed); ids = list(by_problem)
    point = j_per_policy(ids, by_problem)
    draws = {pair: [] for pair in PAIRS}
    for _ in range(resamples):
        J = j_per_policy([rng.choice(ids) for _ in ids], by_problem)
        for a, b in PAIRS:
            if J[a] is not None and J[b] is not None:   # skip replicates with zero successes
                draws[(a, b)].append(J[a] / J[b])
    out = {}
    for a, b in PAIRS:
        d = sorted(draws[(a, b)]); k = len(d)
        pt = point[a] / point[b] if point[a] is not None and point[b] is not None else None
        if k == 0:
            out[(a, b)] = (pt, None, None, 0)
        else:
            out[(a, b)] = (pt, d[int(0.025 * k)], d[max(int(0.975 * k) - 1, 0)], k)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs_csv")
    ap.add_argument("--resamples", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for (x, y), (pt, lo, hi, k) in bootstrap(load(a.runs_csv), a.resamples, a.seed).items():
        if k == 0: print(f"{x} / {y}: no valid resamples (a policy had zero successes)")
        else: print(f"{x} / {y}: point {pt:.3f}, 95% percentile CI [{lo:.3f}, {hi:.3f}]  ({k} valid resamples)")
