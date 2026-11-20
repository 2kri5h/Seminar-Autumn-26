#!/usr/bin/env python3
"""Derive every stopping policy from the branched trajectories and compare them.

Inputs: runs_<tag>.jsonl (harness.py), hidden_scores.json (score_hidden.py), energy.json (energy.py)
Usage:  python analyze.py --runs runs_7b.jsonl runs_1.5b.jsonl --hidden hidden_scores.json \
                          --energy energy.json --out results.json [--resamples 10000]

Policies (all share the same first attempt of a unit):
  one              submit the first program
  <b>_cap<c>       b in {full, short, retry}: continue branch b until a visible pass or c attempts;
                   submit the passing program, else the last one (as in the pilot)
  <b>_prog<K>      progress-based stop on branch b, cap 4: stop on a visible pass, or after K
                   attempts in a row without raising the number of visible checks passed;
                   on a non-accepting stop submit the EARLIEST program with the best count
A submission is correct iff it passes every visible check AND the hidden EvalPlus base+plus tests.
"""
import argparse, json, random, statistics
from collections import defaultdict

BRANCHES = ("full", "short", "retry")


def policies():
    ps = ["one"]
    for b in BRANCHES:
        ps += [f"{b}_cap2", f"{b}_cap4", f"{b}_prog1", f"{b}_prog2"]
    return ps


def outcome(u, policy, H, E):
    tag, tid, s = u["tag"], u["task_id"], u["seed"]
    key = lambda b, k: f"{tag}|{tid}|{s}|{b}|{k}"

    def correct(a, b):
        h = H[key(b, a["attempt"])]
        return a["visible"]["pass"] and h["base"] == "pass" and h["plus"] == "pass"

    a1 = u["first"]
    e = E[key("first", 1)]["J"]
    if policy == "one" or a1["visible"]["pass"]:
        return {"correct": correct(a1, "first"), "J": e, "requests": 1, "entered": False, "submitted": 1}
    b, rule = policy.split("_")
    traj = u["branches"][b]
    seq = [("first", a1)] + [(b, a) for a in traj]
    if rule.startswith("cap"):
        cap = int(rule[3:])
        used = seq[:cap]
        sub = next(((bb, a) for bb, a in used if a["visible"]["pass"]), used[-1])
    else:
        K = int(rule[4:])
        best, best_i, c, used = a1["visible"]["v"], 0, 0, seq[:1]
        sub = None
        for i in range(1, len(seq)):
            used = seq[: i + 1]
            a = seq[i][1]
            if a["visible"]["pass"]:
                sub = seq[i]; break
            if a["visible"]["v"] > best:
                best, best_i, c = a["visible"]["v"], i, 0
            else:
                c += 1
            if c >= K:
                break
        if sub is None:
            sub = seq[best_i]
    J = sum(E[key(bb, a["attempt"])]["J"] for bb, a in used)
    return {"correct": correct(sub[1], sub[0]), "J": J, "requests": len(used), "entered": True,
            "submitted": sub[1]["attempt"]}


def summarize(units, H, E, resamples, rng):
    P = policies()
    out = {p: [outcome(u, p, H, E) for u in units] for p in P}
    n = len(units)
    res = {"units": n, "problems": len({u["task_id"] for u in units}),
           "first_fail": sum(not u["first"]["visible"]["pass"] for u in units), "policies": {}}
    for p in P:
        o = out[p]
        c = sum(x["correct"] for x in o); J = sum(x["J"] for x in o)
        res["policies"][p] = {"correct": c, "success": c / n, "J_per_run": J / n,
                              "J_per_correct": J / c if c else None,
                              "requests": sum(x["requests"] for x in o)}
    # rescues: units whose first attempt failed but the policy's submission is correct
    for p in P:
        res["policies"][p]["rescues"] = sum(x["correct"] and x["entered"] and x["submitted"] > 1 for x in out[p])
    # rescues given up by progress stops relative to the same branch at cap 4
    for b in BRANCHES:
        for K in (1, 2):
            lost = sum(cap["correct"] and not pr["correct"] for cap, pr in zip(out[f"{b}_cap4"], out[f"{b}_prog{K}"]))
            res["policies"][f"{b}_prog{K}"]["lost_vs_cap4"] = lost
    # identical resubmissions in repair branches (does feedback change the program?)
    same = defaultdict(lambda: [0, 0])
    for u in units:
        for b, traj in u["branches"].items():
            prev = u["first"]["code"].strip()
            for a in traj:
                same[b][1] += 1; same[b][0] += a["code"].strip() == prev; prev = a["code"].strip()
    res["identical_resubmission"] = {b: {"identical": v[0], "attempts": v[1]} for b, v in same.items()}
    # task-cluster bootstrap
    by_prob = defaultdict(list)
    for i, u in enumerate(units): by_prob[u["task_id"]].append(i)
    probs = list(by_prob)
    comps = [(p, "one") for p in P if p != "one"] + [("short_cap4", "full_cap4"), ("full_cap2", "full_cap4"),
                                                      ("short_cap2", "short_cap4"), ("full_prog1", "full_cap4"),
                                                      ("short_prog1", "short_cap4"), ("retry_cap4", "short_cap4")]
    draws = {c: {"Jratio": [], "dsucc": []} for c in comps}
    for _ in range(resamples):
        idx = [i for pr in (rng.choice(probs) for _ in probs) for i in by_prob[pr]]
        agg = {}
        for p in P:
            o = out[p]
            cc = sum(o[i]["correct"] for i in idx); JJ = sum(o[i]["J"] for i in idx)
            agg[p] = (cc / len(idx), JJ / cc if cc else None)
        for a, b in comps:
            draws[(a, b)]["dsucc"].append(agg[a][0] - agg[b][0])
            if agg[a][1] and agg[b][1]: draws[(a, b)]["Jratio"].append(agg[a][1] / agg[b][1])
    def ci(xs):
        xs = sorted(xs); k = len(xs)
        return [xs[int(0.025 * k)], xs[max(int(0.975 * k) - 1, 0)]] if k else None
    res["comparisons"] = {}
    for a, b in comps:
        pa, pb = res["policies"][a], res["policies"][b]
        res["comparisons"][f"{a} vs {b}"] = {
            "Jratio": pa["J_per_correct"] / pb["J_per_correct"] if pa["J_per_correct"] and pb["J_per_correct"] else None,
            "Jratio_ci95": ci(draws[(a, b)]["Jratio"]),
            "dsucc_pp": 100 * (pa["success"] - pb["success"]),
            "dsucc_ci95_pp": [100 * x for x in ci(draws[(a, b)]["dsucc"])]}
    # request-level descriptive stats
    reqs = [(u["first"], "first")] + []
    J1 = [E[f"{u['tag']}|{u['task_id']}|{u['seed']}|first|1"]["J"] for u in units]
    res["first_request_J_median"] = statistics.median(J1)
    res["first_request_s_median"] = statistics.median(u["first"]["request"]["t_end"] - u["first"]["request"]["t_start"] for u in units)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--hidden", required=True)
    ap.add_argument("--energy", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H = json.load(open(a.hidden)); E = json.load(open(a.energy))
    units = defaultdict(list)
    for p in a.runs:
        for line in open(p, encoding="utf-8"):
            u = json.loads(line); units[u["tag"]].append(u)
    rng = random.Random(a.seed)
    results = {tag: summarize(us, H, E, a.resamples, rng) for tag, us in units.items()}
    json.dump(results, open(a.out, "w"), indent=1)
    for tag, r in results.items():
        print(f"\n== {tag}: {r['units']} units on {r['problems']} problems; first attempt failed in {r['first_fail']}")
        print(f"{'policy':14s} {'correct':>8s} {'success':>8s} {'J/run':>8s} {'J/correct':>10s} {'reqs':>6s} {'rescues':>8s}")
        for p, v in r["policies"].items():
            jc = f"{v['J_per_correct']:.1f}" if v["J_per_correct"] else "-"
            print(f"{p:14s} {v['correct']:8d} {100*v['success']:7.1f}% {v['J_per_run']:8.1f} {jc:>10s} {v['requests']:6d} {v['rescues']:8d}")
        for k, v in r["comparisons"].items():
            if v["Jratio"]:
                print(f"  {k:28s} J ratio {v['Jratio']:.3f} {v['Jratio_ci95']}  dsucc {v['dsucc_pp']:+.1f}pp {[round(x,1) for x in v['dsucc_ci95_pp']]}")


if __name__ == "__main__":
    main()
