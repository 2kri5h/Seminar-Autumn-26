#!/usr/bin/env python3
"""Cross-model comparison on matched units (same problems, seed 11) plus descriptive counts.

Usage: python crossmodel.py --runs runs_1.5b.jsonl runs_7b.jsonl [runs_gemini.jsonl] \
           --hidden hidden_scores.json --energy energy.json --out crossmodel.json
Models without energy entries (hosted APIs) get accuracy, token and latency figures only.
"""
import argparse, json, random, statistics
from collections import defaultdict

from analyze import outcome


def load(path, seed):
    return [u for u in map(json.loads, open(path, encoding="utf-8")) if u["seed"] == seed]


def ci(xs):
    xs = sorted(x for x in xs if x is not None); k = len(xs)
    return [xs[int(0.025 * k)], xs[max(int(0.975 * k) - 1, 0)]] if k else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--hidden", required=True)
    ap.add_argument("--energy", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()
    H = json.load(open(a.hidden)); E = json.load(open(a.energy))
    rng = random.Random(0)
    models = {}
    for p in a.runs:
        us = load(p, a.seed)
        if us: models[us[0]["tag"]] = {u["task_id"]: u for u in us}
    res = {"seed": a.seed, "models": {}, "paired": {}}

    def corr(u, b, att, cand):
        h = H[f"{u['tag']}|{u['task_id']}|{u['seed']}|{b}|{att}"]
        return cand["visible"]["pass"] and h["base"] == "pass" and h["plus"] == "pass"

    for tag, us in models.items():
        units = list(us.values())
        f = [u["first"] for u in units]
        req = [x["request"] for x in f]
        d = {"units": len(units),
             "first_visible_pass": sum(x["visible"]["pass"] for x in f),
             "first_correct": sum(corr(u, "first", 1, u["first"]) for u in units),
             "visible_pass_hidden_fail": sum(u["first"]["visible"]["pass"] and not corr(u, "first", 1, u["first"])
                                            for u in units),
             "first_latency_s_median": statistics.median(r["t_end"] - r["t_start"] for r in req),
             "first_output_tokens_median": statistics.median(r.get("eval_count") or 0 for r in req),
             "first_prompt_tokens_median": statistics.median(r.get("prompt_eval_count") or 0 for r in req)}
        th = [r.get("thoughts_count") for r in req if r.get("thoughts_count") is not None]
        if th:
            d["first_thinking_tokens_median"] = statistics.median(th)
            d["first_thinking_tokens_mean"] = statistics.mean(th)
            d["first_total_output_tokens_mean"] = statistics.mean((r.get("eval_count") or 0) + (r.get("thoughts_count") or 0)
                                                                  for r in req)
        key1 = lambda u: f"{u['tag']}|{u['task_id']}|{u['seed']}|first|1"
        if all(key1(u) in E for u in units):
            J = [E[key1(u)]["J"] for u in units]
            tok = [u["first"]["request"].get("eval_count") or 0 for u in units]
            d["first_J_median"] = statistics.median(J)
            d["J_per_output_token"] = sum(J) / sum(tok)
        res["models"][tag] = d

    # paired comparison of local models and policies on the common problems
    local = [t for t in models if all(f"{u['tag']}|{u['task_id']}|{u['seed']}|first|1" in E for u in models[t].values())]
    common = sorted(set.intersection(*(set(models[t]) for t in local)))
    arms = [(t, p) for t in local for p in ("one", "retry_cap4", "short_cap4")]
    O = {arm: [outcome(models[arm[0]][tid], arm[1], H, E) for tid in common] for arm in arms}
    for arm in arms:
        o = O[arm]; c = sum(x["correct"] for x in o)
        res["paired"].setdefault("arms", {})[f"{arm[0]}:{arm[1]}"] = {
            "correct": c, "success": c / len(o), "J_per_correct": sum(x["J"] for x in o) / c}
    pairs = []
    if "1.5b" in local and "7b" in local:
        pairs = [(("1.5b", "retry_cap4"), ("7b", "one")), (("1.5b", "one"), ("7b", "one")),
                 (("7b", "retry_cap4"), ("7b", "one")), (("1.5b", "retry_cap4"), ("1.5b", "one"))]
    n = len(common)
    for A, B in pairs:
        ds, jr = [], []
        for _ in range(a.resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            ca = sum(O[A][i]["correct"] for i in idx); cb = sum(O[B][i]["correct"] for i in idx)
            ja = sum(O[A][i]["J"] for i in idx); jb = sum(O[B][i]["J"] for i in idx)
            ds.append(100 * (ca - cb) / n)
            jr.append((ja / ca) / (jb / cb) if ca and cb else None)
        sa, sb = res["paired"]["arms"][f"{A[0]}:{A[1]}"], res["paired"]["arms"][f"{B[0]}:{B[1]}"]
        res["paired"][f"{A[0]}:{A[1]} vs {B[0]}:{B[1]}"] = {
            "dsucc_pp": 100 * (sa["success"] - sb["success"]), "dsucc_ci95_pp": ci(ds),
            "Jratio": sa["J_per_correct"] / sb["J_per_correct"], "Jratio_ci95": ci(jr)}
    res["paired"]["problems"] = n

    # hosted models: compare on the problems they have reached so far
    for t in models:
        if t in local: continue
        tids = sorted(models[t])
        row = {"problems": len(tids)}
        for m in [t] + local:
            row[m] = sum(corr(models[m][x], "first", 1, models[m][x]["first"]) for x in tids if x in models[m])
        res["models"][t]["same_problems_first_correct"] = row
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
