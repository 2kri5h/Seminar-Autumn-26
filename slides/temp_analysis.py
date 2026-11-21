#!/usr/bin/env python3
"""Compare full-history repair at T=0.2 (original runs) and a higher temperature (temp_experiment.py), seed 11.
Usage: python slides/temp_analysis.py <runs_orig.jsonl> <runs_temp.jsonl> [hidden_scores.json]"""
import json, sys
orig = {u["task_id"]: u for u in map(json.loads, open(sys.argv[1])) if u["seed"] == 11 and not u["first"]["visible"]["pass"]}
temp = {u["task_id"]: u for u in map(json.loads, open(sys.argv[2]))}
H = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
def stats(units, tagkey):
    same = att = vis = hid = toks = dur = 0
    for tid, u in units.items():
        prev = u["first"]["code"].strip(); traj = u["branches"]["full"]
        for a in traj:
            att += 1; same += a["code"].strip() == prev; prev = a["code"].strip()
            toks += a["request"].get("eval_count") or 0; dur += a["request"]["t_end"] - a["request"]["t_start"]
        if traj and traj[-1]["visible"]["pass"]:
            vis += 1
            if H is not None:
                k = f"{tagkey(u)}|{tid}|{u['seed']}|full|{traj[-1]['attempt']}"
                h = H.get(k); hid += bool(h and h["base"] == "pass" and h["plus"] == "pass")
    return dict(units=len(units), attempts=att, unchanged=same, unchanged_pct=100 * same / max(att, 1),
                visible_rescues=vis, hidden_rescues=hid if H is not None else None,
                tokens_per_attempt=toks / max(att, 1), seconds_per_attempt=dur / max(att, 1))
common = sorted(set(orig) & set(temp))
o = stats({t: orig[t] for t in common}, lambda u: u["tag"])
t = stats({k: temp[k] for k in common}, lambda u: u["tag"])
print(json.dumps({"T0.2": o, "Thigh": t}, indent=1))
