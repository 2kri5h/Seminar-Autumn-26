#!/usr/bin/env python3
"""Attach GPU board energy (J) to every model request in runs_*.jsonl.

Power log: written by nvidia-smi during the run (see run_experiment.ps1):
  nvidia-smi --query-gpu=timestamp,power.draw.instant --format=csv,noheader,nounits -lms 100 -f power_log.csv
Timestamps there are LOCAL time; requests in runs_*.jsonl use Unix epoch seconds. Pass
--utc-offset-hours for the machine that logged power (India = 5.5) when running elsewhere.

Energy of a request = trapezoid integral of board power over [t_start, t_end], with power
linearly interpolated at both window edges (Eq. 3 of the report).

Usage: python energy.py power_log.csv runs_7b.jsonl [...] --utc-offset-hours 5.5 --out energy.json
Writes {"<tag>|<task_id>|<seed>|<branch>|<attempt>": {"J": ..., "window_s": ..., "n_samples": ...}}.
"""
import argparse, bisect, datetime as dt, json


def load_power(path, offset_h):
    ts, ps = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            t = dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            p = float(parts[1])
        except ValueError:
            continue  # header, '[N/A]' or a torn line
        if offset_h is None:
            e = t.timestamp()  # interpret as this machine's local time
        else:
            e = (t - dt.timedelta(hours=offset_h)).replace(tzinfo=dt.timezone.utc).timestamp()
        ts.append(e); ps.append(p)
    order = sorted(range(len(ts)), key=ts.__getitem__)
    return [ts[i] for i in order], [ps[i] for i in order]


def interp(ts, ps, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0: return ps[0]
    if i >= len(ts): return ps[-1]
    t0, t1 = ts[i - 1], ts[i]
    return ps[i - 1] + (ps[i] - ps[i - 1]) * (t - t0) / (t1 - t0) if t1 > t0 else ps[i]


def integrate(ts, ps, a, b):
    i, j = bisect.bisect_right(ts, a), bisect.bisect_left(ts, b)
    xs = [a] + ts[i:j] + [b]
    ys = [interp(ts, ps, a)] + ps[i:j] + [interp(ts, ps, b)]
    E = sum((ys[k] + ys[k + 1]) / 2 * (xs[k + 1] - xs[k]) for k in range(len(xs) - 1))
    return E, j - i


def requests(path):
    for line in open(path, encoding="utf-8"):
        u = json.loads(line)
        key = f"{u['tag']}|{u['task_id']}|{u['seed']}"
        yield f"{key}|first|1", u["first"]["request"]
        for b, traj in u["branches"].items():
            for a in traj:
                yield f"{key}|{b}|{a['attempt']}", a["request"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("power"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--utc-offset-hours", type=float, default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts, ps = load_power(a.power, a.utc_offset_hours)
    gaps = [ts[k + 1] - ts[k] for k in range(len(ts) - 1)]
    print(f"{len(ts)} power samples, median interval {sorted(gaps)[len(gaps)//2]*1000:.0f} ms, "
          f"max gap {max(gaps):.2f} s")
    out, outside, sparse = {}, 0, 0
    for k, r in (x for p in a.runs for x in requests(p)):
        s, e = r["t_start"], r["t_end"]
        if s < ts[0] or e > ts[-1]: outside += 1
        E, n = integrate(ts, ps, s, e)
        if n < 2: sparse += 1
        out[k] = {"J": E, "window_s": e - s, "n_samples": n}
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} requests; {outside} fall outside the power log; {sparse} have <2 samples inside")


if __name__ == "__main__":
    main()
