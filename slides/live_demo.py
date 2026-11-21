#!/usr/bin/env python3
"""Live demo for the talk: one agent loop on the laptop GPU, with a live energy meter.

Run from the experiment folder (needs harness.py, Ollama and nvidia-smi):
    python live_demo.py                       # collision problem, Qwen 1.5B
    python live_demo.py --task HumanEval/41 --model qwen2.5-coder:7b --mode retry
Each attempt prints the model's code, the visible-test result and the GPU energy it used.
Press Enter between steps so you control the pace while presenting.
"""
import argparse, subprocess, sys, threading, time

import harness as H


class PowerMeter:
    """Samples GPU board power with nvidia-smi every 100 ms and integrates it (trapezoid rule)."""
    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _run(self):
        p = self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits",
                                          "-lms", "100"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, text=True)
        for line in p.stdout:
            if self._stop.is_set(): break
            try: self.samples.append((time.time(), float(line.strip())))
            except ValueError: pass
        p.terminate()

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); time.sleep(0.5); return self

    def __exit__(self, *a):
        time.sleep(0.2); self._stop.set()
        try:
            self.proc.kill(); self.proc.wait(timeout=5)
        except Exception as e:
            print(f"(power meter: could not stop nvidia-smi: {e})")

    def joules(self, t0, t1):
        s = [x for x in self.samples if t0 - 0.1 <= x[0] <= t1 + 0.1]
        return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(s, s[1:]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="HumanEval/41"); ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--mode", choices=["repair", "retry"], default="repair"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-pause", action="store_true")
    a = ap.parse_args()
    pause = (lambda m="": None) if a.no_pause else (lambda m="": input(m + "  [Enter] "))

    prob = next(p for p in H.load_problems() if p["task_id"] == a.task)
    test_module, checks = H.split_checks(prob["test"])
    spec = {"entry_point": prob["entry_point"], "test_module": test_module, "checks": checks}
    header = H.header_of(prob)
    first = [{"role": "system", "content": H.SYSTEM}, {"role": "user", "content": H.FIRST.format(prompt=prob["prompt"])}]
    H.chat(a.model, [{"role": "user", "content": "Say OK."}], 0)  # load the model before the audience watches

    total = 0.0
    with PowerMeter() as pm:
        msgs, last = list(first), None
        for k in range(1, H.CAP + 1):
            pause(f"\nAttempt {k}: ask the model")
            if k > 1 and a.mode == "repair":
                check, detail = H.describe(last["visible"])
                msgs = msgs + [{"role": "assistant", "content": last["raw"]},
                               {"role": "user", "content": H.FEEDBACK.format(check=check, detail=detail)}]
            use = msgs if a.mode == "repair" else first
            seed = a.seed + (k - 1 if a.mode == "repair" else 1000 * (k - 1))
            last = H.attempt(a.model, use, seed, prob, spec, header, a.mode, k)
            r = last["request"]; j = pm.joules(r["t_start"], r["t_end"]); total += j
            print("-" * 60); print(last["code"].strip()); print("-" * 60)
            vis = last["visible"]
            print(f"visible checks passed: {vis['v']}/{vis['n']}"
                  f"   -> {'PASS' if vis['pass'] else 'FAIL'}")
            print(f"this attempt: {j:6.1f} J  ({r['t_end'] - r['t_start']:.1f} s, {r.get('eval_count')} tokens)"
                  f"   running total: {total:6.1f} J")
            if vis["pass"]: break
    print(f"\nTotal GPU energy for this answer: {total:.1f} J  (= {total / 3600:.3f} Wh)")


if __name__ == "__main__":
    main()
