#!/usr/bin/env python3
"""Follow-up: does a higher temperature stop repair from resubmitting the same program?

Takes every seed-11 unit whose first answer failed a visible check (from runs_<tag>.jsonl),
reuses that exact first answer, and reruns the full-history repair branch (cap 4) at a new
temperature. Seeds per attempt match the original branch (seed + k - 1), so only the
temperature changes. Writes one JSON line per unit, in the same format as harness.py.

Usage: python temp_experiment.py --model qwen2.5-coder:1.5b --tag 1.5b --runs runs_1.5b.jsonl --temp 0.8 --out runs_temp_1.5b.jsonl
"""
import argparse, json, os, sys, urllib.request

import harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    def chat(model, messages, seed):
        body = json.dumps({"model": model, "messages": messages, "stream": False, "keep_alive": "60m",
                           "options": {"temperature": a.temp, "seed": seed, "num_ctx": 8192, "num_predict": 1024}}).encode()
        req = urllib.request.Request(H.OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        import time
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        t1 = time.time()
        meta = {k: resp.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                                         "eval_count", "eval_duration")}
        return resp["message"]["content"], {"t_start": t0, "t_end": t1, **meta}
    H.chat = chat

    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    probs = {p["task_id"]: p for p in H.load_problems()}
    units = [u for u in map(json.loads, open(a.runs, encoding="utf-8"))
             if u["seed"] == a.seed and not u["first"]["visible"]["pass"]]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [u for u in units if u["task_id"] not in done]
    print(f"[{a.tag} T={a.temp}] {len(units)} failed first answers, {len(done)} done, {len(todo)} to run", flush=True)
    chat(a.model, [{"role": "user", "content": "Say OK."}], 0)  # warm-up, not recorded

    with open(a.out, "a", encoding="utf-8") as f:
        for i, u in enumerate(todo, 1):
            prob = probs[u["task_id"]]; seed = u["seed"]
            test_module, checks = H.split_checks(prob["test"])
            spec = {"entry_point": prob["entry_point"], "test_module": test_module, "checks": checks}
            header = H.header_of(prob)
            first_msgs = [{"role": "system", "content": H.SYSTEM},
                          {"role": "user", "content": H.FIRST.format(prompt=prob["prompt"])}]
            msgs, last, traj = list(first_msgs), u["first"], []
            for k in range(2, H.CAP + 1):
                check, detail = H.describe(last["visible"])
                msgs = msgs + [{"role": "assistant", "content": last["raw"]},
                               {"role": "user", "content": H.FEEDBACK.format(check=check, detail=detail)}]
                last = H.attempt(a.model, msgs, seed + k - 1, prob, spec, header, "full", k)
                traj.append(last)
                if last["visible"]["pass"]: break
            out = {"task_id": u["task_id"], "seed": seed, "first": u["first"], "branches": {"full": traj},
                   "model": a.model, "tag": f"{a.tag}_T{a.temp}", "temperature": a.temp}
            f.write(json.dumps(out) + "\n"); f.flush()
            print(f"[{a.tag} T={a.temp}] {i}/{len(todo)} {u['task_id']} attempts={len(traj)} "
                  f"pass={'yes' if traj[-1]['visible']['pass'] else 'no'}", flush=True)
    print(f"[{a.tag} T={a.temp}] finished", flush=True)


if __name__ == "__main__":
    main()
