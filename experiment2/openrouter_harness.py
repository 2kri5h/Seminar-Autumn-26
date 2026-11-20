"""Runs the same branched-repair experiment through OpenRouter (OpenAI-compatible API).

Same problems, prompts, visible checks, branches and cap as harness.py; only the model call changes.
Hosted model: records tokens (including reasoning tokens), latency and outcomes, not energy.
The key is read from OPENROUTER_API_KEY. On a daily-limit error the run stops cleanly; re-run to resume.

    $env:OPENROUTER_API_KEY = "<your key>"
    python openrouter_harness.py --model nvidia/<model>:free --tag nemotron --out runs_nemotron.jsonl --seeds 11 --rpm 15
"""
import argparse, json, os, random, sys, time, urllib.error, urllib.request

import harness as H

API = "https://openrouter.ai/api/v1/chat/completions"
MAX_OUT = 16384          # room for long reasoning (an 8,192 cap truncated Gemini's reasoning)
_min_gap = 0.0
_last_call = 0.0


class DailyQuota(Exception):
    pass


def or_chat(model, messages, seed):
    global _last_call
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2, "seed": int(seed) % 2**31,
                       "max_tokens": MAX_OUT, "usage": {"include": True}}).encode()
    key = os.environ["OPENROUTER_API_KEY"]
    for tries in range(8):
        wait = _min_gap - (time.time() - _last_call)
        if wait > 0: time.sleep(wait)
        req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + key,
                                                             "X-Title": "Seminar-Autumn-26 experiment"})
        t0 = time.time(); _last_call = t0
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            t1 = time.time()
            if "error" in resp and not resp.get("choices"):
                raise RuntimeError(json.dumps(resp["error"])[:400])
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 429:
                if "per-day" in msg or "per day" in msg.lower() or "free-models-per-day" in msg:
                    raise DailyQuota(msg[:300])
                time.sleep(min(120, 15 * 2 ** tries)); continue
            if e.code in (500, 502, 503, 504, 408):
                time.sleep(min(120, 10 * 2 ** tries)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg[:500]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(120, 10 * 2 ** tries)); continue
    else:
        raise RuntimeError("gave up after repeated rate-limit or server errors")
    ch = resp["choices"][0]
    u = resp.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    meta = {"prompt_eval_count": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "thoughts_count": det.get("reasoning_tokens"),
            "eval_count": (u.get("completion_tokens") or 0) - (det.get("reasoning_tokens") or 0),
            "finish_reason": ch.get("finish_reason"), "model_version": resp.get("model"),
            "provider": resp.get("provider")}
    return ch["message"].get("content") or "", {"t_start": t0, "t_end": t1, **meta}


def main():
    global _min_gap
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", default="11"); ap.add_argument("--rpm", type=float, default=15)
    a = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY in this shell first.")
    _min_gap = 60.0 / a.rpm
    H.chat = or_chat
    probs = [p for p in H.load_problems() if int(p["task_id"].split("/")[1]) not in H.PILOT_HELDOUT]
    units = [(p, s) for p in probs for s in (int(x) for x in a.seeds.split(","))]
    random.Random(2026).shuffle(units)
    if a.limit: units = units[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: r = json.loads(l); done.add((r["task_id"], r["seed"]))
            except ValueError: pass
    todo = [(p, s) for p, s in units if (p["task_id"], s) not in done]
    print(f"[{a.tag}] {len(units)} units, {len(done)} done, {len(todo)} to run", flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, (p, s) in enumerate(todo, 1):
            try:
                u = H.run_unit(a.model, p, s)
            except DailyQuota as e:
                print(f"[{a.tag}] daily limit reached after {i-1} units; re-run tomorrow to resume. ({e})", flush=True)
                return
            u["model"] = a.model; u["tag"] = a.tag
            f.write(json.dumps(u) + "\n"); f.flush()
            nb = sum(len(v) for v in u["branches"].values())
            print(f"[{a.tag}] {i}/{len(todo)} {p['task_id']} s{s} first={'pass' if u['first']['visible']['pass'] else 'FAIL'}"
                  f" extra_requests={nb}", flush=True)
    print(f"[{a.tag}] finished", flush=True)


if __name__ == "__main__":
    main()
