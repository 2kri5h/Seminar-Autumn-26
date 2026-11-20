"""Runs the same branched-repair experiment against the Gemini API.

Same problems, prompts, visible checks, branches (full / short / retry) and cap as harness.py;
only the model call changes. Energy cannot be measured for a hosted model, so this records
tokens, latency and outcomes; energy is estimated later and labelled as an estimate.

The API key is read from the GEMINI_API_KEY environment variable (never pass it on the command line).
Free-tier limits are per project and change over time: check them in Google AI Studio and set
--rpm accordingly. On a daily-quota error the run stops cleanly; re-run the same command the next
day and it resumes where it stopped.

    $env:GEMINI_API_KEY = "<your key>"      # PowerShell, typed by you
    python gemini_harness.py --model <model-id> --tag gemini --out runs_gemini.jsonl --seeds 11 --rpm 8
"""
import argparse, json, os, random, sys, time, urllib.error, urllib.request

import harness as H

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_OUT = 8192          # thinking models spend output tokens on reasoning; 1024 would truncate them
_min_gap = 0.0
_last_call = 0.0


class DailyQuota(Exception):
    pass


def gemini_chat(model, messages, seed):
    global _last_call
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    body = {"contents": contents,
            "generationConfig": {"temperature": 0.2, "seed": int(seed) % 2**31, "maxOutputTokens": MAX_OUT}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    data = json.dumps(body).encode()
    key = os.environ["GEMINI_API_KEY"]
    for tries in range(8):
        wait = _min_gap - (time.time() - _last_call)
        if wait > 0: time.sleep(wait)
        req = urllib.request.Request(API.format(model=model), data=data,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        t0 = time.time(); _last_call = t0
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read())
            t1 = time.time()
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 429:
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuota(msg[:300])
                time.sleep(min(120, 15 * 2 ** tries)); continue
            if e.code in (500, 503):
                time.sleep(min(120, 10 * 2 ** tries)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg[:500]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(120, 10 * 2 ** tries)); continue
    else:
        raise RuntimeError("gave up after repeated rate-limit or server errors")
    cand = (resp.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    u = resp.get("usageMetadata", {})
    meta = {"prompt_eval_count": u.get("promptTokenCount"), "eval_count": u.get("candidatesTokenCount"),
            "thoughts_count": u.get("thoughtsTokenCount"), "total_tokens": u.get("totalTokenCount"),
            "finish_reason": cand.get("finishReason"), "model_version": resp.get("modelVersion")}
    return text, {"t_start": t0, "t_end": t1, **meta}


def main():
    global _min_gap
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", default="11"); ap.add_argument("--rpm", type=float, default=8,
                    help="requests per minute to stay under (set just below your free-tier limit)")
    a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY in this shell first.")
    _min_gap = 60.0 / a.rpm
    H.chat = gemini_chat                     # same harness, different model call
    probs = [p for p in H.load_problems() if int(p["task_id"].split("/")[1]) not in H.PILOT_HELDOUT]
    seeds = [int(x) for x in a.seeds.split(",")]
    units = [(p, s) for p in probs for s in seeds]
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
                print(f"[{a.tag}] daily quota reached after {i-1} units; re-run tomorrow to resume. ({e})", flush=True)
                return
            u["model"] = a.model; u["tag"] = a.tag
            f.write(json.dumps(u) + "\n"); f.flush()
            nb = sum(len(v) for v in u["branches"].values())
            print(f"[{a.tag}] {i}/{len(todo)} {p['task_id']} s{s} first={'pass' if u['first']['visible']['pass'] else 'FAIL'}"
                  f" extra_requests={nb}", flush=True)
    print(f"[{a.tag}] finished", flush=True)


if __name__ == "__main__":
    main()
