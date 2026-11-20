#!/usr/bin/env python3
"""Follow-up experiment: branched repair policies on HumanEval with a local Ollama model.

For every (problem, seed) unit the model makes ONE first attempt. If that attempt fails
the visible tests, three continuations branch from that SAME failed program:
  full  - repair, appending each answer and failure message to the conversation
  short - repair, sending only task + latest code + latest failure message
  retry - fresh resample of the original prompt with a new seed
Each continuation runs until a visible pass or 4 attempts in total. Policies with caps
1/2/4 and a progress-based stop are derived afterwards as prefixes of these trajectories.

Every model request records wall-clock start/end times so GPU energy can be integrated
from an independent nvidia-smi power log (see README). Output: one JSON line per unit.

Usage (Windows or Linux, Python 3.10+, no extra packages):
  python harness.py --model qwen2.5-coder:7b --tag 7b --out runs_7b.jsonl
"""
import argparse, ast, gzip, json, os, random, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
PILOT_HELDOUT = {0, 8, 14, 31, 41, 42, 49, 66, 72, 79, 83, 86, 87, 97, 103, 108, 110, 111, 113,
                 139, 142, 151, 152, 160}
SEEDS = (11, 29, 47)
CAP = 4
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SYSTEM = "You are an expert Python programmer."
FIRST = ("Complete the following Python function. Reply with the complete function (including any "
         "imports it needs) in a single ```python code block and nothing else.\n\n```python\n{prompt}```")
FEEDBACK = ("The code failed this test:\n{check}\n{detail}\nFix the function. Reply with the complete "
            "corrected function in a single ```python code block.")
SHORT = ("Complete the following Python function.\n\n```python\n{prompt}```\n\nMy previous attempt:\n"
         "```python\n{code}\n```\nIt failed this test:\n{check}\n{detail}\nWrite a corrected complete "
         "function in a single ```python code block and nothing else.")

RUNNER = r'''
import json, sys
src = open(sys.argv[1], encoding="utf-8").read()
spec = json.load(open(sys.argv[2], encoding="utf-8"))
g = {"__name__": "__candidate__"}
try:
    exec(compile(src, "candidate", "exec"), g)
    g["candidate"] = g[spec["entry_point"]]
    exec(compile(spec["test_module"], "tests", "exec"), g)
except BaseException as e:
    print(json.dumps({"compile_error": (type(e).__name__ + ": " + str(e))[:300]}), flush=True); sys.exit(0)
for i, c in enumerate(spec["checks"]):
    try:
        exec(compile(c["pre"] + "\n" + c["code"], "check", "exec"), g)
        print(json.dumps({"i": i, "ok": True}), flush=True)
    except BaseException as e:
        got = None
        if c.get("lhs"):
            try: got = repr(eval(c["lhs"], g))[:200]
            except BaseException as e2: got = "raised " + type(e2).__name__ + ": " + str(e2)[:150]
        print(json.dumps({"i": i, "ok": False, "err": (type(e).__name__ + ": " + str(e))[:200], "got": got}), flush=True)
'''


def load_problems():
    path = os.path.join(HERE, "HumanEval.jsonl.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def split_checks(test_src):
    """Turn HumanEval's check(candidate) into a list of individually runnable checks."""
    mod = ast.parse(test_src)
    module_parts, checks = [], []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            pre = []
            for st in node.body:
                if isinstance(st, ast.Assert):
                    if isinstance(st.test, ast.Constant) and st.test.value is True:
                        continue  # 'assert True' placeholders carry no information
                    lhs = None
                    t = st.test
                    if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
                        lhs = ast.unparse(t.left)
                    checks.append({"pre": "\n".join(pre), "code": ast.unparse(st), "lhs": lhs})
                elif isinstance(st, (ast.For, ast.While, ast.With, ast.If, ast.Try)):
                    checks.append({"pre": "\n".join(pre), "code": ast.unparse(st), "lhs": None})
                else:
                    pre.append(ast.unparse(st))
            module_parts.append(ast.unparse(node))
        else:
            module_parts.append(ast.unparse(node))
    return "\n".join(module_parts), checks


def header_of(prob):
    p, ep = prob["prompt"], prob["entry_point"]
    i = p.find("def " + ep)
    return p[:i] if i >= 0 else ""


def extract_code(text, entry_point):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    for b in reversed(blocks):
        if "def " + entry_point in b:
            return b.strip("\n")
    if blocks:
        return blocks[-1].strip("\n")
    return text.strip()


def run_visible(full_src, spec, timeout=20):
    d = tempfile.mkdtemp(prefix="he_")
    sp, cp, rp = (os.path.join(d, n) for n in ("cand.py", "spec.json", "runner.py"))
    open(sp, "w", encoding="utf-8").write(full_src)
    json.dump(spec, open(cp, "w", encoding="utf-8"))
    open(rp, "w", encoding="utf-8").write(RUNNER)
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "-I", rp, sp, cp], capture_output=True, text=True,
                             timeout=timeout, cwd=d).stdout
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        timed_out = True
    n = len(spec["checks"])
    res = [None] * n
    compile_error = None
    for line in out.splitlines():
        try: r = json.loads(line)
        except ValueError: continue
        if "compile_error" in r: compile_error = r["compile_error"]
        elif "i" in r: res[r["i"]] = r
    for i in range(n):
        if res[i] is None:
            res[i] = {"i": i, "ok": False, "err": "Timeout: no result within %ds" % timeout if timed_out else
                      (compile_error or "no result")}
    v = sum(r["ok"] for r in res) if not compile_error else 0
    first_fail = next((r for r in res if not r["ok"]), None)
    return {"v": v, "n": n, "pass": compile_error is None and v == n, "compile_error": compile_error,
            "first_fail": None if first_fail is None else {"check": spec["checks"][first_fail["i"]]["code"],
                                                           "err": compile_error or first_fail["err"],
                                                           "got": first_fail.get("got")},
            "check_seconds": round(time.time() - t0, 3)}


def describe(vis):
    f = vis["first_fail"]
    if vis["compile_error"]:
        return "(the code could not be loaded)", "Error: " + vis["compile_error"]
    detail = "Error: " + f["err"]
    if f.get("got") is not None:
        detail += "\nYour function returned: " + f["got"]
    return f["check"], detail


def chat(model, messages, seed):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "keep_alive": "60m",
                       "options": {"temperature": 0.2, "seed": seed, "num_ctx": 8192, "num_predict": 1024}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    t1 = time.time()
    meta = {k: resp.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                                     "eval_count", "eval_duration")}
    return resp["message"]["content"], {"t_start": t0, "t_end": t1, **meta}


def attempt(model, messages, seed, prob, spec, header, branch, k):
    text, req = chat(model, messages, seed)
    code = extract_code(text, prob["entry_point"])
    vis = run_visible(header + "\n" + code, spec)
    return {"branch": branch, "attempt": k, "seed": seed, "request": req, "raw": text, "code": code,
            "solution": header + "\n" + code, "visible": vis}


def run_unit(model, prob, seed):
    test_module, checks = split_checks(prob["test"])
    spec = {"entry_point": prob["entry_point"], "test_module": test_module, "checks": checks}
    header = header_of(prob)
    first_msgs = [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": FIRST.format(prompt=prob["prompt"])}]
    a1 = attempt(model, first_msgs, seed, prob, spec, header, "first", 1)
    unit = {"task_id": prob["task_id"], "seed": seed, "n_checks": len(checks), "first": a1, "branches": {}}
    if a1["visible"]["pass"]:
        return unit
    # full history
    msgs, last, traj = list(first_msgs), a1, []
    for k in range(2, CAP + 1):
        check, detail = describe(last["visible"])
        msgs = msgs + [{"role": "assistant", "content": last["raw"]},
                       {"role": "user", "content": FEEDBACK.format(check=check, detail=detail)}]
        last = attempt(model, msgs, seed + k - 1, prob, spec, header, "full", k)
        traj.append(last)
        if last["visible"]["pass"]: break
    unit["branches"]["full"] = traj
    # short history
    last, traj = a1, []
    for k in range(2, CAP + 1):
        check, detail = describe(last["visible"])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": SHORT.format(prompt=prob["prompt"], code=last["code"], check=check, detail=detail)}]
        last = attempt(model, msgs, seed + k - 1, prob, spec, header, "short", k)
        traj.append(last)
        if last["visible"]["pass"]: break
    unit["branches"]["short"] = traj
    # fresh retry
    traj = []
    for k in range(2, CAP + 1):
        last = attempt(model, first_msgs, seed + 1000 * (k - 1), prob, spec, header, "retry", k)
        traj.append(last)
        if last["visible"]["pass"]: break
    unit["branches"]["retry"] = traj
    return unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--seeds", default="11,29,47", help="comma-separated seeds")
    a = ap.parse_args()
    probs = [p for p in load_problems() if int(p["task_id"].split("/")[1]) not in PILOT_HELDOUT]
    seeds = [int(x) for x in a.seeds.split(",")]
    units = [(p, s) for p in probs for s in seeds]
    random.Random(2026).shuffle(units)
    if a.limit: units = units[:a.limit]
    if os.name == "nt":  # keep the laptop awake while this process runs (no settings are changed)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: r = json.loads(l); done.add((r["task_id"], r["seed"]))
            except ValueError: pass
    todo = [(p, s) for p, s in units if (p["task_id"], s) not in done]
    print(f"[{a.tag}] {len(units)} units, {len(done)} done, {len(todo)} to run", flush=True)
    # warm-up request (not recorded) so the first unit does not pay the model load
    chat(a.model, [{"role": "user", "content": "Say OK."}], 0)
    ps = json.loads(urllib.request.urlopen(OLLAMA + "/api/ps", timeout=30).read())
    m = next((x for x in ps.get("models", []) if x.get("name", "").startswith(a.model)), None)
    vram = (m or {}).get("size_vram", 0); size = (m or {}).get("size", 0)
    print(f"[{a.tag}] model in GPU memory: {vram/2**30:.2f} of {size/2**30:.2f} GiB", flush=True)
    if not a.allow_cpu and (not size or vram < 0.95 * size):
        sys.exit("Model is not fully on the GPU (Ollama is using the CPU). Enable the NVIDIA GPU and restart Ollama.")
    with open(a.out, "a", encoding="utf-8") as f:
        for i, (p, s) in enumerate(todo, 1):
            u = run_unit(a.model, p, s); u["model"] = a.model; u["tag"] = a.tag
            f.write(json.dumps(u) + "\n"); f.flush()
            nb = sum(len(v) for v in u["branches"].values())
            print(f"[{a.tag}] {i}/{len(todo)} {p['task_id']} s{s} first={'pass' if u['first']['visible']['pass'] else 'FAIL'}"
                  f" extra_requests={nb}", flush=True)
    print(f"[{a.tag}] finished", flush=True)


if __name__ == "__main__":
    main()
