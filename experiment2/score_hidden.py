#!/usr/bin/env python3
"""Score every candidate program in runs_*.jsonl on the hidden EvalPlus tests (base + plus inputs).

Runs on Linux (EvalPlus uses Unix resource limits). Needs:  pip install evalplus==0.3.1
and the HumanEval+ release file:  HUMANEVAL_OVERRIDE_PATH=/path/HumanEvalPlus.jsonl

Usage: python score_hidden.py runs_7b.jsonl [runs_1.5b.jsonl ...] --out hidden_scores.json
Writes {"<tag>|<task_id>|<seed>|<branch>|<attempt>": {"base": status, "plus": status}}.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth

PROBLEMS = None
EXPECTED = None


def _init():
    global PROBLEMS, EXPECTED
    PROBLEMS = get_human_eval_plus()
    EXPECTED = get_groundtruth(PROBLEMS, get_human_eval_plus_hash(), [])


def _score(key, task_id, solution):
    r = check_correctness("humaneval", 0, PROBLEMS[task_id], solution, EXPECTED[task_id])
    return key, {"base": r["base"][0], "plus": r["plus"][0]}


def candidates(path):
    for line in open(path, encoding="utf-8"):
        u = json.loads(line)
        tag, tid, s = u["tag"], u["task_id"], u["seed"]
        yield f"{tag}|{tid}|{s}|first|1", tid, u["first"]["solution"]
        for b, traj in u["branches"].items():
            for a in traj:
                yield f"{tag}|{tid}|{s}|{b}|{a['attempt']}", tid, a["solution"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [c for p in a.runs for c in candidates(p) if c[0] not in done]
    print(f"{len(done)} already scored, {len(todo)} to score", flush=True)
    _init()  # build ground truth once in the parent (cached on disk by evalplus)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        futs = [ex.submit(_score, *c) for c in todo]
        for i, f in enumerate(as_completed(futs), 1):
            k, v = f.result(); done[k] = v
            if i % 200 == 0:
                json.dump(done, open(a.out, "w")); print(f"{i}/{len(todo)}", flush=True)
    json.dump(done, open(a.out, "w"), indent=0)
    print("done", flush=True)


if __name__ == "__main__":
    main()
