"""Code-correctness tests on tiny SYNTHETIC data (not experimental results)."""
import sys, pathlib, csv, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import bootstrap_ratios as b

def make(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    w = csv.writer(f); w.writerow(["problem_id", "seed", "policy", "correct", "energy_J"]); w.writerows(rows); f.close()
    return f.name

def test_point_estimate_and_identical_policies():
    rows = [(p, s, pol, 1, 10.0) for p in range(6) for s in (1, 2, 3) for pol in b.POLICIES]
    out = b.bootstrap(b.load(make(rows)), resamples=200, seed=1)
    for (pt, lo, hi, k) in out.values():
        assert pt == 1.0 and lo == 1.0 and hi == 1.0 and k == 200

def test_known_ratio():
    # policy B uses exactly twice the energy for the same successes -> ratio 2.0 in every resample
    rows = []
    for p in range(8):
        for s in (1, 2, 3):
            rows += [(p, s, "one_attempt", 1, 10.0), (p, s, "full_history", 1, 20.0), (p, s, "short_history", 1, 15.0)]
    out = b.bootstrap(b.load(make(rows)), resamples=100, seed=2)
    assert abs(out[("full_history", "one_attempt")][0] - 2.0) < 1e-12
    assert abs(out[("short_history", "full_history")][0] - 0.75) < 1e-12

def test_zero_success_replicates_skipped():
    rows = [(p, 1, pol, 0, 5.0) for p in range(3) for pol in b.POLICIES]
    out = b.bootstrap(b.load(make(rows)), resamples=50, seed=3)
    assert all(v[3] == 0 for v in out.values())

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
