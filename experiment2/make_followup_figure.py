#!/usr/bin/env python3
"""Figure for the follow-up experiment, from data/followup/results.json."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#6b7378",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = json.load(open(ROOT / "data" / "followup" / "results.json"))
C = {"one": "#2a78d6", "full": "#eb6834", "short": "#1baf7a", "retry": "#7b5cd6"}
NAME = {"one": "One attempt", "full": "Full history", "short": "Short history", "retry": "Fresh retry"}
M = {"1.5b": ("o", "1.5B"), "7b": ("s", "7B")}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.25, 1, 1]})

a = ax[0]
for t, (mk, lab) in M.items():
    P = R[t]["policies"]
    pts = [("one", P["one"])] + [(b, P[f"{b}_cap4"]) for b in ("full", "short", "retry")]
    a.plot([p["J_per_correct"] for _, p in pts[:1]] * 0, [])
    for k, p in pts:
        a.scatter(p["J_per_correct"], 100 * p["success"], marker=mk, s=40, color=C[k], edgecolor="white",
                  linewidth=1, zorder=3)
    a.text(P["one"]["J_per_correct"] - 8, 100 * P["one"]["success"] + 1.6, lab, fontsize=8, fontweight="bold",
           color="#0b0b0b", ha="right")
a.set_xlabel("GPU energy per correct solution (J)"); a.set_ylabel("Success (%)")
a.set_xlim(60, 420); a.set_ylim(62, 84)
a.yaxis.grid(True, color="#e6e5e0", linewidth=0.6); a.set_axisbelow(True)
a.set_title("(a) Cost and success, cap 4", loc="left", fontsize=9, fontweight="bold")
for k in C:
    a.scatter([], [], color=C[k], s=30, label=NAME[k])
a.legend(fontsize=6.8, frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1)

for j, (key, title) in enumerate([("identical", "(b) Program unchanged (%)"), ("rescues", "(c) Rescued failures (%)")]):
    a = ax[j + 1]
    w = 0.36
    for i, t in enumerate(("1.5b", "7b")):
        vals = []
        for b in ("full", "short", "retry"):
            if key == "identical":
                d = R[t]["identical_resubmission"][b]; vals.append(100 * d["identical"] / d["attempts"])
            else:
                vals.append(100 * R[t]["policies"][f"{b}_cap4"]["rescues"] / R[t]["first_fail"])
        xs = [x + (i - 0.5) * w for x in range(3)]
        a.bar(xs, vals, width=w, color=[C[b] for b in ("full", "short", "retry")], alpha=1 if t == "7b" else 0.55,
              edgecolor="white", linewidth=1)
        for x, v in zip(xs, vals):
            a.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.8, color="#0b0b0b")
    a.set_xticks(range(3)); a.set_xticklabels(["Full", "Short", "Retry"])
    a.set_ylim(0, 115 if key == "identical" else 40)
    a.yaxis.grid(True, color="#e6e5e0", linewidth=0.6); a.set_axisbelow(True)
    a.set_title(title, loc="left", fontsize=9, fontweight="bold")
    a.text(0.98, 0.97, "light: 1.5B   dark: 7B", transform=a.transAxes, ha="right", va="top", fontsize=6.5,
           color="#52514e")
plt.tight_layout(w_pad=1.2)
out = ROOT / "report" / "figs"
plt.savefig(out / "followup.pdf"); plt.savefig(out / "followup.png", dpi=160)
print("ok")
