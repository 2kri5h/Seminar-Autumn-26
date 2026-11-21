#!/usr/bin/env python3
"""Charts for the seminar slides, drawn from the repository data in data/followup/.

Usage (from the repo root): python slides/make_slide_figs.py
Writes PDFs into slides/figs/.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ["/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf"]:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Latin Modern Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555", "pdf.fonttype": 42})
NAVY, RED, TEAL, GREY, LIGHT = "#0d2659", "#ec5858", "#5eb2a8", "#9ea8bd", "#c9d0de"
D = "data/followup"; OUT = "slides/figs"; os.makedirs(OUT, exist_ok=True)
R = json.load(open(f"{D}/results.json"))
C = json.load(open(f"{D}/crossmodel_all.json"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.close(fig)


# 1. The model repeats itself: share of continuation attempts with an unchanged program.
fig, ax = plt.subplots(figsize=(6.4, 3.2))
labels = ["Full history", "Short history", "Fresh retry"]; keys = ["full", "short", "retry"]
for i, (m, col) in enumerate([("1.5b", NAVY), ("7b", TEAL)]):
    ir = R[m]["identical_resubmission"]
    vals = [100 * ir[k]["identical"] / ir[k]["attempts"] for k in keys]
    xs = [x + (i - 0.5) * 0.36 for x in range(3)]
    bars = ax.bar(xs, vals, 0.34, color=col, label=f"Qwen {m.upper()}")
    for x, v in zip(xs, vals): ax.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=12)
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylim(0, 105)
ax.set_ylabel("Program unchanged (%)"); ax.legend(frameon=False, loc="upper right")
save(fig, "unchanged")

# 2. Rescued failures per continuation (cap 4).
fig, ax = plt.subplots(figsize=(6.4, 3.0))
pol = {"Full history": "full_cap4", "Short history": "short_cap4", "Fresh retry": "retry_cap4"}
for i, (m, col) in enumerate([("1.5b", NAVY), ("7b", TEAL)]):
    ff = R[m]["first_fail"]
    vals = [100 * R[m]["policies"][p]["rescues"] / ff for p in pol.values()]
    xs = [x + (i - 0.5) * 0.36 for x in range(3)]
    ax.bar(xs, vals, 0.34, color=col, label=f"Qwen {m.upper()}")
    for x, v in zip(xs, vals): ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=12)
ax.set_xticks(range(3)); ax.set_xticklabels(list(pol)); ax.set_ylim(0, 35)
ax.set_ylabel("Failures rescued (%)"); ax.legend(frameon=False, loc="upper left")
save(fig, "rescues")

# 3. Success vs energy per correct solution (local models, cap 4 and one attempt).
fig, ax = plt.subplots(figsize=(6.4, 3.4))
arms = C["paired"]["arms"]
pts = {"1.5b:one": ("1.5B, one attempt", NAVY, "o"), "1.5b:retry_cap4": ("1.5B, 4 fresh retries", NAVY, "s"),
       "7b:one": ("7B, one attempt", TEAL, "o"), "7b:retry_cap4": ("7B, 4 fresh retries", TEAL, "s")}
for k, (lab, col, mk) in pts.items():
    a = arms[k]; ax.scatter(a["J_per_correct"], 100 * a["success"], s=110, color=col, marker=mk, zorder=3)
    ax.annotate(lab, (a["J_per_correct"], 100 * a["success"]), textcoords="offset points", xytext=(8, -4), fontsize=11)
ax.set_xlabel("GPU energy per correct solution (J)"); ax.set_ylabel("Success (%)")
ax.set_xlim(80, 430); ax.set_ylim(62, 86)
save(fig, "tradeoff")

# 4. Four models on the same 140 problems.
fig, ax = plt.subplots(figsize=(6.4, 3.2))
M = C["models"]; order = [("1.5b", "Qwen 1.5B\n(laptop)"), ("7b", "Qwen 7B\n(laptop)"),
                          ("kimi", "Kimi K3.1 Pro\n(2.8T, hosted)"), ("nemotron", "Nemotron 3 Ultra\n(550B, hosted)")]
for i, (k, lab) in enumerate(order):
    corr = 100 * M[k]["first_correct"] / M[k]["units"]; vis = 100 * M[k]["visible_pass_hidden_fail"] / M[k]["units"]
    ax.bar(i, corr, 0.6, color=NAVY if i < 2 else RED)
    ax.bar(i, vis, 0.6, bottom=corr, color=LIGHT, hatch="///", edgecolor=GREY)
    ax.text(i, corr / 2, f"{corr:.1f}%", ha="center", color="white", fontsize=13, weight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in order], fontsize=11)
ax.set_ylim(0, 105); ax.set_ylabel("First answer correct (%)")
ax.bar([0], [0], color=LIGHT, hatch="///", edgecolor=GREY, label="passed visible tests, failed hidden")
ax.legend(frameon=False, loc="upper left", fontsize=11)
save(fig, "fourmodels")

# 5. Hidden reasoning: thinking vs answer tokens (means per first request).
fig, ax = plt.subplots(figsize=(6.4, 2.6))
rows = [("Qwen 7B", 0, M["7b"]["first_output_tokens_median"]),
        ("Kimi K3.1 Pro", M["kimi"]["first_thinking_tokens_median"], M["kimi"]["first_output_tokens_median"]),
        ("Nemotron 3 Ultra", M["nemotron"]["first_thinking_tokens_median"], M["nemotron"]["first_output_tokens_median"])]
for i, (lab, th, ans) in enumerate(rows):
    ax.barh(i, ans, color=NAVY); ax.barh(i, th, left=ans, color=RED)
    ax.text(ans + th + 10, i, f"{ans + th:.0f}", va="center", fontsize=12)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("Median tokens per first request"); ax.set_xlim(0, 860)
ax.barh([0], [0], color=NAVY, label="visible answer"); ax.barh([0], [0], color=RED, label="hidden reasoning")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "tokens")
print("figures written to", OUT)

# 6. Temperature follow-up (data/temperature/, seed-11 failed first answers, full-history repair).
import subprocess
T = {}
for m in ["1.5b", "7b"]:
    out = subprocess.run(["python3", "slides/temp_analysis.py", f"{D}/runs_{m}.jsonl", f"data/temperature/runs_temp_{m}.jsonl"],
                         capture_output=True, text=True, check=True).stdout
    T[m] = json.loads(out)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for i, (lab, key, col) in enumerate([("T = 0.2 (original)", "T0.2", NAVY), ("T = 0.8 (new run)", "Thigh", RED)]):
    vals = [T[m][key]["unchanged_pct"] for m in ["1.5b", "7b"]]
    xs = [x + (i - 0.5) * 0.36 for x in range(2)]
    ax.bar(xs, vals, 0.34, color=col, label=lab)
    for x, v in zip(xs, vals): ax.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=12)
ax.set_xticks(range(2)); ax.set_xticklabels(["Qwen 1.5B (42 failures)", "Qwen 7B (23 failures)"])
ax.set_ylim(0, 105); ax.set_ylabel("Program unchanged (%)"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "temperature")
json.dump(T, open(f"{OUT}/temperature.json", "w"), indent=1)
