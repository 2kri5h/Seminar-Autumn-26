"""Regenerate report/figs/results.pdf (Fig. 3 of the report) from data/reported_results.json."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.edgecolor":"#6b7378","axes.labelcolor":"#0b0b0b",
 "xtick.color":"#52514e","ytick.color":"#52514e","axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8})
C={"One attempt":"#2a78d6","Full history":"#eb6834","Short history":"#1baf7a"}
pol=["One attempt","Full history","Short history"]
ROOT=pathlib.Path(__file__).resolve().parent.parent
R=json.load(open(ROOT/"data"/"reported_results.json"))["policies"]
keys=["one_attempt","full_history","short_history"]
correct=[R[k]["correct"] for k in keys]; jrun=[R[k]["mean_J_per_run"] for k in keys]
jpc=[j*72/c for j,c in zip(jrun,correct)]
print([round(x,2) for x in jpc])
fig,ax=plt.subplots(1,3,figsize=(7.1,2.35),gridspec_kw={"width_ratios":[1.05,1.15,1.1]})
# (a)
a=ax[0]
b=a.bar(range(3),jpc,color=[C[p] for p in pol],width=0.68,edgecolor="white",linewidth=1.5)
for i,(v,c) in enumerate(zip(jpc,correct)):
    a.text(i,v+4,f"{v:.0f} J",ha="center",va="bottom",fontsize=8.5,color="#0b0b0b",fontweight="bold")
    a.text(i,8,f"{c}/72",ha="center",va="bottom",fontsize=7.5,color="white",fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(["One\nattempt","Full\nhistory","Short\nhistory"])
a.set_ylabel("GPU J per correct solution"); a.set_ylim(0,215)
a.yaxis.grid(True,color="#e6e5e0",linewidth=0.6); a.set_axisbelow(True)
a.set_title("(a) Energy per success",loc="left",fontsize=9,fontweight="bold")
# (b) break-even plane
a=ax[1]
import numpy as np
xs=np.linspace(0.95,1.6,50)
a.fill_between(xs,0.95,xs,color="#1baf7a",alpha=0.10,linewidth=0)
a.plot(xs,xs,color="#52514e",lw=1,ls="--")
a.text(1.46,1.13,"repair is cheaper\nper success",fontsize=7,color="#52514e",ha="center")
pts={"Full history":(correct[1]/correct[0],jrun[1]/jrun[0]),"Short history":(correct[2]/correct[0],jrun[2]/jrun[0])}
for k,(x,y) in pts.items():
    a.scatter([x],[y],s=46,color=C[k],edgecolor="white",linewidth=1.2,zorder=3)
    a.annotate(f"{k}\n({x:.2f}, {y:.2f})",(x,y),xytext=(9,2 if k=="Full history" else -16),textcoords="offset points",fontsize=7,color="#0b0b0b")
a.set_xlim(0.97,1.6); a.set_ylim(0.97,1.6)
a.set_xlabel("Success ratio  $\\hat p_B/\\hat p_A$"); a.set_ylabel("Energy-per-run ratio  $\\bar E_B/\\bar E_A$")
a.set_title("(b) Break-even test, Eq. (5)",loc="left",fontsize=9,fontweight="bold")
# (c) cap runs
a=ax[2]
cats=["Full history","Short history"]; runs=[R["full_history"]["cap_share_of_runs_pct"],R["short_history"]["cap_share_of_runs_pct"]]; en=[R["full_history"]["cap_share_of_energy_pct"],R["short_history"]["cap_share_of_energy_pct"]]
y=np.arange(2)
a.barh(y+0.18,runs,height=0.34,color="#b9b8b0",edgecolor="white",label="share of runs")
a.barh(y-0.18,en,height=0.34,color=[C[c] for c in cats],edgecolor="white",label="share of energy")
for i in range(2):
    a.text(runs[i]+1,y[i]+0.18,f"{runs[i]}% of runs",va="center",fontsize=7.5,color="#52514e")
    a.text(en[i]+1,y[i]-0.18,f"{en[i]}% of energy",va="center",fontsize=7.5,color="#0b0b0b",fontweight="bold")
a.set_yticks(y); a.set_yticklabels(["Full\nhistory","Short\nhistory"]); a.invert_yaxis()
a.set_xlim(0,72); a.set_xlabel("Percent of policy total")
a.set_title("(c) Runs that hit the cap",loc="left",fontsize=9,fontweight="bold")
plt.tight_layout(w_pad=1.6)
out=ROOT/"report"/"figs"; out.mkdir(parents=True,exist_ok=True)
plt.savefig(out/"results.pdf"); plt.savefig(out/"results.png",dpi=160)
