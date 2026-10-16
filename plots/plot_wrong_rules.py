"""PNG for (B): output distribution of 4 accept/reject rules vs target p, one pair, side by side."""
import json, os
import numpy as np
from style import plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "task1_lossless", "wrong_rules.json")))
PAIR = "moderate mismatch, V=6 (chart pair)"
TITLES = {"1 correct": "1. correct rule\nmin(1,p/q), else max(0,p-q)",
          "2 accept if p>=q": "2. accept if p(x) >= q(x)\nelse max(0,p-q)",
          "3 resample from p": "3. min(1,p/q)\nelse resample from p",
          "4 argmax check": "4. keep x if x = argmax p\nelse argmax p"}
e = d[PAIR]
p, q = np.array(e["p"]), np.array(e["q"])
x = np.arange(len(p))
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), dpi=200, sharey=True)
for ax, (rule, title) in zip(axes, TITLES.items()):
    emp = np.array(e[rule]["empirical"])
    ax.bar(x, emp, width=0.62, color="0.82", edgecolor="black", linewidth=0.6, label="output of rule")
    ax.bar(x, p, width=0.62, fill=False, edgecolor="black", linewidth=1.0, linestyle="-", label="target p")
    ax.plot(x, q, "x", color="black", ms=5, mew=0.8, label="draft q")
    ax.set_title(title, fontsize=10, loc="left")
    pv = e[rule]["chi2_p"]
    ax.text(0.98, 0.97, f"TV {e[rule]['tv']:.3f}\nchi2 p {'%.2f' % pv if pv > 1e-3 else '< 1e-300' if pv == 0 else '%.0e' % pv}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_xticks(x, [f"t{i}" for i in x])
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("probability")
axes[0].legend(loc="center right", fontsize=8.5)
fig.text(0.5, 0.005, f"target p = {p.tolist()}   draft q = {q.tolist()}   2,000,000 trials per rule",
         ha="center", fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = os.path.join(ROOT, "task1_lossless", "wrong_rules.png")
fig.savefig(out)
print("wrote", out)
