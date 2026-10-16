"""Large-font versions of the slide 3 and slide 11 charts, for reading on a phone.
Same data as plot_roofline.py and plot_verify_vs_h100.py; only the typography and labels change."""
import json, os
import numpy as np
from style import plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(os.path.dirname(LAB), "assets")
plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1})
ANN = 14
FOOT = dict(fontsize=10.5, color="0.35", ha="right", transform=None)


def footer(ax, text):
    ax.text(1.0, -0.2, text, transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="0.35",
            linespacing=1.3)


# ---------- slide 3: H100 roofline ----------
BW, PEAK = 3.35, 989.0
RIDGE = PEAK / BW
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
ai = np.logspace(-0.3, 3.6, 400)
ax.plot(ai, np.minimum(PEAK, BW * ai), color="black", lw=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.5, 4000); ax.set_ylim(1, 4000)
ax.set_xticks([1, 8, 100, RIDGE, 1000], ["1", "8", "100", "295", "1000"])
ax.set_yticks([1, 3.35, 27, 100, 989], ["1", "3.4", "27", "100", "989"])
ax.minorticks_off()
ax.set_xlabel("FLOPs per byte read")
ax.set_ylabel("TFLOPS used")
ax.axvline(RIDGE, color="0.6", lw=0.8, ls=":")
ax.annotate("ridge: 989 / 3.35\n= ~295 FLOP/byte", (RIDGE, PEAK), xytext=(-215, 4), textcoords="offset points",
            fontsize=ANN, arrowprops=dict(arrowstyle="-", lw=0.9))
ax.text(1300, PEAK * 1.4, "compute-bound", fontsize=ANN, ha="center")
ax.text(0.6, 180, "memory-bound", fontsize=ANN, ha="left")
for x, label, off in [(1, "write 1 token\n4.8 ms", (-22, 30)),
                      (8, "check 8 tokens\nstill ~4.8 ms", (16, -66))]:
    ax.plot([x], [BW * x], "o", color="black", ms=8)
    ax.annotate(label, (x, BW * x), xytext=off, textcoords="offset points", fontsize=ANN,
                arrowprops=dict(arrowstyle="-", lw=0.9))
footer(ax, "H100, Llama 3.1 8B, 16-bit, batch 1. Both points read the same 16 GB of weights,\n"
           "so both take ~4.8 ms. 8x the math is still only 2.7% of the step.")
fig.tight_layout()
out = os.path.join(ASSETS, "05_AI_chart_roofline_h100_big.png")
fig.savefig(out); print("wrote", out)

# ---------- slide 11: cost of checking k tokens, Mac (measured) vs H100 (theory) ----------
d = json.load(open(os.path.join(LAB, "task2_roofline", "verify_cost_curve.json")))
T = "mlx-community/Qwen3-8B-4bit"
ks = sorted(int(k) for k in d[T] if k.isdigit())
mac = np.array([d[T][str(k)]["median_ms"] for k in ks]) / d[T]["1"]["median_ms"]
kk = np.array(ks, dtype=float)
read_ms, math_ms = 16e9 / 3.35e12 * 1e3, 16e9 / 989e12 * 1e3
kd = np.logspace(0, np.log2(512), 400, base=2)
h100 = np.maximum(read_ms, kd * math_ms) / read_ms

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
ax.plot(kk, kk, ":", color="0.5", lw=1.5, label="write k, one at a time")
ax.plot(kk, mac, "-o", color="black", lw=1.8, ms=5, mfc="white", mew=1.2, label="my Mac: check k in 1 pass")
ax.plot(kd, h100, "--", color="#c62828", lw=2.2, label="H100 16-bit: check k (theory)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
pow2 = [k for k in ks if k & (k - 1) == 0]
ax.set_xticks(pow2, [str(k) for k in pow2])
ax.set_yticks([1, 4, 10, 100, 500], ["1x", "4x", "10x", "100x", "500x"])
ax.minorticks_off()
ax.set_ylim(0.75, 900)
ax.set_xlabel("tokens per pass (k)")
ax.set_ylabel("cost vs writing 1")
i2, i8 = ks.index(2), ks.index(8)
# labels sit in open space (data coords): "check 2" above the curves on the left, "check 8" between Mac and H100
for i, txt, at in [(i2, f"check 2: {mac[i2]:.2f}x", (1.05, 7)), (i8, f"check 8: {mac[i8]:.1f}x", (13, 1.9))]:
    ax.plot([kk[i]], [mac[i]], "o", color="black", ms=8)
    ax.annotate(txt, (kk[i], mac[i]), xytext=at, textcoords="data", fontsize=ANN,
                arrowprops=dict(arrowstyle="-", lw=0.9, shrinkA=2, shrinkB=4))
ax.annotate("flat to ~295", (90, 1.0), xytext=(0, 12), textcoords="offset points", fontsize=ANN, color="#c62828")
ax.legend(loc="upper left", fontsize=13, handlelength=1.8)
footer(ax, f"Mac: Apple M4 Pro, mlx {d['mlx']}, Qwen3-8B 4-bit, {d['context_tokens']} tokens in KV cache, "
           "median of 15 runs.\nH100: peak-spec arithmetic for an 8B model.")
fig.tight_layout()
out = os.path.join(ASSETS, "05_AI_chart_verify_cost_vs_h100_big.png")
fig.savefig(out); print("wrote", out)
