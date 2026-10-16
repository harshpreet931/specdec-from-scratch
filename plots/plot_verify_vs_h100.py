"""PNG for slide 11: cost of one pass of k tokens, relative to writing 1 token.
Mac = measured (Qwen3-8B 4-bit). H100 = theory from peak specs (8B model, 16-bit)."""
import json, os
import numpy as np
from style import plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(LAB), "assets", "05_AI_chart_verify_cost_vs_h100.png")

d = json.load(open(os.path.join(LAB, "task2_roofline", "verify_cost_curve.json")))
T = "mlx-community/Qwen3-8B-4bit"
ks = sorted(int(k) for k in d[T] if k.isdigit())
mac = np.array([d[T][str(k)]["median_ms"] for k in ks]) / d[T]["1"]["median_ms"]

read_ms, math_ms = 16e9 / 3.35e12 * 1e3, 16e9 / 989e12 * 1e3     # per pass / per token
kk = np.array(ks, dtype=float)
kd = np.logspace(0, np.log2(512), 400, base=2)
h100 = np.maximum(read_ms, kd * math_ms) / read_ms

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
ax.plot(kk, kk, ":", color="0.55", lw=1.0, label="write k tokens one at a time")
ax.plot(kk, mac, "-o", color="black", ms=3.5, mfc="white", mew=0.8, label="my Mac: check k in one pass (measured)")
ax.plot(kd, h100, "--", color="#c62828", lw=1.4, label="H100, 16-bit: check k in one pass (theory)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ks, [str(k) for k in ks])
ax.set_yticks([1, 2, 4, 10, 30, 100, 500], ["1x", "2x", "4x", "10x", "30x", "100x", "500x"])
ax.minorticks_off()
ax.set_ylim(0.8, 700)
ax.set_xlabel("tokens per pass (k)")
ax.set_ylabel("cost vs writing 1 token")

i2, i8 = ks.index(2), ks.index(8)
for i, txt, off in [(i2, f"check 2: {mac[i2]:.2f}x", (34, -1)), (i8, f"check 8: {mac[i8]:.1f}x", (18, -38))]:
    ax.plot([kk[i]], [mac[i]], "o", color="black", ms=5.5)
    ax.annotate(txt, (kk[i], mac[i]), xytext=off, textcoords="offset points", fontsize=10,
                arrowprops=dict(arrowstyle="-", lw=0.7))
ax.annotate("flat until ~295 tokens", (128, 1.0), xytext=(-40, 22), textcoords="offset points",
            fontsize=9.5, color="#c62828")
ax.legend(loc="upper left", fontsize=9.5)
ax.text(1.0, -0.15, f"Mac: Apple M4 Pro, mlx {d['mlx']}, Qwen3-8B 4-bit, {d['context_tokens']} tokens in KV cache, "
        "median of 15 runs.  H100: peak-spec arithmetic for an 8B model.", transform=ax.transAxes,
        ha="right", fontsize=7.5, color="0.35")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
