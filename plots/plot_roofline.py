"""PNG for slide 3: H100 roofline with decode (1 token) and check (8 tokens) on the memory slope."""
import os
import numpy as np
from style import plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "assets", "05_AI_chart_roofline_h100.png")

BW, PEAK = 3.35, 989.0          # H100 SXM: TB/s, dense BF16 TFLOPS
RIDGE = PEAK / BW               # ~295 FLOP/byte

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
ai = np.logspace(-0.3, 3.6, 400)
ax.plot(ai, np.minimum(PEAK, BW * ai), color="black", lw=1.4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.5, 4000)
ax.set_ylim(1, 3000)
ax.set_xticks([1, 8, 100, RIDGE, 1000], ["1", "8", "100", "295", "1000"])
ax.set_yticks([1, 3.35, 10, 26.8, 100, 989], ["1", "3.4", "10", "27", "100", "989"])
ax.minorticks_off()
ax.set_xlabel("arithmetic intensity (FLOPs per byte read)")
ax.set_ylabel("TFLOPS actually used")

ax.axvline(RIDGE, color="0.6", lw=0.6, ls=":")
ax.annotate("ridge point\n989 TFLOPS / 3.35 TB/s\n= ~295 FLOP/byte", (RIDGE, PEAK), xytext=(-190, -10),
            textcoords="offset points", fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.7))
ax.text(1200, PEAK * 1.25, "compute-bound", fontsize=9.5, ha="center")
ax.text(6, 250, "memory-bound\n(slope = 3.35 TB/s)", fontsize=9.5, ha="center")

for x, label, off in [(1, "write 1 token\n~1 FLOP/byte\n4.8 ms", (-30, 28)),
                      (8, "check 8 tokens\n~8 FLOP/byte\nstill ~4.8 ms", (14, -62))]:
    y = BW * x
    ax.plot([x], [y], "o", color="black", ms=6)
    ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=9.5,
                arrowprops=dict(arrowstyle="-", lw=0.7))

ax.text(1.0, -0.15, "Llama 3.1 8B, 16-bit, batch 1. Same weights read once (16 GB) -> same time; "
        "8x the math is still only 2.7% of the step.", transform=ax.transAxes, ha="right",
        fontsize=7.5, color="0.35")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
