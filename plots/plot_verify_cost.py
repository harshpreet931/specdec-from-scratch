"""PNG for (A): ms per forward pass vs tokens per pass (log2 x), target + draft."""
import json, os
from style import plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "task2_roofline", "verify_cost_curve.json")))
T, D = "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-0.6B-4bit"


def series(name, kmax=10**6):
    ks = sorted(int(k) for k in d[name] if k.isdigit() and int(k) <= kmax)
    return ks, [d[name][str(k)]["median_ms"] for k in ks]


def draw(logy, path, kmax=10**6, offs=((12, 28), (-118, 26))):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    kt, mt = series(T, kmax)
    kd, md = series(D, kmax)
    ax.plot(kt, mt, "-o", color="black", ms=3.5, mfc="white", mew=0.8, label="Qwen3-8B 4-bit (target)")
    ax.plot(kd, md, "--s", color="black", ms=3, mfc="white", mew=0.8, label="Qwen3-0.6B 4-bit (draft)")
    ax.axhline(mt[0], color="0.6", lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(kt, [str(k) for k in kt])
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000], ["2", "5", "10", "20", "50", "100", "200", "500", "1000", "2000"])
        ax.set_ylim(min(md) * 0.7, max(mt) * 1.4)
    else:
        ax.set_ylim(0, max(mt) * 1.08)
    ax.minorticks_off() if logy else None
    ax.set_xlabel("tokens per pass")
    ax.set_ylabel("ms per forward pass")
    t1, t8 = d[T]["1"]["median_ms"], d[T]["8"]["median_ms"]
    for k, t, txt, off in [(1, t1, f"write 1 token\n{t1:.1f} ms", offs[0]),
                           (8, t8, f"check 8 tokens\n{t8:.1f} ms ({t8 / t1:.1f}x)", offs[1])]:
        ax.plot([k], [t], "o", color="black", ms=5.5)
        ax.annotate(txt, (k, t), xytext=off, textcoords="offset points", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(1.0, -0.13, f"{d['device']}, mlx {d['mlx']}, {d['context_tokens']} tokens already in KV cache, "
            f"median of 15 runs", transform=ax.transAxes, ha="right", fontsize=7.5, color="0.35")
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


draw(True, os.path.join(ROOT, "task2_roofline", "verify_cost_curve.png"))
draw(False, os.path.join(ROOT, "task2_roofline", "verify_cost_curve_zoom_linear_y.png"), kmax=32,
     offs=((14, 60), (-150, 30)))
