"""Shared chart style: white background, thin black lines, monospace labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Andale Mono", "Menlo", "DejaVu Sans Mono"],
    "font.size": 11,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelcolor": "black",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.0, "legend.frameon": False,
})
