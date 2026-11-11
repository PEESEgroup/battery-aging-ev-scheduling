"""Redraw Fig. 6 (directional charging ablations) in the visual style of Fig. 8.

Style taken from the embedded Fig. 8 SVG (matplotlib output): Times New Roman, 7.7 pt panel titles,
7.2 pt axis labels and y tick labels, 6.7 pt x tick labels rotated 30 deg, left/bottom spines (0.6 pt),
2-pt outward ticks (0.8 pt), faint y grid (#b0b0b0, alpha 0.18, 0.5 pt), borderless bars from the
same sequential blue palette, 252-pt (3.5-in) figure width.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

src, out_svg, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif"],
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.unicode_minus": False,
})

GROUPS = ["joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage"]
GROUP_LABELS = ["Normal", "Evening peak", "High price", "Capacity shortage"]
VARIANTS = [("full", "Full"), ("no_aging", "PINN off"), ("no_risk", "Risk off")]
# Fig. 8 blue sequence (#d9e6f0 ... #3b78a7); darkest tone = proposed method, as for GPT-6 Sol in Fig. 8
COLORS = {"full": "#3b78a7", "no_aging": "#7faacb", "no_risk": "#bed6e8"}
ROWS = [("tracked", "Tracking (%)", 100, [0, 50, 100], (0, 108)),
        ("delivered", "Delivery (%)", 100, [0, 50, 100], (0, 108)),
        ("mean_soh_loss", "SOH loss (%)", 100, [0, 2, 4], (0, 5.0))]
DIRS = [("down", "Downward"), ("up", "Upward")]

W, H = 252, 214  # pt
fig, axes = plt.subplots(3, 2, figsize=(W / 72, H / 72), sharex=True)
fig.subplots_adjust(left=37.8 / W, right=244.44 / W, top=1 - 27 / H, bottom=40 / H, wspace=0.36, hspace=0.30)

x = np.arange(len(GROUPS))
bw, step = 0.23, 0.26
for r, (col, ylabel, scale, yticks, ylim) in enumerate(ROWS):
    for c, (d, title) in enumerate(DIRS):
        ax = axes[r, c]
        for k, (v, _) in enumerate(VARIANTS):
            vals = [df[(df.group == g) & (df.direction == d) & (df.variant == v)][col].iloc[0] * scale for g in GROUPS]
            ax.bar(x + (k - 1) * step, vals, width=bw, color=COLORS[v], edgecolor="none", linewidth=0, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=7.2, pad=2)
        ax.tick_params(axis="x", labelsize=6.7, pad=2)
        ax.grid(axis="y", color="#b0b0b0", alpha=0.18, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=7.2, labelpad=3)
        if r == 0:
            ax.set_title(title, fontsize=7.7, pad=3)
        if r == len(ROWS) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(GROUP_LABELS, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.tick_params(axis="x", length=0)

handles = [Patch(facecolor=COLORS[v], edgecolor="none", label=lab) for v, lab in VARIANTS]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.2,
           bbox_to_anchor=(0.5 * (37.8 + 244.44) / W, 1.0), handlelength=1.0, handleheight=0.8,
           handletextpad=0.4, columnspacing=1.4, borderaxespad=0.2)

fig.savefig(out_svg, format="svg")
fig.savefig(out_png, format="png", dpi=600)
