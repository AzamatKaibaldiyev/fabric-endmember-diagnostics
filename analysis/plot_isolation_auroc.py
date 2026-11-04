"""plot_isolation_auroc.py -- detectability versus endmember isolation.

    python analysis/plot_isolation_auroc.py [out_prefix]   (default: fig_isolation)

One point per held-out fibre at matched unknown fraction (30-50%): AUROC of the
reconstruction residual and of classifier max-probability, mean +- s.d. over
3 seeds, against the isolation of the fibre's endmember in the closed-set model
(angle to its nearest other endmember, log axis). The residual follows a
log-linear law in isolation; confidence does not. Values are the measured ones
in the novelty-detection table; replace them if the runs are repeated.

Single-column (3.5 in) print figure: no title, two series distinguished by
colour and by filled/hollow markers, chance line marked.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fibre: (isolation deg, residual mean, sd, max-prob mean, sd)
DATA = {
    "Spandex": (26.73, 0.887, 0.010, 0.727, 0.093),
    "Acrylic": (6.08, 0.667, 0.004, 0.618, 0.048),
    "Wool":    (5.37, 0.606, 0.006, 0.874, 0.032),
    "Silk":    (3.21, 0.563, 0.012, 0.578, 0.073),
    "Linen":   (1.24, 0.347, 0.005, 0.762, 0.022),
}
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_isolation"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

names = list(DATA)
iso = np.array([DATA[n][0] for n in names])
res, res_sd = np.array([DATA[n][1] for n in names]), np.array([DATA[n][2] for n in names])
mp, mp_sd = np.array([DATA[n][3] for n in names]), np.array([DATA[n][4] for n in names])

fig, ax = plt.subplots(figsize=(3.5, 2.15))
ax.axhspan(0.0, 0.5, color="#f0efec", zorder=0)
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(1.05, 0.51, "chance", ha="left", va="bottom", color=INK2, fontsize=6.3)

# log-linear fit of the residual
b, a = np.polyfit(np.log(iso), res, 1)
r = np.corrcoef(np.log(iso), res)[0, 1]
xx = np.geomspace(1.0, 36, 100)
ax.plot(xx, a + b * np.log(xx), color=BLUE, lw=0.9, ls=(0, (2, 2)), alpha=0.8, zorder=2)

ax.errorbar(iso, res, yerr=res_sd, fmt="o", color=BLUE, ms=4.5, mec="white", mew=0.7,
            capsize=2, elinewidth=0.8, zorder=4,
            label=f"reconstruction error (dashed: log fit, r = {r:.2f})")
ax.errorbar(iso, mp, yerr=mp_sd, fmt="s", color=ORANGE, mfc="white", mec=ORANGE, ms=4.2,
            mew=1.1, capsize=2, elinewidth=0.8, zorder=3, label="classifier confidence (max. probability)")

LABEL = {"Spandex": (1.0, 0.045, "center"), "Acrylic": (1.0, 0.045, "center"),
         "Wool": (1.0, -0.06, "center"), "Silk": (1 / 1.13, 0.0, "right"),
         "Linen": (1.0, -0.055, "center")}
for n, x, y in zip(names, iso, res):
    fx, dy, ha = LABEL[n]
    ax.text(x * fx, y + dy, n, ha=ha, va="center", color=INK, fontsize=6.3)

ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20], ["1°", "2°", "5°", "10°", "20°"])
ax.minorticks_off()
ax.set_xlim(1.0, 36)
ax.set_ylim(0.25, 1.0)
ax.set_xlabel("Endmember isolation (angle to nearest other endmember)")
ax.set_ylabel("AUROC, 30–50% unknown")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=6.3, handlelength=1.4,
          borderaxespad=0.3, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
print("wrote", OUT + ".{pdf,png}", f"(r = {r:.3f}, slope = {b:.3f} per log-degree)")
