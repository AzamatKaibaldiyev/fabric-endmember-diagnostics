"""plot_fig4.py -- Figure 4: novelty-detection AUROC versus unknown fraction.

    python analysis/plot_fig4.py [out_prefix] [--column]   (default: fig4_detection)

--column draws a 3.5-inch single-column version (IEEE two-column papers).


Uses the measured 3-seed values embedded in plot_openset_curves.py, so the
figure and Table 9 come from the same numbers. Print figure: no title (the
caption carries it), one categorical hue per fibre in fixed order plus a
distinct marker, legend always present, thin recessive grid.
"""
import importlib.util, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("poc", os.path.join(_here, "plot_openset_curves.py"))
poc = importlib.util.module_from_spec(spec); spec.loader.exec_module(poc)
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
COLUMN = "--column" in sys.argv[1:]
OUT = _args[0] if _args else "fig4_detection"

ORDER = ["SP", "A", "W", "S", "L"]                      # by isolation, high -> low
COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated slots 1-5
MARK = ["o", "s", "^", "D", "v"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

FS = 7 if COLUMN else 8.5          # base font
FS2 = 6.3 if COLUMN else 7.5       # annotations and legend
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": FS,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(3.5, 2.45) if COLUMN else (5.5, 3.4))
x = np.arange(len(poc.BINS))
ax.axhspan(0.0, 0.5, color="#f0efec", zorder=0)
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(len(x) - 0.55, 0.488, "chance", ha="right", va="top", color=INK2, fontsize=FS2)
ax.text(-0.35, 0.265, "below chance" if COLUMN else "below chance: scored lower than known material",
        ha="left", va="bottom", color=INK2, fontsize=FS2)

for c, mk, key in zip(COLOR, MARK, ORDER):
    iso, bins = poc.DATA[key]
    xs, ms, sds = [], [], []
    for i, b in enumerate(poc.BINS):
        if b in bins:
            v = np.asarray(bins[b], float)
            xs.append(i); ms.append(v.mean()); sds.append(v.std(ddof=1))
    ax.errorbar(xs, ms, yerr=sds, color=c, lw=1.3 if COLUMN else 1.6, marker=mk,
                ms=4.2 if COLUMN else 5.5,
                mec="white", mew=0.8, capsize=2, elinewidth=0.9, zorder=3,
                label=f"{poc.FULL[key]} ({iso:.1f}°)")

ax.set_xticks(x, [b + "%" for b in poc.BINS])
ax.set_xlim(-0.4, len(x) - 0.6)
ax.set_ylim(0.25, 0.93)
ax.set_xlabel("Held-out fibre fraction of the sample" if COLUMN
              else "Fraction of the sample made up of the held-out fibre")
ax.set_ylabel("Residual AUROC" if COLUMN else "AUROC of the reconstruction residual")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
leg = ax.legend(title="Held-out fibre (isolation)", loc="upper left", frameon=False,
                fontsize=FS2, title_fontsize=FS2, labelcolor=INK,
                handlelength=1.6 if COLUMN else 2.0, borderaxespad=0.2)
leg.get_title().set_color(INK2)
fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
print("wrote", OUT + ".{pdf,png}")
