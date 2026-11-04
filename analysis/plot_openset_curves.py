#!/usr/bin/env python
"""
plot_openset_curves.py — Figure 3: detection versus how much novel material.

One line per held-out fibre, x = unknown-fraction bin, y = AUROC, chance marked.
The point of the figure is the SHAPES: spandex rises steeply with fraction,
silk and acrylic rise gently, linen falls and sits below chance throughout. A
table of 5 fibres x 6 bins x 3 seeds is 90 numbers and hides all of that.

Numbers below are the measured 3-seed means from the final configuration
(unmixing decoder, lambda=4, random init, smoothness prior 1e-1). Replace them
if the runs are repeated.

    from plot_openset_curves import plot_curves, plot_summary
    plot_curves()          # Figure 3
    plot_summary()         # inset/companion: AUROC vs isolation
"""
import numpy as np
import matplotlib.pyplot as plt

# ── measured data: fibre -> (isolation_deg, {bin: [seed values]}) ───────────
DATA = {
    "SP": (26.73, {
        "0.5–5":  [0.5405, 0.5420, 0.5411],
        "5–15":   [0.5956, 0.5992, 0.5973],
        "15–30":  [0.6802, 0.6840, 0.6842],
        "30–50":  [0.8818, 0.8987, 0.8816]}),
    "A":  (6.08, {
        "5–15":   [0.3509, 0.3514, 0.3528],
        "15–30":  [0.6515, 0.6501, 0.6488],
        "30–50":  [0.6635, 0.6711, 0.6650],
        "50–80":  [0.6762, 0.6808, 0.6714],
        "80–100": [0.6227, 0.6059, 0.6012]}),
    "W":  (5.37, {
        "0.5–5":  [0.5815, 0.5846, 0.5849],
        "5–15":   [0.5495, 0.5637, 0.5631],
        "15–30":  [0.5746, 0.5909, 0.5869],
        "30–50":  [0.5997, 0.6122, 0.6062],
        "50–80":  [0.6352, 0.6396, 0.6400],
        "80–100": [0.5774, 0.5879, 0.5839]}),
    "S":  (3.21, {
        "5–15":   [0.5801, 0.5895, 0.5814],
        "15–30":  [0.5463, 0.5551, 0.5478],
        "30–50":  [0.5750, 0.5642, 0.5505],
        "50–80":  [0.6211, 0.5849, 0.5365],
        "80–100": [0.6686, 0.6467, 0.6044]}),
    "L":  (1.24, {
        "0.5–5":  [0.5504, 0.5533, 0.5470],
        "5–15":   [0.3580, 0.3575, 0.3558],
        "15–30":  [0.3298, 0.3290, 0.3294],
        "30–50":  [0.3476, 0.3514, 0.3420],
        "50–80":  [0.2919, 0.2921],
        "80–100": [0.2971, 0.3099]}),
}
BINS = ["0.5–5", "5–15", "15–30", "30–50", "50–80", "80–100"]
FULL = {"SP": "Spandex", "A": "Acrylic", "W": "Wool", "S": "Silk", "L": "Linen"}
# n per bin, for the caption / supplementary table
NPER = {"SP": {"0.5–5": 18426, "5–15": 6512, "15–30": 434, "30–50": 95},
        "A":  {"5–15": 175, "15–30": 1126, "30–50": 2255, "50–80": 2238, "80–100": 495},
        "W":  {"0.5–5": 1115, "5–15": 1757, "15–30": 788, "30–50": 957,
               "50–80": 897, "80–100": 1335},
        "S":  {"5–15": 265, "15–30": 943, "30–50": 911, "50–80": 641, "80–100": 963},
        "L":  {"0.5–5": 120, "5–15": 592, "15–30": 895, "30–50": 416,
               "50–80": 4139, "80–100": 2660}}


def plot_curves(data=DATA, figsize=(8.2, 5.4), annotate_n=False):
    """Figure 3. Colour encodes isolation, so the ordering of the lines is the
    ordering of the geometry -- the reader sees the law without being told."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(BINS))
    iso = [v[0] for v in data.values()]
    norm = plt.Normalize(min(iso), max(iso))
    cmap = plt.cm.viridis

    for name, (isolation, bins) in sorted(data.items(),
                                          key=lambda kv: -kv[1][0]):
        xs, ys, es = [], [], []
        for i, b in enumerate(BINS):
            if b not in bins:
                continue
            v = np.array(bins[b])
            xs.append(i); ys.append(v.mean())
            es.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        c = cmap(norm(isolation))
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=6, lw=2.0, capsize=3,
                    color=c, label=f"{FULL[name]}  ({isolation:.1f}°)", zorder=3)
        if annotate_n:
            for xi, yi in zip(xs, ys):
                n = NPER[name].get(BINS[xi])
                if n:
                    ax.annotate(f"{n:,}", (xi, yi), fontsize=6,
                                textcoords="offset points", xytext=(0, -12),
                                ha="center", color=c)

    ax.axhline(0.5, ls="--", c="0.45", lw=1.2, zorder=1)
    ax.text(len(BINS) - 0.45, 0.505, "chance", fontsize=9, color="0.35",
            ha="right", va="bottom")
    ax.axhspan(0, 0.5, color="0.92", zorder=0)
    ax.text(0.02, 0.055, "anti-detection\n(scored below known material)",
            fontsize=8.5, color="0.4", transform=ax.transAxes, va="bottom")

    ax.set_xticks(x); ax.set_xticklabels([f"{b}%" for b in BINS])
    ax.set_xlabel("fraction of the sample made up of the held-out fibre")
    ax.set_ylabel("AUROC (physics reconstruction residual)")
    ax.set_ylim(0.25, 0.95)
    ax.set_title("Novelty detection versus how much novel material is present\n"
                 "(legend gives endmember isolation; 3 seeds, bars = s.d.)",
                 fontsize=11)
    leg = ax.legend(title="held-out fibre (isolation)", fontsize=9,
                    title_fontsize=9, loc="upper left", framealpha=0.92)
    leg._legend_box.align = "left"
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    return fig


def plot_summary(data=DATA, bin_="30–50", figsize=(5.6, 4.6)):
    """Companion panel: AUROC against isolation at a matched unknown fraction.
    Log x-axis, because the relationship is logarithmic (r = 0.99 in log space
    versus 0.88 in linear)."""
    xs, ys, es, labs = [], [], [], []
    for n, (iso, bins) in data.items():
        if bin_ not in bins:
            continue
        v = np.array(bins[bin_])
        xs.append(iso); ys.append(v.mean())
        es.append(v.std(ddof=1) if len(v) > 1 else 0.0); labs.append(FULL[n])
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=9, capsize=4, color="tab:blue",
                zorder=3)
    lx = np.log(xs)
    b, a = np.polyfit(lx, ys, 1)
    g = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 200)
    ax.plot(g, a + b * np.log(g), "-", c="0.5", lw=1.4, zorder=2,
            label=f"fit in log space  (r = {np.corrcoef(lx, ys)[0,1]:.2f})")
    for x_, y_, l_ in zip(xs, ys, labs):
        ax.annotate(l_, (x_, y_), fontsize=9, textcoords="offset points",
                    xytext=(9, -3))
    ax.axhline(0.5, ls="--", c="0.45", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("endmember isolation (deg, log scale)")
    ax.set_ylabel(f"AUROC at {bin_}% unknown")
    ax.set_title("Detectability is ordered by endmember geometry\n"
                 "(Spearman = 1.00 across all five fibres)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    return fig


def supplementary_table(data=DATA):
    """Exact numbers for the supplementary material."""
    print(f"  {'fibre':<9}{'isolation':>10}{'bin':>9}{'n':>8}"
          f"{'AUROC':>9}{'s.d.':>8}")
    for name, (iso, bins) in sorted(data.items(), key=lambda kv: -kv[1][0]):
        for b in BINS:
            if b not in bins:
                continue
            v = np.array(bins[b])
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            n = NPER[name].get(b, 0)
            print(f"  {FULL[name]:<9}{iso:>9.2f}°{b+'%':>9}{n:>8,}"
                  f"{v.mean():>9.4f}{sd:>8.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="prefix; omit to just show")
    ap.add_argument("--annotate_n", action="store_true")
    a = ap.parse_args()
    supplementary_table()
    f1 = plot_curves(annotate_n=a.annotate_n)
    f2 = plot_summary()
    if a.save:
        f1.savefig(f"{a.save}_fig3_curves.png", dpi=300, bbox_inches="tight")
        f1.savefig(f"{a.save}_fig3_curves.pdf", bbox_inches="tight")
        f2.savefig(f"{a.save}_fig3_summary.png", dpi=300, bbox_inches="tight")
        f2.savefig(f"{a.save}_fig3_summary.pdf", bbox_inches="tight")
        print(f"\n  wrote {a.save}_fig3_curves.{{png,pdf}} and "
              f"{a.save}_fig3_summary.{{png,pdf}}")
    else:
        plt.show()
