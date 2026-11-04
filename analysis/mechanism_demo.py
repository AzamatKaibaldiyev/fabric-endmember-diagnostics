#!/usr/bin/env python
"""
mechanism_demo.py — what do the labels actually MEAN on a plot?

Generates pairs of synthetic spectra that differ in exactly one controlled way,
then reports how each metric responds. Run it, look at the panels, then compare
against your real endmember figure: whichever synthetic panel your fibre
resembles tells you what its label should be.

    from mechanism_demo import demo
    demo()                 # all cases, plots + table
    demo(plot=False)       # table only

The four metrics:
    SAM raw        angle between the spectra as measured. Sensitive to
                   everything: level, scale, slope, curvature, peaks.
    SAM SNV        after (x - mean)/std. Removes LEVEL and SCALE.
                   Still sensitive to SLOPE and CURVATURE.
    SAM detrend1   after removing a fitted straight line. Also removes SLOPE.
    SAM detrend2   after removing a fitted quadratic. Also removes CURVATURE.
                   This is what EMSC does in chemometrics.

Reading the table:
    big raw, small SNV                 -> only the overall level differs
    big SNV, small detrend1            -> baseline SLOPE differs (scatter)
    big detrend1, small detrend2       -> baseline CURVATURE differs (scatter)
    big detrend2                       -> the ABSORPTION PEAKS genuinely differ
    all big on a low-contrast spectrum -> too little structure to judge
"""
import numpy as np
import matplotlib.pyplot as plt

WL = np.linspace(960, 1660, 200)


def sam(u, v):
    u = np.asarray(u, float).ravel(); v = np.asarray(v, float).ravel()
    return float(np.degrees(np.arccos(
        np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def snv(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def detrend(v, order=1, wl=WL):
    v = np.asarray(v, float)
    x = (wl - wl.mean()) / wl.std()
    return v - np.polyval(np.polyfit(x, v, order), x)


def contrast(v):
    v = np.asarray(v, float)
    return float((v.max() - v.min()) / (abs(v.mean()) + 1e-12))


def base(peaks=((1480, 45, 0.16), (1200, 35, 0.05)), floor=0.16):
    """A cellulosic-like spectrum: flat floor plus O-H style absorption peaks."""
    y = np.full_like(WL, floor)
    for c, w, h in peaks:
        y = y + h * np.exp(-0.5 * ((WL - c) / w) ** 2)
    return y


def cases():
    a = base()
    x = (WL - WL.mean()) / WL.std()
    return [
        ("identical", a, a.copy(),
         "the ideal: every metric near zero"),
        ("level offset", a, a + 0.08,
         "raw sees it; SNV removes it entirely"),
        ("scale x1.4", a, a * 1.4,
         "SAM is scale-invariant, so even raw ignores this"),
        ("baseline SLOPE", a, a + 0.05 * x,
         "SNV large, detrend1 kills it -> SCATTER, not chemistry"),
        ("baseline CURVATURE", a, a + 0.04 * x ** 2,
         "SNV and detrend1 large, detrend2 kills it -> still SCATTER"),
        ("peak SHIFTED 30nm", a, base(((1510, 45, 0.16), (1200, 35, 0.05))),
         "survives every correction -> PEAKS GENUINELY DIFFER"),
        ("peak HEIGHT halved", a, base(((1480, 45, 0.08), (1200, 35, 0.05))),
         "survives every correction -> PEAKS GENUINELY DIFFER"),
        ("featureless + noise", np.full_like(WL, 0.30) + 0.004 * np.sin(WL / 7),
         np.full_like(WL, 0.30) + 0.004 * np.cos(WL / 9),
         "tiny contrast: every normalised metric explodes on nothing"),
    ]


def demo(plot=True):
    rows = []
    print(f"  {'case':<22}{'contrast':>9}{'SAM raw':>9}{'SAM SNV':>9}"
          f"{'detrend1':>10}{'detrend2':>10}   verdict")
    print("  " + "-" * 96)
    for name, a, b, _ in cases():
        r = sam(a, b)
        s = sam(snv(a), snv(b))
        d1 = sam(snv(detrend(a, 1)), snv(detrend(b, 1)))
        d2 = sam(snv(detrend(a, 2)), snv(detrend(b, 2)))
        c = contrast(a)
        if c < 0.25:
            v = "featureless - not assessable"
        elif s < 3:
            v = "recovered"
        elif d1 < 3:
            v = "scatter: baseline SLOPE"
        elif d2 < 3:
            v = "scatter: baseline CURVATURE"
        else:
            v = "PEAKS DIFFER"
        print(f"  {name:<22}{c:>9.2f}{r:>9.2f}{s:>9.2f}{d1:>10.2f}{d2:>10.2f}   {v}")
        rows.append((name, a, b, r, s, d1, d2, v))

    print("\n  How to use this against your real figure:")
    print("    find the synthetic panel your fibre's red/black pair resembles;")
    print("    that panel's verdict is the label your fibre should carry.")

    if not plot:
        return rows
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.0), squeeze=False)
    for i, (name, a, b, r, s, d1, d2, v) in enumerate(rows):
        ax = axes[0][i]
        ax.plot(WL, a, "k", lw=1.8, label="reference")
        ax.plot(WL, b, "tab:red", lw=1.5, label="comparison")
        ax.set_title(f"{name}\nraw {r:.1f}°  SNV {s:.1f}°", fontsize=9)
        if i == 0:
            ax.set_ylabel("raw absorbance"); ax.legend(fontsize=7)
        ax2 = axes[1][i]
        ax2.plot(WL, snv(a), "k", lw=1.8)
        ax2.plot(WL, snv(b), "tab:red", lw=1.5)
        ax2.set_title(f"detr1 {d1:.1f}°  detr2 {d2:.1f}°", fontsize=9)
        ax2.set_xlabel("wavelength (nm)")
        if i == 0:
            ax2.set_ylabel("SNV")
        for a_ in (ax, ax2):
            a_.tick_params(labelsize=7)
    fig.suptitle("What each disagreement mechanism looks like "
                 "(top: raw, bottom: SNV)", fontsize=12)
    fig.tight_layout()
    return rows, fig


if __name__ == "__main__":
    out = demo(plot=True)
    plt.show()
