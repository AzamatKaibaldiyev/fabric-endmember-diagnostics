#!/usr/bin/env python
"""
plot_endmembers.py — the interpretability figure.

For each fibre, overlays:
    thin grey   : individual measured PURE samples (shows natural variability)
    black       : mean of those pure samples          (the reference)
    coloured    : the LEARNED endmember               (mean over seeds)
    shaded band : min-max across seeds                (reproducibility)

Two views:
    RAW  — the space the reconstruction loss actually operates in (primary)
    SNV  — shape only, scatter and baseline offset removed (supplementary)

Works from a notebook:

    from plot_endmembers import load_all, plot_grid, plot_fibre
    D = load_all("train.py", "/path/to/data",
                 ["r0.pt", "r1.pt", "r2.pt"])
    plot_grid(D)                 # all 12, raw
    plot_grid(D, snv=True)       # all 12, shape-only
    plot_fibre(D, "CA")          # one fibre, large, both channels

or from the shell:

    python plot_endmembers.py --script train.py --data_root <DIR> \
        --ckpt r0.pt r1.pt r2.pt
"""
import importlib.util, os, sys, types
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, not Type 3 (IEEE PDF eXpress)

# Average centre wavelengths of the 200 retained DLP2010NIR points, from the
# FabricSpectra authors. Non-uniform: ~4.3 nm spacing at the blue end shrinking
# to ~3.0 nm at the red end.
WAVELENGTHS_NM = np.array([
    961.2, 965.48, 969.32, 973.26, 977.35, 981.18, 985.16, 989.06, 992.88, 996.7,
    1000.87, 1004.75, 1008.85, 1012.69, 1016.61, 1020.41, 1024.31, 1028.23, 1032.01, 1036.02,
    1039.85, 1043.77, 1047.65, 1051.41, 1055.29, 1059.28, 1063.14, 1066.88, 1070.88, 1074.61,
    1078.56, 1082.32, 1086.03, 1089.74, 1093.55, 1097.25, 1101.26, 1105.32, 1109.02, 1112.85,
    1116.69, 1120.48, 1124.14, 1127.81, 1131.47, 1135.12, 1139.23, 1142.95, 1146.68, 1150.53,
    1154.39, 1158.06, 1161.68, 1165.3, 1168.92, 1172.86, 1176.66, 1180.46, 1184.06, 1187.77,
    1191.37, 1195.09, 1198.67, 1202.28, 1206.07, 1209.89, 1213.64, 1217.2, 1220.92, 1224.53,
    1228.08, 1231.73, 1235.28, 1239.03, 1242.8, 1246.5, 1250.02, 1253.59, 1257.28, 1260.86,
    1264.36, 1267.85, 1271.63, 1275.23, 1278.79, 1282.41, 1286.17, 1289.65, 1293.18, 1296.76,
    1300.21, 1303.95, 1307.4, 1310.99, 1314.53, 1317.97, 1321.45, 1324.98, 1328.4, 1332.16,
    1335.85, 1339.26, 1342.85, 1346.25, 1349.75, 1353.18, 1356.58, 1360.03, 1363.51, 1367.16,
    1370.64, 1374.2, 1377.56, 1380.92, 1384.42, 1387.86, 1391.21, 1394.78, 1398.33, 1401.7,
    1405.1, 1408.53, 1411.91, 1415.26, 1418.79, 1422.1, 1425.53, 1429.03, 1432.31, 1435.68,
    1438.96, 1442.38, 1445.9, 1449.17, 1452.53, 1456.05, 1459.56, 1462.81, 1466.13, 1469.37,
    1472.61, 1475.89, 1479.26, 1482.58, 1486.06, 1489.28, 1492.55, 1495.83, 1499.34, 1502.54,
    1505.8, 1509.03, 1512.39, 1515.86, 1519.07, 1522.31, 1525.54, 1528.7, 1531.86, 1535.01,
    1538.2, 1541.53, 1544.86, 1548.31, 1551.53, 1554.88, 1558.0, 1561.12, 1564.24, 1567.39,
    1570.55, 1573.84, 1576.94, 1580.24, 1583.5, 1586.65, 1589.76, 1592.83, 1596.04, 1599.23,
    1602.48, 1605.63, 1608.68, 1611.88, 1615.09, 1618.13, 1621.28, 1624.41, 1627.47, 1630.68,
    1633.69, 1636.71, 1639.87, 1643.03, 1646.09, 1649.18, 1652.23, 1655.36, 1658.63, 1661.64])

CH_NAMES = {0: "absorbance", 1: "reflectance -> absorbance"}


def _load_module(path, name="_plot_mod"):
    if "wandb" not in sys.modules:
        wb = types.ModuleType("wandb")
        for fn in ("init", "log", "finish"):
            setattr(wb, fn, lambda *a, **k: None)
        wb.run = None
        sys.modules["wandb"] = wb
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detrend(v, wl=None, order=1):
    """Remove a linear baseline (offset AND slope) fitted against wavelength.

    SNV removes offset and scale but NOT slope, so two spectra with identical
    absorption peaks sitting on different baseline tilts still produce a large
    spectral angle. In NIR that tilt is a SCATTER signature (particle size,
    surface texture), not chemistry -- so comparing after detrending separates
    'the peaks disagree' from 'the baseline slope disagrees'."""
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(wl if wl is not None else np.arange(len(v)), dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    c = np.polyfit(x, v, order)
    return v - np.polyval(c, x)


def snv(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    m = v.mean(axis=axis, keepdims=True)
    s = v.std(axis=axis, keepdims=True)
    return (v - m) / (s + 1e-9)


def sam_deg(u, v):
    u = np.asarray(u).ravel(); v = np.asarray(v).ravel()
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(u @ v / d, -1.0, 1.0))))


def load_all(script, data_root, ckpts, channels=2, channel_idx=1,
             purity=99.0, n_examples=25, max_scan=120000, split="train"):
    """Gather everything the plots need: learned endmembers (per seed), the
    measured pure-sample mean per fibre, and a sample of individual pure
    spectra for the grey background."""
    import torch
    M = _load_module(script)
    ds = M.NIRDataset(
        os.path.join(data_root, f"{split}_data.npy"),
        os.path.join(data_root, f"{split}_multi_label_classification_label.npy"),
        os.path.join(data_root, f"{split}_composition_label.npy"),
        channels=channels, channel_idx=channel_idx)
    refl = M.reflectance_channel_indices(channels, channel_idx)

    Es = []
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        key = next((k for k in sd if k.endswith("decoder.endmembers")), None)
        if key is None:
            print(f"  !! {os.path.basename(p)}: no endmembers, skipped")
            continue
        Es.append(torch.nn.functional.softplus(sd[key]).numpy())
    if not Es:
        raise RuntimeError("no checkpoint contained decoder.endmembers")
    Es = np.stack(Es)                                   # (S, K, C, L)
    K = Es.shape[1]
    names = list(M.FIBER_NAMES)[:K]

    # measured pure samples, converted to the SAME space the decoder targets
    idx = np.arange(min(max_scan, len(ds.X)))
    comps = np.asarray(ds.comps)[idx]
    pure_mean = np.zeros_like(Es[0])
    pure_examples, n_pure = {}, np.zeros(K, dtype=int)
    for k in range(K):
        sel = idx[comps[:, k] >= purity]
        n_pure[k] = len(sel)
        if len(sel) == 0:
            pure_examples[names[k]] = None
            continue
        take = sel[:min(len(sel), 400)]
        raw = np.stack([ds.X[i] for i in take]).astype(np.float32)
        t = torch.from_numpy(raw)
        t = M.to_absorbance(t, refl)
        arr = t.numpy()
        pure_mean[k] = arr.mean(axis=0)
        rs = np.random.RandomState(0)
        pick = rs.choice(len(arr), min(n_examples, len(arr)), replace=False)
        pure_examples[names[k]] = arr[pick]

    L = Es.shape[-1]
    wl = WAVELENGTHS_NM if L == len(WAVELENGTHS_NM) else np.arange(L)
    if L != len(WAVELENGTHS_NM):
        print(f"  note: {L} points != 200, using index on the x-axis")
    print(f"  loaded {Es.shape[0]} seed(s), {K} fibres, {Es.shape[2]} channel(s)")
    print(f"  pure-sample counts: " +
          "  ".join(f"{names[k]}:{n_pure[k]}" for k in range(K)))
    return dict(E=Es, pure=pure_mean, examples=pure_examples, names=names,
                n_pure=n_pure, wl=wl, K=K, C=Es.shape[2])


def _panel(ax, D, k, ch, use_snv, show_examples=True, legend=False,
           title_fs=10, tick_fs=8, lw_scale=1.0, compact=False, bw=False):
    names, wl = D["names"], D["wl"]
    Es, pure = D["E"], D["pure"]
    learned = Es[:, k, ch]                       # (S, L)
    ref = pure[k, ch]
    has_ref = D["n_pure"][k] > 0

    f = (lambda v: snv(v)) if use_snv else (lambda v: np.asarray(v, float))

    if show_examples and D["examples"][names[k]] is not None:
        for ex in D["examples"][names[k]]:
            ax.plot(wl, f(ex[ch]), color="0.75", lw=0.5 * lw_scale, alpha=0.5, zorder=1)
    if has_ref:
        ax.plot(wl, f(ref), color="k", lw=2.0 * lw_scale, zorder=3,
                label="measured pure (mean)")
    lm = f(learned.mean(0))
    if learned.shape[0] > 1:
        lo = np.min([f(learned[s]) for s in range(learned.shape[0])], axis=0)
        hi = np.max([f(learned[s]) for s in range(learned.shape[0])], axis=0)
        ax.fill_between(wl, lo, hi, color="tab:red", alpha=0.25, zorder=2,
                        label="learned, seed range")
    ax.plot(wl, lm, color="tab:red", lw=1.8 * lw_scale, zorder=4, label="learned endmember",
            ls=(0, (3, 1.5)) if bw else "-")   # dashed survives greyscale printing

    if has_ref:
        s_raw = sam_deg(learned.mean(0), ref)
        s_snv = sam_deg(snv(learned.mean(0)), snv(ref))
        t = (f"{names[k]}  n={D['n_pure'][k]}  {s_snv if use_snv else s_raw:.2f}°" if compact
             else f"{names[k]}  n={D['n_pure'][k]}   SAM {s_snv if use_snv else s_raw:.2f}°")
    else:
        t = (f"{names[k]}  never pure" if compact
             else f"{names[k]}  n=0  (never pure — no reference)")
    ax.set_title(t, fontsize=title_fs)
    if compact:   # narrow panels: label every other tick so the numbers do not collide
        ax.set_xticks([1000, 1200, 1400, 1600], ["1000", "", "1400", ""])
    ax.tick_params(labelsize=tick_fs)
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.9)


def plot_grid(D, ch=0, snv=False, show_examples=True, ncols=4, figsize=None,
              paper=False, fibres=None, width=None, bw=False):
    """All fibres in a grid. ch=0 absorbance, ch=1 reflectance->absorbance.

    paper=True: journal size (7.2 in wide, 3 columns), labels readable at print
    size, no suptitle (the caption carries it), one shared legend on top.
    width (inches) overrides the paper width; below 6 in, panel titles and
    fonts are compacted (e.g. 4.8 in = ESANN's 12.2 cm text width).
    bw=True draws the learned endmember dashed so it survives greyscale print."""
    idx = ([D["names"].index(f) for f in fibres] if fibres
           else list(range(D["K"])))
    K = len(idx)
    if paper:
        ncols = min(K, 4) if fibres else 3
    nrows = int(np.ceil(K / ncols))
    compact = bool(paper and width and width < 6)
    if paper:
        w = width or 7.2
        h_row = (2.05 if not fibres else 1.95) * (w / 7.2) ** 0.5 if not compact else 1.5
        figsize = figsize or (w, h_row * nrows)
        fs = (dict(title_fs=6.3, tick_fs=5.5, lw_scale=0.6, compact=True, bw=bw) if compact
              else dict(title_fs=7.5, tick_fs=6.5, lw_scale=0.7, bw=bw))
    else:
        figsize = figsize or (4.2 * ncols, 2.9 * nrows)
        fs = {}
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for j, k in enumerate(idx):
        _panel(axes[j], D, k, ch, snv, show_examples,
               legend=(j == 0 and not paper), **fs)
    for j in range(K, len(axes)):
        axes[j].axis("off")
    lab_fs = (6 if compact else 7) if paper else 9
    for j in range(max(0, len(axes) - ncols), len(axes)):
        axes[j].set_xlabel("wavelength (nm)", fontsize=lab_fs)
    lab = "SNV (shape only)" if snv else "absorbance"
    for j in range(0, len(axes), ncols):
        axes[j].set_ylabel(lab, fontsize=lab_fs)
    if paper:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([], [], color="0.75", lw=0.8, label="measured pure spectra"),
                   Line2D([], [], color="k", lw=1.4, label="measured mean"),
                   Line2D([], [], color="tab:red", lw=1.3, label="learned endmember",
                          ls=(0, (3, 1.5)) if bw else "-"),
                   Patch(facecolor="tab:red", alpha=0.25, label="seed range")]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   fontsize=6 if compact else 7, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.8 if compact else 2.0, columnspacing=1.2 if compact else 2.0)
        fig.tight_layout(rect=[0, 0, 1, 0.975 if nrows > 1 else (0.89 if compact else 0.9)],
                         w_pad=0.4 if compact else None)
        return fig
    fig.suptitle(
        f"Learned endmembers vs measured pure fibre spectra — "
        f"channel {ch} ({CH_NAMES.get(ch, ch)}), {'SNV' if snv else 'raw'} space",
        fontsize=12, y=1.002)
    fig.tight_layout()
    return fig


def plot_fibre(D, name, show_examples=True, figsize=(11, 4)):
    """One fibre, raw and SNV side by side, all channels stacked."""
    k = D["names"].index(name)
    C = D["C"]
    fig, axes = plt.subplots(C, 2, figsize=(figsize[0], figsize[1] * C),
                             squeeze=False)
    for ch in range(C):
        _panel(axes[ch][0], D, k, ch, False, show_examples, legend=(ch == 0))
        _panel(axes[ch][1], D, k, ch, True, show_examples)
        axes[ch][0].set_ylabel(f"ch{ch} raw", fontsize=9)
        axes[ch][1].set_ylabel(f"ch{ch} SNV", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (nm)", fontsize=9)
    fig.suptitle(f"{name}: learned endmember vs measured pure spectra",
                 fontsize=12, y=1.001)
    fig.tight_layout()
    return fig


def fibre_metrics(D):
    """Per-fibre numbers behind the diagnosis: shape agreement (SNV SAM),
    reproducibility (cross-seed SAM), and how FEATUREFUL the measured spectrum
    is in this window. A flat spectrum carries little shape to learn AND gets
    its small wiggles amplified by SNV, so contrast must be reported next to
    the SAM or the SNV number is easy to over-read."""
    import itertools
    K, Es, pure = D["K"], D["E"], D["pure"]
    out = {}
    for k in range(K):
        nm = D["names"][k]
        pairs = [sam_deg(Es[i, k], Es[j, k])
                 for i, j in itertools.combinations(range(Es.shape[0]), 2)]
        cs = float(np.mean(pairs)) if pairs else float("nan")
        if D["n_pure"][k] > 0:
            lm, rf = Es[:, k].mean(0), pure[k]
            raw_sam = sam_deg(lm, rf)
            snv_sam = sam_deg(snv(lm), snv(rf))
            # detrended: offset AND slope removed, per channel, vs wavelength
            wl = D["wl"]
            dl = np.stack([detrend(lm[c], wl, 1) for c in range(lm.shape[0])])
            dr = np.stack([detrend(rf[c], wl, 1) for c in range(rf.shape[0])])
            det_sam = sam_deg(snv(dl), snv(dr))
            # QUADRATIC detrend also removes baseline CURVATURE. A linear
            # detrend can make a curved disagreement look WORSE, because
            # subtracting a straight line from a curve leaves more relative
            # difference, not less. This is what EMSC removes in chemometrics.
            ql = np.stack([detrend(lm[c], wl, 2) for c in range(lm.shape[0])])
            qr = np.stack([detrend(rf[c], wl, 2) for c in range(rf.shape[0])])
            det2_sam = sam_deg(snv(ql), snv(qr))
            # NATURAL SPREAD: how much do individual measured pure samples of
            # this fibre differ from EACH OTHER? That is the scale against
            # which any disagreement should be judged -- an endmember closer to
            # the mean than real samples are to each other is indistinguishable
            # from a real sample, and no fixed threshold has to be invented.
            # NATURAL SPREAD must be measured in the SAME space as the metric it
            # judges. SNV divides by the standard deviation, so for a flat
            # spectrum two identical samples measured twice look far apart once
            # normalised -- 20x further than in raw space. Comparing an SNV
            # disagreement against a RAW-space bar is therefore far too strict
            # for exactly the low-contrast fibres, which is the case that matters.
            ex = D["examples"][nm]
            if ex is not None and len(ex) >= 3:
                m = min(len(ex), 12)
                pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
                natural = float(np.median(
                    [sam_deg(ex[i, 0], ex[j, 0]) for i, j in pairs]))
                natural_snv = float(np.median(
                    [sam_deg(snv(ex[i, 0]), snv(ex[j, 0])) for i, j in pairs]))
                natural_d1 = float(np.median(
                    [sam_deg(snv(detrend(ex[i, 0], wl, 1)),
                             snv(detrend(ex[j, 0], wl, 1))) for i, j in pairs]))
                natural_d2 = float(np.median(
                    [sam_deg(snv(detrend(ex[i, 0], wl, 2)),
                             snv(detrend(ex[j, 0], wl, 2))) for i, j in pairs]))
                # Uncertainty on the REFERENCE itself. The reference is a mean
                # over n_pure samples that scatter by `natural`, so its standard
                # error is natural/sqrt(n_pure). For a fibre with 90 pure
                # samples this is an order of magnitude larger than for one with
                # 13,717, and a ratio near 1 should be read with that in mind.
                ref_se = natural / np.sqrt(max(D["n_pure"][k], 1))
            else:
                natural = natural_snv = natural_d1 = natural_d2 = float("nan")
                ref_se = float("nan")
            ref = pure[k, 0]
            contrast = float((ref.max() - ref.min()) / (abs(ref.mean()) + 1e-9))
        else:
            raw_sam = snv_sam = det_sam = det2_sam = contrast = float("nan")
            natural = natural_snv = natural_d1 = natural_d2 = float("nan")
            ref_se = float("nan")
        # Order matters. A featureless spectrum has almost no structure to
        # compare, so BOTH SNV and detrended SAM become unreliable there and we
        # must say so rather than declare a disagreement. A pure baseline tilt
        # is a SCATTER signature, not a chemical error, so it counts as
        # recovered-with-a-caveat.
        # Thresholds are DATA-DRIVEN where possible: `natural` is the median
        # angle between two real pure samples of the same fibre, so it is the
        # spread the material itself shows. Fixed fallbacks are used only when
        # too few pure samples exist to estimate it.
        # each metric is judged against the natural spread IN ITS OWN SPACE.
        # If the natural spread is itself huge, the comparison has no power:
        # two REAL samples of that fibre already disagree by more than any
        # plausible model error, so no conclusion can be drawn in that space.
        # Acrylic is the case -- natSNV 33 deg -- and a 6 deg shift in the
        # measurement flips its label without the spectra meaningfully changing.
        nn = lambda v, fb: (v if not np.isnan(v) else fb)
        ratio = snv_sam / natural_snv if (not np.isnan(snv_sam) and
                                          not np.isnan(natural_snv) and
                                          natural_snv > 1e-9) else float("nan")
        if D["n_pure"][k] == 0:
            mech = "no reference"
        elif not np.isnan(natural_snv) and natural_snv > 20.0:
            mech = ("recovered (raw); shape not assessable"
                    if (not np.isnan(raw_sam) and not np.isnan(natural)
                        and raw_sam <= 2 * natural)
                    else "shape not assessable")
        elif cs > nn(natural, 3.0):
            mech = "not identifiable"
        elif snv_sam <= nn(natural_snv, 10.0):
            mech = "recovered"
        elif det_sam <= nn(natural_d1, 10.0):
            mech = "recovered, scatter slope"
        elif det2_sam <= nn(natural_d2, 10.0):
            mech = "recovered, scatter curvature"
        else:
            mech = "peaks differ"
        # --- simple verdict: reproducibility, then agreement in RAW space ---
        raw_ratio = (raw_sam / natural if (not np.isnan(raw_sam)
                     and not np.isnan(natural) and natural > 1e-9)
                     else float("nan"))
        if D["n_pure"][k] == 0:
            simple = "no reference"
        elif not np.isnan(natural) and cs > natural:
            simple = "not identifiable"
        elif np.isnan(raw_ratio):
            simple = "no reference"
        elif raw_ratio <= 1.0:
            simple = "recovered"
        elif raw_ratio <= 1.3:
            simple = "marginal"
        else:
            simple = "not recovered"
        # --- SNV note: informative only where the SNV reference is stable ---
        if np.isnan(natural_snv):
            note = ""
        elif natural_snv > 20:
            note = "SNV reference unstable"
        elif snv_sam <= natural_snv:
            note = "shape agrees"
        elif det_sam <= natural_d1:
            note = "shape differs by baseline slope"
        elif det2_sam <= natural_d2:
            note = "shape differs by baseline curvature"
        else:
            note = "shape differs"
        out[nm] = dict(simple=simple, raw_ratio=raw_ratio, note=note,
                       ref_se=ref_se,
                       raw_sam=raw_sam, snv_sam=snv_sam, det_sam=det_sam,
                       det2_sam=det2_sam, natural=natural,
                       natural_snv=natural_snv, natural_d1=natural_d1,
                       natural_d2=natural_d2, ratio=ratio, cross_seed=cs,
                       contrast=contrast, n_pure=int(D["n_pure"][k]), mech=mech)
    return out


MECH_COLOR = {"recovered": "tab:blue", "recovered, scatter slope": "tab:cyan",
              "recovered, scatter curvature": "tab:green",
              "recovered (raw); shape not assessable": "tab:olive",
              "shape not assessable": "tab:brown",
              "featureless (not assessable)": "tab:purple",
              "peaks differ": "tab:orange", "not identifiable": "tab:red",
              "no reference": "0.5"}
# SIMPLE scheme (default, used in the paper). Recovery is judged in RAW space:
# it is the space the reconstruction loss optimises AND the only space with a
# stable reference for every fibre. SNV analysis is reported as a separate NOTE
# column rather than as a verdict, because for several fibres the SNV reference
# is unusable (see natSNV).
MECH_SIMPLE = ["recovered", "marginal", "not recovered",
               "not identifiable", "no reference"]
MECH_SIMPLE_COLOR = {"recovered": "tab:blue", "marginal": "tab:cyan",
                     "not recovered": "tab:orange",
                     "not identifiable": "tab:red", "no reference": "0.5"}

MECH_ORDER = ["recovered", "recovered, scatter slope",
              "recovered, scatter curvature",
              "recovered (raw); shape not assessable",
              "shape not assessable", "featureless (not assessable)",
              "peaks differ", "not identifiable", "no reference"]


def plot_seed_spread(D, ch=0, figsize=(8, 4.2), by_mechanism=True, paper=False):
    """Cross-seed spectral angle per fibre — the reproducibility bar chart."""
    if D["E"].shape[0] < 2:
        print("  need >=2 seeds for the spread plot"); return None
    Mt = fibre_metrics(D)
    names = sorted(Mt, key=lambda n: Mt[n]["cross_seed"])
    vals = [Mt[n]["cross_seed"] for n in names]
    cols = ([MECH_SIMPLE_COLOR[Mt[n]["simple"]] for n in names] if by_mechanism
            else ["tab:red" if v > 3 else "tab:blue" for v in vals])
    if paper:
        figsize = (5.5, 2.6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, vals, color=cols)
    _nat = [Mt[n]["natural"] for n in names if not np.isnan(Mt[n]["natural"])]
    _bar = float(np.median(_nat)) if _nat else 3.0
    ax.axhline(_bar, ls="--", c="0.4", lw=1)
    if paper:
        ax.set_ylim(0, max(_bar, max(vals)) * 1.2)
    ax.text(0.01, _bar * 1.05,
            f"median natural spread between real pure samples ({_bar:.1f}$^\\circ$)",
            fontsize=7 if paper else 8, color="0.3", transform=ax.get_yaxis_transform())
    ax.set_ylabel("mean cross-seed SAM (deg)", fontsize=8 if paper else None)
    if paper:
        ax.tick_params(labelsize=7)
    else:
        ax.set_title("Endmember reproducibility across seeds\n"
                     "(low bar = the data determines it; colour = recovery verdict)")
    if by_mechanism:
        from matplotlib.patches import Patch
        used = [m for m in MECH_SIMPLE if any(Mt[n]["simple"] == m for n in names)]
        ax.legend(handles=[Patch(facecolor=MECH_SIMPLE_COLOR[m], label=m) for m in used],
                  fontsize=7 if paper else 8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_diagnostic_map(D, figsize=(7.5, 5.5)):
    """The two failure modes as a 2-D map.
        x = reproducibility  (cross-seed SAM)
        y = shape agreement  (SNV SAM vs the measured pure spectrum)
    bottom-left  = recovered
    top-left     = the runs AGREE with each other but not with reality
    right        = the runs do not even agree with each other
    Marker size encodes spectral contrast: small marker = featureless spectrum
    in this window, which is the usual reason for a top-left point."""
    Mt = fibre_metrics(D)
    fig, ax = plt.subplots(figsize=figsize)
    for n, m in Mt.items():
        if np.isnan(m["snv_sam"]):
            continue
        sz = 40 + 400 * min(m["contrast"], 1.5) / 1.5
        ax.scatter(m["cross_seed"], m["snv_sam"], s=sz,
                   color=MECH_SIMPLE_COLOR[m["simple"]], alpha=0.75,
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(n, (m["cross_seed"], m["snv_sam"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=10)
    _nx = [m["natural"] for m in Mt.values() if not np.isnan(m["natural"])]
    _ny = [m.get("natural_snv", np.nan) for m in Mt.values()]
    _ny = [v for v in _ny if not np.isnan(v)]
    ax.axvline(float(np.median(_nx)) if _nx else 3.0, ls="--", c="0.5", lw=1)
    ax.axhline(float(np.median(_ny)) if _ny else 10.0, ls="--", c="0.5", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("cross-seed SAM (deg)   ->   runs disagree with EACH OTHER")
    ax.set_ylabel("SNV SAM vs measured (deg)   ->   disagrees with REALITY")
    ax.set_title("Two distinct failure modes\n"
                 "(marker size = spectral contrast; small = featureless here)")
    from matplotlib.patches import Patch
    used = [m for m in MECH_SIMPLE if any(v["simple"] == m for v in Mt.values())]
    ax.legend(handles=[Patch(facecolor=MECH_SIMPLE_COLOR[m], label=m) for m in used],
              fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def print_metrics(D):
    """Text table behind the figures — paste straight into the paper."""
    Mt = fibre_metrics(D)
    print(f"  {'fibre':<6}{'n_pure':>7}{'raw':>7}{'nat':>7}{'ratio':>7}"
          f"{'+-ref':>7}{'x-seed':>8}{'SNV':>7}{'natSNV':>8}  "
          f"{'verdict':<17}SNV note")
    for n in sorted(Mt, key=lambda z: (Mt[z]["mech"], -Mt[z]["snv_sam"]
                                       if not np.isnan(Mt[z]["snv_sam"]) else 0)):
        m = Mt[n]
        f = lambda v: "    n/a" if np.isnan(v) else f"{v:7.2f}"
        g = lambda v: "    n/a" if np.isnan(v) else f"{v:6.2f}"
        h = lambda v: "    n/a" if np.isnan(v) else f"{v:6.2f}"
        print(f"  {n:<6}{m['n_pure']:>7}{h(m['raw_sam'])}"
              f"{h(m.get('natural', float('nan')))}"
              f"{h(m.get('raw_ratio', float('nan')))}"
              f"{h(m.get('ref_se', float('nan')))}"
              f"{m['cross_seed']:>8.2f}{h(m['snv_sam'])}"
              f"{h(m.get('natural_snv', float('nan'))):>8}  "
              f"{m.get('simple',''):<17}{m.get('note','')}")
    ok = [m for m in Mt.values() if not np.isnan(m["snv_sam"])]
    print(f"\n  mean SAM raw {np.mean([m['raw_sam'] for m in ok]):.2f}°   "
          f"mean SAM SNV {np.mean([m['snv_sam'] for m in ok]):.2f}°")
    print(f"  mean SAM detrend1 {np.mean([m['det_sam'] for m in ok]):.2f}°   "
          f"detrend2 {np.mean([m['det2_sam'] for m in ok]):.2f}°")
    nat = [m['natural'] for m in ok if not np.isnan(m['natural'])]
    if nat:
        print(f"  median NATURAL spread between two real pure samples: "
              f"{np.median(nat):.2f}° -- this is the scale disagreements are "
              f"judged against, instead of an invented threshold.")
    print("\n  +-ref = standard error of the REFERENCE mean (natural/sqrt(n_pure)).")
    print("  Fibres with few pure samples have a noisy reference, so a ratio near")
    print("  1 there carries real uncertainty; for cotton it does not.")
    print("\n  RATIO = SNV SAM / natural SNV spread. Below 1 means the endmember is")
    print("  closer to the measured mean than two real samples are to each other.")
    print("  Values near 1 are MARGINAL: the categorical label there rests on a")
    print("  crossing of a fraction of a degree and should not be over-read.")
    print("  RAW is the primary number: it is the space the reconstruction loss")
    print("  optimises. SNV removes offset and scale. DETRENDED also removes the")
    print("  baseline SLOPE, so a large SNV SAM with a small detrended SAM means")
    print("  the absorption peaks agree and only the scatter baseline differs.")
    from collections import Counter
    cnt = Counter(m["simple"] for m in Mt.values())
    ref = len(Mt) - cnt.get("no reference", 0)
    ok = cnt.get("recovered", 0) + cnt.get("marginal", 0)
    print(f"\n  RECOVERY (raw space, judged against each fibre's natural spread)")
    print(f"    {ok} of {ref} referenced fibres recovered or marginal")
    for m in MECH_SIMPLE:
        if cnt.get(m):
            print(f"    {cnt[m]:>2}  {m}")
    nu = sum(1 for m in Mt.values() if m.get("note") == "SNV reference unstable")
    if nu:
        print(f"\n  {nu} fibre(s) have an unstable SNV reference (natSNV > 20 deg):")
        print("    two real pure samples disagree by more than any plausible model")
        print("    error, so shape agreement cannot be assessed for them at all.")

    drop = [(n, m["snv_sam"] - m["det_sam"]) for n, m in Mt.items()
            if not np.isnan(m.get("det_sam", np.nan)) and m["snv_sam"] > 10]
    if drop:
        print("\n  fibres whose disagreement is mostly baseline slope:")
        for n, d in sorted(drop, key=lambda z: -z[1]):
            print(f"    {n:<4} SNV {Mt[n]['snv_sam']:6.2f}° -> detrended "
                  f"{Mt[n]['det_sam']:6.2f}°   ({d:+.2f}° explained by tilt)")

    # Does the SNV gap track how much shape there was to learn?
    # MSE penalises ABSOLUTE error, so a flat spectrum exerts almost no shape
    # pressure on the endmember -- and SNV then magnifies whatever difference
    # remains. If that is the mechanism, SNV SAM must fall as contrast rises.
    # If it does NOT, the SNV number is a metric artefact and should be dropped.
    st = [m for m in Mt.values()
          if not np.isnan(m["snv_sam"]) and m["cross_seed"] < 3]
    if len(st) >= 4:
        c = np.array([m["contrast"] for m in st])
        y = np.array([m["snv_sam"] for m in st])
        r = float(np.corrcoef(c, y)[0, 1])
        print(f"\n  corr(spectral contrast, SNV SAM) = {r:+.3f}   "
              f"(identifiable fibres only, n={len(st)})")
        if r < -0.5:
            print("  Strongly NEGATIVE -> shape disagreement is concentrated on")
            print("  FEATURELESS spectra. That is an information limit of the")
            print("  960-1660 nm window, not a flaw in the metric.")
        elif r > -0.2:
            print("  NOT negative -> high SNV SAM also appears on high-contrast")
            print("  fibres, so it is not explained by featurelessness. Treat the")
            print("  SNV column with caution and lead with raw.")
        else:
            print("  Weakly negative -> partial support; report both and say so.")
    return Mt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--ch", type=int, default=0)
    ap.add_argument("--save", default=None, help="prefix; omit to just show")
    ap.add_argument("--fibres", default=None,
                    help="comma-separated subset for the raw/SNV grids, e.g. SP,M,W,C "
                         "(with --paper: one row of up to 4 panels)")
    ap.add_argument("--width", type=float, default=None,
                    help="figure width in inches for --paper (e.g. 4.8 for ESANN's 12.2 cm)")
    ap.add_argument("--bw", action="store_true",
                    help="dashed learned endmember, for proceedings printed in black and white")
    ap.add_argument("--paper", action="store_true",
                    help="journal-size figures: 3-column grid, print-readable labels, "
                         "no in-figure titles (captions carry them)")
    a = ap.parse_args()
    D = load_all(a.script, a.data_root, a.ckpt, a.channels, a.channel_idx)
    print_metrics(D)
    _fib = a.fibres.split(",") if a.fibres else None
    figs = [("raw", plot_grid(D, ch=a.ch, snv=False, paper=a.paper, fibres=_fib,
                              width=a.width, bw=a.bw)),
            ("snv", plot_grid(D, ch=a.ch, snv=True, paper=a.paper, fibres=_fib,
                              width=a.width, bw=a.bw)),
            ("seedspread", plot_seed_spread(D, ch=a.ch, paper=a.paper)),
            ("diagnostic", plot_diagnostic_map(D))]
    if a.save:
        for nm, f in figs:
            if f is not None:
                f.savefig(f"{a.save}_{nm}.png", dpi=300, bbox_inches="tight")
                f.savefig(f"{a.save}_{nm}.pdf", bbox_inches="tight")
                print(f"  wrote {a.save}_{nm}.png")
    else:
        plt.show()
