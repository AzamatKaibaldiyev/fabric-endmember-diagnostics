#!/usr/bin/env python
"""
openset_stratified.py — is the novelty detector actually failing, or is the
TEST unfair?

The problem with the plain held-out evaluation
----------------------------------------------
It labels a sample "unknown" if it contains ANY amount of the held-out fibre.
But elastane is typically 2-10% of a fabric, so a "95% cotton + 5% spandex"
sample is 95% explainable by the remaining endmembers. Scoring those as
positives drags AUROC toward chance even if the detector is perfect on
genuinely novel material.

What this script does
---------------------
Re-scores a held-out checkpoint (no retraining) and reports detection
performance STRATIFIED BY THE TRUE FRACTION of the held-out fibre, plus a
blend-size-matched comparison that removes the "complex spectra reconstruct
badly" confound.

Two clean outcomes, both publishable:
  * AUROC rises sharply with unknown fraction -> the detector works; its
    limitation is quantified (needs >= X% novel material). Report the curve.
  * AUROC stays flat near 0.5 -> the residual genuinely does not detect novel
    fibres in this wavelength range. Report the negative result.

Usage
-----
  python openset_stratified.py --script train.py --data_root <DIR> \
      --ckpt holdout_SP_seed0.pt --holdout SP
"""
import argparse, importlib.util, os, sys, types
import numpy as np
import torch
from sklearn import metrics as skm


def load_module(path, name="_os_mod"):
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


def auroc_safe(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(skm.roc_auc_score(y, s))


def fpr_at_tpr(y, s, tpr=0.95):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    thr = np.percentile(s[y == 1], 100 * (1 - tpr))
    return float((s[y == 0] >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_mode", default="auto",
                    choices=["auto", "raw", "snv", "raw+snv"],
                    help="backbone input the checkpoint was trained with. auto = read it from the checkpoint metadata / filename tag")
    ap.add_argument("--holdout", required=True,
                    help="the fibre that was held out, e.g. SP / L / S")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="openset_stratified.csv")
    a = ap.parse_args()

    M = load_module(a.script)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hold = M.parse_holdout(a.holdout)
    keep = [k for k in range(12) if k not in hold]
    hname = ", ".join(M.FIBER_NAMES[k] for k in hold)

    # ORIGINAL 12-class test set: we need the true held-out fraction
    X = np.load(os.path.join(a.data_root, "test_data.npy"))
    Yf = np.load(os.path.join(a.data_root, "test_multi_label_classification_label.npy"))
    Cf = np.load(os.path.join(a.data_root, "test_composition_label.npy"))
    X = X.reshape((Yf.shape[0], 2, 200)).astype(np.float32)
    if a.channels == 1:
        X = X[:, a.channel_idx:a.channel_idx + 1, :]
    frac = Cf[:, hold].sum(axis=1)                 # true % of the held-out fibre
    n_fib = (Cf > 0.5).sum(axis=1)                 # blend size (12-class truth)
    print(f"held out: {hname}   test {len(X):,} samples, "
          f"{int((frac > 0.5).sum()):,} contain it")

    # rebuild the reduced-class model
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    arch = M.infer_arch_from_ckpt(sd) if hasattr(M, "infer_arch_from_ckpt") else {}
    class A: pass
    args = A()
    for k, v in dict(channels=a.channels, channel_idx=a.channel_idx,
                     model=arch.get("model") or "atto", drop_path=0.0,
                     multiscale=arch.get("multiscale", True), student_script=None,
                     student_arch="conv", comp_head="linear", physics_decoder=True,
                     physics_weight=0.0, physics_mixing="linear",
                     physics_recon_loss="mse", physics_km_target=False,
                     physics_abundance="normalize", physics_abundance_softplus=False,
                     endmember_init="random", scatter_correct="none",
                     km_transform=False, consistency_weight=0.0,
                     evidential_weight=0.0, emsc_degree=2,
                     _num_classes=len(keep)).items():
        setattr(args, k, v)
    if hasattr(M, "configure_input_from_checkpoint"):
        # SNV has no weights: the input mode must come from the checkpoint
        M.configure_input_from_checkpoint(args, ck, a.ckpt, a.input_mode)
    else:
        print("!! WARNING: the --script module has no configure_input_from_checkpoint;"
              " input preprocessing is NOT checked. An SNV-trained checkpoint will be"
              " fed RAW spectra and every network number below is meaningless.")
    base, _ = M.build_student(args)
    model = M.maybe_wrap_physics(base, args, train_set=None, length=X.shape[-1],
                                 num_classes=len(keep))
    miss, unexp = model.load_state_dict(sd, strict=False)
    core = [k for k in miss if not k.startswith(("decoder.", "emsc."))]
    if core:
        print(f"!! {len(core)} backbone keys missing -> ABORT"); sys.exit(1)
    model = model.to(dev).eval()

    res, maxp, je = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), a.batch):
            xb = torch.from_numpy(X[s:s + a.batch]).float().to(dev)
            mlc, reg, aux = M.fwd(model, xb)
            res.append(aux["residual"].cpu().numpy())
            maxp.append(torch.sigmoid(mlc).max(dim=1).values.cpu().numpy())
            # JointEnergy (Wang et al., NeurIPS 2021): the multi-label OOD
            # baseline. Sums the per-label binary energies -log(1+exp(f_i)).
            # max-prob is a softmax construct; this is the right comparison for
            # a head of K independent sigmoids.
            je.append((-torch.nn.functional.softplus(mlc)).sum(1).cpu().numpy())
    res = np.concatenate(res); maxp = np.concatenate(maxp); je = np.concatenate(je)

    # ── score fusion ────────────────────────────────────────────────────
    # The physics residual and classifier confidence fail in OPPOSITE regimes
    # (confidence works when little novel material is present; the residual
    # works when a lot is). Their raw scales differ by ~2 orders of magnitude,
    # so combine them as PERCENTILE RANKS -- scale-free and nothing to fit.
    def to_rank(v):
        r = np.empty(len(v), dtype=np.float64)
        r[np.argsort(v)] = np.arange(len(v))
        return r / max(len(v) - 1, 1)
    r_res, r_maxp, r_je = to_rank(res), to_rank(-maxp), to_rank(-je)
    fuse_mp = 0.5 * r_res + 0.5 * r_maxp        # residual + max-prob
    fuse_je = 0.5 * r_res + 0.5 * r_je          # residual + JointEnergy
    SCORES = [("physics residual", res), ("classifier max-prob", -maxp),
              ("JointEnergy", -je), ("FUSE res+maxprob", fuse_mp),
              ("FUSE res+JointEnergy", fuse_je)]

    # ── 1. stratify by the TRUE fraction of held-out material ────────────
    print("\n" + "=" * 76)
    print("  DETECTION vs UNKNOWN FRACTION")
    print("  (each row: pure-known samples as negatives, samples whose held-out")
    print("   fraction falls in the bin as positives)")
    print("=" * 76)
    known = frac <= 0.5
    hdr = f"  {'held-out fraction':<18}{'n':>7}" + "".join(f"{n[:14]:>16}" for n, _ in SCORES)
    print(hdr)
    rows = []
    bins = [(0.5, 5), (5, 15), (15, 30), (30, 50), (50, 80), (80, 100.1)]
    for lo, hi in bins:
        sel = (frac > lo) & (frac <= hi)
        if sel.sum() < 20:
            continue
        m = known | sel
        y = sel[m].astype(int)
        aucs = [auroc_safe(y, sc[m]) for _, sc in SCORES]
        best = int(np.nanargmax(aucs))
        cells = "".join(f"{v:>16.4f}" if i != best else f"{('*%.4f' % v):>16}"
                        for i, v in enumerate(aucs))
        print(f"  {f'{lo:g}-{hi:g}%':<18}{int(sel.sum()):>7}" + cells)
        rows.append([f"{lo:g}-{hi:g}%", int(sel.sum())] + [f"{v:.4f}" for v in aucs]
                    + [f"{fpr_at_tpr(y, res[m]):.4f}"])
    print("  (* = best score in that row)")
    print(f"  {'(known reference)':<22}{int(known.sum()):>8}{'':>11}{'':>12}{'':>9}"
          f"   {np.median(res[known]):.5f}")

    # ── 2. remove the blend-size confound ───────────────────────────────
    print("\n" + "=" * 76)
    print("  BLEND-SIZE-MATCHED  (unknowns vs knowns with the SAME fibre count)")
    print("  Residual grows with composition complexity, so an unmatched")
    print("  comparison can be dominated by blend size rather than novelty.")
    print("=" * 76)
    print(f"  {'blend size':<12}{'n unk':>7}{'n known':>9}" +
          "".join(f"{n[:14]:>16}" for n, _ in SCORES))
    for k in range(1, 7):
        u = (frac > 0.5) & (n_fib == k)
        kn = (frac <= 0.5) & (n_fib == k)
        if u.sum() < 20 or kn.sum() < 20:
            continue
        m = u | kn
        y = u[m].astype(int)
        aucs = [auroc_safe(y, sc[m]) for _, sc in SCORES]
        best = int(np.nanargmax(aucs))
        cells = "".join(f"{v:>16.4f}" if i != best else f"{('*%.4f' % v):>16}"
                        for i, v in enumerate(aucs))
        print(f"  {k} fibre(s){'':<2}{int(u.sum()):>7}{int(kn.sum()):>9}" + cells)

    # ── 3. the cleanest possible case: the held-out fibre is DOMINANT ────
    print("\n" + "=" * 76)
    print("  CLEANEST CASE: held-out fibre is the MAJORITY of the sample")
    print("=" * 76)
    for thr in (50, 70, 90):
        u = frac > thr
        if u.sum() < 20:
            print(f"  >{thr}%: only {int(u.sum())} samples — too few to score")
            continue
        m = known | u
        y = u[m].astype(int)
        aucs = [auroc_safe(y, sc[m]) for _, sc in SCORES]
        print(f"  >{thr}% held-out (n={int(u.sum()):,}): " +
              "  ".join(f"{n.split()[0][:8]}={v:.4f}" for (n, _), v in zip(SCORES, aucs)))

    with open(a.out, "w") as f:
        f.write("bin,n," + ",".join(n.replace(" ", "_") for n, _ in SCORES) + ",fpr95\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n  wrote {a.out}")
    print("\n  READING IT: if AUROC climbs steadily with the fraction bin, the")
    print("  detector works and you can state the minimum detectable fraction.")
    print("  If it stays flat near 0.5 even above 90%, the residual genuinely")
    print("  cannot see novel fibres here — report that as the negative result.")


if __name__ == "__main__":
    main()
