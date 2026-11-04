#!/usr/bin/env python
"""
endmember_report.py — turn learned endmembers into publishable evidence.

Two claims it can support, one of which needs NO reference spectra:

  (A) AGREEMENT  : learned endmember vs the measured pure-fibre mean (SAM, deg).
                   CONFOUNDED if endmember_init="data" (they started there).
                   Only unconfounded when the run used endmember_init="random".

  (B) STABILITY  : do INDEPENDENT SEEDS converge to the same endmember?
                   Reference-free, so it works even for Spandex, which has zero
                   pure samples and therefore no meaningful reference at all.
                   Low cross-seed SAM = the endmember is identifiable from the
                   data; high = it is an unconstrained slack channel.

Usage
-----
  python endmember_report.py --script train.py --data_root <DIR> \
      --ckpt run_seed0.pt run_seed1.pt run_seed2.pt \
      [--out endmembers]

Writes <out>_sam.csv (per-fibre numbers) and <out>_spectra.csv (curves for
plotting: learned mean across seeds + measured pure mean, per fibre/channel).
"""
import argparse, importlib.util, itertools, os, sys, types
import numpy as np
import torch


def load_module(path, name="_em_mod"):
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


def sam_deg(u, v):
    u = np.asarray(u).ravel(); v = np.asarray(v).ravel()
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(u @ v / d, -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one checkpoint per seed (same config)")
    ap.add_argument("--model", default="atto")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--n", type=int, default=120000,
                    help="samples scanned when counting pure spectra. Must match "
                         "plot_endmembers.load_all(max_scan=...) or the two "
                         "scripts will report different n_pure.")
    ap.add_argument("--purity", type=float, default=99.0)
    ap.add_argument("--out", default="endmembers")
    ap.add_argument("--snv_compare", action="store_true",
                    help="ALSO report SAM after SNV-normalising both the learned "
                         "endmember and the measured pure mean. Raw space is what "
                         "the model optimises; SNV space isolates SHAPE agreement "
                         "by removing per-sample scatter and baseline offset.")
    a = ap.parse_args()

    M = load_module(a.script)
    ds = M.NIRDataset(
        os.path.join(a.data_root, "train_data.npy"),
        os.path.join(a.data_root, "train_multi_label_classification_label.npy"),
        os.path.join(a.data_root, "train_composition_label.npy"),
        channels=a.channels, channel_idx=a.channel_idx)
    # sequential, not random, so the two scripts scan the SAME samples
    idx = np.arange(min(a.n, len(ds)))
    refl = M.reflectance_channel_indices(a.channels, a.channel_idx)
    pure, cnt = M.estimate_endmembers_from_data(
        ds, idx, 12, a.channels, refl, purity=a.purity)

    # ── pull the endmember tensor straight out of each checkpoint ───────
    mats = []
    for p in a.ckpt:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        key = next((k for k in sd if k.endswith("decoder.endmembers")), None)
        if key is None:
            print(f"  !! {os.path.basename(p)}: no decoder.endmembers "
                  f"(was physics_decoder on for this run?) -- skipped")
            continue
        E = torch.nn.functional.softplus(sd[key]).numpy()   # stored pre-softplus
        mats.append((os.path.basename(p), E))
        print(f"  loaded endmembers from {os.path.basename(p)}  shape={E.shape}")
    if not mats:
        print("no usable checkpoints; nothing to report."); return

    names = M.FIBER_NAMES
    K = len(names)
    Es = np.stack([m[1] for m in mats])                     # (S, K, C, L)
    S = Es.shape[0]

    # ── report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  ENDMEMBER REPORT   ({S} seed(s))")
    print("=" * 78)
    print(f"  {'fibre':<7}{'n_pure':>8}{'SAM vs measured':>18}{'cross-seed SAM':>17}"
          f"{'  verdict'}")
    rows = []
    for k in range(K):
        agree = sam_deg(Es[:, k].mean(0), pure[k]) if cnt[k] > 0 else float("nan")
        pairs = [sam_deg(Es[i, k], Es[j, k]) for i, j in itertools.combinations(range(S), 2)]
        stab = float(np.mean(pairs)) if pairs else float("nan")
        if cnt[k] == 0:
            verdict = "no reference (never pure)"
        elif not np.isnan(stab) and stab < 3:
            verdict = "identifiable" if agree < 10 else "stable but shifted"
        elif not np.isnan(stab) and stab < 10:
            verdict = "moderately stable"
        else:
            verdict = "UNSTABLE / slack channel"
        a_s = "   n/a" if np.isnan(agree) else f"{agree:6.2f}"
        s_s = "   n/a" if np.isnan(stab) else f"{stab:6.2f}"
        print(f"  {names[k]:<7}{int(cnt[k]):>8}{a_s:>15} deg{s_s:>14} deg   {verdict}")
        rows.append((names[k], int(cnt[k]), agree, stab, verdict))

    ok = [r for r in rows if r[1] > 0]
    if ok:
        print(f"\n  mean SAM vs measured (fibres with pure samples): "
              f"{np.nanmean([r[2] for r in ok]):.2f} deg")
        if a.snv_compare:
            svs = [sam_deg(snv(Es[:, k].mean(0)), snv(pure[k]))
                   for k in range(K) if cnt[k] > 0]
            print(f"  mean SAM after SNV (shape only):                 "
                  f"{np.nanmean(svs):.2f} deg")
            print("  RAW is the honest number (it is the space the reconstruction")
            print("  loss operates in). SNV isolates shape by removing scatter and")
            print("  baseline offset; a big drop means part of the raw gap is")
            print("  scatter, not a genuine mismatch of spectral shape.")
    if S > 1:
        print(f"  mean cross-seed SAM (ALL fibres):                {np.nanmean([r[3] for r in rows]):.2f} deg")
        print("\n  Cross-seed agreement is REFERENCE-FREE: it is evidence the data")
        print("  determines the endmember, and it is the only usable evidence for")
        print("  fibres that are never pure (e.g. Spandex).")
    else:
        print("\n  (pass several seed checkpoints to get the reference-free"
              " stability column)")

    # ── pairwise degeneracy: which fibres are spectrally indistinguishable? ──
    # If two endmembers are nearly collinear, the linear mixing model cannot
    # tell them apart: mass can be traded between them without changing the
    # reconstruction. That is a PHYSICAL explanation for a confusion pair, and
    # it predicts both endmember instability and classifier error.
    print("\n" + "=" * 78)
    print("  PAIRWISE ENDMEMBER DEGENERACY  (SAM between learned endmembers)")
    print("=" * 78)
    Em = Es.mean(0)
    P = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i != j:
                P[i, j] = sam_deg(Em[i], Em[j])
    print(f"  {'fibre':<7}{'nearest other fibre':>22}{'SAM':>9}   "
          f"{'2nd nearest':>14}{'SAM':>9}")
    deg_rows = []
    for i in range(K):
        o = np.argsort(np.nan_to_num(P[i], nan=1e9))
        a1, a2 = o[0], o[1]
        print(f"  {names[i]:<7}{names[a1]:>22}{P[i,a1]:>8.2f}°   "
              f"{names[a2]:>14}{P[i,a2]:>8.2f}°")
        deg_rows.append((names[i], names[a1], P[i, a1], names[a2], P[i, a2]))
    off = P[~np.isnan(P)]
    print(f"\n  median pairwise SAM across all fibre pairs: {np.median(off):.2f}°")
    print("  A fibre whose nearest neighbour is FAR below this median is")
    print("  spectrally degenerate with it -> expect confusion + unstable")
    print("  endmembers for that pair.")
    with open(f"{a.out}_pairwise.csv", "w") as f:
        f.write("fibre_i,fibre_j,sam_deg\n")
        for i in range(K):
            for j in range(K):
                if i != j:
                    f.write(f"{names[i]},{names[j]},{P[i,j]:.4f}\n")

    # ── Gram matrix / conditioning: how ill-posed is the unmixing itself? ──
    # x = E a is solvable for a only to the extent that E has independent
    # columns. cond(G), G = E_n E_n^T on L2-normalised endmembers, quantifies
    # that: a large condition number means some abundance directions are nearly
    # unidentifiable, so mass can slide between fibres with almost no change in
    # the reconstruction. This is the matrix-level version of the pairwise table
    # and is what actually governs FCLS stability.
    print("\n" + "=" * 78)
    print("  ENDMEMBER GRAM MATRIX / CONDITIONING")
    print("=" * 78)
    V = Em.reshape(K, -1)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    G = Vn @ Vn.T
    ev = np.clip(np.linalg.eigvalsh(G), 1e-12, None)[::-1]
    cond = float(ev[0] / ev[-1])
    p_ = ev / ev.sum()
    eff_rank = float(np.exp(-(p_ * np.log(p_)).sum()))

    # MEAN-CENTRED version. Endmembers are NON-NEGATIVE spectra sharing a large
    # baseline/DC component, which inflates every cosine toward 1 regardless of
    # how different the fibres actually are. Centring removes that shared
    # offset, so:
    #   RAW      conditioning -> predicts FCLS/abundance instability (FCLS
    #                            solves with the raw E, so this is the honest
    #                            number for "is unmixing well-posed?")
    #   CENTRED  conditioning -> measures genuine SHAPE discriminability
    #                            ("are these fibres actually distinguishable?")
    Vc = V - V.mean(axis=1, keepdims=True)
    Vcn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
    Gc = Vcn @ Vcn.T
    evc = np.clip(np.linalg.eigvalsh(Gc), 1e-12, None)[::-1]
    condc = float(evc[0] / evc[-1])
    pc = evc / evc.sum()
    eff_rank_c = float(np.exp(-(pc * np.log(pc)).sum()))

    print(f"  {'':<26}{'RAW (governs FCLS)':>22}{'MEAN-CENTRED (shape)':>24}")
    print(f"  {'condition number':<26}{cond:>22,.1f}{condc:>24,.1f}")
    print(f"  {'effective rank / ' + str(K):<26}{eff_rank:>22.2f}{eff_rank_c:>24.2f}")
    print(f"  {'largest eigenvalue':<26}{ev[0]:>22.3f}{evc[0]:>24.3f}")
    print(f"  {'smallest eigenvalue':<26}{ev[-1]:>22.2e}{evc[-1]:>24.2e}")
    if cond > 1e3:
        print("\n  RAW: SEVERELY ill-conditioned -> abundances are only weakly")
        print("       identifiable; expect FCLS to be unstable and rare fibres to")
        print("       be unrecoverable by least squares.")
    if condc < cond / 10:
        print("  CENTRED conditioning is much better than raw, so a large part of")
        print("  the raw ill-conditioning is the shared non-negative baseline, not")
        print("  true fibre similarity. Report both numbers, not just the raw one.")
    off_diag = np.abs(G - np.eye(K))
    offc = np.abs(Gc - np.eye(K))
    print(f"\n  {'fibre':<6}{'raw max|cos|':>14}{'with':>6}{'centred max|cos|':>19}{'with':>6}")
    for k in np.argsort(-off_diag.max(axis=1))[:6]:
        j = int(np.argmax(off_diag[k])); jc = int(np.argmax(offc[k]))
        print(f"  {names[k]:<6}{G[k, j]:>+14.4f}{names[j]:>6}"
              f"{Gc[k, jc]:>+19.4f}{names[jc]:>6}")
    with open(f"{a.out}_gram.csv", "w") as f:
        f.write("," + ",".join(names) + "\n")
        for i in range(K):
            f.write(names[i] + "," + ",".join(f"{G[i,j]:.6f}" for j in range(K)) + "\n")
    with open(f"{a.out}_gram_centred.csv", "w") as f:
        f.write("," + ",".join(names) + "\n")
        for i in range(K):
            f.write(names[i] + "," + ",".join(f"{Gc[i,j]:.6f}" for j in range(K)) + "\n")

    # ── dump for plotting ───────────────────────────────────────────────
    with open(f"{a.out}_sam.csv", "w") as f:
        f.write("fibre,n_pure,sam_vs_measured_deg,cross_seed_sam_deg,verdict\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]}\n")
    Em = Es.mean(0)
    with open(f"{a.out}_spectra.csv", "w") as f:
        f.write("fibre,channel,point,learned,measured_pure,"
                "learned_snv,measured_pure_snv\n")
        for k in range(K):
            for c in range(Em.shape[1]):
                lv = (Em[k, c] - Em[k, c].mean()) / (Em[k, c].std() + 1e-9)
                pv = (pure[k, c] - pure[k, c].mean()) / (pure[k, c].std() + 1e-9)
                for i in range(Em.shape[2]):
                    f.write(f"{names[k]},{c},{i},{Em[k,c,i]:.6f},{pure[k,c,i]:.6f},"
                            f"{lv[i]:.6f},{pv[i]:.6f}\n")
    print(f"\n  wrote {a.out}_sam.csv and {a.out}_spectra.csv")
    print("  (plot learned vs measured_pure per fibre -> your interpretability figure)")


if __name__ == "__main__":
    main()
