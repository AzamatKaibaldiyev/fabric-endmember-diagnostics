#!/usr/bin/env python
"""
abundance_audit.py -- what does the DECODER actually receive?

The decoder never sees the input spectrum. It sees only the 12 normalised
proportions the encoder predicts, and it must reconstruct the same raw
absorbance target whatever the input mode. So if a fibre's endmember is stable
under one input mode and unstable under another, the cause has to be in those
proportions (or in what they fail to explain). This script measures them
directly, per checkpoint, and groups checkpoints by input mode.

Per fibre (on the same fixed set of spectra for every checkpoint):
  mass ratio      sum of predicted proportion / sum of true proportion
  MAE present     mean |pred - true| (percentage points) where the fibre is present
  false mass      predicted mass on samples WITHOUT the fibre, as % of its true mass
  r               correlation of predicted vs true proportion over all samples
  x-seed r        correlation of the predicted proportion between seeds of the
                  same input mode (does every seed hand the decoder the SAME mass?)

Per checkpoint:
  residual        reconstruction residual (median)
  rho(level)      Spearman correlation of the residual with each spectrum's mean
                  absorbance; rho(scale) with its spread. High values mean the
                  residual is scatter the proportions could not explain.

Hypotheses for "CA/M endmembers unstable under SNV-only input":
  H1 proportions   SNV hands the decoder worse or less consistent CA/M mass
                   -> MAE / false mass up, r or x-seed r down for CA/M
  H2 scatter       raw input lets the encoder bend proportions to explain
                   scatter; SNV cannot -> SNV residual higher and more
                   correlated with level/scale
  H3 seed luck     none of the above differ -> look at which seed is the outlier

Usage
-----
  python abundance_audit.py --script train.py --data_root <DIR> \\
      --ckpt raw_s0.pt raw_s1.pt raw_s2.pt snv_s0.pt snv_s1.pt snv_s2.pt \\
             rawsnv_s0.pt rawsnv_s1.pt rawsnv_s2.pt
  (input mode is read from each checkpoint; --groups overrides the grouping)
"""
import argparse, importlib.util, os, re, sys, types, warnings
from itertools import combinations
import numpy as np
import torch


def load_module(path, name="_aa_mod"):
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


def ranks(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def pearson(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build(M, a, ck, ckpt_path, length):
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not any(k.endswith("decoder.endmembers") for k in sd):
        raise SystemExit(f"!! {ckpt_path}: no decoder.endmembers -- physics decoder off?")
    arch = M.infer_arch_from_ckpt(sd)
    class A: pass
    args = A()
    for k, v in dict(channels=a.channels, channel_idx=a.channel_idx,
                     model=arch.get("model") or "atto", drop_path=0.0,
                     multiscale=arch.get("multiscale", False), student_script=None,
                     student_arch="conv", comp_head="linear", physics_decoder=True,
                     physics_weight=0.0, physics_mixing="linear",
                     physics_recon_loss="mse", physics_km_target=False,
                     physics_abundance="normalize", physics_abundance_softplus=False,
                     endmember_init="random", scatter_correct="none",
                     km_transform=False, consistency_weight=0.0,
                     evidential_weight=0.0, emsc_degree=2, _num_classes=12).items():
        setattr(args, k, v)
    mode = M.configure_input_from_checkpoint(args, ck, ckpt_path, a.input_mode)
    base, _ = M.build_student(args)
    model = M.maybe_wrap_physics(base, args, train_set=None, length=length,
                                 num_classes=12)
    miss, _ = model.load_state_dict(sd, strict=False)
    core = [k for k in miss if not k.startswith(("decoder.", "emsc."))]
    if core:
        raise SystemExit(f"!! {ckpt_path}: {len(core)} backbone keys missing -> ABORT")
    return model, mode


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--groups", default=None,
                    help="comma list, one label per --ckpt (overrides input-mode grouping)")
    ap.add_argument("--input_mode", default="auto",
                    choices=["auto", "raw", "snv", "raw+snv"])
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="train = the data the decoder was actually fitted on")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--focus", default="CA,M", help="fibres to flag in the verdict")
    ap.add_argument("--neighbours", default="R,T",
                    help="fibres to watch for mass the focus fibres lose")
    a = ap.parse_args()

    M = load_module(a.script)
    if not hasattr(M, "configure_input_from_checkpoint"):
        raise SystemExit("!! --script module is too old (no configure_input_from_checkpoint)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(M.FIBER_NAMES)
    pre = a.split
    ds = M.NIRDataset(
        os.path.join(a.data_root, f"{pre}_data.npy"),
        os.path.join(a.data_root, f"{pre}_multi_label_classification_label.npy"),
        os.path.join(a.data_root, f"{pre}_composition_label.npy"),
        channels=a.channels, channel_idx=a.channel_idx)
    idx = np.sort(np.random.RandomState(0).choice(len(ds), min(a.n, len(ds)),
                                                  replace=False))
    X = ds.X[idx]
    T = np.asarray(ds.comps, dtype=np.float64)[idx] / 100.0         # true proportions
    print(f"auditing {len(idx):,} {pre} spectra on {dev}\n")

    if a.groups:
        glabels = [g.strip() for g in a.groups.split(",")]
        if len(glabels) != len(a.ckpt):
            raise SystemExit("!! --groups needs exactly one label per --ckpt")
    runs = []
    for i, p in enumerate(a.ckpt):
        print(f"[{i + 1}/{len(a.ckpt)}] {os.path.basename(p)}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model, mode = build(M, a, ck, p, X.shape[-1])
        model = model.to(dev).eval()
        A_, R_, lev, scl = [], [], [], []
        with torch.no_grad():
            for s in range(0, len(X), a.batch):
                xb = torch.from_numpy(X[s:s + a.batch]).float().to(dev)
                _, _, aux = M.fwd(model, xb)
                A_.append(aux["abundance"].cpu().numpy())
                R_.append(aux["residual"].cpu().numpy())
                tgt = aux["recon_target"]
                lev.append(tgt.mean(dim=(1, 2)).cpu().numpy())
                scl.append(tgt.std(dim=2).mean(dim=1).cpu().numpy())
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        E = torch.nn.functional.softplus(
            sd[next(k for k in sd if k.endswith("decoder.endmembers"))]).numpy()
        m_seed = re.search(r"seed(\d+)", os.path.basename(p))
        runs.append(dict(name=os.path.basename(p), E=E.reshape(E.shape[0], -1),
                         seed=f"seed{m_seed.group(1)}" if m_seed else f"#{i}",
                         group=glabels[i] if a.groups else mode,
                         A=np.vstack(A_).astype(np.float64), R=np.concatenate(R_),
                         lev=np.concatenate(lev), scl=np.concatenate(scl)))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ── per-checkpoint metrics ───────────────────────────────────────────
    K = T.shape[1]
    tmass = T.sum(0)
    for r in runs:
        A_ = r["A"]
        present = T > 0.005
        r["mass"] = A_.sum(0) / np.maximum(tmass, 1e-9)
        r["mae"] = np.array([np.abs(A_[present[:, k], k] - T[present[:, k], k]).mean() * 100
                             if present[:, k].any() else np.nan for k in range(K)])
        r["false"] = np.array([A_[~present[:, k], k].sum() / max(tmass[k], 1e-9) * 100
                               for k in range(K)])
        r["r"] = np.array([pearson(A_[:, k], T[:, k]) for k in range(K)])
        r["res_med"] = float(np.median(r["R"]))
        r["rho_lev"] = spearman(r["R"], r["lev"])
        r["rho_scl"] = spearman(r["R"], r["scl"])

    order = []
    for r in runs:
        if r["group"] not in order:
            order.append(r["group"])
    if "raw" in order:
        order.remove("raw"); order.insert(0, "raw")
    G = {g: [r for r in runs if r["group"] == g] for g in order}
    ref = order[0]

    def stat(g, key, k=None):
        v = np.array([r[key][k] if k is not None else r[key] for r in G[g]], float)
        return np.nanmean(v), (np.nanstd(v, ddof=1) if len(v) > 1 else np.nan)

    def xseed_r(g, k):
        rs = G[g]
        if len(rs) < 2:
            return np.nan, np.nan
        v = [pearson(p["A"][:, k], q["A"][:, k]) for p, q in combinations(rs, 2)]
        return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan)

    def flag(g, key, k=None, fn=None):
        """'*' when |mean_g - mean_ref| > 2 x pooled seed SD (a heuristic, 3 seeds)."""
        if g == ref:
            return " "
        (m1, s1), (m0, s0) = (fn(g, k), fn(ref, k)) if fn else (stat(g, key, k), stat(ref, key, k))
        if np.isnan(s1) or np.isnan(s0):
            return " "
        pooled = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
        return "*" if abs(m1 - m0) > 2 * max(pooled, 1e-12) else " "

    W = 20
    print("\n" + "=" * (8 + W * len(order)))
    print("  WHAT THE DECODER RECEIVES  (mean +- SD over seeds; * = differs from "
          f"'{ref}' by > 2 pooled SD)")
    print("  groups: " + ",  ".join(f"{g} (n={len(G[g])})" for g in order))
    print("=" * (8 + W * len(order)))
    blocks = [("mass ratio  (pred / true; 1 = right total mass)", "mass", "{:.3f}"),
              ("MAE where present  (percentage points)", "mae", "{:.2f}"),
              ("false mass on absent samples  (% of true mass)", "false", "{:.1f}"),
              ("r  (predicted vs true proportion)", "r", "{:.3f}")]
    for title, key, fmt in blocks:
        print(f"\n  {title}")
        print("  " + f"{'fibre':<6}" + "".join(f"{g:>{W}}" for g in order))
        for k in range(K):
            cells = []
            for g in order:
                m, s = stat(g, key, k)
                cell = fmt.format(m) + ("" if np.isnan(s) else " +-" + fmt.format(s))
                cells.append(f"{cell + flag(g, key, k):>{W}}")
            print("  " + f"{names[k]:<6}" + "".join(cells))
    print("\n  x-seed r  (same input mode, different seeds: is the mass CONSISTENT?)")
    print("  " + f"{'fibre':<6}" + "".join(f"{g:>{W}}" for g in order))
    for k in range(K):
        cells = []
        for g in order:
            m, s = xseed_r(g, k)
            cell = "n/a" if np.isnan(m) else f"{m:.3f}" + ("" if np.isnan(s) else f" +-{s:.3f}")
            cells.append(f"{cell + flag(g, None, k, fn=xseed_r):>{W}}")
        print("  " + f"{names[k]:<6}" + "".join(cells))

    print("\n  reconstruction residual  (H2: scatter the proportions cannot explain)")
    print("  " + f"{'':<14}" + "".join(f"{g:>{W}}" for g in order))
    for title, key, fmt in [("median", "res_med", "{:.5f}"),
                            ("rho(level)", "rho_lev", "{:+.3f}"),
                            ("rho(scale)", "rho_scl", "{:+.3f}")]:
        cells = []
        for g in order:
            m, s = stat(g, key)
            cell = fmt.format(m) + ("" if np.isnan(s) else " +-" + fmt.format(s).lstrip("+"))
            cells.append(f"{cell + flag(g, key):>{W}}")
        print("  " + f"{title:<14}" + "".join(cells))

    # ── per-seed detail: does the seed that starves a fibre lose its endmember? ──
    def angle(u, v):
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    foc = [names.index(f.strip()) for f in a.focus.split(",") if f.strip() in names]
    nbr = [names.index(f.strip()) for f in a.neighbours.split(",")
           if f.strip() in names and names.index(f.strip()) not in foc]
    for g in order:
        rs = G[g]
        for r in rs:
            # angle to the NEAREST other seed: an outlier seed is far from all
            # others, while the seeds that agree are near each other. (Comparing
            # with the mean of the others lets the outlier inflate everyone.)
            r["loo"] = {}
            for k in foc:
                others = [angle(r["E"][k], q["E"][k]) for q in rs if q is not r]
                r["loo"][k] = min(others) if others else np.nan
    ref_mass = {k: np.mean([r["mass"][k] for r in G[ref]]) for k in foc}
    print("\n" + "=" * 78)
    print("  PER-SEED DETAIL")
    print("  mass  = predicted / true mass of that fibre (what the decoder receives)")
    print("  angle = this seed's endmember vs the NEAREST other seed in its group (deg)")
    print(f"  !     = mass below half of the '{ref}' mean, or the group's outlier angle")
    print("=" * 78)
    hdr = f"  {'group':<9}{'seed':<8}"
    for k in foc:
        hdr += f"{names[k] + ' mass':>10}{names[k] + ' MAE':>9}{names[k] + ' ang':>9}"
    for k in nbr:
        hdr += f"{names[k] + ' mass':>10}{names[k] + ' false':>10}"
    print(hdr)
    for g in order:
        rs = G[g]
        for r in rs:
            line = f"  {g:<9}{r['seed']:<8}"
            for k in foc:
                angs = sorted([q["loo"][k] for q in rs], reverse=True)
                out_ang = (len(angs) > 2 and r["loo"][k] == angs[0]
                           and angs[0] > 2 * max(angs[1], 1e-9))
                mflag = "!" if r["mass"][k] < 0.5 * ref_mass[k] else " "
                line += (f"{r['mass'][k]:>9.3f}{mflag}{r['mae'][k]:>9.2f}"
                         f"{r['loo'][k]:>8.2f}{'!' if out_ang else ' '}")
            for k in nbr:
                line += f"{r['mass'][k]:>10.3f}{r['false'][k]:>10.1f}"
            print(line)
        print()
    print("  If the seed marked ! for mass is also the ! angle, the endmember failed")
    print("  because the decoder was starved of that fibre's mass, not because the")
    print("  spectra changed (the decoder target is identical in every run).")

    # ── verdict for the focus fibres ─────────────────────────────────────
    focus = [names.index(f.strip()) for f in a.focus.split(",") if f.strip() in names]
    others = [g for g in order if g != ref]
    print("\n" + "=" * 78)
    print(f"  READ-OUT for {', '.join(names[k] for k in focus)}  (vs '{ref}')")
    print("=" * 78)
    for g in others:
        h1 = []
        for k in focus:
            for key, worse in (("mae", +1), ("false", +1), ("r", -1)):
                (m1, _), (m0, _) = stat(g, key, k), stat(ref, key, k)
                if flag(g, key, k) == "*" and np.sign(m1 - m0) == worse:
                    h1.append(f"{names[k]} {key}")
            (m1, _), (m0, _) = xseed_r(g, k), xseed_r(ref, k)
            if flag(g, None, k, fn=xseed_r) == "*" and m1 < m0:
                h1.append(f"{names[k]} x-seed r")
        (rm1, _), (rm0, _) = stat(g, "res_med"), stat(ref, "res_med")
        h2 = (flag(g, "res_med") == "*" and rm1 > rm0) or any(
            flag(g, key) == "*" and abs(stat(g, key)[0]) > abs(stat(ref, key)[0])
            for key in ("rho_lev", "rho_scl"))
        print(f"  {g}:")
        print(f"    H1 (worse/less consistent proportions): "
              f"{'SUPPORTED -> ' + ', '.join(h1) if h1 else 'no clear difference'}")
        print(f"    H2 (unexplained scatter in residual):   "
              f"{'SUPPORTED' if h2 else 'no clear difference'}")
        if not h1 and not h2:
            print("    -> neither: check per-seed endmember angles (H3, one outlier seed)")
    print("\n  With 3 seeds the '*' flag is a heuristic, not a significance test.")


if __name__ == "__main__":
    main()
