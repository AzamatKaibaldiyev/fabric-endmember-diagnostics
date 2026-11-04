#!/usr/bin/env python
"""
fcls_vs_classifier.py — attribute rare-class failure to PRIOR BIAS or to SPECTRA.

The logic
---------
Your learned endmembers define a purely PHYSICAL classifier: solve fully
constrained least squares (FCLS -- abundances non-negative and summing to one,
Heinz & Chang, IEEE TGRS 2001) of each measured spectrum against the 12
endmembers, then call a fibre "present" if its abundance clears a threshold.

That physical classifier has NO learned class prior. It cannot prefer polyester
just because polyester is 16x more common. So:

  * FCLS RECOVERS a rare fibre that the network drops  -> the spectrum carries
    the information and the network's failure is a PRIOR/imbalance effect
    (fixable by logit adjustment / DB loss).
  * FCLS ALSO fails on it                              -> the information is not
    separable in this wavelength range: a SPECTRAL limit, not a training bug.

The per-class gap (FCLS recall - classifier recall) is therefore a direct
attribution diagnostic.

Usage
-----
  python fcls_vs_classifier.py --script train.py --data_root <DIR> \
      --ckpt run_seed0.pt [--split test] [--n 20000]
"""
import argparse, importlib.util, os, sys, types
import numpy as np
import torch
from scipy.optimize import nnls
from sklearn import metrics as skm


def infer_arch_from_ckpt(sd):
    """Read the architecture back OUT of the checkpoint instead of assuming it.

    The 8-missing-key failure came from hardcoding multiscale=True: ms_proj is
    4 Linear layers = 8 params, so a checkpoint trained WITHOUT multiscale
    leaves exactly 8 keys unfilled and silently randomises the head.
    """
    keys = list(sd.keys())
    strip = lambda k: k[len("base."):] if k.startswith("base.") else k
    ks = [strip(k) for k in keys]
    multiscale = any(k.startswith("ms_proj") for k in ks)
    in_chans, dims0 = None, None
    for k in ks:
        if k.startswith("downsample_layers.0.0.weight"):
            w = sd.get(k) if k in sd else sd.get("base." + k)
            if w is not None:
                dims0, in_chans = int(w.shape[0]), int(w.shape[1])
            break
    size = None
    for nm, cfg in (("atto", 40), ("femto", 48), ("pico", 64),
                    ("nano", 80), ("tiny", 96), ("base", 128)):
        if dims0 == cfg:
            size = nm
            break
    return {"multiscale": multiscale, "in_chans": in_chans,
            "dims0": dims0, "model": size}


def load_module(path, name="_fcls_mod"):
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


def fcls(X, E, delta=1e3):
    """Fully Constrained Least Squares.

    min_a ||E a - x||^2   s.t.  a >= 0,  sum(a) = 1

    The sum-to-one constraint is imposed by the classical augmentation trick:
    append a row of `delta` to E and a `delta` entry to x, then solve NNLS.
    As delta grows the equality constraint is enforced ever more tightly.

    X: (N, D) spectra.  E: (D, K) endmembers.  Returns (N, K) abundances.
    """
    D, K = E.shape
    E_aug = np.vstack([E, delta * np.ones((1, K))])
    A = np.zeros((len(X), K))
    for i, x in enumerate(X):
        x_aug = np.concatenate([x, [delta]])
        A[i], _ = nnls(E_aug, x_aug)
        if (i + 1) % 5000 == 0:
            print(f"    FCLS {i+1}/{len(X)}", flush=True)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_mode", default="auto",
                    choices=["auto", "raw", "snv", "raw+snv"],
                    help="backbone input the checkpoint was trained with. auto = read it from the checkpoint metadata / filename tag")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model", default="atto")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--physics_km_target", action="store_true")
    ap.add_argument("--out", default="fcls_report.csv")
    ap.add_argument("--fcls_snv", action="store_true",
                    help="SNV-normalise spectra AND endmembers before solving. "
                         "Removes multiplicative scatter, which the sum-to-one "
                         "constraint otherwise cannot absorb.")
    ap.add_argument("--fcls_no_asc", action="store_true",
                    help="drop the sum-to-one constraint (plain NNLS, then "
                         "renormalise). Lets each sample choose its own overall "
                         "scale, which ASC forbids.")
    a = ap.parse_args()

    M = load_module(a.script)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pre = "test" if a.split == "test" else "train"
    ds = M.NIRDataset(
        os.path.join(a.data_root, f"{pre}_data.npy"),
        os.path.join(a.data_root, f"{pre}_multi_label_classification_label.npy"),
        os.path.join(a.data_root, f"{pre}_composition_label.npy"),
        channels=a.channels, channel_idx=a.channel_idx)
    idx = np.random.RandomState(0).choice(len(ds), min(a.n, len(ds)), replace=False)
    names = M.FIBER_NAMES
    print(f"scoring {len(idx):,} {pre} spectra on {dev}")

    # ── endmembers straight from the checkpoint ─────────────────────────
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    key = next((k for k in sd if k.endswith("decoder.endmembers")), None)
    if key is None:
        print("!! no decoder.endmembers in this checkpoint -- was physics_decoder on?")
        return
    E = torch.nn.functional.softplus(sd[key]).numpy()          # (K, C, L)
    K = E.shape[0]
    E_flat = E.reshape(K, -1).T                                # (D, K)
    print(f"endmembers: {E.shape} -> design matrix {E_flat.shape}")

    # ── network predictions ─────────────────────────────────────────────
    arch = infer_arch_from_ckpt(sd)
    if arch["model"]:
        print(f"  checkpoint architecture detected: model={arch['model']} "
              f"multiscale={arch['multiscale']} in_chans={arch['in_chans']}")
        a.model = arch["model"]
        if arch["in_chans"]:
            a.channels = arch["in_chans"]
    class A: pass
    args = A()
    for k, v in dict(channels=a.channels, channel_idx=a.channel_idx, model=a.model,
                     drop_path=0.0, multiscale=arch["multiscale"], student_script=None,
                     student_arch="conv", comp_head="linear", physics_decoder=True,
                     physics_weight=0.0, physics_mixing="linear",
                     physics_recon_loss="mse", physics_km_target=a.physics_km_target,
                     physics_abundance="normalize", physics_abundance_softplus=False,
                     endmember_init="random", scatter_correct="none",
                     km_transform=False, consistency_weight=0.0,
                     evidential_weight=0.0, emsc_degree=2).items():
        setattr(args, k, v)
    if hasattr(M, "configure_input_from_checkpoint"):
        # SNV has no weights: the input mode must come from the checkpoint
        M.configure_input_from_checkpoint(args, ck, a.ckpt, a.input_mode)
    else:
        print("!! WARNING: the --script module has no configure_input_from_checkpoint;"
              " input preprocessing is NOT checked. An SNV-trained checkpoint will be"
              " fed RAW spectra and every network number below is meaningless.")
    base, _ = M.build_student(args)
    model = M.maybe_wrap_physics(base, args, train_set=None,
                                 length=ds.X.shape[-1], num_classes=K)
    miss, unexp = model.load_state_dict(sd, strict=False)
    core = [k for k in miss if not k.startswith(("decoder.", "emsc."))]
    if core:
        print(f"!! {len(core)} backbone keys missing -- classifier side is UNRELIABLE")
        for k in core[:10]:
            print(f"     missing: {k}")
        print("   ABORTING: the network-vs-FCLS comparison would be meaningless.")
        sys.exit(1)
    model = model.to(dev).eval()

    logits, Y, Xraw = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            b = idx[s:s + a.batch]
            x = torch.stack([torch.from_numpy(ds.X[i]).float() for i in b]).to(dev)
            mlc, reg, aux = M.fwd(model, x)
            logits.append(mlc.cpu().numpy())
            Y.append(np.stack([ds.Y[i] for i in b]))
            tgt = (M.kubelka_munk(x, model.refl_idx) if a.physics_km_target
                   else M.to_absorbance(x, model.refl_idx))
            Xraw.append(tgt.cpu().numpy().reshape(len(b), -1))
    logits = np.vstack(logits); Y = np.vstack(Y); Xraw = np.vstack(Xraw)
    pred_net = (logits > 0).astype(int)

    # ── FCLS: the physics-only classifier ───────────────────────────────
    # ── fairness pre-processing ─────────────────────────────────────────
    # ASC (sum-to-one) fixes the total magnitude, so a sample that is simply
    # brighter or darker than average CANNOT be fitted. On fabrics that scatter
    # variation is large, so we offer two standard remedies and report which
    # was used -- otherwise a poor FCLS score may just be a scale artefact.
    Xs, Es = Xraw.copy(), E_flat.copy()
    if a.fcls_snv:
        Xs = (Xs - Xs.mean(1, keepdims=True)) / (Xs.std(1, keepdims=True) + 1e-8)
        Es = (Es - Es.mean(0, keepdims=True)) / (Es.std(0, keepdims=True) + 1e-8)
        print("  [fair] SNV applied to spectra and endmembers")
    print("solving FCLS (no learned prior involved) ...")
    if a.fcls_no_asc:
        print("  [fair] ASC dropped: plain NNLS then renormalise")
        Ab = np.zeros((len(Xs), K))
        for i, x in enumerate(Xs):
            Ab[i], _ = nnls(Es, x)
            if (i + 1) % 5000 == 0:
                print(f"    NNLS {i+1}/{len(Xs)}", flush=True)
        Ab = Ab / (Ab.sum(1, keepdims=True) + 1e-9)
    else:
        Ab = fcls(Xs, Es)

    # sweep the abundance threshold; pick the best by exact match
    best_t, best_acc = None, -1
    for t in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
              0.25, 0.30, 0.35, 0.40, 0.50]:
        acc = skm.accuracy_score(Y, (Ab > t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    pred_fcls = (Ab > best_t).astype(int)
    print(f"FCLS abundance threshold {best_t} (best of sweep)")

    # ── the attribution table ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  PER-CLASS RECALL:  network (has a learned prior)  vs  FCLS (has none)")
    print("=" * 78)
    n_pos = Y.sum(axis=0)
    print(f"  {'fibre':<6}{'n_present':>10}{'net recall':>12}{'FCLS recall':>13}"
          f"{'gap':>9}   attribution")
    rows = []
    for k in range(K):
        r_net = skm.recall_score(Y[:, k], pred_net[:, k], zero_division=0)
        r_fcl = skm.recall_score(Y[:, k], pred_fcls[:, k], zero_division=0)
        gap = r_fcl - r_net
        if gap > 0.10:
            att = "PRIOR BIAS (spectra carry it, net drops it)"
        elif r_net < 0.6 and r_fcl < 0.6:
            att = "SPECTRAL limit (neither can see it)"
        elif gap < -0.10:
            att = "net beats physics (nonlinear//context)"
        else:
            att = "agree"
        print(f"  {names[k]:<6}{int(n_pos[k]):>10}{r_net:>12.3f}{r_fcl:>13.3f}"
              f"{gap:>+9.3f}   {att}")
        rows.append((names[k], int(n_pos[k]), r_net, r_fcl, gap, att))

    print(f"\n  exact-match  network={skm.accuracy_score(Y, pred_net):.4f}"
          f"   FCLS={skm.accuracy_score(Y, pred_fcls):.4f}")
    print(f"  macro-F1     network="
          f"{skm.f1_score(Y, pred_net, average='macro', zero_division=0):.4f}"
          f"   FCLS={skm.f1_score(Y, pred_fcls, average='macro', zero_division=0):.4f}")
    rare = np.argsort(n_pos)[:3]
    print(f"\n  the 3 RAREST fibres ({', '.join(names[k] for k in rare)}):")
    print(f"     network recall = "
          f"{np.mean([skm.recall_score(Y[:,k],pred_net[:,k],zero_division=0) for k in rare]):.3f}")
    print(f"     FCLS    recall = "
          f"{np.mean([skm.recall_score(Y[:,k],pred_fcls[:,k],zero_division=0) for k in rare]):.3f}")
    print("\n  If FCLS recall is clearly HIGHER on the rare fibres, the spectra do")
    print("  contain the signal and the network is discarding it because of the")
    print("  class prior -> logit adjustment / DB loss is the right fix.")

    with open(a.out, "w") as f:
        f.write("fibre,n_present,net_recall,fcls_recall,gap,attribution\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]}\n")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
