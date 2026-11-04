#!/usr/bin/env python
"""
misclassification_detection.py — the closed-set companion to the held-out
open-set experiment.

A model trained on all 12 fibres has NO unknowns in its test set, so there is
no OOD task to score. The useful closed-set question instead is:

    can the physics residual tell us WHICH samples the model is about to get
    wrong, before we know the answer?

That is "failure prediction" / misclassification detection. It matters
practically: on a sorting line you would route high-score samples to manual
inspection rather than trusting the prediction.

Scores compared (same four as the open-set script, so the tables match):
    physics residual      -- reconstruction error from the unmixing decoder
    classifier max-prob   -- 1 - max sigmoid, the usual confidence baseline
    JointEnergy           -- the proper multi-label confidence baseline
    rank fusion           -- percentile-rank average of residual + each

Usage
-----
  python misclassification_detection.py --script train.py \
      --data_root <DIR> --ckpt headline_seed0.pt
"""
import argparse, importlib.util, os, sys, types
import numpy as np
import torch
from sklearn import metrics as skm


def load_module(path, name="_mis_mod"):
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


def to_rank(v):
    r = np.empty(len(v), dtype=np.float64)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_mode", default="auto",
                    choices=["auto", "raw", "snv", "raw+snv"],
                    help="backbone input the checkpoint was trained with. auto = read it from the checkpoint metadata / filename tag")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="misclassification.csv")
    a = ap.parse_args()

    M = load_module(a.script)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = M.NIRDataset(
        os.path.join(a.data_root, "test_data.npy"),
        os.path.join(a.data_root, "test_multi_label_classification_label.npy"),
        os.path.join(a.data_root, "test_composition_label.npy"),
        channels=a.channels, channel_idx=a.channel_idx)

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
                     evidential_weight=0.0, emsc_degree=2, _num_classes=12).items():
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
                                 length=ds.X.shape[-1], num_classes=12)
    miss, unexp = model.load_state_dict(sd, strict=False)
    core = [k for k in miss if not k.startswith(("decoder.", "emsc."))]
    if core:
        print(f"!! {len(core)} backbone keys missing -> ABORT"); sys.exit(1)
    model = model.to(dev).eval()

    res, maxp, je, preds = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(ds.X), a.batch):
            xb = torch.from_numpy(ds.X[s:s + a.batch]).float().to(dev)
            mlc, reg, aux = M.fwd(model, xb)
            res.append(aux["residual"].cpu().numpy())
            p = torch.sigmoid(mlc)
            maxp.append(p.max(dim=1).values.cpu().numpy())
            je.append((-torch.nn.functional.softplus(mlc)).sum(1).cpu().numpy())
            preds.append((p > 0.5).int().cpu().numpy())
    res = np.concatenate(res); maxp = np.concatenate(maxp)
    je = np.concatenate(je); preds = np.vstack(preds)
    Y = np.asarray(ds.Y)
    wrong = (preds != Y).any(axis=1).astype(int)      # exact-match FAILURE
    n_fib = (np.asarray(ds.comps) > 0.5).sum(axis=1)

    print(f"test samples: {len(Y):,}   exact-match accuracy: {1-wrong.mean():.4f}"
          f"   ({int(wrong.sum()):,} wrong)")

    SC = [("physics residual", res), ("classifier max-prob", -maxp),
          ("JointEnergy", -je),
          ("FUSE res+maxprob", 0.5*to_rank(res) + 0.5*to_rank(-maxp)),
          ("FUSE res+JointEnergy", 0.5*to_rank(res) + 0.5*to_rank(-je))]

    print("\n" + "=" * 74)
    print("  CAN THE SCORE PREDICT A MISCLASSIFICATION?  (positives = wrong)")
    print("=" * 74)
    print(f"  {'score':<24}{'AUROC':>9}{'AUPR':>9}{'FPR@95TPR':>12}")
    rows = []
    for nm, sc in SC:
        auroc = skm.roc_auc_score(wrong, sc)
        aupr = skm.average_precision_score(wrong, sc)
        thr = np.percentile(sc[wrong == 1], 5)
        fpr = float((sc[wrong == 0] >= thr).mean())
        print(f"  {nm:<24}{auroc:>9.4f}{aupr:>9.4f}{fpr:>12.4f}")
        rows.append((nm, auroc, aupr, fpr))

    print("\n" + "=" * 74)
    print("  BY BLEND SIZE  (does the residual flag failures on hard blends?)")
    print("=" * 74)
    print(f"  {'blend':<10}{'n':>8}{'err rate':>10}" +
          "".join(f"{n[:13]:>15}" for n, _ in SC))
    for k in range(1, 7):
        m = n_fib == k
        if m.sum() < 50 or wrong[m].sum() < 10 or (1 - wrong[m]).sum() < 10:
            continue
        aucs = [skm.roc_auc_score(wrong[m], sc[m]) for _, sc in SC]
        print(f"  {k} fibre{'':<4}{int(m.sum()):>8}{wrong[m].mean():>10.3f}" +
              "".join(f"{v:>15.4f}" for v in aucs))

    print("\n" + "=" * 74)
    print("  PRACTICAL VIEW: send the highest-scoring X% for manual inspection")
    print("=" * 74)
    print(f"  {'flag rate':<12}{'errors caught (physics)':>26}{'errors caught (max-prob)':>28}")
    for pct in (1, 5, 10, 20):
        k = int(len(res) * pct / 100)
        for nm, sc in SC[:1]:
            idx = np.argsort(-sc)[:k]
            r1 = wrong[idx].sum() / max(wrong.sum(), 1)
        idx2 = np.argsort(maxp)[:k]
        r2 = wrong[idx2].sum() / max(wrong.sum(), 1)
        print(f"  top {pct:>2}%{'':<6}{r1*100:>24.1f}%{r2*100:>27.1f}%")
    print("\n  (random flagging would catch X% of errors by flagging X% of samples)")

    with open(a.out, "w") as f:
        f.write("score,auroc,aupr,fpr95\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f}\n")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
