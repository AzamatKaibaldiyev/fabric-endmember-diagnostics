#!/usr/bin/env python
"""
physics_probe.py — calibrate and sanity-check the Beer-Lambert unmixing decoder
BEFORE burning GPU-hours on an ablation that cannot move.

It answers four questions on YOUR data:
  1. What magnitude is the reconstruction loss vs the task losses?
     -> the physics_weight that actually makes the term visible.
  2. Is the composition on the simplex? (if not, the physics loss is
     scale-invariant and cannot constrain composition at all)
  3. How good are the data-initialised endmembers? (spectral angle vs the
     measured pure-fibre means -> the interpretability figure)
  4. How separable is the reconstruction residual? (the OOD/novelty claim)

Usage
-----
    python physics_probe.py --script train.py \
        --data_root /path/to/dataverse_files/ \
        [--ckpt chkpts/....pt] [--comp_head gated] [--n 20000]

With --ckpt it probes a trained model (most accurate). Without, it probes an
untrained one, which still gives correct loss SCALES for weight calibration.
"""
import argparse, importlib.util, os, sys, types
import numpy as np
import torch
import torch.nn.functional as F


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


def load_module(path, name="_probe_mod"):
    # stub wandb so importing the training script does not try to log
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train.py",
                    help="path to your training script")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="optional trained checkpoint (.pt with ['model'])")
    ap.add_argument("--input_mode", default="auto",
                    choices=["auto", "raw", "snv", "raw+snv"],
                    help="backbone input the checkpoint was trained with. auto = read it from the checkpoint metadata / filename tag")
    ap.add_argument("--model", default="atto")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--channel_idx", type=int, default=1)
    ap.add_argument("--multiscale", action="store_true", default=True)
    ap.add_argument("--comp_head", default="gated",
                    choices=["linear", "gated", "softmax", "sparsemax",
                             "simplex", "dirichlet"])
    ap.add_argument("--physics_mixing", default="linear")
    ap.add_argument("--physics_km_target", action="store_true")
    ap.add_argument("--n", type=int, default=20000, help="samples to probe")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    M = load_module(a.script)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── data ────────────────────────────────────────────────────────────
    root = a.data_root
    ds = M.NIRDataset(
        os.path.join(root, "train_data.npy"),
        os.path.join(root, "train_multi_label_classification_label.npy"),
        os.path.join(root, "train_composition_label.npy"),
        channels=a.channels, channel_idx=a.channel_idx)
    n = min(a.n, len(ds))
    idx = np.random.RandomState(0).choice(len(ds), n, replace=False)
    print(f"probing {n:,} of {len(ds):,} training spectra  (device={dev})")

    # ── args shim for the wrapper ───────────────────────────────────────
    class A: pass
    args = A()
    for k, v in dict(channels=a.channels, channel_idx=a.channel_idx,
                     model=a.model, drop_path=0.0, multiscale=a.multiscale,
                     student_script=None, student_arch="conv",
                     comp_head=a.comp_head, physics_decoder=True,
                     physics_weight=1.0, physics_mixing=a.physics_mixing,
                     physics_recon_loss="mse",
                     physics_km_target=a.physics_km_target,
                     endmember_init="data", endmember_purity=99.0,
                     endmember_smooth_weight=0.0, endmember_volume_weight=0.0,
                     endmember_anchor_weight=0.0, evidential_weight=0.0,
                     consistency_weight=0.0, consistency_tau=0.01,
                     scatter_correct="none", emsc_degree=2, km_transform=False,
                     synth_blend=False, mlc_weight=1.0).items():
        setattr(args, k, v)

    if a.ckpt:
        _ck0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        _sd0 = _ck0["model"] if isinstance(_ck0, dict) and "model" in _ck0 else _ck0
        _arch = infer_arch_from_ckpt(_sd0)
        if _arch["model"]:
            print(f"  checkpoint architecture detected: model={_arch['model']} "
                  f"multiscale={_arch['multiscale']} in_chans={_arch['in_chans']}")
            args.model = _arch["model"]
            args.multiscale = _arch["multiscale"]
            if _arch["in_chans"]:
                args.channels = _arch["in_chans"]
    if a.ckpt and hasattr(M, "configure_input_from_checkpoint"):
        # SNV has no weights: the input mode must come from the checkpoint
        M.configure_input_from_checkpoint(args, _ck0, a.ckpt, a.input_mode)
    elif a.ckpt:
        print("!! WARNING: the --script module has no configure_input_from_checkpoint;"
              " input preprocessing is NOT checked. An SNV-trained checkpoint will be"
              " fed RAW spectra and every network number below is meaningless.")
    if a.input_mode != "auto" and not a.ckpt:
        args.input_mode = a.input_mode
    base, _ = M.build_student(args)
    model = M.maybe_wrap_physics(base, args, train_set=ds, train_idx=idx,
                                 length=ds.X.shape[-1], num_classes=12)

    # ── checkpoint load (into the WRAPPED model, so trained endmembers count) ──
    loaded_ok, trained_endmembers = False, False
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        ck_wrapped = any(k.startswith("base.") for k in sd)
        tgt_wrapped = hasattr(model, "base")
        if ck_wrapped and not tgt_wrapped:
            sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        elif tgt_wrapped and not ck_wrapped:
            sd = {f"base.{k}": v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        # ignore the decoder's own params when judging whether the BACKBONE loaded
        miss_core = [k for k in miss if not k.startswith(("decoder.", "emsc."))]
        trained_endmembers = not any(k.startswith("decoder.") for k in miss)
        loaded_ok = len(miss_core) == 0
        print(f"checkpoint: {len(miss)} missing / {len(unexp)} unexpected; "
              f"backbone fully loaded = {loaded_ok}; "
              f"trained endmembers = {trained_endmembers}")
        if not loaded_ok:
            print("  !! MISSING BACKBONE KEYS -> part of the model is RANDOM.")
            print("     Loss magnitudes and simplex stats below are MEANINGLESS.")
            for k in miss_core[:12]:
                print(f"       missing: {k}")
            if len(miss_core) > 12:
                print(f"       ... and {len(miss_core)-12} more")
            print("     Likely cause: the probe's architecture flags differ from the")
            print("     run that produced the checkpoint (e.g. --multiscale, --model,")
            print("     --channels). Re-run the probe with matching flags.")
        if unexp:
            for k in list(unexp)[:6]:
                print(f"       unexpected: {k}")
    else:
        print("no --ckpt given: loss SCALES are still usable for weight calibration,")
        print("but leakage / endmember quality reflect an UNTRAINED model.")
    model = model.to(dev).eval()

    # ── forward pass ────────────────────────────────────────────────────
    L_mlc = L_reg = L_rec = 0.0
    nb = 0
    regs, comps, resids, neg_mass, sums = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, n, a.batch):
            b = idx[s:s + a.batch]
            x = torch.stack([torch.from_numpy(ds.X[i]).float() for i in b]).to(dev)
            y = torch.stack([torch.tensor(ds.Y[i]) for i in b]).float().to(dev)
            c = torch.stack([torch.from_numpy(ds.comps[i]).float() for i in b]).to(dev)
            mlc, reg, aux = M.fwd(model, x)
            L_mlc += F.multilabel_soft_margin_loss(mlc, y).item()
            L_reg += F.smooth_l1_loss(reg, c).item()
            if "recon" in aux:
                L_rec += F.mse_loss(aux["recon"], aux["recon_target"]).item()
                resids.append(aux["residual"].cpu().numpy())
            raw = aux.get("reg_raw", reg)
            neg_mass.append((raw.clamp(max=0).abs().sum(1)).cpu().numpy())
            sums.append(reg.sum(1).cpu().numpy())
            regs.append(reg.cpu().numpy()); comps.append(c.cpu().numpy())
            nb += 1
    L_mlc /= nb; L_reg /= nb; L_rec /= nb
    reg = np.vstack(regs); comp = np.vstack(comps)
    sums = np.concatenate(sums); neg_mass = np.concatenate(neg_mass)

    # ── 1. weight calibration ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  1. LOSS MAGNITUDES  ->  physics_weight calibration")
    print("=" * 72)
    task = L_mlc + L_reg
    print(f"  loss_mlc  = {L_mlc:.5f}")
    print(f"  loss_reg  = {L_reg:.5f}      (task total = {task:.5f})")
    print(f"  recon MSE = {L_rec:.6f}")
    if L_rec > 0:
        print(f"\n  physics_weight so the term is X% of the task loss:")
        for share in (0.02, 0.05, 0.10, 0.20, 0.50):
            print(f"    {int(share*100):>3}%  ->  physics_weight = {share*task/L_rec:>10,.1f}")
        print(f"\n  your current 0.1 gives a share of "
              f"{100*0.1*L_rec/(task+0.1*L_rec):.4f}%  <- effectively zero")
        print(f"  RECOMMENDED START: --physics_weight {0.10*task/L_rec:,.0f}   (~10% share)")

    # ── 2. simplex check ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  2. IS THE COMPOSITION ON THE SIMPLEX?")
    print("=" * 72)
    print(f"  comp_head            = {a.comp_head}")
    print(f"  predicted sum: mean={sums.mean():8.3f}  std={sums.std():7.3f}  "
          f"min={sums.min():8.3f}  max={sums.max():8.3f}   (target 100)")
    print(f"  negative mass /sample (raw head): mean={neg_mass.mean():.3f}")
    if abs(sums.mean() - 100) > 5 or sums.std() > 5:
        print("  !! NOT on the simplex -> the reconstruction loss is scale-INVARIANT")
        print("     (endmembers just rescale to absorb it) and cannot constrain")
        print("     composition. Use --comp_head gated / simplex / sparsemax.")
    else:
        print("  OK: on the simplex -> the physics term genuinely constrains composition.")

    # ── 3. leakage + endmember quality ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  3. LEAKAGE  &  ENDMEMBER QUALITY  (the interpretability figure)")
    print("=" * 72)
    print(f"  leakage (mass on truly-absent fibres) = {M.leakage_metric(reg, comp):.4f}")
    dec = getattr(model, "decoder", None)
    if dec is not None:
        E = dec.get_endmembers().detach().cpu().numpy()
        refl = M.reflectance_channel_indices(a.channels, a.channel_idx)
        pure, cnt = M.estimate_endmembers_from_data(
            ds, idx, 12, a.channels, refl, purity=99.0,
            km=a.physics_km_target)
        def sam(u, v):
            u = u.ravel(); v = v.ravel()
            return float(np.degrees(np.arccos(np.clip(
                u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))))
        if not trained_endmembers:
            print("  NOTE: trained endmembers were NOT loaded, so the SAM column below")
            print("        is tautological (endmembers were just initialised FROM these")
            print("        same pure means). Provide a checkpoint saved from a run with")
            print("        physics_decoder=True to get a real comparison.")
        print(f"\n  {'fibre':<8}{'n_pure':>8}{'SAM vs measured pure':>24}   endmember status")
        for k in range(12):
            nm = M.FIBER_NAMES[k]
            if cnt[k] == 0:
                status = "UNUSABLE (no pure sample -> global mean)"
            elif cnt[k] < 30:
                status = f"WEAK (only {int(cnt[k])} pure spectra)"
            elif cnt[k] < 100:
                status = "thin"
            else:
                status = "ok"
            print(f"  {nm:<8}{int(cnt[k]):>8}{sam(E[k], pure[k]):>21.2f} deg   {status}")
        bad = [M.FIBER_NAMES[k] for k in range(12) if cnt[k] < 30]
        if bad:
            print(f"\n  !! {len(bad)} endmember(s) are unreliable: {', '.join(bad)}")
            print("     These produce noisy reconstruction gradients. Do NOT set")
            print("     endmember_anchor_weight > 0 -- it pins them to a bad estimate.")

    # ── 4. residual spread (OOD potential) ──────────────────────────────
    if resids:
        r = np.concatenate(resids)
        print("\n" + "=" * 72)
        print("  4. RECONSTRUCTION RESIDUAL  (novelty / OOD score)")
        print("=" * 72)
        for p in (50, 90, 99, 100):
            print(f"    p{p:<3} = {np.percentile(r, p):.6f}")
        print(f"    ratio p99/p50 = {np.percentile(r,99)/max(np.percentile(r,50),1e-12):.1f}x")
        print("    (a NOVEL material should land far above p99; use this as the")
        print("     threshold when you score hemp/jute/leather from the FX17 set)")
    print()


if __name__ == "__main__":
    main()
