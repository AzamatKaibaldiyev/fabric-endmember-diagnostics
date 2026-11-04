#!/usr/bin/env python
"""
drop_pure_audit.py — which fibres actually lost data to --drop_pure, and why?

The filter removes samples by max(composition) >= threshold, NOT by fibre. So a
sample that is 99% cotton + 1% spandex is removed even though spandex is never
the pure component. Any fibre that commonly appears as a trace additive in an
otherwise-pure fabric therefore loses training data it was never credited with.

This script separates two kinds of loss for every fibre:

  OWN-PURE  loss : samples where THIS fibre was the >= threshold component.
                   These are the "anchors" the ablation intends to remove.
  COLLATERAL loss: samples where this fibre was PRESENT but some OTHER fibre
                   was >= threshold. Unintended, and invisible in n_pure.

It also reports the CONTRAST PAIRS a fibre loses: for a minor additive like
elastane, the endmember is identifiable mainly through the difference between
a blend and the near-pure spectrum of its host fibre. Removing the host's pure
samples destroys that differencing signal even if the blend itself survives.

Usage
-----
  python drop_pure_audit.py --data_root <DIR> [--threshold 99]
  python drop_pure_audit.py --data_root <DIR> --threshold 99 \
      --degradation "SP:3.44:11.31,CA:0.65:15.87,M:0.82:2.72,T:0.23:1.51"
"""
import argparse, os
import numpy as np

FIBER_NAMES = ["SP", "CA", "L", "R", "M", "T", "P", "W", "N", "A", "S", "C"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--threshold", type=float, default=99.0)
    ap.add_argument("--present_thr", type=float, default=0.5,
                    help="composition %% above which a fibre counts as present")
    ap.add_argument("--degradation", default=None,
                    help="optional 'NAME:before:after,...' cross-seed SAM values, "
                         "to correlate loss against instability")
    ap.add_argument("--out", default="drop_pure_audit.csv")
    a = ap.parse_args()

    C = np.load(os.path.join(a.data_root, "train_composition_label.npy"))
    K = C.shape[1]
    names = FIBER_NAMES[:K]
    present = C > a.present_thr
    dominant = C.argmax(axis=1)
    removed = C.max(axis=1) >= a.threshold          # the drop_pure mask

    print(f"training samples: {len(C):,}")
    print(f"removed by --drop_pure {a.threshold:g}: {int(removed.sum()):,} "
          f"({100*removed.mean():.1f}%)")

    print("\n" + "=" * 92)
    print("  WHERE EACH FIBRE'S TRAINING DATA WENT")
    print("=" * 92)
    print(f"  {'fibre':<6}{'present':>9}{'own-pure':>10}{'collateral':>12}"
          f"{'total lost':>12}{'% of its data':>15}")
    rows = []
    for k in range(K):
        n_present = int(present[:, k].sum())
        own = int((removed & (dominant == k)).sum())
        collat = int((removed & present[:, k] & (dominant != k)).sum())
        lost = own + collat
        pct = 100 * lost / max(n_present, 1)
        print(f"  {names[k]:<6}{n_present:>9,}{own:>10,}{collat:>12,}"
              f"{lost:>12,}{pct:>14.1f}%")
        rows.append((names[k], n_present, own, collat, lost, pct))

    print("\n  own-pure  = samples where this fibre WAS the >= threshold component")
    print("              (the anchors the ablation intends to remove)")
    print("  collateral= samples where this fibre was present but ANOTHER fibre")
    print("              was >= threshold -- unintended loss, invisible in n_pure")

    # ── which fibres are hit mainly by collateral loss? ──────────────────
    print("\n" + "=" * 92)
    print("  FIBRES HURT WITHOUT LOSING ANY 'PURE' SAMPLES OF THEIR OWN")
    print("=" * 92)
    hidden = [r for r in rows if r[2] == 0 and r[3] > 0]
    if hidden:
        for nm, npr, own, col, lost, pct in sorted(hidden, key=lambda z: -z[3]):
            print(f"  {nm:<4} own-pure=0 but lost {col:,} samples "
                  f"({pct:.1f}% of its {npr:,}) as collateral")
        print("\n  These fibres appear as TRACE additives in otherwise-pure fabrics.")
        print("  n_pure reports 0 for them, so the ablation looks like it should")
        print("  not affect them -- but it removes their data anyway.")
    else:
        print("  none: every fibre that lost data also lost own-pure samples.")

    # ── contrast pairs: does a fibre lose its differencing reference? ────
    print("\n" + "=" * 92)
    print("  LOST CONTRAST REFERENCE")
    print("  A minor additive is identifiable mainly via the difference between a")
    print("  blend and the near-pure spectrum of its HOST fibre. If the host's pure")
    print("  samples are removed, that differencing signal disappears even when the")
    print("  blend itself survives.")
    print("=" * 92)
    print(f"  {'fibre':<6}{'median %':>10}{'top hosts (share of its samples)':>44}")
    for k in range(K):
        sel = present[:, k]
        if sel.sum() == 0:
            continue
        med = float(np.median(C[sel, k]))
        # which fibres co-occur as the dominant partner
        co = C[sel].copy(); co[:, k] = -1
        host = co.argmax(axis=1)
        cnt = np.bincount(host, minlength=K)
        top = np.argsort(-cnt)[:3]
        hs = ", ".join(f"{names[j]} {100*cnt[j]/max(sel.sum(),1):.0f}%"
                       for j in top if cnt[j] > 0)
        flag = "  <- minor additive" if med <= 10 else ""
        print(f"  {names[k]:<6}{med:>9.1f}%{hs:>44}{flag}")

    # ── optional: does loss predict instability? ─────────────────────────
    if a.degradation:
        print("\n" + "=" * 92)
        print("  DOES DATA LOSS EXPLAIN THE ENDMEMBER DEGRADATION?")
        print("=" * 92)
        deg = {}
        for tok in a.degradation.split(","):
            nm, b, af = tok.split(":")
            deg[nm.strip()] = (float(b), float(af))
        xs, ys, labs = [], [], []
        print(f"  {'fibre':<6}{'% data lost':>13}{'x-seed before':>15}"
              f"{'after':>9}{'ratio':>9}")
        for nm, npr, own, col, lost, pct in rows:
            if nm not in deg:
                continue
            b, af = deg[nm]
            xs.append(pct); ys.append(af / max(b, 1e-9)); labs.append(nm)
            print(f"  {nm:<6}{pct:>12.1f}%{b:>15.2f}{af:>9.2f}{af/max(b,1e-9):>9.2f}x")
        if len(xs) >= 3:
            r = float(np.corrcoef(xs, ys)[0, 1])
            print(f"\n  corr(%data lost, instability ratio) = {r:+.3f}  (n={len(xs)})")
            if r > 0.5:
                print("  POSITIVE -> the fibres that lost the most data destabilised")
                print("  the most. Data loss is the parsimonious explanation and")
                print("  should be reported instead of a joint-estimation argument.")
            elif r < 0.2:
                print("  NOT positive -> data loss does NOT explain the pattern.")
                print("  Something else (e.g. loss of contrast references, or joint")
                print("  identifiability) is driving it.")
            else:
                print("  Weak -> partial explanation; report both mechanisms.")

    with open(a.out, "w") as f:
        f.write("fibre,n_present,own_pure_lost,collateral_lost,total_lost,pct_lost\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f}\n")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
