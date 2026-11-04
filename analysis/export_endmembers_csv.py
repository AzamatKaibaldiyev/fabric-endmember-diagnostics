#!/usr/bin/env python
"""
export_endmembers_csv.py -- write the learned endmember spectra to CSV.

The decoder stores its endmembers as unconstrained parameters and applies a
softplus to keep them non-negative; this script applies the same softplus, so
the CSV holds exactly the spectra the decoder mixes. Channel 0 is absorbance;
channel 1 is reflectance mapped to absorbance (A = -log10 R).

Usage
-----
  python analysis/export_endmembers_csv.py --ckpt run_seed0.pt run_seed1.pt run_seed2.pt \
      --out endmembers_csv

Writes one file per checkpoint and channel (<checkpoint>_ch<c>.csv) and, for
several checkpoints, their mean (endmembers_mean_ch<c>.csv). Columns:
wavelength_nm, then one column per fibre.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_endmembers import WAVELENGTHS_NM            # published centre wavelengths

FIBRES = ["SP", "CA", "L", "R", "M", "T", "P", "W", "N", "A", "S", "C"]


def endmembers(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    key = next((k for k in sd if k.endswith("decoder.endmembers")), None)
    if key is None:
        raise SystemExit(f"!! {path}: no decoder.endmembers (trained without --physics_decoder?)")
    return torch.nn.functional.softplus(sd[key].float()).numpy()      # (K, C, L)


def write(path, wl, E_ch, names):
    with open(path, "w") as f:
        f.write("wavelength_nm," + ",".join(names) + "\n")
        for i, w in enumerate(wl):
            f.write(f"{w:.2f}," + ",".join(f"{E_ch[k, i]:.6f}" for k in range(len(names))) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="endmembers_csv")
    ap.add_argument("--names", default=None,
                    help="comma-separated fibre names; needed only for held-out-class "
                         "checkpoints, which have fewer than 12 endmembers")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Es = [endmembers(p) for p in a.ckpt]
    K, C, L = Es[0].shape
    names = a.names.split(",") if a.names else (FIBRES if K == 12 else None)
    if names is None or len(names) != K:
        raise SystemExit(f"!! {K} endmembers: pass --names with {K} fibre names")
    wl = WAVELENGTHS_NM if L == len(WAVELENGTHS_NM) else np.arange(L, dtype=float)
    for p, E in zip(a.ckpt, Es):
        stem = os.path.splitext(os.path.basename(p))[0]
        for c in range(C):
            write(os.path.join(a.out, f"{stem}_ch{c}.csv"), wl, E[:, c], names)
    if len(Es) > 1:
        M = np.mean(Es, axis=0)
        for c in range(C):
            write(os.path.join(a.out, f"endmembers_mean_ch{c}.csv"), wl, M[:, c], names)
    print(f"wrote {len(Es) * C + (C if len(Es) > 1 else 0)} CSV files to {a.out}/")


if __name__ == "__main__":
    main()
