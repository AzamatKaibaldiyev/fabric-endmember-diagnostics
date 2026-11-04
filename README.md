# Learned endmembers make a spectral network inspectable

Code and supplementary material for *Learned endmembers make a spectral network
inspectable: physics-informed unmixing for near-infrared fibre analysis*
(submitted to ESANN 2027).

A 1-D ConvNeXtV2 network predicts which of twelve fibres a fabric contains and
in what proportions. A decoder based on the Beer–Lambert law then has to rebuild
the measured near-infrared spectrum from those proportions and twelve learned
pure-fibre spectra (endmembers). In chemometric terms the decoder is a classical
least-squares (K-matrix) model learned end-to-end. It leaves accuracy unchanged,
and the learned endmembers become something you can check: against measured
pure spectra, across random seeds, and as a predictor of which unseen fibres the
model will notice.

## Supplementary material

All twelve learned endmembers against the measured pure spectra (the paper's
Fig. 2 shows four of them):

- [`supplementary/endmembers_all_raw.pdf`](supplementary/endmembers_all_raw.pdf):
  raw absorbance, the space the decoder is trained in.
- [`supplementary/endmembers_all_snv.pdf`](supplementary/endmembers_all_snv.pdf):
  after standard normal variate (SNV) correction, which removes level and scale
  and compares shape only.

See [`supplementary/README.md`](supplementary/README.md) for how to read them.

## Install

```bash
pip install -r requirements.txt
```

Training needs a CUDA GPU. The analysis scripts also run on CPU.

## Data

The experiments use the public FabricSpectra corpus (Kim and Chi, ICASSP 2026).
Download it and place the six `.npy` files in `data/FabricSpectra/`; see
[`data/README.md`](data/README.md). The data are not redistributed here.

## Training

One seed of the paper's final configuration:

```bash
python train.py --data_root data/FabricSpectra/ --model atto --no_multiscale --reg_loss l1 \
  --physics_decoder --physics_weight 4 --endmember_init random \
  --physics_mixing linear --physics_abundance normalize \
  --endmember_smooth_weight 1e-1 --seed 0
```

Every training run behind the paper:

```bash
bash scripts/train_paper_runs.sh accuracy   # with/without decoder, MLP control (3-5 seeds)
bash scripts/train_paper_runs.sh holdout    # one fibre held out at a time (novelty detection)
```

The same script also contains the runs for the further analyses below
(`smoothness`, `input`, `nopure`).

- **Split.** A tenth of the fabrics in the training set is held out for
  checkpoint selection; all scans of one fabric stay on the same side. The
  official test split is scored once at the end. Split indices are cached in
  [`splits/`](splits/README.md); the files for seeds 0–4 are the ones used in
  the paper, and each run prints the split's fingerprint.
- **Outputs.** Checkpoints go to `chkpts/`, named after every flag that differs
  from the defaults. The final test metrics are printed at the end of each run.
- **Logging.** Weights & Biases is off by default; set `WANDB_MODE=online` to
  log.

## Reproducing the paper

Below, `D=data/FabricSpectra/` and `FINAL` stands for the three final-configuration
checkpoints (seeds 0–2), e.g.
`chkpts/cleanv3conv-atto_ch2_seed{0,1,2}_fabricVF10_single_ablfull_multistep_regl1_unmixlinear4_best_acc.pt`.

| Result | Command |
|---|---|
| Accuracy (Sec. 3.1) | test metrics printed at the end of the `accuracy` runs |
| Table 1 and Fig. 2 | `python analysis/plot_endmembers.py --data_root $D --ckpt $FINAL --paper --fibres SP,M,W,C --width 4.8 --bw --save fig_row` (prints the Table 1 metrics; Fig. 2 is `fig_row_raw.pdf`) |
| Supplementary figures | `python analysis/plot_endmembers.py --data_root $D --ckpt $FINAL --paper --save endmembers_all` |
| Fig. 3 | `python analysis/plot_isolation_auroc.py fig3_isolation` (draws the measured values stored in the script) |
| Detection AUROC per fraction (Sec. 3.3) | `python analysis/openset_stratified.py --data_root $D --ckpt <held-out checkpoint> --holdout SP` (per fibre and seed, from the `holdout` runs) |
| Predicting the model's own errors (Sec. 3.3) | `python analysis/misclassification_detection.py --data_root $D --ckpt <checkpoint>` |
| Fig. 1 | `figures/fig_architecture_wide.tex` (TikZ) |

`--bw` draws the learned endmember dashed, so the figure still reads when printed
in black and white.

## Further analyses

Scripts for analyses beyond the conference paper:

| Analysis | Script |
|---|---|
| Endmember similarity and conditioning (Gram matrix) | `analysis/endmember_report.py --data_root $D --ckpt $FINAL` |
| Smoothness sweep and decoder variants | `endmember_report.py` on three checkpoints of each `smoothness` variant |
| Fully constrained least squares vs. the network | `analysis/fcls_vs_classifier.py --data_root $D --ckpt <checkpoint>` |
| Input representation (raw, SNV, raw+SNV) | test metrics of the `input` runs |
| Composition mass passed to the decoder, per fibre and seed | `analysis/abundance_audit.py --data_root $D --ckpt <checkpoints>` |
| Training without pure spectra | `analysis/drop_pure_audit.py --data_root $D --threshold 99`, and `endmember_report.py` on the `nopure` checkpoints |
| Detection AUROC against unknown fraction (curves) | `analysis/plot_fig4.py` (values stored in `plot_openset_curves.py`) |
| Physics-weight calibration | `analysis/physics_probe.py --data_root $D --comp_head linear` |
| Learned endmembers as CSV | `analysis/export_endmembers_csv.py --ckpt $FINAL` |
| What raw, SNV and detrended angles respond to (synthetic) | `analysis/mechanism_demo.py` |

Checkpoints record their input representation (`raw`, `snv`, `raw+snv`). The
analysis scripts read it from the checkpoint, and `--input_mode` overrides it.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT; see [`LICENSE`](LICENSE).
# fabric-endmember-diagnostics
