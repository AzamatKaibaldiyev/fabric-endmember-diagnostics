#!/usr/bin/env bash
# Every training run behind the paper and the further analyses. Each block is
# independent; run the ones you need. One run trains for 160 epochs on one GPU.
#
#   bash scripts/train_paper_runs.sh accuracy   # or: holdout smoothness input nopure all
#
# Checkpoints go to chkpts/ with names encoding every flag, e.g.
#   cleanv3conv-atto_ch2_seed0_fabricVF10_single_ablfull_multistep_regl1_unmixlinear4_best_acc.pt
set -euo pipefail

DATA=${DATA:-data/FabricSpectra/}
COMMON="--data_root $DATA --model atto --no_multiscale --reg_loss l1"
# Unmixing decoder: lambda = 4, random endmember initialisation, linear mixing.
DEC="--physics_decoder --physics_weight 4 --endmember_init random \
     --physics_mixing linear --physics_abundance normalize"
# Final configuration of the paper: decoder + smoothness prior 1e-1.
FINAL="$DEC --endmember_smooth_weight 1e-1"

run() { echo ">>> python train.py $*"; python train.py "$@"; }

accuracy() {   # accuracy with and without the decoder, and the unconstrained-decoder control
  for s in 0 1 2 3 4; do run $COMMON --seed $s; done                    # no decoder
  for s in 0 1 2;     do run $COMMON $DEC --seed $s; done               # decoder, no prior
  for s in 0 1 2;     do run $COMMON $FINAL --seed $s; done             # final configuration
  run $COMMON --physics_decoder --physics_weight 4 --physics_mixing mlp --seed 0   # MLP control
}

smoothness() {   # smoothness sweep and decoder variants (the no-prior runs are in 'accuracy')
  for w in 1e-3 1e-2 1e0; do
    for s in 0 1 2; do run $COMMON $DEC --endmember_smooth_weight $w --seed $s; done
  done
  for s in 0 1 2; do
    run $COMMON $DEC --physics_mixing bilinear --seed $s
    run $COMMON $DEC --physics_mixing ppnm --seed $s
    run $COMMON $DEC --physics_mixing bilinear --endmember_smooth_weight 1e-2 --seed $s
  done
}

input() {   # input representation (the raw-input runs are in 'accuracy')
  for s in 0 1 2; do
    run $COMMON $FINAL --input_mode snv --seed $s
    run $COMMON $FINAL --input_mode raw+snv --seed $s
  done
}

nopure() {   # no pure spectra in training
  for s in 0 1 2; do run $COMMON $FINAL --drop_pure 99 --seed $s; done
}

holdout() {   # one fibre held out of training (novelty detection)
  for f in SP A W S L; do
    for s in 0 1 2; do run $COMMON $FINAL --holdout_classes $f --seed $s; done
  done
}

case "${1:-}" in
  accuracy) accuracy ;; holdout) holdout ;; smoothness) smoothness ;; input) input ;; nopure) nopure ;;
  all) accuracy; holdout; smoothness; input; nopure ;;
  *) echo "usage: $0 {accuracy|holdout|smoothness|input|nopure|all}"; exit 1 ;;
esac
