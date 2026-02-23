#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep07_test \
    model=metric_vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.lossconfig.use_gan=False \
    model.lossconfig.gan_weight=0 \
    model.lossconfig.reconstruction_loss="L1" \
    model.lossconfig.pips_weight=0 \
    model.lossconfig.kld_weight=5,10,25 \
    model.lossconfig.schedule_metric=zip(True,False) \
    model.lossconfig.schedule_kld=zip(True,False)