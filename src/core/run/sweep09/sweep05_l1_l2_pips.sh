#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep09_sched \
    model=metric_vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.dec_use_local_attn=False \
    model.ddconfig.latent_dim=128 \
    model.lossconfig.use_gan=False \
    model.lossconfig.gan_weight=0 \
    model.lossconfig.reconstruction_loss="L1","L2" \
    model.lossconfig.schedule_kld=False,True \
    model.lossconfig.pips_weight=0