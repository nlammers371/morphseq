#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep06_l3_pips \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.dec_use_local_attn=False \
    model.ddconfig.latent_dim=128 \
    model.lossconfig.use_gan=False \
    model.lossconfig.gan_weight=0 \
    model.lossconfig.reconstruction_loss="L2" \
    model.lossconfig.pips_weight=0,3.25,7.5,15