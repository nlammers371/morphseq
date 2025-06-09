#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep04_patch4_attn \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.dec_use_local_attn=True \
    model.ddconfig.latent_dim=64,128 \
    model.lossconfig.gan_net="patch4scale" \
    model.lossconfig.gan_weight=0.05,0.1