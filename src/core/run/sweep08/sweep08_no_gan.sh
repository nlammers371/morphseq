#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep08_no_gan \
    model=metric_vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.dec_use_local_attn=False,True \
    model.ddconfig.latent_dim=128 \
    model.dataconfig.batch_size=128 \
    model.lossconfig.gan_net="ms_patch" \
    model.lossconfig.gan_weight=0 \
    model.lossconfig.kld_weight=5 \
    model.lossconfig.pips_weight=0,7.5 \
    model.lossconfig.use_gan=False 