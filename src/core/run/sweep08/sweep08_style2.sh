#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep08_style2 \
    model=metric_vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.dec_use_local_attn=False,True \
    model.ddconfig.latent_dim=128 \
    model.dataconfig.batch_size=128 \
    model.lossconfig.gan_net="style2" \
    model.lossconfig.gan_weight=0.1 \
    model.lossconfig.kld_weight=5 \
    model.lossconfig.pips_weight=0,7.5 