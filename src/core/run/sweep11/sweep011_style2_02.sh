#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep11_style2_03 \
    model.wandb.project="sweep11" \
    model=metric_vae_timm \
    model.lossconfig.pips_net="vgg" \
    model.lossconfig.gan_weight=0.175 \
    model.lossconfig.schedule_metric=False \
    model.lossconfig.kld_weight=5 \
    model.lossconfig.pips_weight=4,10 \
    model.ddconfig.latent_dim=128 \
    model.dataconfig.batch_size=128 \
    model.ddconfig.dec_use_local_attn=True \
    model.lossconfig.gan_net="style2" \
    model.ddconfig.name="Swin-Tiny" \
    model.trainconfig.max_epochs=50 