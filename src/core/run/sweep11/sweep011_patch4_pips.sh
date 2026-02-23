#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep11_patch4 \
    model.wandb.project="sweep11" \
    model=metric_vae_timm \
    model.lossconfig.pips_net="vgg" \
    model.lossconfig.schedule_metric=False \
    model.lossconfig.gan_weight=0.1 \
    model.lossconfig.kld_weight=5 \
    model.lossconfig.pips_weight=4,7,10 \
    model.ddconfig.latent_dim=128 \
    model.dataconfig.batch_size=128 \
    model.ddconfig.dec_use_local_attn=True \
    model.lossconfig.gan_net="patch4scale" \
    model.ddconfig.name="Swin-Tiny" \
    model.trainconfig.max_epochs=50 