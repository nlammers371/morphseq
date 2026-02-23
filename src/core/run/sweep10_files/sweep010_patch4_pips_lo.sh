#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-Blurg}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep10_pips_lo \
    model=metric_vae_timm \
    model.wandb.project="sweep10" \
    model.lossconfig.pips_net="squeeze","vgg" \
    model.lossconfig.lambda_feat_match=10,0 \
    model.lossconfig.gan_weight=0.1 \
    model.lossconfig.kld_weight=5 \
    model.lossconfig.pips_weight=1 \
    model.ddconfig.latent_dim=128 \
    model.dataconfig.batch_size=128 \
    model.ddconfig.dec_use_local_attn=True \
    model.lossconfig.gan_net="patch4scale" \
    model.ddconfig.name="Swin-Tiny" \
    model.trainconfig.max_epochs=60 