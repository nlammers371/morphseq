#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep01_gan00_pips \
    model=vae_timm \
    model.ddconfig.name="ConvNeXt-Tiny","Swin-Tiny","MaxViT-Tiny","Swin-Large","MaxViT-Small","Vit-Large" \
    model.lossconfig.gan_net="style2" \
    model.lossconfig.pips_weight=1.0 \
    model.lossconfig.schedule_gan=False \
    model.trainconfig.max_epochs=1.0