#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-SwinT}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep03_SwinT \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.lossconfig.gan_net="ms_patch","style2_big" \
    model.lossconfig.gan_weight=0.05,0.01