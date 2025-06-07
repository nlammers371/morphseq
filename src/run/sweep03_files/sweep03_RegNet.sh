#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-RegNet}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep03_RegNet \
    model=vae_timm \
    model.ddconfig.name="RegNet-Y" \
    model.lossconfig.gan_net="ms_patch","style2_big" \
    model.lossconfig.gan_weight=0.05,0.01