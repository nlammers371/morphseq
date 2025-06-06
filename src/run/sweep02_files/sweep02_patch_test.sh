#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep02_patch_pips \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.lossconfig.gan_net="patch" \
    model.lossconfig.gan_weight=0.05