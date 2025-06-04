#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training --multirun \
    hydra.job.name=sweep01_gan00_pips \
    model=vae_timm \
    model.ddconfig.name="MaxViT-Tiny" \
    model.lossconfig.gan_net="style2" \
    model.lossconfig.gan_weight=1.0 \
    model.trainconfig.max_epochs=7.0