#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep01_gan00_pips \
    model=vae_timm \
    model.lossconfig.pips_weight=7.5 \
    model.lossconfig.kld_weight=1.0 \
    model.trainconfig.max_epochs=35 