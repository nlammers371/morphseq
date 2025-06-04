#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_cluster --multirun \
    hydra.job.name=sweep01_mspatch_pips \
    model=vae_timm \
    model.ddconfig.name="Efficient-B0-RA","Efficient-B4","ConvNeXt-Tiny","Swin-Tiny","MaxViT-Tiny","Swin-Large","MaxViT-Small","Vit-Large" \
    model.lossconfig.gan_net="ms_patch" \
    model.lossconfig.gan_weight=0.05,0.25