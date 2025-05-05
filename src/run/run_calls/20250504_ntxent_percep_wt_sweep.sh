#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_cluster --run \
    hydra.job.name=ntxent_percep02 \
    model.lossconfig.pips_weight=0.2


# === Second run ===
echo ">>> Starting second run: run_name=${2:-runB}"
python -m src.run.training_cluster --run \
    hydra.job.name=ntxent_00_n50_T01_bio \
    model=morph_vae_big_cluster \
    model.lossconfig.metric_weight=50.0 \
    model.lossconfig.pips_weight=0.8 \
    model.ddconfig.n_out_channels=16 \
    model.ddconfig.frac_nuisance_latents=0.5 \
    model.dataconfig.batch_size=256 \
    model.lossconfig.kld_weight=5.0 \
    model.lossconfig.bio_only_kld=True \
    model.trainconfig.max_epochs=125

echo "✅ Both runs complete."