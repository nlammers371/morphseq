#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training --run \
    hydra.job.name=ntxent_squeeze_percep005_beta01_margin2 \
    model.lossconfig.pips_weight=0.05 \
    model.lossconfig.pips_net="squeeze" \
    model.lossconfig.kld_weight=1.0 \
    model.lossconfig.margin=2 \
    model.ddconfig.n_out_channels=32


# === Second run ===
echo ">>> Starting second run: run_name=${2:-runB}"
python -m src.run.training --run \
    hydra.job.name=ntxent_squeeze_percep005_beta01_margin2_T1 \
    model.lossconfig.pips_weight=0.05 \
    model.lossconfig.pips_net="squeeze" \
    model.lossconfig.kld_weight=1.0 \
    model.lossconfig.margin=2 \
    model.lossconfig.temperature=1 \
    model.ddconfig.n_out_channels=32


echo ">>> Starting second run: run_name=${3:-runB}"
python -m src.run.training --run \
    hydra.job.name=ntxent_percep0000 \
    model.lossconfig.pips_weight=1e-5 \
    model.lossconfig.kld_weight=3.0 \
    model.lossconfig.margin=2 \
    model.lossconfig.temperature=1 \
    model.ddconfig.n_out_channels=32

echo "✅ Both runs complete."