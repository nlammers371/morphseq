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
    hydra.job.name=ntxent_percep005 \
    model.lossconfig.pips_weight=0.05


echo ">>> Starting second run: run_name=${3:-runB}"
python -m src.run.training_cluster --run \
    hydra.job.name=ntxent_percep001 \
    model.lossconfig.pips_weight=0.01

echo "✅ Both runs complete."