#!/usr/bin/env bash
set -euo pipefail

# === First run ===
echo ">>> Starting first run: run_name=${1:-runA}"
python -m src.run.training_ldm_cluster --run \
    hydra.job.name=ldm_long \
    model.lossconfig.pips_weight=0.05 \
    model.dataconfig.batch_size=16 \
    model.lossconfig.kld_weight=3.0 \
    model.trainconfig.max_epochs=40 \
    model.lossconfig.pips_warmup=10 \
    model.lossconfig.pips_rampup=5 \
    model.lossconfig.kld_warmup= 5 \
    model.lossconfig.kld_rampup= 5


## === Second run ===
#echo ">>> Starting second run: run_name=${2:-runB}"
python -m src.run.training_ldm_cluster --run \
    hydra.job.name=ldm_short \
    model.lossconfig.pips_weight=0.05 \
    model.dataconfig.batch_size=16 \
    model.lossconfig.kld_weight=3.0 \
    model.trainconfig.max_epochs=15
#
#echo "✅ Both runs complete."