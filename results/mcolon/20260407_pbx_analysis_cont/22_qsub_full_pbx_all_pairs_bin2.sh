#!/usr/bin/env bash
#$ -N pbx_allpairs_bin2
#$ -q trapnell-login.q
#$ -l mfree=8G
#$ -l h_rt=120:00:00
#$ -j y
#$ -pe serial 16
#$ -cwd
#$ -V
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/logs
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/logs

set -euo pipefail

cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# Override at submission time if needed, e.g.:
#   qsub -pe serial 24 -v N_JOBS=24,CLASS_SET=canonical,RUN_TAG=feature4_v2 results/mcolon/20260407_pbx_analysis_cont/22_qsub_full_pbx_all_pairs_bin2.sh
N_JOBS="${N_JOBS:-${NSLOTS:-16}}"
CLASS_SET="${CLASS_SET:-both}"
N_PERM="${N_PERM:-500}"
RUN_TAG="${RUN_TAG:-feature4}"

export N_JOBS CLASS_SET N_PERM RUN_TAG
export PYTHONWARNINGS="ignore:.*multi_class.*deprecated.*:FutureWarning"

results/mcolon/20260407_pbx_analysis_cont/21_run_full_pbx_all_pairs_bin2.sh
