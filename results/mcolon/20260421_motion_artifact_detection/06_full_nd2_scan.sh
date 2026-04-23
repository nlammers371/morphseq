#!/bin/bash
#$ -N motion_qc_scan
#$ -q trapnell-short.q
#$ -pe serial 16
#$ -l h_rt=02:00:00
#$ -l mfree=8G
#$ -cwd
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/qsub.out
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/qsub.err
#$ -V

# Estimated runtime: ~10,735 stacks × 6.7s ÷ 16 workers ≈ 75 min wall time
# h_rt=2h gives ~60% headroom

set -euo pipefail

MORPHSEQ=/net/trapnell/vol1/home/mdcolon/proj/morphseq
SCRIPT=$MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_full_nd2_scan.py

mkdir -p $MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_scan_output

echo "[$(date)] Starting full ND2 scan — 16 workers"
echo "Host: $(hostname)"
echo "NSLOTS: $NSLOTS"

conda run -n segmentation_grounded_sam --no-capture-output \
    python $SCRIPT --workers 16

echo "[$(date)] Done."
