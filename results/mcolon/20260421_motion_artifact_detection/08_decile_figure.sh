#!/bin/bash
#$ -N decile_figure
#$ -q trapnell-long.q
#$ -pe serial 1
#$ -l mem_free=48G
#$ -cwd
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -V

echo "[$(date)] Starting decile figure  host=$(hostname)"

conda run -n segmentation_grounded_sam --no-capture-output \
    python /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/08_decile_ranked_figure.py

echo "[$(date)] Done."
