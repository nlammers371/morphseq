#!/bin/bash
#$ -N focus_decile_figure
#$ -q trapnell-long.q
#$ -pe serial 1
#$ -l mem_free=48G
#$ -cwd
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260423_focus_artifact_detection/10_scan_output/logs/
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260423_focus_artifact_detection/10_scan_output/logs/
#$ -V

echo "[$(date)] Starting focus rel_entropy decile figures  host=$(hostname)"

conda run -n segmentation_grounded_sam --no-capture-output \
    python /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260423_focus_artifact_detection/09_rel_entropy_decile_bins.py

echo "[$(date)] Done."
