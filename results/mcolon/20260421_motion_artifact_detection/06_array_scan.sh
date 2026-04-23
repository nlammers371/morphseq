#!/bin/bash
#$ -N motion_qc_array
#$ -q trapnell-long.q
#$ -pe serial 16
#$ -t 1-12
#$ -cwd
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -V

# T=16–112 (97 timepoints) split across 12 tasks, ~8 timepoints each.
# T=0–15 already done (cache hits if revisited).
# Each task writes its own chunk_summaries_tXXX_tYYY.csv — no write collisions.
# .npz grid writes are safe: each (t,p) file is unique.

set -euo pipefail

MORPHSEQ=/net/trapnell/vol1/home/mdcolon/proj/morphseq
SCRIPT=$MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_full_nd2_scan.py
LOG_DIR=$MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs
mkdir -p $LOG_DIR

# Map SGE_TASK_ID (1-12) → t_start, t_end
# Chunks: [16,24) [24,32) [32,40) [40,48) [48,56) [56,64)
#         [64,72) [72,80) [80,88) [88,96) [96,104) [104,113)
declare -A T_START T_END
T_START[1]=16;  T_END[1]=24
T_START[2]=24;  T_END[2]=32
T_START[3]=32;  T_END[3]=40
T_START[4]=40;  T_END[4]=48
T_START[5]=48;  T_END[5]=56
T_START[6]=56;  T_END[6]=64
T_START[7]=64;  T_END[7]=72
T_START[8]=72;  T_END[8]=80
T_START[9]=80;  T_END[9]=88
T_START[10]=88; T_END[10]=96
T_START[11]=96; T_END[11]=104
T_START[12]=104;T_END[12]=113

TS=${T_START[$SGE_TASK_ID]}
TE=${T_END[$SGE_TASK_ID]}

echo "[$(date)] Task $SGE_TASK_ID: T=${TS}–$((TE-1))  host=$(hostname)  NSLOTS=$NSLOTS"

conda run -n segmentation_grounded_sam --no-capture-output \
    python $SCRIPT \
    --t-start $TS \
    --t-end   $TE \
    --workers 16

echo "[$(date)] Task $SGE_TASK_ID done."
