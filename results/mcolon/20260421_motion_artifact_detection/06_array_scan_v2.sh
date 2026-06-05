#!/bin/bash
#$ -N motion_qc_v2
#$ -q trapnell-long.q
#$ -pe serial 4
#$ -l mem_free=32G
#$ -t 1-12
#$ -cwd
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs/
#$ -V

# V2: 4 workers instead of 16 to avoid OOM.
# Peak memory per worker: ~1.5 GB (288 MB raw stack + NCC intermediates).
# 4 workers × ~1.5 GB = ~6 GB working set + headroom → request 32G to be safe.
#
# Missing t-values (84 total): 20-55, 57-79, 82-87, 90-95, 98-103, 106-112
# Split into 12 roughly equal chunks of ~7 t-values each.

set -euo pipefail

MORPHSEQ=/net/trapnell/vol1/home/mdcolon/proj/morphseq
SCRIPT=$MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_full_nd2_scan.py
LOG_DIR=$MORPHSEQ/results/mcolon/20260421_motion_artifact_detection/06_scan_output/array_logs
mkdir -p $LOG_DIR

# Map SGE_TASK_ID (1-12) → t_start, t_end (within missing ranges)
# Chunk plan covers: 20-55, 57-79, 82-87, 90-95, 98-103, 106-112
# Task 1:  t=20-26   (7 t-values = 665 stacks)
# Task 2:  t=27-33   (7 t-values = 665 stacks)
# Task 3:  t=34-40   (7 t-values = 665 stacks)
# Task 4:  t=41-47   (7 t-values = 665 stacks)
# Task 5:  t=48-55   (8 t-values = 760 stacks)
# Task 6:  t=57-63   (7 t-values = 665 stacks)
# Task 7:  t=64-71   (8 t-values = 760 stacks)
# Task 8:  t=72-79   (8 t-values = 760 stacks)
# Task 9:  t=82-87   (6 t-values = 570 stacks)
# Task 10: t=90-95   (6 t-values = 570 stacks)
# Task 11: t=98-103  (6 t-values = 570 stacks)
# Task 12: t=106-112 (7 t-values = 665 stacks)
declare -A T_START T_END
T_START[1]=20;  T_END[1]=27
T_START[2]=27;  T_END[2]=34
T_START[3]=34;  T_END[3]=41
T_START[4]=41;  T_END[4]=48
T_START[5]=48;  T_END[5]=56
T_START[6]=57;  T_END[6]=64
T_START[7]=64;  T_END[7]=72
T_START[8]=72;  T_END[8]=80
T_START[9]=82;  T_END[9]=88
T_START[10]=90; T_END[10]=96
T_START[11]=98; T_END[11]=104
T_START[12]=106;T_END[12]=113

TS=${T_START[$SGE_TASK_ID]}
TE=${T_END[$SGE_TASK_ID]}

echo "[$(date)] Task $SGE_TASK_ID: T=${TS}–$((TE-1))  host=$(hostname)  NSLOTS=$NSLOTS"

conda run -n segmentation_grounded_sam --no-capture-output \
    python $SCRIPT \
    --t-start $TS \
    --t-end   $TE \
    --workers 4

echo "[$(date)] Task $SGE_TASK_ID done."
