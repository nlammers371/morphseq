#!/bin/bash
# filepath: /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/run_gdino.qsub.sh

# run from the current working directory
#$ -cwd

# job name
#$ -N gdino_detect

# queue and resources
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=30G
#$ -pe serial 1

# stdout & stderr
#$ -o logs/gdino_detect.$JOB_ID.out
#$ -e logs/gdino_detect.$JOB_ID.err

# load any modules you need
# Initialize conda properly for qsub environment
CONDA_BASE="/net/trapnell/vol1/home/mdcolon/software/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate segmentation_grounded_sam

# Verify activation worked
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python3

# run your python script
python3 /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/03_initial_gdino_detections.py \
  --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml \
  --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json \
  --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations.json \
  --prompts "individual embryo"