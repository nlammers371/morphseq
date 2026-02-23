#!/bin/bash
#$ -q trapnell-login.q
#$ -l mfree=45G
#$ -l h_rt=96:0:0
#$ -pe serial 1
#$ -N test_gen_metrics

# Load conda environment
source /net/trapnell/vol1/home/mdcolon/software/miniconda3/miniconda.sh # Modify this to match your conda installation path
conda activate vae_env_cluster

# Run Python script
python /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20241216/test_gen_perform_metrics.py