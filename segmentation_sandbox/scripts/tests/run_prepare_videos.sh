#!/bin/bash

# Run MorphSeq Pipeline Script 1: Prepare Videos and Create Metadata

python 01_prepare_videos.py \
    --directory_with_experiments /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images \
    --output_parent_dir /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data \
    --experiments_to_process "20250612_30hpf_ctrl_atf6" \
    --verbose
