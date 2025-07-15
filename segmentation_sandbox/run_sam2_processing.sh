#!/bin/bash
# SAM2 Video Processing Script
# ============================
#
# This script runs the SAM2 video processing pipeline using the corrected paths
# and generates tracking annotations for embryo segmentation.

set -e  # Exit on any error

# Configuration paths
SANDBOX_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox"
CONFIG_PATH="$SANDBOX_ROOT/configs/pipeline_config.yaml"
ANNOTATIONS_PATH="$SANDBOX_ROOT/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json"
OUTPUT_PATH="$SANDBOX_ROOT/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json"

# Script path
SCRIPT_PATH="$SANDBOX_ROOT/scripts/04_sam2_video_processing.py"

echo "🎬 SAM2 Video Processing Pipeline"
echo "================================="
echo "📁 Sandbox root: $SANDBOX_ROOT"
echo "⚙️  Config: $CONFIG_PATH"
echo "📋 Annotations: $ANNOTATIONS_PATH"
echo "💾 Output: $OUTPUT_PATH"
echo ""

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$ANNOTATIONS_PATH" ]; then
    echo "❌ Annotations file not found: $ANNOTATIONS_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "📁 Created output directory: $OUTPUT_DIR"

# Navigate to sandbox root
cd "$SANDBOX_ROOT"
echo "📂 Working directory: $(pwd)"

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "🔧 Activating conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo ""
echo "🚀 Starting SAM2 video processing..."
echo "⏰ Started at: $(date)"
echo ""

# Run the SAM2 processing script
python scripts/04_sam2_video_processing.py \
    --config "$CONFIG_PATH" \
    --annotations "$ANNOTATIONS_PATH" \
    --output "$OUTPUT_PATH" \
    --target-prompt "individual embryo" \
    --segmentation-format rle \
    --verbose \
    --max-videos 5 \
    --save-interval 10

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SAM2 processing completed successfully!"
    echo "⏰ Finished at: $(date)"
    echo "📁 Results saved to: $OUTPUT_PATH"
    
    # Check output file size
    if [ -f "$OUTPUT_PATH" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "📊 Output file size: $FILE_SIZE"
    fi
    
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Generate videos: python scripts/video_generation/sam2_video_generator.py"
    echo "   2. Review results in: $OUTPUT_PATH"
    
else
    echo ""
    echo "❌ SAM2 processing failed!"
    echo "⏰ Failed at: $(date)"
    exit 1
fi



# Configuration paths
SANDBOX_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox"
CONFIG_PATH="$SANDBOX_ROOT/configs/pipeline_config.yaml"
ANNOTATIONS_PATH="$SANDBOX_ROOT/data/annotation_and_masks/gdino_annotations/gdino_annotations.json"
OUTPUT_PATH="$SANDBOX_ROOT/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations.json"

# Script path
SCRIPT_PATH="$SANDBOX_ROOT/scripts/04_sam2_video_processing.py"

echo "🎬 SAM2 Video Processing Pipeline"
echo "================================="
echo "📁 Sandbox root: $SANDBOX_ROOT"
echo "⚙️  Config: $CONFIG_PATH"
echo "📋 Annotations: $ANNOTATIONS_PATH"
echo "💾 Output: $OUTPUT_PATH"
echo ""

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$ANNOTATIONS_PATH" ]; then
    echo "❌ Annotations file not found: $ANNOTATIONS_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "📁 Created output directory: $OUTPUT_DIR"

# Navigate to sandbox root
cd "$SANDBOX_ROOT"
echo "📂 Working directory: $(pwd)"

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "🔧 Activating conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo ""
echo "🚀 Starting SAM2 video processing..."
echo "⏰ Started at: $(date)"
echo ""

# Run the SAM2 processing script
python scripts/04_sam2_video_processing.py \
    --config "$CONFIG_PATH" \
    --annotations "$ANNOTATIONS_PATH" \
    --output "$OUTPUT_PATH" \
    --target-prompt "individual embryo" \
    --segmentation-format rle \
    --verbose \
    --max-videos 5 \
    --save-interval 10

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SAM2 processing completed successfully!"
    echo "⏰ Finished at: $(date)"
    echo "📁 Results saved to: $OUTPUT_PATH"
    
    # Check output file size
    if [ -f "$OUTPUT_PATH" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "📊 Output file size: $FILE_SIZE"
    fi
    
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Generate videos: python scripts/video_generation/sam2_video_generator.py"
    echo "   2. Review results in: $OUTPUT_PATH"
    
else
    echo ""
    echo "❌ SAM2 processing failed!"
    echo "⏰ Failed at: $(date)"
    exit 1
fi

