#!/bin/bash

# Playground Build03A Test - True Isolated Environment
# Purpose: Test snip extraction in completely isolated environment with no pre-existing outputs

echo "🎮 Playground Build03A Test - True Isolation"
echo "=============================================="

# Ensure conda environment
if [[ "$CONDA_DEFAULT_ENV" != "segmentation_grounded_sam" ]]; then
    echo "❌ Wrong conda environment. Please run: conda activate segmentation_grounded_sam"
    exit 1
fi

# Change to project directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

PLAYGROUND_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground"

echo "🔍 Testing playground environment setup:"
echo "📁 Root: $PLAYGROUND_ROOT"
echo "🔗 Symlinks: $(find $PLAYGROUND_ROOT -type l | wc -l) symlinks created"
echo "📊 Images available: $(ls $PLAYGROUND_ROOT/built_image_data/stitched_FF_images/20250612_30hpf_ctrl_atf6/ | wc -l)"
echo ""

echo "🧪 Critical Validation - Empty Output Directories:"
echo "📂 training_data/bf_embryo_snips: $(ls $PLAYGROUND_ROOT/training_data/bf_embryo_snips/ 2>/dev/null | wc -l) files"
echo "📂 metadata/combined_metadata_files: $(ls $PLAYGROUND_ROOT/metadata/combined_metadata_files/ 2>/dev/null | wc -l) files" 
echo ""

if [ $(ls $PLAYGROUND_ROOT/training_data/bf_embryo_snips/ 2>/dev/null | wc -l) -gt 0 ]; then
    echo "⚠️  WARNING: Output directory not empty - removing for true isolation"
    rm -rf $PLAYGROUND_ROOT/training_data/bf_embryo_snips/*
fi

if [ $(ls $PLAYGROUND_ROOT/metadata/combined_metadata_files/ 2>/dev/null | wc -l) -gt 0 ]; then
    echo "⚠️  WARNING: Metadata output directory not empty - removing for true isolation"
    rm -rf $PLAYGROUND_ROOT/metadata/combined_metadata_files/*
fi

echo "✅ Output directories confirmed empty - ready for true validation"
echo ""

echo "🧪 Running Build03A with 2 embryos, 1 frame each:"
echo "Expected: Should create snips in playground, not find pre-existing files"
echo ""

python -m src.run_morphseq_pipeline.cli build03 \
  --root "$PLAYGROUND_ROOT" \
  --exp 20250612_30hpf_ctrl_atf6 \
  --sam2-csv "$PLAYGROUND_ROOT/sam2_metadata_playground.csv" \
  --by-embryo 2 \
  --frames-per-embryo 1

RESULT=$?

echo ""
echo "🔍 POST-TEST VALIDATION:"
echo "========================"

if [ $RESULT -eq 0 ]; then
    echo "✅ Build03A completed without errors"
    
    # Check metadata output
    DF01_FILE="$PLAYGROUND_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv"
    if [ -f "$DF01_FILE" ]; then
        echo "✅ df01.csv created: $(wc -l < "$DF01_FILE") rows"
        echo "📅 Created: $(stat -c %y "$DF01_FILE")"
    else
        echo "❌ df01.csv NOT created"
    fi
    
    # Check snip outputs
    SNIPS_DIR="$PLAYGROUND_ROOT/training_data/bf_embryo_snips"
    if [ -d "$SNIPS_DIR" ]; then
        snip_count=$(find "$SNIPS_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.tiff" | wc -l)
        echo "📊 Snip images created: $snip_count"
        
        if [ $snip_count -gt 0 ]; then
            echo "🎉 SNIP EXTRACTION SUCCESS!"
            echo "📅 Sample file timestamp:"
            find "$SNIPS_DIR" -name "*.jpg" | head -1 | xargs stat -c "%y %n"
            echo "📋 Sample files:"
            find "$SNIPS_DIR" -name "*.jpg" | head -3
            
            echo ""
            echo "🎯 VALIDATION COMPLETE: Pipeline working with true isolation ✅"
            
        else
            echo "❌ NO SNIP IMAGES FOUND"
            echo "🔍 This confirms snip extraction is still failing"
        fi
    else
        echo "❌ Snips directory not created"
    fi
    
else
    echo "❌ Build03A FAILED with exit code $RESULT"
    echo "🔍 Check errors above for root cause"
fi

echo ""
echo "📊 PLAYGROUND STATUS SUMMARY:"
echo "  Inputs (symlinks): $(find $PLAYGROUND_ROOT -type l | wc -l)"
echo "  Output files: $(find $PLAYGROUND_ROOT -type f -newer "$0" | wc -l)"
echo "  Total disk usage: $(du -sh $PLAYGROUND_ROOT | cut -f1)"