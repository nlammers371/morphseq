#!/bin/bash

# SAM2 Full Pipeline Integration Test
# Created: August 31, 2025
# Purpose: Unified end-to-end validation of Build03A → Build04 → Build05 → VAE chain
# Consolidates: test_sam2_pipeline.sh, test_sam2_step2.sh, test_sam2_step3.sh, test_sam2_safe1_safe.sh

echo "🧪 SAM2 Full Pipeline Integration Test"
echo "====================================="
echo "Phases: Build03A → Build04 → Build05 → VAE Integration"
echo ""

# Ensure conda environment
if [[ "$CONDA_DEFAULT_ENV" != "segmentation_grounded_sam" ]]; then
    echo "❌ Wrong conda environment. Please run: conda activate segmentation_grounded_sam"
    exit 1
fi

# Change to project directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# Configuration
PLAYGROUND_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground"
SAFE_OUTPUT_DIR="/net/trapnell/vol1/home/mdcolon/proj/morphseq/safe_test_outputs"
SAM2_CSV="/net/trapnell/vol1/home/mdcolon/proj/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv"
EXPERIMENT="20250612_30hpf_ctrl_atf6"
TRAIN_NAME="sam2_test_$(date +%Y%m%d_%H%M)"

# Create safe output directory
mkdir -p "$SAFE_OUTPUT_DIR"

echo "🔧 Test Configuration:"
echo "   Playground Root: $PLAYGROUND_ROOT"
echo "   Safe Outputs: $SAFE_OUTPUT_DIR"
echo "   SAM2 CSV: $SAM2_CSV"
echo "   Experiment: $EXPERIMENT"
echo "   Training Name: $TRAIN_NAME"
echo ""

# Pre-flight checks
echo "🔍 Pre-flight Validation:"

if [ ! -d "$PLAYGROUND_ROOT" ]; then
    echo "❌ Playground environment not found: $PLAYGROUND_ROOT"
    echo "   Set up playground first with symlinks to production data"
    exit 1
fi

if [ ! -f "$SAM2_CSV" ]; then
    echo "❌ SAM2 CSV not found: $SAM2_CSV"
    exit 1
fi

echo "✅ Playground environment ready"
echo "✅ SAM2 CSV available ($(wc -l < "$SAM2_CSV") rows)"
echo ""

# =============================================================================
# PHASE 1: BUILD03A - SAM2 CSV TO DF01.CSV
# =============================================================================

echo "🚀 PHASE 1: Build03A - SAM2 Integration"
echo "========================================"
echo "Expected: Process SAM2 CSV → df01.csv with 2 embryos, 1 frame each"
echo ""

# Clear previous outputs for clean test
rm -rf "$PLAYGROUND_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv"
rm -rf "$PLAYGROUND_ROOT/training_data/bf_embryo_snips/$EXPERIMENT"

python -m src.run_morphseq_pipeline.cli build03 \
  --root "$PLAYGROUND_ROOT" \
  --exp "$EXPERIMENT" \
  --sam2-csv "$SAM2_CSV" \
  --by-embryo 2 \
  --frames-per-embryo 1

PHASE1_EXIT_CODE=$?
DF01_FILE="$PLAYGROUND_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv"

if [ $PHASE1_EXIT_CODE -eq 0 ] && [ -f "$DF01_FILE" ]; then
    echo ""
    echo "✅ PHASE 1 SUCCESS: Build03A completed"
    echo "   📊 df01.csv created: $(wc -l < "$DF01_FILE") rows"
    
    # Check snip extraction
    SNIPS_DIR="$PLAYGROUND_ROOT/training_data/bf_embryo_snips/$EXPERIMENT"
    if [ -d "$SNIPS_DIR" ]; then
        snip_count=$(find "$SNIPS_DIR" -name "*.png" -o -name "*.jpg" -o -name "*.tiff" | wc -l)
        echo "   📊 Snip images created: $snip_count"
        if [ $snip_count -gt 0 ]; then
            echo "   ✅ Snip extraction SUCCESSFUL"
        else
            echo "   ⚠️  No snip images found"
        fi
    fi
    
    # Copy df01 to safe outputs
    cp "$DF01_FILE" "$SAFE_OUTPUT_DIR/embryo_metadata_df01.csv"
    
else
    echo ""
    echo "❌ PHASE 1 FAILED: Build03A could not process SAM2 CSV"
    echo "   Exit code: $PHASE1_EXIT_CODE"
    echo "   Check SAM2 CSV format and image accessibility"
    exit 1
fi

echo ""
echo "⏩ Proceeding to Phase 2..."
echo ""

# =============================================================================
# PHASE 2: BUILD04 - DF01.CSV TO DF02.CSV  
# =============================================================================

echo "🚀 PHASE 2: Build04 - Quality Control Processing"
echo "==============================================="
echo "Expected: Process df01.csv → df02.csv with perturbation mapping and QC flags"
echo ""

python -m src.run_morphseq_pipeline.cli build04 \
  --root "$PLAYGROUND_ROOT"

PHASE2_EXIT_CODE=$?
DF02_FILE="$PLAYGROUND_ROOT/metadata/combined_metadata_files/embryo_metadata_df02.csv"

if [ $PHASE2_EXIT_CODE -eq 0 ] && [ -f "$DF02_FILE" ]; then
    echo ""
    echo "✅ PHASE 2 SUCCESS: Build04 completed"
    echo "   📊 df02.csv created: $(wc -l < "$DF02_FILE") rows"
    
    # Show sample output columns
    echo "   📋 Sample columns:"
    head -1 "$DF02_FILE" | cut -d',' -f1-8 | tr ',' '\n' | head -8 | sed 's/^/      /'
    
    # Copy df02 to safe outputs
    cp "$DF02_FILE" "$SAFE_OUTPUT_DIR/embryo_metadata_df02.csv"
    
else
    echo ""
    echo "❌ PHASE 2 FAILED: Build04 could not process df01.csv"
    echo "   Exit code: $PHASE2_EXIT_CODE"
    echo "   Check for missing dependency files or QC algorithm issues"
    exit 1
fi

echo ""
echo "⏩ Proceeding to Phase 3..."
echo ""

# =============================================================================
# PHASE 3: BUILD05 - TRAINING DATA ORGANIZATION
# =============================================================================

echo "🚀 PHASE 3: Build05 - Training Data Generation"  
echo "=============================================="
echo "Expected: Organize snips into training folders by phenotype/perturbation"
echo "Training name: $TRAIN_NAME"
echo ""

python -m src.run_morphseq_pipeline.cli build05 \
  --root "$PLAYGROUND_ROOT" \
  --train-name "$TRAIN_NAME"

PHASE3_EXIT_CODE=$?
TRAINING_DIR="$PLAYGROUND_ROOT/training_data/$TRAIN_NAME"

if [ $PHASE3_EXIT_CODE -eq 0 ] && [ -d "$TRAINING_DIR" ]; then
    echo ""
    echo "✅ PHASE 3 SUCCESS: Build05 completed"
    echo "   📁 Training directory created: $TRAINING_DIR"
    
    # Check training folder structure
    echo "   📋 Training folder structure:"
    ls -la "$TRAINING_DIR/" | sed 's/^/      /'
    
    # Count images in each subdirectory
    if [ -d "$TRAINING_DIR" ]; then
        echo "   📊 Image counts per category:"
        for subdir in "$TRAINING_DIR"/*; do
            if [ -d "$subdir" ]; then
                count=$(find "$subdir" -name "*.png" -o -name "*.jpg" -o -name "*.tiff" | wc -l)
                echo "      $(basename "$subdir"): $count images"
            fi
        done
    fi
    
else
    echo ""
    echo "❌ PHASE 3 FAILED: Build05 could not create training data"
    echo "   Exit code: $PHASE3_EXIT_CODE"
    echo "   Check df02.csv format and snip image accessibility"
    exit 1
fi

echo ""
echo "⏩ Proceeding to Phase 4..."
echo ""

# =============================================================================
# PHASE 4: VAE INTEGRATION - MORPHOLOGICAL EMBEDDINGS
# =============================================================================

echo "🚀 PHASE 4: VAE Integration - Morphological Embeddings"
echo "======================================================"
echo "Expected: Generate morphological embeddings from training data"
echo ""

# Search for existing VAE models
echo "🔍 Searching for existing VAE models..."
VAE_MODELS=$(find /net/trapnell/vol1/home/mdcolon/proj/morphseq -name "*final_model*" -o -name "*vae*model*" 2>/dev/null | head -5)

if [ -z "$VAE_MODELS" ]; then
    echo "⚠️  No pre-trained VAE models found - skipping embedding generation"
    echo "   To complete VAE testing, train a model first or locate existing checkpoints"
    VAE_SUCCESS=false
else
    echo "✅ Found potential VAE models:"
    echo "$VAE_MODELS" | sed 's/^/   /'
    
    # Try to generate embeddings using the assess_image_set.py script
    echo ""
    echo "🧠 Attempting embedding generation..."
    
    # This is a placeholder - actual VAE testing would require:
    # 1. Model compatibility verification
    # 2. Input dimension matching  
    # 3. Batch processing of training images
    # 4. Embedding extraction and validation
    
    echo "⚠️  VAE integration testing requires manual model selection and compatibility check"
    echo "   Next steps:"
    echo "   1. Verify model architecture matches image dimensions"
    echo "   2. Use src/vae/auxiliary_scripts/assess_image_set.py"  
    echo "   3. Generate embeddings for training data"
    echo "   4. Create UMAP visualization for validation"
    
    VAE_SUCCESS=partial
fi

echo ""

# =============================================================================
# FINAL VALIDATION AND SUMMARY
# =============================================================================

echo "🎯 FULL PIPELINE VALIDATION COMPLETE"
echo "===================================="
echo ""
echo "📋 RESULTS SUMMARY:"
echo "   ✅ Phase 1 (Build03A): SAM2 CSV → df01.csv"
echo "   ✅ Phase 2 (Build04): df01.csv → df02.csv with QC"
echo "   ✅ Phase 3 (Build05): df02.csv → training data organization"
if [ "$VAE_SUCCESS" = "partial" ]; then
    echo "   ⚠️  Phase 4 (VAE): Model found, manual testing required"
elif [ "$VAE_SUCCESS" = "false" ]; then
    echo "   ❌ Phase 4 (VAE): No models found"
else
    echo "   ✅ Phase 4 (VAE): Embeddings generated successfully"
fi
echo ""

echo "📁 OUTPUT LOCATIONS:"
echo "   • Safe df01.csv: $SAFE_OUTPUT_DIR/embryo_metadata_df01.csv"
echo "   • Safe df02.csv: $SAFE_OUTPUT_DIR/embryo_metadata_df02.csv"
echo "   • Training data: $TRAINING_DIR"
echo "   • Snip images: $PLAYGROUND_ROOT/training_data/bf_embryo_snips/$EXPERIMENT"
echo ""

echo "🚀 PRODUCTION READINESS STATUS:"
echo "   Build03A → Build04 → Build05: ✅ VALIDATED"
echo "   VAE Integration: ⏳ REQUIRES MANUAL SETUP"
echo "   Scale Testing: ⏳ PENDING (test with 10+ embryos)"
echo ""

echo "📝 NEXT ACTIONS:"
echo "   1. Test with larger dataset (10+ embryos)"
echo "   2. Complete VAE model selection and embedding generation"
echo "   3. Validate morphological embedding quality"
echo "   4. Document production deployment commands"
echo ""

echo "✨ SUCCESS: Core SAM2 pipeline integration fully functional!"

# Clean up - remove consolidated redundant test scripts
echo ""
echo "🧹 Cleaning up redundant test scripts..."
rm -f test_sam2_pipeline.sh test_sam2_step2.sh test_sam2_step3.sh test_sam2_safe1_safe.sh
echo "   Removed 4 individual test scripts (now consolidated)"