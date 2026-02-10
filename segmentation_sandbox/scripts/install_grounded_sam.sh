#!/bin/bash
# SAM2 + Open-Grounded-DINO Installation Script (macOS Compatible)
# Run from morphseq directory to keep everything relative

# =================================================================
# USING OPEN-GROUNDINGDINO & OFFICIAL GROUNDED-DINO
# =================================================================
#
# This script installs:
# - SAM2 from Facebook Research (for segmentation)
# - Official Grounded-DINO from IDEA-Research
# - Open-GroundingDino which contains GroundingDINO code for inference
#
# We clone all three at the same level under segmentation_sandbox/models:
#   sam2/  
#   Grounded-DINO/
#   Open-GroundingDino/
#
# =================================================================
# PRE-RUN SETUP
# =================================================================
# 1. cd /path/to/morphseq
# 2. conda create -n segmentation_grounded_sam python=3.10 -y
# 3. conda activate segmentation_grounded_sam
# 4. bash install_grounded_sam.sh
#
# IMPORTANT: Install with CUDA environment 
# if you compile with CUDA you have to use the CUDA, and same for cpu
# =================================================================

set -e  # Exit on any error

echo "=== Step 1: Set Up Morphseq Home Directory ==="
if [ -z "$MORPHSEQ_HOME" ]; then
    MORPHSEQ_HOME="$(pwd)"
    echo "MORPHSEQ_HOME not set, using current dir: $MORPHSEQ_HOME"
else
    echo "Using existing MORPHSEQ_HOME: $MORPHSEQ_HOME"
fi
export MORPHSEQ_HOME

echo "=== Step 2: Check Conda Environment ==="
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "segmentation_grounded_sam" ]; then
  echo "ERROR: Activate 'segmentation_grounded_sam' environment first."
  exit 1
else
  echo "✓ Activated conda env: $CONDA_DEFAULT_ENV"
fi

echo "=== Step 3: Install Core Python Packages ==="
conda install pytorch torchvision torchaudio -c pytorch -y
pip install matplotlib jupyterlab
# Install libvips system dependency and pyvips for faster image I/O
conda install -c conda-forge libvips -y
# Install core dependencies with specific versions to avoid conflicts
pip install supervision==0.25.0 opencv-python numpy scipy pillow pandas pyvips scikit-image tqdm pyyaml pycocotools addict timm transformers safetensors

echo "=== Step 4: Create models directory ==="
mkdir -p "$MORPHSEQ_HOME/segmentation_sandbox/models"
cd "$MORPHSEQ_HOME/segmentation_sandbox/models"

echo "✓ Working in: $(pwd)"

# ----------------------------------------------------------------------------
# Step 5: Install SAM2
# ----------------------------------------------------------------------------
echo "=== Step 5: Install SAM2 ==="
# Remove and recreate the directory to ensure it's clean
rm -rf "$MORPHSEQ_HOME/segmentation_sandbox/models/sam2"
mkdir -p "$MORPHSEQ_HOME/segmentation_sandbox/models/sam2"
cd "$MORPHSEQ_HOME/segmentation_sandbox/models/sam2"
git clone https://github.com/facebookresearch/sam2.git .
pip install -e .
pip install jupyter matplotlib
# Download checkpoints
mkdir -p checkpoints
cd checkpoints
echo "Downloading SAM2 checkpoints..."
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Verify SAM2 checkpoint downloads
echo "Verifying SAM2 checkpoint downloads..."
for checkpoint in sam2.1_hiera_large.pt sam2.1_hiera_base_plus.pt sam2.1_hiera_small.pt; do
    if [ ! -f "$checkpoint" ]; then
        echo "❌ ERROR: $checkpoint not found!"
        exit 1
    fi
    
    file_size=$(stat -c%s "$checkpoint" 2>/dev/null || stat -f%z "$checkpoint" 2>/dev/null || echo "0")
    if [ "$file_size" -eq 0 ]; then
        echo "❌ ERROR: $checkpoint is empty (0 bytes)!"
        echo "Download may have failed. Please check your internet connection and try again."
        exit 1
    fi
    
    # Convert bytes to MB for display
    file_size_mb=$((file_size / 1024 / 1024))
    echo "✓ $checkpoint: ${file_size_mb}MB"
done

cd ../..

echo "✓ SAM2 installed with verified checkpoints"

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Step 6: Install Official GroundingDINO (IDEA-Research)
# ----------------------------------------------------------------------------
echo "=== Step 6: Install Official GroundingDINO ==="
# Remove and recreate the directory to ensure it's clean
rm -rf "$MORPHSEQ_HOME/segmentation_sandbox/models/GroundingDINO"
mkdir -p "$MORPHSEQ_HOME/segmentation_sandbox/models/GroundingDINO"
cd "$MORPHSEQ_HOME/segmentation_sandbox/models/GroundingDINO"
git clone https://github.com/IDEA-Research/GroundingDINO.git .
git pull --ff-only || echo "⚠ Could not pull latest changes"
pip install -r requirements.txt
# If setup.py exists, install in editable mode
if [ -f setup.py ]; then
  pip install -e .
fi
mkdir weights
cd weights
echo "Downloading GroundingDINO weights..."
curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl -L -o groundingdino_swinb_cogcoor.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Verify downloads
echo "Verifying GroundingDINO weights downloads..."
for weights_file in groundingdino_swint_ogc.pth groundingdino_swinb_cogcoor.pth; do
    if [ ! -f "$weights_file" ]; then
        echo "❌ ERROR: $weights_file not found!"
        exit 1
    fi
    
    file_size=$(stat -c%s "$weights_file" 2>/dev/null || stat -f%z "$weights_file" 2>/dev/null || echo "0")
    if [ "$file_size" -eq 0 ]; then
        echo "❌ ERROR: $weights_file is empty (0 bytes)!"
        echo "Download may have failed. Please check your internet connection and try again."
        exit 1
    fi
    
    # Convert bytes to MB for display
    file_size_mb=$((file_size / 1024 / 1024))
    echo "✓ $weights_file: ${file_size_mb}MB"
done

cd ..
echo "✓ Official GroundingDINO installed with verified weights"

# ----------------------------------------------------------------------------
# Step 7: Install Open-GroundingDino
# ----------------------------------------------------------------------------
echo "=== Step 7: Install Open-GroundingDino ==="
# Remove and recreate the directory to ensure it's clean
rm -rf "$MORPHSEQ_HOME/segmentation_sandbox/models/Open-GroundingDino"
mkdir -p "$MORPHSEQ_HOME/segmentation_sandbox/models/Open-GroundingDino"
cd "$MORPHSEQ_HOME/segmentation_sandbox/models/Open-GroundingDino"
git clone https://github.com/longzw1997/Open-GroundingDino.git .
# Fix supervision version conflict in requirements.txt
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/supervision==0.6.0/supervision==0.25.0/' requirements.txt
else
    # Linux
    sed -i 's/supervision==0.6.0/supervision==0.25.0/' requirements.txt
fi
pip install -r requirements.txt
# CUDA fix - detect OS and use appropriate sed syntax
cd models/GroundingDINO/ops
[ ! -f setup.py.backup ] && cp setup.py setup.py.backup
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/raise NotImplementedError('Cuda is not availabel')/print('Compiling without CUDA - CPU only mode'); return []/" setup.py
else
    # Linux
    sed -i "s/raise NotImplementedError('Cuda is not availabel')/print('Compiling without CUDA - CPU only mode'); return []/" setup.py
fi
cd ../..
python setup.py build install || echo "⚠ ops build failed, proceeding"
cd ..
echo "✓ Open-GroundingDino installed"

# ----------------------------------------------------------------------------
# Step 8: Verify Package Installation
# ----------------------------------------------------------------------------
echo "=== Step 8: Verify Package Installation ==="
pip list | grep -E "supervision|groundingdino|torch|opencv|numpy"
echo "✓ Package verification complete"

# ----------------------------------------------------------------------------
# Step 8.5: Final Model Files Verification
# ----------------------------------------------------------------------------
echo "=== Step 8.5: Final Model Files Verification ==="
cd "$MORPHSEQ_HOME/segmentation_sandbox/models"

# Check GroundingDINO weights
echo "Checking GroundingDINO weights..."
gdino_weights_dir="GroundingDINO/weights"
for weights_file in groundingdino_swint_ogc.pth groundingdino_swinb_cogcoor.pth; do
    weights_path="$gdino_weights_dir/$weights_file"
    if [ ! -f "$weights_path" ]; then
        echo "❌ CRITICAL ERROR: $weights_path not found!"
        exit 1
    fi
    
    file_size=$(stat -c%s "$weights_path" 2>/dev/null || stat -f%z "$weights_path" 2>/dev/null || echo "0")
    if [ "$file_size" -eq 0 ]; then
        echo "❌ CRITICAL ERROR: $weights_path is empty!"
        exit 1
    fi
    
    file_size_mb=$((file_size / 1024 / 1024))
    echo "✓ $weights_file: ${file_size_mb}MB"
done

# Check SAM2 checkpoints
echo "Checking SAM2 checkpoints..."
sam2_checkpoints_dir="sam2/checkpoints"
for checkpoint in sam2.1_hiera_large.pt sam2.1_hiera_base_plus.pt sam2.1_hiera_small.pt; do
    checkpoint_path="$sam2_checkpoints_dir/$checkpoint"
    if [ ! -f "$checkpoint_path" ]; then
        echo "❌ CRITICAL ERROR: $checkpoint_path not found!"
        exit 1
    fi
    
    file_size=$(stat -c%s "$checkpoint_path" 2>/dev/null || stat -f%z "$checkpoint_path" 2>/dev/null || echo "0")
    if [ "$file_size" -eq 0 ]; then
        echo "❌ CRITICAL ERROR: $checkpoint_path is empty!"
        exit 1
    fi
    
    file_size_mb=$((file_size / 1024 / 1024))
    echo "✓ $checkpoint: ${file_size_mb}MB"
done

echo "✅ All model files verified successfully!"

# ----------------------------------------------------------------------------
# Step 9: Smoke Tests
# ----------------------------------------------------------------------------
echo "=== Step 9: Smoke Tests ==="
cd "$MORPHSEQ_HOME"
python - << 'EOF'
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
from sam2.build_sam import build_sam2; print('SAM2 import OK')
from groundingdino.models import build_model; print('Official Grounded-DINO import OK')
import groundingdino.datasets.transforms as T; print('Open-GroundingDino import OK')
EOF

echo "🎉 Installation complete!"
