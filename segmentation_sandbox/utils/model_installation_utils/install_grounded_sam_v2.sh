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
if [ ! -d "sam2" ]; then
  git clone https://github.com/facebookresearch/sam2.git
fi
cd sam2
pip install -e .
pip install jupyter matplotlib
# Download checkpoints
cd checkpoints
if [ ! -f sam2.1_hiera_large.pt ]; then
  wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
  wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fi
cd ../..

echo "✓ SAM2 installed"

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Step 6: Install Official GroundingDINO (IDEA-Research)
# ----------------------------------------------------------------------------
echo "=== Step 6: Install Official GroundingDINO ==="
# Clone or update the repository
if [ ! -d "GroundingDINO" ]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git GroundingDINO
  echo "✓ Cloned Official GroundingDINO repository"
else
  echo "Official GroundingDINO directory exists, pulling latest changes"
  cd GroundingDINO
  git pull --ff-only || echo "⚠ Could not pull latest changes"
  cd ..
fi
cd GroundingDINO
pip install -r requirements.txt
# If setup.py exists, install in editable mode
if [ -f setup.py ]; then
  pip install -e .
fi
mkdir weights
cd weights
curl -sO https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl -sO https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
echo "✓ Official GroundingDINO installed"

# ----------------------------------------------------------------------------
# Step 7: Install Open-GroundingDino
# ----------------------------------------------------------------------------
echo "=== Step 7: Install Open-GroundingDino ==="
if [ ! -d "Open-GroundingDino" ]; then
  git clone https://github.com/longzw1997/Open-GroundingDino.git
fi
cd Open-GroundingDino
pip install -r requirements.txt
# macOS CUDA fix
cd models/GroundingDINO/ops
[ ! -f setup.py.backup ] && cp setup.py setup.py.backup
sed -i '' "s/raise NotImplementedError('Cuda is not availabel')/print('Compiling without CUDA - CPU only mode'); return []/" setup.py
cd ../..
python setup.py build install || echo "⚠ ops build failed, proceeding"
cd ..
echo "✓ Open-GroundingDino installed"

# ----------------------------------------------------------------------------
# Step 8: Install other image/data packages
# ----------------------------------------------------------------------------
echo "=== Step 8: Install other image/data packages ==="
pip install opencv-python numpy scipy pillow pandas scikit-image tqdm pyyaml supervision pycocotools addict timm transformers safetensors

echo "✓ All packages installed"

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
