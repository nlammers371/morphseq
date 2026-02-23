#!/bin/bash

# Keyence Test Data Extraction Script
# =====================================
# Purpose: Extract 10 frames from well A12 of experiment 20250612_24hpf_ctrl_atf6
# Created: 2025-11-06
# Agent: Wave 1, Agent 5

set -e  # Exit on error

# Configuration
EXPERIMENT_ID="20250612_24hpf_ctrl_atf6"
WELL="A12"
NUM_FRAMES=10
CHANNEL="BF"  # Brightfield channel

# Paths
NETWORK_ROOT="/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
SOURCE_DIR="${NETWORK_ROOT}/raw_image_data/Keyence/${EXPERIMENT_ID}"
DEST_DIR="$(dirname "$0")/raw_image_data/Keyence/test_keyence_001"

echo "========================================"
echo "Keyence Test Data Extraction"
echo "========================================"
echo ""
echo "Source Experiment: ${EXPERIMENT_ID}"
echo "Well: ${WELL}"
echo "Frames to extract: ${NUM_FRAMES}"
echo "Channel: ${CHANNEL}"
echo ""
echo "Source: ${SOURCE_DIR}"
echo "Destination: ${DEST_DIR}"
echo ""

# Check if network mount is accessible
if [ ! -d "${NETWORK_ROOT}" ]; then
    echo "❌ ERROR: Network mount not accessible"
    echo "   Expected: ${NETWORK_ROOT}"
    echo ""
    echo "Please mount the network drive first:"
    echo "   # On Linux/Mac:"
    echo "   sudo mkdir -p /net/trapnell/vol1/home/nlammers/projects/data"
    echo "   sudo mount -t nfs trapnell-vol1.gs.washington.edu:/vol1/home/nlammers/projects/data /net/trapnell/vol1/home/nlammers/projects/data"
    echo ""
    echo "   # Or use sshfs:"
    echo "   sshfs username@server:/net/trapnell/vol1/home/nlammers/projects/data /net/trapnell/vol1/home/nlammers/projects/data"
    exit 1
fi

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "❌ ERROR: Source experiment directory not found"
    echo "   Expected: ${SOURCE_DIR}"
    echo ""
    echo "Available Keyence experiments:"
    ls -1 "${NETWORK_ROOT}/raw_image_data/Keyence/" | head -10
    exit 1
fi

# Create destination directory
mkdir -p "${DEST_DIR}"

echo "✓ Source directory found"
echo "✓ Destination directory ready"
echo ""

# Explore source structure to determine file naming pattern
echo "Analyzing source file structure..."
echo ""

# List files for well A12
echo "Files for well ${WELL}:"
find "${SOURCE_DIR}" -name "*${WELL}*" -type f | head -5
echo ""

# Try different common Keyence naming patterns
PATTERN1="${SOURCE_DIR}/*${WELL}*${CHANNEL}*.tif"
PATTERN2="${SOURCE_DIR}/${WELL}/*${CHANNEL}*.tif"
PATTERN3="${SOURCE_DIR}/*/*${WELL}*${CHANNEL}*.tif"

# Find matching files
MATCHED_FILES=""
for pattern in "${PATTERN1}" "${PATTERN2}" "${PATTERN3}"; do
    files=$(find "$(dirname "${pattern}")" -name "$(basename "${pattern}")" 2>/dev/null | sort | head -${NUM_FRAMES})
    if [ ! -z "$files" ]; then
        MATCHED_FILES="$files"
        echo "✓ Found files matching pattern: $pattern"
        break
    fi
done

if [ -z "$MATCHED_FILES" ]; then
    echo "❌ ERROR: No files found for well ${WELL} and channel ${CHANNEL}"
    echo ""
    echo "Tried patterns:"
    echo "  - ${PATTERN1}"
    echo "  - ${PATTERN2}"
    echo "  - ${PATTERN3}"
    echo ""
    echo "Please check the actual file naming convention and update this script."
    echo ""
    echo "Example search:"
    echo "  find ${SOURCE_DIR} -name '*${WELL}*' | head -20"
    exit 1
fi

# Count matched files
NUM_MATCHED=$(echo "$MATCHED_FILES" | wc -l)
echo "Found ${NUM_MATCHED} files for extraction"
echo ""

if [ "$NUM_MATCHED" -lt "$NUM_FRAMES" ]; then
    echo "⚠️  WARNING: Found only ${NUM_MATCHED} files, expected ${NUM_FRAMES}"
    echo "   Will extract all available files"
    echo ""
fi

# Extract files
echo "Extracting files..."
file_count=0
echo "$MATCHED_FILES" | while IFS= read -r source_file; do
    if [ -f "$source_file" ]; then
        file_count=$((file_count + 1))
        # Get original filename
        filename=$(basename "$source_file")

        # Create standardized filename
        # Format: test_keyence_001_A12_TXXXX_BF.tif
        frame_num=$(printf "%04d" $file_count)
        dest_filename="test_keyence_001_${WELL}_T${frame_num}_${CHANNEL}.tif"

        # Copy file
        cp "$source_file" "${DEST_DIR}/${dest_filename}"
        echo "  ✓ Frame ${file_count}: ${filename} → ${dest_filename}"
    fi
done

echo ""
echo "========================================"
echo "Extraction Complete!"
echo "========================================"
echo ""

# Verify extraction
extracted_count=$(ls -1 "${DEST_DIR}"/*.tif 2>/dev/null | wc -l)
echo "Files extracted: ${extracted_count}"
echo ""

if [ "$extracted_count" -eq 0 ]; then
    echo "❌ ERROR: No files were extracted"
    exit 1
fi

# Show extracted files
echo "Extracted files:"
ls -lh "${DEST_DIR}"
echo ""

# Verify images are readable (requires Python with skimage)
echo "Verifying image readability..."
python3 << 'EOF'
import sys
from pathlib import Path

try:
    import skimage.io as skio
    import numpy as np

    img_dir = Path("test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001")
    images = sorted(img_dir.glob("*.tif"))

    if not images:
        print("❌ No images found to verify")
        sys.exit(1)

    print(f"Verifying {len(images)} images...")

    shapes = set()
    dtypes = set()
    sizes = []

    for img_path in images:
        img = skio.imread(img_path)
        shapes.add(img.shape)
        dtypes.add(str(img.dtype))
        sizes.append(img_path.stat().st_size)

    print(f"  ✓ All images readable")
    print(f"  Image shapes: {shapes}")
    print(f"  Image dtypes: {dtypes}")
    print(f"  File sizes: {min(sizes)/1024/1024:.2f} - {max(sizes)/1024/1024:.2f} MB")

    if len(shapes) > 1:
        print("  ⚠️  WARNING: Images have different shapes!")
    if len(dtypes) > 1:
        print("  ⚠️  WARNING: Images have different data types!")

    print("\n✅ Image verification complete")

except ImportError:
    print("⚠️  Skipping image verification (skimage not installed)")
    print("  To verify images, install: pip install scikit-image")
except Exception as e:
    print(f"❌ Error verifying images: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo ""
echo "1. Update experiment config to include test_keyence_001"
echo "2. Run pipeline through Phase 2:"
echo ""
echo "   snakemake \\"
echo "       --config experiments=test_keyence_001 \\"
echo "       --until rule_generate_image_manifest \\"
echo "       --cores 2"
echo ""
echo "3. Validate outputs:"
echo ""
echo "   python scripts/validate_phase2_outputs.py \\"
echo "       --exp test_keyence_001 \\"
echo "       --microscope keyence"
echo ""
echo "✅ Extraction script completed successfully!"
