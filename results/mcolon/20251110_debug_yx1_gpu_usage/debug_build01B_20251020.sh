#!/bin/bash
# Debug runner for build01B on experiment 20251020
# Tests GPU usage on interactive GPU node (no qsub needed)

set -euo pipefail

# ============================================================================
# Environment Setup
# ============================================================================

export REPO_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
export DATA_ROOT="$REPO_ROOT/morphseq_playground"

echo "=== Environment Setup ==="
echo "REPO_ROOT: $REPO_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Check conda environment
echo "=== Checking Conda Environment ==="
if [[ "$CONDA_DEFAULT_ENV" == "segmentation_grounded_sam" ]]; then
    echo "Already in segmentation_grounded_sam environment"
else
    echo "Activating segmentation_grounded_sam..."
    # Try to activate (this may fail if conda command not available in non-interactive shell)
    if command -v conda &> /dev/null; then
        conda activate segmentation_grounded_sam || {
            echo "WARNING: conda activate failed - assuming environment is already active"
        }
    else
        echo "WARNING: conda command not found - assuming environment is already active"
    fi
fi
echo "Current conda environment: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# Set PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# GPU Check
# ============================================================================

echo "=== GPU Availability Check ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
else
    echo "WARNING: nvidia-smi not found - GPU may not be available"
    echo ""
fi

# ============================================================================
# Run Build01B with Debug Logging
# ============================================================================

echo "=== Starting Build01B for Experiment 20251020 ==="
echo "Running with MSEQ_YX1_DEBUG=1 for diagnostic output"
echo "Start time: $(date)"
echo ""

# Enable debug logging
export MSEQ_YX1_DEBUG=1

# Change to repo root so profiler traces are saved there
cd "$REPO_ROOT"

# Allow experiment override (default 20251020)
# export EXP="${1:-20251020}"
export EXP="${1:-20250501}"
export MICROSCOPE="${MICROSCOPE:-YX1}"
export OVERWRITE="${OVERWRITE:-1}"

echo "Experiment : ${EXP}"
echo "Microscope : ${MICROSCOPE}"
echo "Overwrite  : ${OVERWRITE}"
echo ""

# Run the build by calling run_build01() directly (module exposes no CLI)
python - <<'PY'
import os
from src.run_morphseq_pipeline.steps.run_build01 import run_build01

root = os.environ["DATA_ROOT"]
exp = os.environ.get("EXP", "20251020")
microscope = os.environ.get("MICROSCOPE", "YX1")
overwrite = os.environ.get("OVERWRITE", "0") in {"1", "true", "True", "TRUE"}

print(f"🔧 Dispatching run_build01(root={root}, exp={exp}, microscope={microscope}, overwrite={overwrite})")
run_build01(root=root, exp=exp, microscope=microscope, overwrite=overwrite)
PY

echo ""
echo "=== Build Complete ==="
echo "End time: $(date)"
echo ""
echo "Check profiler traces and logs for GPU usage details."
echo "Profiler traces saved to: $REPO_ROOT/profiler_trace_*.json"
echo ""
echo "To view profiler traces:"
echo "  1. Open Chrome browser"
echo "  2. Go to chrome://tracing"
echo "  3. Load the profiler_trace_*.json files"
echo "  4. Look for 'aten::' operations - CUDA ops will show GPU activity"
