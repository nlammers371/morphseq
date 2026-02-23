#!/bin/bash

# MorphSeq Experiment Manager - Complete Pipeline Processing
# Wrapper script for intelligent batch processing using ExperimentManager

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script metadata
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Parameters
data_root=""
repo_root=""
experiments=""

show_usage() {
    cat << EOF
MorphSeq Experiment Manager - Complete Pipeline Processing

USAGE:
  $SCRIPT_NAME --data-root PATH --repo-root PATH --experiments EXPERIMENTS

PARAMETERS:
  --data-root PATH     Path to MorphSeq data directory (required)
  --repo-root PATH     Path to MorphSeq repository root (required) 
  --experiments LIST   Experiments to process: "all" or "exp1,exp2,exp3" (required)
  --help              Show this help message

EXAMPLES:
  # Process all experiments
  $SCRIPT_NAME --data-root /path/to/morphseq/data --repo-root /path/to/morphseq/repo --experiments "all"
  
  # Process specific experiments
  $SCRIPT_NAME --data-root morphseq_playground --repo-root . --experiments "20250529_30hpf_ctrl_atf6,20240418"

PIPELINE STAGES:
  1. Per-experiment: Raw data -> FF images -> QC masks -> SAM2 segmentation
  2. Per-experiment: Build03 embryo processing -> Latent embeddings  
  3. Global: Build04 QC -> Build06 final merge

The script uses ExperimentManager for intelligent processing - only runs steps that are actually needed.

NOTES:
  - Requires 'segmentation_grounded_sam' conda environment
  - Uses existing experiment state tracking to avoid duplicate work
  - Future versions will support --overwrite and selective step processing

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)
            data_root="$2"
            shift 2
            ;;
        --repo-root)
            repo_root="$2"
            shift 2
            ;;
        --experiments)
            experiments="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Error: Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$data_root" || -z "$repo_root" || -z "$experiments" ]]; then
    echo "❌ Error: Missing required parameters"
    echo ""
    show_usage
    exit 1
fi

# Validate paths exist
if [[ ! -d "$data_root" ]]; then
    echo "❌ Error: Data root directory does not exist: $data_root"
    exit 1
fi

if [[ ! -d "$repo_root" ]]; then
    echo "❌ Error: Repo root directory does not exist: $repo_root"
    exit 1
fi

# Convert to absolute paths to avoid issues
data_root=$(realpath "$data_root")
repo_root=$(realpath "$repo_root")

echo "🚀 MorphSeq Experiment Manager"
echo "   Data root: $data_root"  
echo "   Repo root: $repo_root"
echo "   Experiments: $experiments"
echo ""

# Environment setup
echo "🔧 Setting up environment..."

# Activate conda environment
if ! source "$(conda info --base)/etc/profile.d/conda.sh"; then
    echo "❌ Error: Failed to source conda"
    exit 1
fi

if ! conda activate segmentation_grounded_sam; then
    echo "❌ Error: Failed to activate segmentation_grounded_sam environment"
    echo "   Please ensure the environment exists: conda env list"
    exit 1
fi

echo "✅ Environment activated: segmentation_grounded_sam"

# Set working directory
if ! cd "$repo_root"; then
    echo "❌ Error: Failed to change to repo root: $repo_root"
    exit 1
fi

echo "✅ Working directory: $repo_root"
echo ""

# Create and execute Python orchestration script
echo "🐍 Starting Python orchestration..."

python_script=$(mktemp /tmp/morphseq_process_XXXXXX.py)

# Trap to ensure cleanup
trap 'rm -f "$python_script"' EXIT

cat > "$python_script" << 'PYTHON_EOF'
import sys
import traceback
from pathlib import Path

# Add repo root to Python path
repo_root = sys.argv[1]
data_root = sys.argv[2] 
experiments_arg = sys.argv[3]

sys.path.insert(0, repo_root)

try:
    from src.build.pipeline_objects import ExperimentManager
except ImportError as e:
    print(f"❌ Error: Failed to import ExperimentManager: {e}")
    print(f"   Make sure you're running from the correct repository root")
    sys.exit(1)

def main():
    try:
        print("📋 Initializing ExperimentManager...")
        
        # Initialize ExperimentManager
        manager = ExperimentManager(root=data_root)
        
        # Explicitly discover and sync experiments (ensuring fresh state)
        manager.discover_experiments()
        manager.update_experiment_status()
        
        print(f"🔍 Discovered {len(manager.experiments)} experiments")
        
        # Determine target experiments
        if experiments_arg.lower() == "all":
            target_experiments = list(manager.experiments.keys())
            print(f"📋 Processing ALL {len(target_experiments)} experiments")
        else:
            target_experiments = [exp.strip() for exp in experiments_arg.split(",") if exp.strip()]
            print(f"📋 Processing {len(target_experiments)} specific experiments: {target_experiments}")
            
            # Validate experiments exist
            missing = [exp for exp in target_experiments if exp not in manager.experiments]
            if missing:
                print(f"❌ Error: Experiments not found: {missing}")
                available = list(manager.experiments.keys())
                print(f"   Available experiments: {available[:10]}{'...' if len(available) > 10 else ''}")
                return 1
        
        if not target_experiments:
            print("❌ Error: No experiments to process")
            return 1
        
        # Process experiments through pipeline
        success = process_experiments_through_pipeline(manager, target_experiments)
        
        if success:
            print("✅ Pipeline processing complete!")
            return 0
        else:
            print("❌ Pipeline failed; see above")
            return 1
        
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")
        traceback.print_exc()
        return 1

def process_experiments_through_pipeline(manager, target_experiments):
    """Process experiments through complete MorphSeq pipeline"""
    
    print("")
    print("🚀 Starting complete pipeline processing...")
    
    success = True
    
    try:
        # Stage 1: Per-experiment basics (only process what's needed)
        print("📦 Stage 1: Basic per-experiment processing...")
        for exp_name in target_experiments:
            if exp_name not in manager.experiments:
                print(f"  ❌ Skipping unknown experiment: {exp_name}")
                success = False
                continue
                
            exp = manager.experiments[exp_name]
            print(f"  Processing {exp_name}...")
            
            # Export/build raw images if needed
            if not exp.flags.get('ff', False):
                print(f"    📸 Exporting images...")
                try:
                    exp.export_images()
                except Exception as e:
                    print(f"    ❌ Export failed: {e}")
                    success = False
                    continue
                    
            # Stitch images if needed (Keyence only)
            if exp.needs_stitch and exp.microscope == "Keyence":
                print(f"    🧩 Stitching images...")
                try:
                    exp.stitch_images()
                except Exception as e:
                    print(f"    ❌ Stitching failed: {e}")
                    success = False
                    continue
                    
            # Generate QC masks if needed (5 UNet models)
            if exp.needs_segment:
                print(f"    🎯 Generating QC masks...")
                try:
                    exp.segment_images()
                except Exception as e:
                    print(f"    ❌ Segmentation failed: {e}")
                    success = False
                    continue
        
        # Stage 2: Advanced per-experiment processing
        print("")
        print("🔬 Stage 2: Advanced per-experiment processing...")
        for exp_name in target_experiments:
            if exp_name not in manager.experiments:
                continue
                
            exp = manager.experiments[exp_name]
            print(f"  Processing {exp_name}...")
            
            # SAM2 segmentation if needed
            if exp.needs_sam2:
                print(f"    🤖 Running SAM2 segmentation...")
                try:
                    exp.run_sam2()
                except Exception as e:
                    print(f"    ❌ SAM2 failed: {e}")
                    success = False
                    continue
                    
            # Build03 embryo processing if needed
            if exp.needs_build03:
                print(f"    🧬 Running Build03 embryo processing...")
                try:
                    exp.run_build03()
                except Exception as e:
                    print(f"    ❌ Build03 failed: {e}")
                    success = False
                    continue
                    
            # Generate latent embeddings if needed
            if not exp.has_latents():
                print(f"    🧠 Generating latent embeddings...")
                try:
                    exp.generate_latents()
                except Exception as e:
                    print(f"    ❌ Latent generation failed: {e}")
                    success = False
                    continue
        
        # Stage 3: Global operations
        print("")
        print("🌍 Stage 3: Global pipeline operations...")
        
        # Build04: df01 -> df02 (global QC)
        if manager.needs_build04:
            print("  📊 Running Build04 (global QC)...")
            try:
                manager.run_build04()
            except Exception as e:
                print(f"  ❌ Build04 failed: {e}")
                success = False
        else:
            print("  ✅ Build04 already complete")
            
        # Build06: df02 + latents -> df03 (final merge)  
        if manager.needs_build06:
            print("  🔗 Running Build06 (final merge)...")
            try:
                manager.run_build06()
            except Exception as e:
                print(f"  ❌ Build06 failed: {e}")
                success = False
        else:
            print("  ✅ Build06 already complete")
            
        return success
        
    except Exception as e:
        print(f"❌ Pipeline orchestration failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <repo_root> <data_root> <experiments>")
        sys.exit(1)
    sys.exit(main())
PYTHON_EOF

# Execute the Python orchestration
echo ""
if python "$python_script" "$repo_root" "$data_root" "$experiments"; then
    echo ""
    echo "🎉 All done! Check your data directory for results."
    exit_code=0
else
    echo ""
    echo "❌ Pipeline failed; see above"
    exit_code=1
fi

# Cleanup happens via trap
exit $exit_code