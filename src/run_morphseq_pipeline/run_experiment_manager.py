#!/usr/bin/env python3
"""
MorphSeq Experiment Manager - Complete Pipeline Processing

Simple script for batch processing experiments through the complete MorphSeq pipeline
using intelligent ExperimentManager orchestration.

Usage:
    python run_experiment_manager.py --data-root PATH --repo-root PATH --experiments LIST
"""

import sys
import argparse
import traceback
from pathlib import Path

def setup_path(repo_root):
    """Add repo root to Python path for imports"""
    repo_path = Path(repo_root).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

# Default configuration - modify these as needed
DEFAULT_DATA_ROOT = "morphseq_playground"
DEFAULT_REPO_ROOT = "."
# DEFAULT_EXPERIMENTS = "all"
DEFAULT_EXPERIMENTS = "20251207_pbx"

def main():
    parser = argparse.ArgumentParser(
        description="MorphSeq Experiment Manager - Complete Pipeline Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process all experiments (using defaults)
  python run_experiment_manager.py
  
  # Process all experiments with custom paths
  python run_experiment_manager.py --data-root morphseq_playground --repo-root . --experiments "all"
  
  # Process specific experiments
  python run_experiment_manager.py --data-root /path/to/data --repo-root /path/to/repo --experiments "20250529_30hpf_ctrl_atf6,20240418"

PIPELINE STAGES:
  1. Per-experiment: Raw data -> FF images -> QC masks -> SAM2 segmentation
  2. Per-experiment: Build03 embryo processing -> Latent embeddings  
  3. Global: Build04 QC -> Build06 final merge

The script uses ExperimentManager for intelligent processing - only runs steps that are actually needed.

DEFAULTS:
  --data-root: {DEFAULT_DATA_ROOT}
  --repo-root: {DEFAULT_REPO_ROOT}
  --experiments: {DEFAULT_EXPERIMENTS}
        """.format(
            DEFAULT_DATA_ROOT=DEFAULT_DATA_ROOT,
            DEFAULT_REPO_ROOT=DEFAULT_REPO_ROOT, 
            DEFAULT_EXPERIMENTS=DEFAULT_EXPERIMENTS
        )
    )
    
    parser.add_argument(
        "--data-root", 
        default=DEFAULT_DATA_ROOT,
        help=f"Path to MorphSeq data directory (default: {DEFAULT_DATA_ROOT})"
    )
    parser.add_argument(
        "--repo-root", 
        default=DEFAULT_REPO_ROOT,
        help=f"Path to MorphSeq repository root (default: {DEFAULT_REPO_ROOT})"
    )
    parser.add_argument(
        "--experiments", 
        default=DEFAULT_EXPERIMENTS,
        help=f'Experiments to process: "all" or comma-separated list like "exp1,exp2,exp3" (default: {DEFAULT_EXPERIMENTS})'
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running any processing steps"
    )
    
    args = parser.parse_args()
    
    # Validate and resolve paths
    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    
    if not data_root.exists():
        print(f"❌ Error: Data root directory does not exist: {data_root}")
        return 1
        
    if not repo_root.exists():
        print(f"❌ Error: Repo root directory does not exist: {repo_root}")
        return 1
    
    print("🚀 MorphSeq Experiment Manager")
    print(f"   Data root: {data_root}")  
    print(f"   Repo root: {repo_root}")
    print(f"   Experiments: {args.experiments}")
    print("")
    
    # Setup Python path and imports
    setup_path(repo_root)
    
    try:
        from src.build.pipeline_objects import ExperimentManager
    except ImportError as e:
        print(f"❌ Error: Failed to import ExperimentManager: {e}")
        print(f"   Make sure you're running from the correct repository root")
        return 1
    
    try:
        print("📋 Initializing ExperimentManager...")
        
        # Initialize ExperimentManager
        manager = ExperimentManager(root=str(data_root))
        
        # Explicitly discover and sync experiments (ensuring fresh state)
        manager.discover_experiments()
        manager.update_experiment_status()
        
        print(f"🔍 Discovered {len(manager.experiments)} experiments")
        
        # Determine target experiments
        if args.experiments.lower() == "all":
            target_experiments = list(manager.experiments.keys())
            print(f"📋 Processing ALL {len(target_experiments)} experiments")
        else:
            target_experiments = [exp.strip() for exp in args.experiments.split(",") if exp.strip()]
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
        success = process_experiments_through_pipeline(manager, target_experiments, dry_run=args.dry_run)
        
        if args.dry_run:
            print("✅ Dry-run complete! Use without --dry-run to actually execute.")
            return 0
        elif success:
            print("✅ Pipeline processing complete!")
            return 0
        else:
            print("❌ Pipeline failed; see above")
            return 1
        
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")
        traceback.print_exc()
        return 1

def process_experiments_through_pipeline(manager, target_experiments, dry_run=False):
    """Process experiments through complete MorphSeq pipeline"""
    
    print("")
    if dry_run:
        print("🔍 DRY RUN: Showing what would be processed...")
    else:
        print("🚀 Starting complete pipeline processing...")
    
    success = True
    
    try:
        # Stage 1: Per-experiment basics (only process what's needed)
        print("📦 Stage 1: Basic per-experiment processing...")
        # Overwrite toggles via environment
        import os
        def _as_bool(v: str) -> bool:
            return str(v).lower() in ("1", "true", "yes", "on")
        overwrite_build01 = _as_bool(os.environ.get("MSEQ_OVERWRITE_BUILD01", "0"))
        for exp_name in target_experiments:
            if exp_name not in manager.experiments:
                print(f"  ❌ Skipping unknown experiment: {exp_name}")
                success = False
                continue
                
            exp = manager.experiments[exp_name]
            print(f"  Processing {exp_name}...")
            # Show concise status like CLI (RAW/FF/STITCH/QC/SAM2/DF01/LATENTS)
            try:
                raw_ok    = bool(exp.raw_path)
                ff_ok     = bool(exp.ff_path)
                st_ok     = bool(exp.stitch_ff_path)
                qc_p, qc_t= exp.qc_mask_status()
                gdino_ok  = exp.gdino_detections_path.exists()
                seg_ok    = exp.sam2_segmentations_path.exists()
                sam2_ok   = exp.sam2_csv_path.exists()
                df01_ok   = exp.is_in_df01()
                lat_ok    = exp.has_latents()
                parts = [
                    f"RAW {'✅' if raw_ok else '❌'}",
                    f"FF {'✅' if ff_ok else '❌'}",
                    f"STITCH {'✅' if st_ok else '❌'}",
                    f"QC {qc_p}/{qc_t}",
                    f"GDINO {'✅' if gdino_ok else '❌'}",
                    f"SAM2-SEG {'✅' if seg_ok else '❌'}",
                    f"SAM2-CSV {'✅' if sam2_ok else '❌'}",
                    f"DF01 {'✅' if df01_ok else '❌'}",
                    f"LATENTS {'✅' if lat_ok else '❌'}",
                ]
                print("    " + " | ".join(parts))
            except Exception:
                pass
            
            # Export/build raw images
            if overwrite_build01 or not exp.flags.get('ff', False):
                print(f"    📸 {'[DRY RUN]' if dry_run else ''} Exporting images{' (overwrite)' if overwrite_build01 else ''}...")
                if not dry_run:
                    try:
                        exp.export_images()
                    except Exception as e:
                        print(f"    ❌ Export failed: {e}")
                        # Always show full traceback for debugging
                        import traceback
                        traceback.print_exc()
                        success = False
                        continue
                    
            # Stitch images if needed (Keyence only)
            if exp.microscope == "Keyence" and (overwrite_build01 or exp.needs_stitch):
                print(f"    🧩 {'[DRY RUN]' if dry_run else ''} Stitching images{' (overwrite)' if overwrite_build01 else ''}...")
                if not dry_run:
                    try:
                        exp.stitch_images()
                    except Exception as e:
                        print(f"    ❌ Stitching failed: {e}")
                        import os
                        if str(os.environ.get("MSEQ_TRACE", "0")).lower() in ("1","true","yes","on"):
                            import traceback
                            traceback.print_exc()
                        success = False
                        continue
                    
            # Generate QC masks if needed (5 UNet models)
            if exp.needs_segment:
                print(f"    🎯 {'[DRY RUN]' if dry_run else ''} Generating QC masks...")
                if not dry_run:
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
                print(f"    🤖 {'[DRY RUN]' if dry_run else ''} Running SAM2 segmentation...")
                if not dry_run:
                    try:
                        exp.run_sam2()
                    except Exception as e:
                        print(f"    ❌ SAM2 failed: {e}")
                        success = False
                        continue
                    
            # Build03 embryo processing if needed
            if exp.needs_build03:
                print(f"    🧬 {'[DRY RUN]' if dry_run else ''} Running Build03 embryo processing...")
                if not dry_run:
                    try:
                        exp.run_build03()
                    except Exception as e:
                        print(f"    ❌ Build03 failed: {e}")
                        success = False
                        continue
                    
            # Generate latent embeddings if needed
            if not exp.has_latents():
                print(f"    🧠 {'[DRY RUN]' if dry_run else ''} Generating latent embeddings...")
                if not dry_run:
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
            print(f"  📊 {'[DRY RUN]' if dry_run else ''} Running Build04 (global QC)...")
            if not dry_run:
                try:
                    manager.run_build04()
                except Exception as e:
                    print(f"  ❌ Build04 failed: {e}")
                    success = False
        else:
            print("  ✅ Build04 already complete")
            
        # Build06: df02 + latents -> df03 (final merge)  
        if manager.needs_build06:
            print(f"  🔗 {'[DRY RUN]' if dry_run else ''} Running Build06 (final merge)...")
            if not dry_run:
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
    sys.exit(main())
