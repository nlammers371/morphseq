
# laud Opus tracking plan 
# Understanding the New Tracking Requirements from the Integration Documents

Based on my analysis of the integration documents, particularly the detailed tracking improvements outlined at the end of `integrate_experiment_manager.md`, here's what needs to be tracked:

## The Critical Tracking Distinction: Per-Experiment vs Global

The documents identify a fundamental issue that the current system doesn't handle: **some pipeline steps produce per-experiment artifacts while others produce global (cohort-level) files**.

### Per-Experiment Artifacts (Need Individual Tracking)

1. **Build02 (QC Masks)**
   - **Artifacts**: 5 UNet model predictions directories
   - **Paths**: 
     - `<data_root>/segmentation/mask_v0_0100_predictions/{exp}/`
     - `<data_root>/segmentation/yolk_v1_0050_predictions/{exp}/`
     - `<data_root>/segmentation/focus_v0_0100_predictions/{exp}/`
     - `<data_root>/segmentation/bubble_v0_0100_predictions/{exp}/`
     - `<data_root>/segmentation/via_v1_0100_predictions/{exp}/`
   - **Tracking**: Check if all 5 directories exist for each experiment

2. **SAM2 Segmentation**
   - **Artifact**: Per-experiment metadata CSV
   - **Path**: `<data_root>/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
   - **Tracking**: Check file existence for each experiment

3. **Build03 (Contributes to df01)**
   - **Special Case**: This step APPENDS to a global file but needs per-experiment tracking
   - **Logic**: Track via timestamp comparison - has this experiment's inputs (SAM2 CSV) changed since last Build03 run?

4. **Build06 Latent Embeddings (Critical Intermediate File)**
   - **Artifact**: Per-experiment embedding file
   - **Path**: `<data_root>/analysis/latent_embeddings/legacy/{model_name}/morph_latents_{exp}.csv`
   - **Tracking**: Check if latent file exists for given model

### Global Artifacts (Need Cohort-Level Tracking)

1. **Build04 (QC & Staging)**
   - **Input**: Combined `df01.csv` (from all Build03 runs)
   - **Output**: Combined `df02.csv`
   - **Tracking**: Compare timestamps - if df01 newer than df02, needs rerun

2. **Build06 Final Combination**
   - **Input**: `df02.csv` + all individual latent files
   - **Output**: Final `df03.csv`
   - **Tracking**: Compare timestamps - if df02 newer than df03, needs rerun

## The Problem This Solves

Currently, if you check whether `df02.csv` exists, you can't tell:
- Which experiments contributed to it
- Whether a new experiment needs to be added
- Whether an experiment's data has been updated since df02 was created

The hybrid tracking model solves this by:
1. Tracking per-experiment steps individually in each experiment's state file
2. Tracking global steps by timestamp comparison at the manager level
3. Using the combination to determine what needs to run

## Implementation Requirements from Documents

### In `Experiment` class (pipeline_objects.py):
```python
# New properties needed:
@property
def sam2_csv_path(self) -> Path:
    return self.data_root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{self.date}.csv"

@property
def needs_build03(self) -> bool:
    # Check if SAM2 output is newer than last Build03 run
    last_run = self.timestamps.get("build03", 0)
    if self.sam2_csv_path.exists():
        newest = self.sam2_csv_path.stat().st_mtime
        return newest > last_run
    return False

def get_latent_path(self, model_name: str) -> Path:
    return (self.data_root / "analysis" / "latent_embeddings" / 
            "legacy" / model_name / f"morph_latents_{self.date}.csv")

def has_latents(self, model_name: str) -> bool:
    return self.get_latent_path(model_name).exists()
```

### In `ExperimentManager` class:
```python
# New properties for global files:
@property
def df01_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"

@property
def df02_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"

@property
def df03_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

@property
def needs_build04(self) -> bool:
    # Global check: is df01 newer than df02?
    if not self.df01_path.exists():
        return False
    if not self.df02_path.exists():
        return True
    return self.df01_path.stat().st_mtime > self.df02_path.stat().st_mtime

@property
def needs_build06(self) -> bool:
    # Global check: is df02 newer than df03?
    if not self.df02_path.exists():
        return False
    if not self.df03_path.exists():
        return True
    return self.df02_path.stat().st_mtime > self.df03_path.stat().st_mtime
```

## The Phased Rollout Strategy

The documents recommend a careful three-phase approach:

**Phase 1**: Implement tracking logic only (no behavior changes)
**Phase 2**: Add read-only `status` command to test tracking
**Phase 3**: Enable actual orchestration with the tracking

This ensures the tracking logic is correct before it drives any actual processing decisions.


# Detailed Phased Rollout Strategy for MorphSeq Pipeline Tracking

## Phase 1: Implement Tracking Logic Only (Days 1-2)
**Goal**: Add all tracking infrastructure without changing any existing behavior

### Step 1.1: Extend `pipeline_objects.py` with New Properties

Add to the `Experiment` class:
```python
# New tracking properties - READ ONLY, no side effects
@property
def sam2_csv_path(self) -> Path:
    """Path where SAM2 output should be for this experiment"""
    return self.data_root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{self.date}.csv"

@property
def needs_sam2(self) -> bool:
    """Check if SAM2 needs to run for this experiment"""
    # Simple existence check for MVP
    return not self.sam2_csv_path.exists()

@property
def needs_build03(self) -> bool:
    """Check if Build03 needs to run based on SAM2 output timestamp"""
    last_run = self.timestamps.get("build03", 0)
    if self.sam2_csv_path.exists():
        newest = self.sam2_csv_path.stat().st_mtime
        return newest > last_run
    # If no SAM2 output, check for legacy masks
    return self.mask_path and self.mask_path.exists() and not self.flags.get("build03", False)

def get_latent_path(self, model_name: str) -> Path:
    """Construct path to latent embedding file"""
    return (self.data_root / "analysis" / "latent_embeddings" / 
            "legacy" / model_name / f"morph_latents_{self.date}.csv")

def has_latents(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
    """Check if latent embeddings exist for this experiment"""
    return self.get_latent_path(model_name).exists()

@property
def has_all_qc_masks(self) -> bool:
    """Check if all 5 Build02 QC masks exist"""
    mask_types = ["mask_v0_0100", "yolk_v1_0050", "focus_v0_0100", 
                  "bubble_v0_0100", "via_v1_0100"]
    seg_root = self.data_root / "segmentation"
    for mask_type in mask_types:
        mask_dir = seg_root / f"{mask_type}_predictions" / self.date
        if not mask_dir.exists():
            return False
    return True
```

Add to the `ExperimentManager` class:
```python
# Global tracking properties
@property
def df01_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"

@property
def df02_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"

@property
def df03_path(self) -> Path:
    return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

@property
def needs_build04(self) -> bool:
    """Check if Build04 needs to run (df01 -> df02)"""
    if not self.df01_path.exists():
        return False  # Can't run without input
    if not self.df02_path.exists():
        return True  # Output doesn't exist
    return self.df01_path.stat().st_mtime > self.df02_path.stat().st_mtime

@property
def needs_build06(self) -> bool:
    """Check if Build06 final combination needs to run (df02 -> df03)"""
    if not self.df02_path.exists():
        return False  # Can't run without input
    if not self.df03_path.exists():
        return True  # Output doesn't exist
    return self.df02_path.stat().st_mtime > self.df03_path.stat().st_mtime
```

### Step 1.2: Create Standalone Test Script

Create `test_tracking_logic.py`:
```python
#!/usr/bin/env python3
"""Test script to verify tracking logic without modifying any data"""

from pathlib import Path
from src.build.pipeline_objects import ExperimentManager

def test_tracking(data_root: str):
    """Test the new tracking properties"""
    manager = ExperimentManager(data_root)
    
    print("=" * 60)
    print("TRACKING LOGIC TEST REPORT")
    print("=" * 60)
    
    # Test per-experiment tracking
    for date, exp in sorted(manager.experiments.items())[:3]:  # Test first 3
        print(f"\nExperiment: {date}")
        print(f"  Microscope: {exp.microscope}")
        print(f"  Has all QC masks: {exp.has_all_qc_masks}")
        print(f"  SAM2 CSV exists: {exp.sam2_csv_path.exists()}")
        print(f"  Needs SAM2: {exp.needs_sam2}")
        print(f"  Needs Build03: {exp.needs_build03}")
        print(f"  Has latents: {exp.has_latents()}")
        
    # Test global tracking
    print(f"\nGlobal Files:")
    print(f"  df01 exists: {manager.df01_path.exists()}")
    print(f"  df02 exists: {manager.df02_path.exists()}")
    print(f"  df03 exists: {manager.df03_path.exists()}")
    print(f"  Needs Build04: {manager.needs_build04}")
    print(f"  Needs Build06 final: {manager.needs_build06_final}")

if __name__ == "__main__":
    test_tracking("/path/to/data")
```

### Step 1.3: Verification
- Run the test script against real data
- Manually verify the output matches expected state
- Check that no files are modified (use `ls -la` timestamps)
- Commit these changes as "feat: add tracking logic (read-only)"

## Phase 2: Add Read-Only Status Command (Day 3)
**Goal**: Integrate tracking into CLI without running any processing

### Step 2.1: Add Status Command to CLI

In `cli.py`, add the new subcommand:
```python
# Add to build_parser()
p_status = sub.add_parser("status", help="Show pipeline status for all experiments")
p_status.add_argument("--data-root", required=True)
p_status.add_argument("--experiments", help="Comma-separated list (default: all)")
p_status.add_argument("--verbose", action="store_true", help="Show detailed status")
```

### Step 2.2: Implement Status Handler

In `cli.py`, add to main():
```python
elif args.cmd == "status":
    from src.build.pipeline_objects import ExperimentManager
    
    manager = ExperimentManager(resolve_root(args))
    
    # Filter experiments if specified
    if args.experiments:
        exp_list = args.experiments.split(",")
        experiments = {k: v for k, v in manager.experiments.items() if k in exp_list}
    else:
        experiments = manager.experiments
    
    print("\n" + "=" * 80)
    print("MORPHSEQ PIPELINE STATUS REPORT")
    print("=" * 80)
    print(f"Data root: {resolve_root(args)}")
    print(f"Total experiments: {len(experiments)}")
    
    # Per-experiment status
    for date, exp in sorted(experiments.items()):
  
  elif args.cmd == "status":
    from src.build.pipeline_objects import ExperimentManager
    import traceback
    
    try:
        manager = ExperimentManager(resolve_root(args))
    except Exception as e:
        print(f"ERROR: Failed to initialize ExperimentManager: {e}")
        return 1
    
    # ... experiment filtering code ...
    
    # Per-experiment status with error handling
    failed_experiments = []
    for date, exp in sorted(experiments.items()):
        try:
            status_line = f"\n{date}: "
            status_bits = []
            
            # Wrap each check in try-except
            try:
                if exp.flags.get("ff", False):
                    status_bits.append("✅ FF")
                else:
                    status_bits.append("❌ FF")
            except Exception:
                status_bits.append("⚠️ FF")
            
            try:
                if exp.has_all_qc_masks:
                    status_bits.append("✅ QC")
                else:
                    status_bits.append("❌ QC")
            except Exception:
                status_bits.append("⚠️ QC")
                
            try:
                if exp.sam2_csv_path.exists():
                    status_bits.append("✅ SAM2")
                else:
                    status_bits.append("❌ SAM2")
            except (PermissionError, OSError):
                status_bits.append("⚠️ SAM2")
            
            print(status_line + " | ".join(status_bits))
            
        except Exception as e:
            print(f"\n{date}: ⚠️ ERROR - Could not check status")
            failed_experiments.append((date, str(e)))
            if args.verbose:
                print(f"  Error details: {e}")
                traceback.print_exc()
    
    # Report failures at the end
    if failed_experiments:
        print(f"\n⚠️ Warning: {len(failed_experiments)} experiments had errors during status check")
        if args.verbose:
            for exp_date, error in failed_experiments:
                print(f"  - {exp_date}: {error}")
```

### Step 2.3: Test Status Command
```bash
# Test on full dataset
python -m src.run_morphseq_pipeline.cli status --data-root /data

# Test on specific experiments
python -m src.run_morphseq_pipeline.cli status --data-root /data --experiments exp1,exp2

# Test verbose mode
python -m src.run_morphseq_pipeline.cli status --data-root /data --verbose
```

### Step 2.4: Verification
- Confirm status output matches manual inspection
- Verify no data files are modified
- Test with various experiment states (complete, partial, missing)
- Commit as "feat: add status command (read-only)"

## Phase 3: Enable Orchestration (Days 4-5)
**Goal**: Use tracking to drive actual pipeline execution

### Step 3.1: Add Step Execution Methods to Experiment

In `pipeline_objects.py`, add execution wrappers:
```python
# In Experiment class
@record("sam2")
def run_sam2(self, workers: int = 8, **kwargs):
    """Execute SAM2 segmentation for this experiment"""
    from ..run_morphseq_pipeline.steps.run_sam2 import run_sam2
    print(f"🎯 Running SAM2 for {self.date}")
    result = run_sam2(root=self.data_root, exp=self.date, workers=workers, **kwargs)
    return result

@record("build03")
def run_build03(self, by_embryo: int = None, frames_per_embryo: int = None, **kwargs):
    """Execute Build03 for this experiment with SAM2/legacy detection"""
    print(f"🔬 Running Build03 for {self.date}")
    
    # Import here to avoid circular dependencies
    from ..run_morphseq_pipeline.steps.run_build03 import run_build03 as run_build03_step
    
    # Determine which path to use
    sam2_csv = None
    if self.sam2_csv_path.exists():
        print(f"  Using SAM2 masks from {self.sam2_csv_path}")
        sam2_csv = str(self.sam2_csv_path)
    else:
        print(f"  Using legacy Build02 masks")
        # Check if legacy masks exist
        if not self.has_all_qc_masks:
            raise RuntimeError(f"No SAM2 CSV and missing QC masks for {self.date}")
    
    # Call the actual Build03 function with proper parameters
    try:
        result = run_build03_step(
            root=str(self.data_root),
            exp=self.date,
            sam2_csv=sam2_csv,  # Will be None for legacy path
            by_embryo=by_embryo,
            frames_per_embryo=frames_per_embryo,
            n_workers=kwargs.get('n_workers', self.num_cpu_workers),
            df01_out=kwargs.get('df01_out', None)  # Use default if not specified
        )
        
        # Update df01 contribution tracking
        if result:
            self.flags['contributed_to_df01'] = True
            self.timestamps['last_df01_contribution'] = datetime.utcnow().isoformat()
            
        return result
        
    except Exception as e:
        print(f"  ❌ Build03 failed: {e}")
        raise
Supporting Code in run_build03.py
The run_build03 function needs to handle both paths:
def run_build03(root: str, exp: str, sam2_csv: str = None, **kwargs):
    """
    Unified Build03 entry point that handles both SAM2 and legacy paths
    """
    root_path = Path(root)
    
    if sam2_csv:
        # SAM2 path - use the CSV directly
        print(f"Processing with SAM2 metadata: {sam2_csv}")
        # Read SAM2 CSV and process
        sam2_df = pd.read_csv(sam2_csv)
        # ... process SAM2 data ...
        
    else:
        # Legacy path - use Build02 masks
        print(f"Processing with legacy Build02 masks")
        from src.build.build03A_process_images import segment_wells, compile_embryo_stats
        
        tracked_df = segment_wells(root=root_path, exp_name=exp)
        stats_df = compile_embryo_stats(root=root_path, tracked_df=tracked_df)
        # ... continue with legacy processing ...
    
    # Common path - append to df01
    df01_path = root_path / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"
    # ... append results to df01 ...
    
    return True


@record("latents")
def generate_latents(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
    """Generate latent embeddings for this experiment"""
    from ..analyze.gen_embeddings import ensure_embeddings_for_experiments
    print(f"🧬 Generating latents for {self.date}")
    success = ensure_embeddings_for_experiments(
        data_root=self.data_root,
        experiments=[self.date],
        model_name=model_name,
        **kwargs
    )
    return success
```

### Step 3.2: Add Pipeline Command

In `cli.py`:
```python
# Add to build_parser()
p_pipe = sub.add_parser("pipeline", help="Orchestrated pipeline execution")
p_pipe.add_argument("--data-root", required=True)
p_pipe.add_argument("action", choices=["e2e", "sam2", "build03", "build04", "build06"])
p_pipe.add_argument("--experiments", help="Comma-separated list")
p_pipe.add_argument("--later-than", type=int, help="Process after YYYYMMDD")
p_pipe.add_argument("--force", action="store_true", help="Force rerun even if not needed")
p_pipe.add_argument("--dry-run", action="store_true", help="Show what would run")
# In cli.py - Add model-name argument to pipeline command
p_pipe.add_argument("--model-name", 
                    default="20241107_ds_sweep01_optimum",
                    help="Model name for embedding generation")

# In the pipeline handler
elif args.cmd == "pipeline":
    model_name = args.model_name  # Get from CLI
    
    if args.action == "e2e":
        for exp in selected:
            # Pass model_name down to the methods
            if args.force or not exp.has_latents(model_name):
                if not args.dry_run:
                    exp.generate_latents(model_name=model_name)
                    
    # Also need to pass to Build06 final
    if manager.needs_build06:
        # Call build06 with model_name parameter
        run_build06(..., model_name=model_name)

# In main()
elif args.cmd == "pipeline":
    from src.build.pipeline_objects import ExperimentManager
    
    manager = ExperimentManager(resolve_root(args))
    
    # Select experiments
    if args.experiments:
        exp_list = args.experiments.split(",")
        selected = [manager.experiments[e] for e in exp_list if e in manager.experiments]
    elif args.later_than:
        selected = [e for e in manager.experiments.values() 
                   if int(e.date[:8]) > args.later_than]
    else:
        selected = list(manager.experiments.values())
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
    
    if args.action == "e2e":
        # Full pipeline for selected experiments
        for exp in selected:
            print(f"\n{'='*60}")
            print(f"Processing {exp.date}")
            
            # Per-experiment steps
            if args.force or exp.needs_sam2:
                if not args.dry_run:
                    exp.run_sam2()
                else:
                    print(f"  Would run SAM2")
                    
            if args.force or exp.needs_build03:
                if not args.dry_run:
                    exp.run_build03()
                else:
                    print(f"  Would run Build03")
                    
            if args.force or not exp.has_latents():
                if not args.dry_run:
                    exp.generate_latents()
                else:
                    print(f"  Would generate latents")
        
        # Global steps (after all experiments processed)
        if args.force or manager.needs_build04:
            if not args.dry_run:
                print("\n🔄 Running Build04 (global)")
                # Call build04 function
            else:
                print("\nWould run Build04 (global)")
                
        if args.force or manager.needs_build06_final:
            if not args.dry_run:
                print("\n🔄 Running Build06 final (global)")
                # Call build06 function
            else:
                print("\nWould run Build06 final (global)")
```

### Step 3.3: Progressive Testing

1. **Dry run first** (no actual execution):
   ```bash
   python -m src.run_morphseq_pipeline.cli pipeline --data-root /data e2e --dry-run --experiments test_exp
   ```

2. **Single experiment test**:
   ```bash
   python -m src.run_morphseq_pipeline.cli pipeline --data-root /data e2e --experiments small_test_exp
   ```

3. **Small batch test**:
   ```bash
   python -m src.run_morphseq_pipeline.cli pipeline --data-root /data e2e --later-than 20250901
   ```

4. **Verify with status**:
   ```bash
   python -m src.run_morphseq_pipeline.cli status --data-root /data --experiments small_test_exp
   ```
# Use default model
python -m src.run_morphseq_pipeline.cli pipeline --data-root /data e2e

# Use different model
python -m src.run_morphseq_pipeline.cli pipeline --data-root /data e2e --model-name new_model_v2

### Step 3.4: Validation Points

After each test:
- Check that only needed steps ran (via logs)
- Verify state files updated correctly
- Confirm output files created where expected
- Run status command to see updated state
- Test resume capability by interrupting and rerunning

## Safety Measures Throughout

1. **All changes are additive** - existing CLI commands remain unchanged
2. **Dry-run capability** at every phase
3. **Progressive rollout** - start with read-only, then single experiments
4. **State verification** after each phase via status command
5. **Easy rollback** - can always fall back to original CLI commands

This phased approach ensures that the tracking logic is thoroughly tested before it drives any actual processing, minimizing risk while delivering the core orchestration benefits.