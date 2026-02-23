from __future__ import annotations
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict, Any

REQUIRED_DF01_COLS = [
    "snip_id", "embryo_id", "experiment_date", "well", "time_int",
    "Time Rel (s)", "predicted_stage_hpf", "surface_area_um",
    "short_pert_name", "phenotype", "control_flag", "temperature",
    "medium", "use_embryo_flag",
]


def check_df01_schema(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_DF01_COLS if c not in df.columns]
    return missing


def units_check(df: pd.DataFrame, n_rows: int = 3, tol: float = 1e-3) -> list[str]:
    """Rough units sanity: surface_area_um ≈ area_px × (um_per_px)^2.

    Returns list of row indices (as strings) that fail tolerance.
    """
    fails: list[str] = []
    sample = df.head(n_rows)
    for idx, row in sample.iterrows():
        try:
            if row.get("Height (px)", 0) and row.get("Height (um)", 0):
                um_per_px = float(row["Height (um)"]) / float(row["Height (px)"])
            else:
                continue
            expected = float(row.get("area_px", 0.0)) * (um_per_px ** 2)
            actual = float(row.get("surface_area_um", 0.0))
            if actual == 0:
                fails.append(str(idx))
                continue
            err = abs(actual - expected) / actual
            if err > tol:
                fails.append(str(idx))
        except Exception:
            fails.append(str(idx))
    return fails


def mask_paths_check(root: Path, df: pd.DataFrame) -> list[str]:
    """Verify exported mask files exist if column present.
    Looks under segmentation_sandbox/data/exported_masks/{exp}/masks.
    Returns list of missing relative paths.
    """
    if "exported_mask_path" not in df.columns:
        return []
    missing: list[str] = []
    for exp in df["experiment_date"].astype(str).unique():
        base = Path(root) / "segmentation_sandbox" / "data" / "exported_masks" / exp / "masks"
        sub = df[df["experiment_date"].astype(str) == exp]
        for rel in sub["exported_mask_path"].unique():
            if not (base / rel).exists():
                missing.append(str((base / rel)))
    return missing


def run_validation(root: str | Path, exp: str | None, df01: str, checks: str) -> None:
    root = Path(root)
    df01_path = Path(df01)
    if not df01_path.is_absolute():
        df01_path = root / df01_path
    if not df01_path.exists():
        raise FileNotFoundError(f"df01 not found: {df01_path}")
    df = pd.read_csv(df01_path)
    checks_list = [c.strip() for c in checks.split(",") if c.strip()]

    if "schema" in checks_list:
        missing = check_df01_schema(df)
        if missing:
            raise SystemExit(f"Schema check failed. Missing columns: {missing}")
        print("✅ Schema check passed.")

    if "units" in checks_list:
        failures = units_check(df)
        if failures:
            raise SystemExit(f"Units check failed for rows: {failures}")
        print("✅ Units check passed.")

    if "paths" in checks_list:
        miss = mask_paths_check(root, df)
        if miss:
            raise SystemExit(f"Mask path check failed. Missing: {miss[:5]}... (total {len(miss)})")
        print("✅ Mask path check passed.")


# ==============================================================================
# SAM2 Pipeline Validation Functions
# ==============================================================================

def validate_stitched_images(root: Path, exp: str) -> List[str]:
    """Check if Build01 stitched images exist for experiment.
    
    Args:
        root: Data root directory
        exp: Experiment name
        
    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []
    stitched_dir = root / "built_image_data" / "stitched_FF_images" / exp
    
    if not stitched_dir.exists():
        errors.append(f"Stitched images directory not found: {stitched_dir}")
        return errors
    
    # Check if directory has image files
    image_files = list(stitched_dir.glob("*.jpg")) + list(stitched_dir.glob("*.png"))
    if not image_files:
        errors.append(f"No image files found in {stitched_dir}")
    
    # Check metadata file exists
    metadata_file = root / "metadata" / "built_metadata_files" / f"{exp}_metadata.csv"
    if not metadata_file.exists():
        errors.append(f"Built metadata file not found: {metadata_file}")
    
    return errors


def validate_build02_masks(root: Path, exp: str, model_names: List[str] = None) -> List[str]:
    """Check if Build02 QC masks exist for experiment.
    
    Args:
        root: Data root directory
        exp: Experiment name  
        model_names: List of model names to check (defaults to all 5 QC models)
        
    Returns:
        List of error messages (empty if validation passes)
    """
    if model_names is None:
        model_names = [
            "mask_v0_0100",  # embryo masks
            "yolk_v1_0050",  # yolk masks  
            "focus_v0_0100", # focus masks
            "bubble_v0_0100", # bubble masks
            "via_v1_0100"    # viability masks
        ]
    
    errors = []
    segmentation_dir = root / "segmentation"
    
    for model_name in model_names:
        model_dir = segmentation_dir / f"{model_name}_predictions" / exp
        if not model_dir.exists():
            errors.append(f"Build02 {model_name} masks not found: {model_dir}")
            continue
            
        # Check if directory has mask files
        mask_files = list(model_dir.glob("*.png")) + list(model_dir.glob("*.tif*"))
        if not mask_files:
            errors.append(f"No mask files found in {model_dir}")
    
    return errors


def validate_sam2_outputs(root: Path, exp: str) -> List[str]:
    """Check if SAM2 pipeline outputs exist for experiment.
    
    Args:
        root: Data root directory
        exp: Experiment name
        
    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []
    sam2_root = root / "sam2_pipeline_files"
    
    # Check main SAM2 data structure
    if not sam2_root.exists():
        errors.append(f"SAM2 pipeline directory not found: {sam2_root}")
        return errors
    
    # Check experiment metadata CSV
    csv_file = sam2_root / "sam2_expr_files" / f"sam2_metadata_{exp}.csv"
    if not csv_file.exists():
        errors.append(f"SAM2 metadata CSV not found: {csv_file}")
    else:
        # Validate CSV schema
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["image_id", "embryo_id", "exported_mask_path", "experiment_id"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"SAM2 CSV missing required columns: {missing_cols}")
        except Exception as e:
            errors.append(f"Error reading SAM2 CSV {csv_file}: {e}")
    
    # Check exported masks directory
    masks_dir = sam2_root / "exported_masks" / exp
    if not masks_dir.exists():
        errors.append(f"SAM2 exported masks directory not found: {masks_dir}")
    else:
        mask_files = list(masks_dir.glob("**/*.png"))
        if not mask_files:
            errors.append(f"No mask files found in {masks_dir}")
    
    # Check core pipeline files
    pipeline_files = [
        sam2_root / "detections" / "gdino_detections.json",
        sam2_root / "embryo_metadata" / "grounded_sam_segmentations.json",
        sam2_root / "embryo_metadata" / "experiment_metadata.json"
    ]
    
    for file_path in pipeline_files:
        if not file_path.exists():
            errors.append(f"SAM2 pipeline file not found: {file_path}")
    
    return errors


def validate_sam2_mask_paths(root: Path, csv_path: Path) -> List[str]:
    """Validate that mask paths in SAM2 CSV actually exist.
    
    Args:
        root: Data root directory
        csv_path: Path to SAM2 metadata CSV
        
    Returns:
        List of missing mask file paths
    """
    errors = []
    
    try:
        df = pd.read_csv(csv_path)
        if "exported_mask_path" not in df.columns:
            return ["SAM2 CSV missing 'exported_mask_path' column"]
        
        sam2_masks_dir = root / "sam2_pipeline_files" / "exported_masks"
        
        missing_count = 0
        for mask_path in df["exported_mask_path"].unique():
            if pd.isna(mask_path):
                continue
                
            full_path = sam2_masks_dir / mask_path
            if not full_path.exists():
                missing_count += 1
                if missing_count <= 5:  # Only report first 5 missing files
                    errors.append(f"SAM2 mask file not found: {full_path}")
        
        if missing_count > 5:
            errors.append(f"... and {missing_count - 5} more missing mask files")
            
    except Exception as e:
        errors.append(f"Error validating SAM2 mask paths: {e}")
    
    return errors


def run_sam2_validation(
    root: str | Path, 
    exp: str, 
    checks: str = "stitched,build02,sam2,mask_paths"
) -> None:
    """Run SAM2 pipeline validation checks.
    
    Args:
        root: Data root directory
        exp: Experiment name
        checks: Comma-separated list of checks to run
               Options: stitched, build02, sam2, mask_paths
    """
    root = Path(root)
    checks_list = [c.strip() for c in checks.split(",") if c.strip()]
    all_errors = []
    
    print(f"🔍 Running SAM2 validation for experiment: {exp}")
    print(f"📁 Data root: {root}")
    print(f"🧪 Checks: {', '.join(checks_list)}")
    print()
    
    if "stitched" in checks_list:
        print("📹 Validating stitched images...")
        errors = validate_stitched_images(root, exp)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"❌ {error}")
        else:
            print("✅ Stitched images validation passed")
    
    if "build02" in checks_list:
        print("🎭 Validating Build02 QC masks...")
        errors = validate_build02_masks(root, exp)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"❌ {error}")
        else:
            print("✅ Build02 masks validation passed")
    
    if "sam2" in checks_list:
        print("🎯 Validating SAM2 outputs...")
        errors = validate_sam2_outputs(root, exp)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"❌ {error}")
        else:
            print("✅ SAM2 outputs validation passed")
    
    if "mask_paths" in checks_list and "sam2" in checks_list:
        print("🔗 Validating SAM2 mask file paths...")
        csv_path = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp}.csv"
        if csv_path.exists():
            errors = validate_sam2_mask_paths(root, csv_path)
            if errors:
                all_errors.extend(errors)
                for error in errors:
                    print(f"❌ {error}")
            else:
                print("✅ SAM2 mask paths validation passed")
        else:
            all_errors.append("SAM2 CSV not found for mask path validation")
            print("❌ SAM2 CSV not found for mask path validation")
    
    print()
    if all_errors:
        print(f"❌ Validation failed with {len(all_errors)} errors:")
        for error in all_errors:
            print(f"  • {error}")
        raise SystemExit("SAM2 validation failed")
    else:
        print("✅ All SAM2 validation checks passed!")

