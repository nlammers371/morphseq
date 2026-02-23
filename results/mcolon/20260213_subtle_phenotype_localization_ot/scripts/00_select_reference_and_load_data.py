#!/usr/bin/env python3
"""Select reference embryo and load WT controls + mutants for 48 hpf pilot.

This script loads pre-existing cohort manifests from the Stream D migration
(results/mcolon/20260213_stream_d_reference_embryo/) and extracts:
- 1 reference WT embryo (rank 1 from reference_wt cohort)
- ≥10 WT control embryos (heldout_wt cohort)
- ≥20 mutant embryos (mutant cohort)

All embryos are selected for target HPF bin (48 hpf ± 1.25).

NOTE: Cohort selection logic is in:
  results/mcolon/20260213_stream_d_reference_embryo/pipeline/01_build_cohort_manifest.py
  
If manifests are not found at expected location, the script searches for them
in results/mcolon/2026* directories (migration-safe).

Outputs:
- data/selected_embryos_48hpf.csv: Selected embryos with roles
- data/reference_embryo_info.json: Reference embryo metadata
- data/embryo_summary.txt: Selection summary
- data/cohort_contract_48hpf.json: Locked cohort contract (params + exact IDs/frames)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _find_project_root(start: Path) -> Path:
    """Find project root by searching for src/ and results/ directories."""
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


def _search_for_file(filename: str, base_dir: Path, pattern: str = "2026*") -> Optional[Path]:
    """Search for a file in subdirectories matching pattern."""
    search_dirs = sorted(base_dir.glob(pattern), reverse=True)
    for d in search_dirs:
        candidates = list(d.rglob(filename))
        if candidates:
            return candidates[0]
    return None


def load_cohort_manifest(
    expected_path: Path,
    search_base: Optional[Path] = None,
) -> pd.DataFrame:
    """Load pre-existing cohort manifest from Stream D.
    
    Args:
        expected_path: Expected location after migration
        search_base: Base directory to search if not found (optional)
    
    Returns:
        DataFrame with columns: embryo_id, genotype, set_type, set_rank, 
                                n_bins_covered, coverage_frac, curvature_median, etc.
    
    Raises:
        FileNotFoundError: If manifest not found in expected or searched locations
    """
    if expected_path.exists():
        print(f"Loading cohort manifest from: {expected_path}")
        return pd.read_csv(expected_path)
    
    if search_base is not None:
        print(f"Manifest not found at {expected_path}, searching in {search_base}...")
        found_path = _search_for_file("cohort_selected_embryos.csv", search_base)
        if found_path is not None:
            print(f"Found manifest at: {found_path}")
            return pd.read_csv(found_path)
    
    raise FileNotFoundError(
        f"Cohort manifest not found at {expected_path}. "
        f"Run results/mcolon/20260213_stream_d_reference_embryo/pipeline/01_build_cohort_manifest.py first."
    )


def load_bin_frame_manifest(
    expected_path: Path,
    search_base: Optional[Path] = None,
) -> pd.DataFrame:
    """Load bin-frame manifest mapping embryo_id → frame_index per HPF bin.
    
    Args:
        expected_path: Expected location after migration
        search_base: Base directory to search if not found (optional)
    
    Returns:
        DataFrame with columns: embryo_id, set_type, set_rank, genotype, bin_hpf,
                                frame_index, matched_stage_hpf, stage_abs_err_hpf, etc.
    
    Raises:
        FileNotFoundError: If manifest not found
    """
    if expected_path.exists():
        print(f"Loading bin-frame manifest from: {expected_path}")
        return pd.read_csv(expected_path)
    
    if search_base is not None:
        print(f"Manifest not found at {expected_path}, searching in {search_base}...")
        found_path = _search_for_file("cohort_bin_frame_manifest.csv", search_base)
        if found_path is not None:
            print(f"Found manifest at: {found_path}")
            return pd.read_csv(found_path)
    
    raise FileNotFoundError(
        f"Bin-frame manifest not found at {expected_path}."
    )


def extract_embryos_for_bin(
    bin_manifest: pd.DataFrame,
    cohort_df: pd.DataFrame,
    target_hpf: float,
    tolerance_hpf: float = 1.25,
) -> pd.DataFrame:
    """Extract embryos for target HPF bin with tolerance.
    
    Args:
        bin_manifest: Per-bin frame assignments
        cohort_df: Cohort metadata (set_type, rank, QC metrics)
        target_hpf: Target HPF bin (e.g., 48.0)
        tolerance_hpf: Stage matching tolerance (default 1.25)
    
    Returns:
        DataFrame with selected embryos at target bin
    """
    # Filter to target bin
    bin_subset = bin_manifest[bin_manifest["bin_hpf"] == target_hpf].copy()
    
    # Remove rows with missing frames
    bin_subset = bin_subset.dropna(subset=["frame_index"])
    
    # Filter by stage tolerance
    bin_subset = bin_subset[bin_subset["stage_abs_err_hpf"] <= tolerance_hpf]
    
    # Join with cohort metadata (redundant but ensures consistency)
    cohort_subset = cohort_df[["embryo_id", "genotype", "set_type", "set_rank", 
                                 "coverage_frac", "curvature_median"]].copy()
    
    merged = bin_subset.merge(cohort_subset, on=["embryo_id", "genotype", "set_type", "set_rank"], how="left")
    
    return merged


def assign_roles(selected_df: pd.DataFrame, n_ref: int = 1) -> pd.DataFrame:
    """Assign roles based on set_type.
    
    Args:
        selected_df: Selected embryos with set_type column
        n_ref: Number of reference embryos to select (default 1 for pilot)
    
    Returns:
        DataFrame with added 'role' column
    """
    df = selected_df.copy()
    
    # Map set_type to role
    role_map = {
        "reference_wt": "reference",
        "heldout_wt": "control",
        "mutant": "mutant",
    }
    df["role"] = df["set_type"].map(role_map)
    
    # For pilot: take only top n_ref reference embryos
    if n_ref < len(df[df["role"] == "reference"]):
        ref_subset = df[df["role"] == "reference"].sort_values("set_rank").iloc[:n_ref]
        non_ref = df[df["role"] != "reference"]
        df = pd.concat([ref_subset, non_ref], ignore_index=True)
    
    return df


def summarize_selection(selected_df: pd.DataFrame) -> Dict:
    """Generate summary statistics for selected embryos."""
    summary = {
        "total_embryos": int(len(selected_df)),
        "by_role": {},
        "by_genotype": {},
        "stage_range_hpf": {
            "min": float(selected_df["matched_stage_hpf"].min()),
            "max": float(selected_df["matched_stage_hpf"].max()),
            "mean": float(selected_df["matched_stage_hpf"].mean()),
        },
    }
    
    for role, g in selected_df.groupby("role"):
        summary["by_role"][role] = {
            "count": int(len(g)),
            "mean_coverage": float(g["coverage_frac"].mean()),
            "median_curvature": float(g["curvature_median"].median()),
            "stage_range": [float(g["matched_stage_hpf"].min()), float(g["matched_stage_hpf"].max())],
        }
    
    for geno, g in selected_df.groupby("genotype"):
        summary["by_genotype"][str(geno)] = int(len(g))
    
    return summary


def _hpf_tag(target_hpf: float) -> str:
    if float(target_hpf).is_integer():
        return f"{int(target_hpf)}hpf"
    return f"{str(target_hpf).replace('.', 'p')}hpf"


def _build_contract(
    selected_df: pd.DataFrame,
    args: argparse.Namespace,
    cohort_path: Path,
    bin_path: Path,
) -> Dict:
    n_ref_selected = int((selected_df["role"] == "reference").sum())
    n_control_selected = int((selected_df["role"] == "control").sum())
    n_mutant_selected = int((selected_df["role"] == "mutant").sum())
    contract = {
        "contract_version": "1.0",
        "description": "Locked cohort contract for subtle phenotype localization pilot.",
        "selection_parameters": {
            "genotype_wt": "cep290_wildtype",
            "genotype_mutant": "cep290_homozygous",
            "target_hpf": float(args.target_hpf),
            "tolerance_hpf": float(args.tolerance_hpf),
            "n_ref_requested": int(args.n_ref),
            "n_control_min_required": int(args.min_controls),
            "n_mutant_min_required": int(args.min_mutants),
            "n_ref_selected": n_ref_selected,
            "n_control_selected": n_control_selected,
            "n_mutant_selected": n_mutant_selected,
        },
        "temporal_window_policy": {
            "mode": "single_2hpf_bin",
            "primary_window_hpf": [float(args.target_hpf) - 1.0, float(args.target_hpf) + 1.0],
            "expanded_window_hpf_optional": [float(args.target_hpf) - 2.0, float(args.target_hpf) + 2.0],
            "multi_frame_collapse_rule": (
                "If expanded window is used, collapse to one row per embryo before stats "
                "using per-embryo median of each feature across frames."
            ),
        },
        "source_manifests": {
            "cohort_selected_embryos_csv": str(cohort_path),
            "cohort_bin_frame_manifest_csv": str(bin_path),
        },
        "cohorts": {
            "reference": [],
            "control": [],
            "mutant": [],
        },
    }

    key_cols = ["set_rank", "embryo_id", "frame_index", "matched_stage_hpf", "stage_abs_err_hpf", "genotype"]
    for role in ["reference", "control", "mutant"]:
        role_df = selected_df[selected_df["role"] == role].copy()
        role_df = role_df.sort_values(["set_rank", "embryo_id"], ascending=[True, True])
        for row in role_df[key_cols].itertuples(index=False):
            contract["cohorts"][role].append(
                {
                    "embryo_id": str(row.embryo_id),
                    "frame_index": int(row.frame_index),
                    "set_rank": int(row.set_rank),
                    "matched_stage_hpf": float(row.matched_stage_hpf),
                    "stage_abs_err_hpf": float(row.stage_abs_err_hpf),
                    "genotype": str(row.genotype),
                }
            )
    return contract


def run(args: argparse.Namespace) -> None:
    """Main execution function."""
    project_root = _find_project_root(Path(__file__).resolve())
    analysis_root = Path(__file__).resolve().parents[1]
    output_dir = analysis_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define expected paths (migration-aware)
    stream_d_root = project_root / "results" / "mcolon" / "20260213_stream_d_reference_embryo"
    cohort_path = stream_d_root / "output" / "cohort_selection" / "cohort_selected_embryos.csv"
    bin_path = stream_d_root / "output" / "cohort_selection" / "cohort_bin_frame_manifest.csv"
    search_base = project_root / "results" / "mcolon"
    
    # Load manifests (with fallback search)
    cohort_df = load_cohort_manifest(cohort_path, search_base=search_base if args.search_fallback else None)
    bin_manifest = load_bin_frame_manifest(bin_path, search_base=search_base if args.search_fallback else None)
    
    # Extract embryos for target bin
    selected = extract_embryos_for_bin(
        bin_manifest,
        cohort_df,
        target_hpf=args.target_hpf,
        tolerance_hpf=args.tolerance_hpf,
    )
    
    # Assign roles
    selected = assign_roles(selected, n_ref=args.n_ref)
    
    # Validate counts
    n_ref = len(selected[selected["role"] == "reference"])
    n_ctrl = len(selected[selected["role"] == "control"])
    n_mut = len(selected[selected["role"] == "mutant"])
    
    print(f"\nSelected embryos at {args.target_hpf} hpf:")
    print(f"  Reference: {n_ref}")
    print(f"  Controls: {n_ctrl}")
    print(f"  Mutants: {n_mut}")
    
    if n_ref < 1:
        raise ValueError(f"Need ≥1 reference embryo, got {n_ref}")
    if n_ctrl < args.min_controls:
        raise ValueError(f"Need ≥{args.min_controls} WT controls for permutation testing, got {n_ctrl}")
    if n_mut < args.min_mutants:
        raise ValueError(f"Need ≥{args.min_mutants} mutants, got {n_mut}")
    
    # Generate summary
    summary = summarize_selection(selected)
    
    # Save outputs
    hpf_tag = _hpf_tag(args.target_hpf)
    selected_path = output_dir / f"selected_embryos_{hpf_tag}.csv"
    selected.to_csv(selected_path, index=False)
    print(f"\nWrote selected embryos to: {selected_path}")
    
    # Save reference embryo info
    ref_embryo = selected[selected["role"] == "reference"].iloc[0]
    ref_info = {
        "embryo_id": str(ref_embryo["embryo_id"]),
        "frame_index": int(ref_embryo["frame_index"]),
        "genotype": str(ref_embryo["genotype"]),
        "matched_stage_hpf": float(ref_embryo["matched_stage_hpf"]),
        "curvature_median": float(ref_embryo["curvature_median"]),
        "coverage_frac": float(ref_embryo["coverage_frac"]),
        "set_rank": int(ref_embryo["set_rank"]),
    }
    ref_path = output_dir / "reference_embryo_info.json"
    with open(ref_path, "w") as f:
        json.dump(ref_info, f, indent=2)
    print(f"Wrote reference embryo info to: {ref_path}")

    contract = _build_contract(
        selected_df=selected,
        args=args,
        cohort_path=cohort_path,
        bin_path=bin_path,
    )
    contract_path = output_dir / f"cohort_contract_{hpf_tag}.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"Wrote cohort contract to: {contract_path}")
    
    # Save summary
    summary_path = output_dir / "embryo_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Embryo Selection Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Target HPF: {args.target_hpf} ± {args.tolerance_hpf}\n")
        f.write(f"Total embryos: {summary['total_embryos']}\n\n")
        f.write("By role:\n")
        for role, stats in summary["by_role"].items():
            f.write(f"  {role:12s}: {stats['count']:3d} embryos, "
                   f"coverage={stats['mean_coverage']:.2f}, "
                   f"curvature={stats['median_curvature']:.6f}\n")
        f.write(f"\nBy genotype:\n")
        for geno, count in summary["by_genotype"].items():
            f.write(f"  {geno:20s}: {count:3d}\n")
        f.write(f"\nStage range: {summary['stage_range_hpf']['min']:.2f} - {summary['stage_range_hpf']['max']:.2f} hpf\n")
        f.write(f"Mean stage: {summary['stage_range_hpf']['mean']:.2f} hpf\n")
    print(f"Wrote summary to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Reference embryo and cohorts selected.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select reference embryo and load WT controls + mutants for 48 hpf pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-hpf",
        type=float,
        default=48.0,
        help="Target HPF bin for pilot study",
    )
    parser.add_argument(
        "--tolerance-hpf",
        type=float,
        default=1.25,
        help="Stage matching tolerance (hpf)",
    )
    parser.add_argument(
        "--n-ref",
        type=int,
        default=1,
        help="Number of reference embryos to select (1 for pilot)",
    )
    parser.add_argument(
        "--min-controls",
        type=int,
        default=10,
        help="Minimum WT controls required for permutation testing",
    )
    parser.add_argument(
        "--min-mutants",
        type=int,
        default=20,
        help="Minimum mutants required",
    )
    parser.add_argument(
        "--search-fallback",
        action="store_true",
        default=True,
        help="Search for manifests in results/mcolon/2026* if not found at expected location",
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_false",
        dest="search_fallback",
        help="Disable fallback search (fail if manifests not at expected location)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
