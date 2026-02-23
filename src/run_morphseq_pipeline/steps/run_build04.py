#!/usr/bin/env python3
"""
Build04 CLI: Process per-experiment embryo QC and staging.

Supports two usage modes:
1. Experiment discovery: --root <root> --exp <experiment_id>
2. Explicit paths: --root <root> --in-csv <path> --out-csv <path>

Default output: metadata/build04_output/qc_staged_{exp}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.build.build04_perform_embryo_qc import build04_stage_per_experiment
from src.data_pipeline.quality_control.config import QC_DEFAULTS


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build04 per-experiment QC and staging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="Data root directory")
    parser.add_argument("--exp", help="Experiment ID (for auto-discovery mode)")
    parser.add_argument("--in-csv", help="Input Build03 CSV path (overrides auto-discovery)")
    parser.add_argument("--out-csv", help="Output Build04 CSV path (overrides default)")
    parser.add_argument("--out-dir", help="Output directory (overrides default)")
    parser.add_argument("--stage-ref", help="Stage reference CSV path")
    parser.add_argument("--dead-lead-time", type=float, default=QC_DEFAULTS['dead_lead_time_hours'], help="Hours before death to retroactively flag")
    parser.add_argument("--sg-window", type=int, default=5, help="Savitzky-Golay window length")
    parser.add_argument("--sg-poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _discover_build03_csv(root: Path, exp: str) -> Path:
    """Discover the Build03 CSV file for the experiment."""
    build03_csv = root / "metadata" / "build03_output" / f"expr_embryo_metadata_{exp}.csv"
    if build03_csv.exists():
        return build03_csv
    
    raise FileNotFoundError(f"Build03 CSV not found for experiment {exp}: {build03_csv}")


def main():
    args = _parse_args()
    
    root = Path(args.root)
    verbose = args.verbose
    
    # Determine input and output paths
    if args.in_csv and args.out_csv:
        # Explicit paths mode
        in_csv = Path(args.in_csv)
        out_csv = Path(args.out_csv)
        exp = args.exp or "unknown"
        if verbose:
            print(f"🔧 Explicit paths mode")
    elif args.exp:
        # Experiment discovery mode
        exp = args.exp
        in_csv = _discover_build03_csv(root, exp)
        
        # Default output directory/path
        if args.out_dir:
            output_dir = Path(args.out_dir)
        else:
            output_dir = root / "metadata" / "build04_output"

        if args.out_csv:
            out_csv = Path(args.out_csv)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_csv = output_dir / f"qc_staged_{exp}.csv"
        
        if verbose:
            print(f"🔍 Experiment discovery mode for: {exp}")
    else:
        print("❌ Error: Must provide either --exp OR both --in-csv and --out-csv")
        return 1
    
    if verbose:
        print(f"📁 Input CSV: {in_csv}")
        print(f"📁 Output CSV: {out_csv}")
    
    # Ensure input exists
    if not in_csv.exists():
        print(f"❌ Input CSV not found: {in_csv}")
        return 1
    
    try:
        out_path = build04_stage_per_experiment(
            root=root,
            exp=exp,
            in_csv=in_csv,
            out_dir=args.out_dir and Path(args.out_dir),
            stage_ref=args.stage_ref and Path(args.stage_ref),
            dead_lead_time=args.dead_lead_time,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
        )
        
        print(f"✅ Build04 completed for {exp}")
        print(f"   📁 Output: {out_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Build04 failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


# Thin function wrapper for CLI imports and programmatic use
def run_build04(
    root: Path | str,
    exp: str | None = None,
    in_csv: Path | str | None = None,
    out_csv: Path | str | None = None,
    out_dir: Path | str | None = None,
    stage_ref: Path | str | None = None,
    dead_lead_time: float = None,
    sg_window: int = 5,
    sg_poly: int = 2,
) -> Path | None:
    """Programmatic entry for Build04 per-experiment QC.

    Prefer passing `exp` to auto-discover input and choose default output.
    Alternatively, specify `in_csv` and `out_csv` explicitly.
    Returns the output path if successful.

    Parameters
    ----------
    root : Path or str
        Data root directory
    exp : str, optional
        Experiment ID for auto-discovery
    in_csv : Path or str, optional
        Input Build03 CSV path
    out_csv : Path or str, optional
        Output Build04 CSV path
    out_dir : Path or str, optional
        Output directory
    stage_ref : Path or str, optional
        Stage reference CSV path
    dead_lead_time : float, optional
        Hours before death to retroactively flag.
        If None, uses QC_DEFAULTS['dead_lead_time_hours'] (default 4.0)
    sg_window : int, default 5
        Savitzky-Golay window length
    sg_poly : int, default 2
        Savitzky-Golay polynomial order
    """
    # Use default from config if not specified
    if dead_lead_time is None:
        dead_lead_time = QC_DEFAULTS['dead_lead_time_hours']
    root_p = Path(root)
    if in_csv and out_csv:
        in_p = Path(in_csv)
        out_p = Path(out_csv)
        sel_exp = exp or in_p.stem.replace("expr_embryo_metadata_", "")
    elif exp:
        sel_exp = exp
        in_p = _discover_build03_csv(root_p, sel_exp)
        if out_dir:
            od = Path(out_dir)
        else:
            od = root_p / "metadata" / "build04_output"
        od.mkdir(parents=True, exist_ok=True)
        out_p = od / f"qc_staged_{sel_exp}.csv"
    else:
        raise ValueError("Provide either exp or both in_csv and out_csv")

    return build04_stage_per_experiment(
        root=root_p,
        exp=sel_exp,
        in_csv=in_p,
        out_dir=out_p.parent,
        stage_ref=Path(stage_ref) if stage_ref else None,
        dead_lead_time=dead_lead_time,
        sg_window=sg_window,
        sg_poly=sg_poly,
    )
