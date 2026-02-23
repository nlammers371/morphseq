#!/usr/bin/env python3
"""
Build06 Per-Experiment CLI: Merge per-experiment df02 + latents → df03.

Usage:
    python -m src.run_morphseq_pipeline.steps.run_build06_per_exp \
      --data-root <root> \
      --exp <experiment_name> \
      --model-name <model> \
      [--no-generate-latents] [--overwrite] [--dry-run]

CLI semantics (MVP):
- Default: generates missing latents automatically for the requested experiment, then merges.
- --no-generate-latents: do not generate; fail if latents missing.
- --overwrite: overwrite the per‑experiment df03 AND force regeneration of latents (even if present).
- --overwrite and --no-generate-latents are mutually exclusive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.run_morphseq_pipeline.services.gen_embeddings import normalize_snip_ids


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build06 per-experiment embeddings merge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--exp", required=True, help="Experiment name/ID")
    parser.add_argument("--model-name", default="20241107_ds_sweep01_optimum", help="Model name for embeddings")
    parser.add_argument("--no-generate-latents", action="store_true", help="Do not generate missing latents (fail instead)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output and force latent regeneration")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show plan without processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _validate_flags(no_generate_latents: bool, overwrite: bool) -> None:
    """Validate mutually exclusive flags."""
    if no_generate_latents and overwrite:
        raise ValueError("❌ --no-generate-latents and --overwrite are mutually exclusive")


def _discover_build04_csv(root: Path, exp: str) -> Path:
    """Discover the Build04 per-experiment CSV file."""
    build04_csv = root / "metadata" / "build04_output" / f"qc_staged_{exp}.csv"
    if not build04_csv.exists():
        raise FileNotFoundError(f"❌ Build04 CSV not found for experiment {exp}: {build04_csv}")
    return build04_csv


def _discover_latents_path(data_root: Path, model_name: str, exp: str) -> Path:
    """Get expected latents path (doesn't check existence)."""
    return data_root / "analysis" / "latent_embeddings" / "legacy" / model_name / f"morph_latents_{exp}.csv"


def _prepare_output_path(root: Path, exp: str, overwrite: bool) -> Path:
    """Prepare and validate output path."""
    output_dir = root / "metadata" / "build06_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"df03_final_output_with_latents_{exp}.csv"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"❌ Output exists (use --overwrite): {output_path}")
    return output_path


def _validate_build04_schema(csv_path: Path, verbose: bool = False) -> dict:
    """Validate required columns in Build04 CSV and return stats."""
    if verbose:
        print(f"📋 Validating Build04 schema: {csv_path}")

    # Load just headers first
    df_head = pd.read_csv(csv_path, nrows=0)
    required_cols = ['snip_id', 'use_embryo_flag', 'experiment_date']
    missing = [col for col in required_cols if col not in df_head.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns in {csv_path}: {missing}")

    # Load full data for stats
    df = pd.read_csv(csv_path)
    stats = {
        'total_rows': len(df),
        'has_use_embryo_flag': 'use_embryo_flag' in df.columns,
        'quality_rows': len(df[df['use_embryo_flag'] == True]) if 'use_embryo_flag' in df.columns else 0,
    }

    if verbose:
        print(f"   ✅ Schema valid: {len(df)} total rows, {stats['quality_rows']} quality embryos")

    return stats


def _ensure_experiment_latents(
    data_root: Path,
    model_name: str,
    exp: str,
    generate_missing: bool,
    overwrite: bool,
    verbose: bool
) -> Path:
    """Generate/ensure latents for single experiment using existing services."""
    import logging
    from src.run_morphseq_pipeline.services.gen_embeddings import ensure_latents_for_experiments

    if verbose:
        print(f"   🤖 Ensuring latents for {exp} with model {model_name}")

    try:
        latent_paths = ensure_latents_for_experiments(
            data_root=data_root,
            model_name=model_name,
            experiments=[exp],
            generate_missing=generate_missing,
            overwrite=overwrite,
            repo_root=data_root,  # Use same root for MVP
            logger=logging.getLogger(__name__)
        )
        return latent_paths[exp]
    except Exception as e:
        raise RuntimeError(f"❌ Failed to ensure latents for {exp}: {e}") from e


def _load_and_filter_build04(csv_path: Path, verbose: bool) -> pd.DataFrame:
    """Load Build04 CSV and apply quality filtering."""
    from src.run_morphseq_pipeline.services.gen_embeddings import filter_high_quality_embryos
    import logging

    if verbose:
        print(f"   📊 Loading and filtering Build04 data")

    df02_raw = pd.read_csv(csv_path)
    df02_filtered = filter_high_quality_embryos(df02_raw, logger=logging.getLogger(__name__))

    if verbose:
        print(f"   ✅ {len(df02_raw)} rows → {len(df02_filtered)} quality embryos")

    return df02_filtered


def _load_experiment_latents(latents_path: Path, verbose: bool) -> pd.DataFrame:
    """Load latents for single experiment - capture ALL z_ columns."""
    df = pd.read_csv(latents_path)

    # Find ALL z_ columns (z_mu, z_sigma, etc.)
    z_cols = [col for col in df.columns if col.startswith('z_')]
    if not z_cols:
        raise ValueError(f"❌ No z_ columns found in {latents_path}")

    if verbose:
        print(f"   🧬 Loaded {len(df)} latent embeddings, {len(z_cols)} z_ dimensions")

    return df[['snip_id'] + z_cols].copy()


def _merge_with_coverage_report(df02: pd.DataFrame, latents: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Direct left-join merge - assume snip_ids are consistent."""
    if verbose:
        print(f"   🔗 Merging {len(df02)} embryos with {len(latents)} latent embeddings")

    # Direct merge without normalization (assume consistent snip_ids)
    df03 = df02.merge(latents, on='snip_id', how='left')

    # Calculate and report coverage using any z_ column as indicator
    z_cols = [col for col in df03.columns if col.startswith('z_')]
    if z_cols:
        matched = df03[z_cols[0]].notna().sum()
        total = len(df03)
        coverage = matched / total if total > 0 else 0

        print(f"   📊 Join coverage: {matched}/{total} ({coverage:.1%})")
        if coverage < 0.9:
            print(f"   ⚠️  Coverage below 90% - check snip_id consistency")
    else:
        print(f"   ⚠️  No z_ columns found after merge")

    return df03


def build06_merge_per_experiment(
    root: Path,
    exp: str,
    model_name: str,
    generate_missing: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Per-experiment Build06 merge: df02 + latents → df03.

    Args:
        root: Data root directory
        exp: Experiment name/ID
        model_name: Model name for embeddings
        generate_missing: Generate missing latents (default: True)
        overwrite: Overwrite output and force latent regeneration
        dry_run: Show plan without processing
        verbose: Verbose output

    Returns:
        Path to output file (or expected output path for dry-run)
    """
    if verbose or dry_run:
        print(f"🎯 Build06 Per-Experiment: {exp}")
        print(f"📊 Model: {model_name}")
        print(f"🔄 Overwrite: {overwrite}")
        print(f"🧬 Generate missing latents: {generate_missing}")
        print()

    # 1) Discover and validate Build04 input
    build04_csv = _discover_build04_csv(root, exp)
    build04_stats = _validate_build04_schema(build04_csv, verbose)

    # 2) Check latents path
    latents_path = _discover_latents_path(root, model_name, exp)
    latents_exist = latents_path.exists()

    if verbose or dry_run:
        print(f"📁 Build04 Input: ✅ {build04_csv}")
        print(f"   📊 {build04_stats['total_rows']} total rows, {build04_stats['quality_rows']} quality embryos")
        print(f"🧬 Latents: {'✅' if latents_exist else '❌'} {latents_path}")
        if not latents_exist:
            if generate_missing:
                print(f"   🤖 Will generate latents with model {model_name}")
            else:
                print(f"   ⚠️  Missing latents (generation disabled)")

    # 3) Validate latent generation logic
    if not latents_exist and not generate_missing:
        raise FileNotFoundError(f"❌ Latents missing and generation disabled: {latents_path}")

    if latents_exist and overwrite:
        if verbose or dry_run:
            print(f"   🔄 Will regenerate latents (overwrite mode)")

    # 4) Prepare output path
    output_path = _prepare_output_path(root, exp, overwrite)

    if verbose or dry_run:
        print(f"📁 Output: ✅ {output_path}")
        if output_path.exists():
            print(f"   🔄 Will overwrite existing file")
        print()

    # 5) Show processing plan for dry-run
    if dry_run:
        print("🔍 Processing Plan:")
        if not latents_exist or overwrite:
            print(f"   1️⃣  Generate latents for {exp}")
        print(f"   2️⃣  Load and filter Build04 data ({build04_stats['quality_rows']} quality embryos)")
        print(f"   3️⃣  Load latent embeddings (all z_ columns)")
        print(f"   4️⃣  Left-join df02 + latents by snip_id (no normalization)")
        print(f"   5️⃣  Calculate and report join coverage")
        print(f"   6️⃣  Write merged df03 to {output_path}")
        print()
        print("✅ Dry-run complete - use without --dry-run to execute")
        return output_path

    # 6) ACTUAL PROCESSING
    try:
        print(f"🔄 Processing {exp}...")

        # Step 1: Ensure latents exist
        if not latents_exist or overwrite:
            latents_path = _ensure_experiment_latents(
                root, model_name, exp, generate_missing, overwrite, verbose
            )

        # Step 2: Load and filter Build04 data
        df02_filtered = _load_and_filter_build04(build04_csv, verbose)

        # Step 3: Load latent embeddings
        latents_df = _load_experiment_latents(latents_path, verbose)

        # Step 4: Merge with coverage reporting
        df03 = _merge_with_coverage_report(df02_filtered, latents_df, verbose)

        # Step 5: Write output
        if verbose:
            print(f"   💾 Writing {len(df03)} rows to {output_path}")

        df03.to_csv(output_path, index=False)

        z_cols = [col for col in df03.columns if col.startswith('z_')]
        print(f"✅ Build06 per-experiment completed")
        print(f"   📊 {len(df03)} embryos with {len(z_cols)} latent dimensions")
        print(f"   📁 Output: {output_path}")

        return output_path

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise


def main():
    args = _parse_args()

    # Validate arguments
    try:
        _validate_flags(args.no_generate_latents, args.overwrite)

        root = Path(args.data_root)
        if not root.exists():
            print(f"❌ Data root does not exist: {root}")
            return 1

        # Convert flags to positive logic
        generate_missing = not args.no_generate_latents

        # Call the merge function
        output_path = build06_merge_per_experiment(
            root=root,
            exp=args.exp,
            model_name=args.model_name,
            generate_missing=generate_missing,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if not args.dry_run:
            print(f"✅ Build06 per-experiment completed")
            print(f"   📁 Output: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ Build06 per-experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
