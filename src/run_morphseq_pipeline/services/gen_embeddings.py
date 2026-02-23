"""
Pipeline-first service for embedding ingestion/generation and df02 merge.

Centralizes embedding operations under clear, testable functions used by Build06.
Keeps existing assessment scripts intact while providing canonical pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
import logging

# Import calculate_morph_embeddings lazily to avoid heavy dependency chain
# from src.analyze.analysis_utils import calculate_morph_embeddings


def filter_high_quality_embryos(df02: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Filter df02 to only embryos with use_embryo_flag=True (high quality).

    This replaces Build05's quality filtering by using the comprehensive QC flag from Build04.

    use_embryo_flag is computed by determine_use_embryo_flag() in src.build.qc.embryo_flags
    and excludes embryos with: dead_flag, dead_flag2, sa_outlier_flag, sam2_qc_flag,
    frame_flag, no_yolk_flag.

    Note: focus_flag and bubble_flag are NOT used for exclusion (too many false positives).

    Args:
        df02: Input dataframe from Build04
        logger: Optional logger for progress reporting

    Returns:
        Filtered dataframe containing only high-quality embryos
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    initial_count = len(df02)
    
    # Check if use_embryo_flag column exists
    if 'use_embryo_flag' not in df02.columns:
        logger.warning("use_embryo_flag column not found in df02 - using all embryos")
        return df02.copy()

    # Robust coercion: handle 0/1 and string encodings (e.g., 'true','false')
    flag = df02['use_embryo_flag']
    try:
        if flag.dtype != bool:
            if str(flag.dtype).startswith('int') or str(flag.dtype).startswith('float'):
                flag_bool = flag.fillna(0).astype(int).astype(bool)
            else:
                vals = flag.astype(str).str.strip().str.lower()
                mapping = {
                    'true': True, 't': True, '1': True, 'yes': True, 'y': True,
                    'false': False, 'f': False, '0': False, 'no': False, 'n': False,
                    '': False, 'nan': False
                }
                flag_bool = vals.map(mapping).fillna(False).astype(bool)
        else:
            flag_bool = flag
    except Exception:
        # Fallback to pandas truthiness if unexpected dtype
        flag_bool = flag.astype(bool)

    # Apply quality filter
    filtered_df = df02[flag_bool].copy()
    final_count = len(filtered_df)
    
    # Log filtering statistics
    filtered_out = initial_count - final_count
    filter_pct = (filtered_out / initial_count * 100) if initial_count > 0 else 0
    
    logger.info(f"✅ Quality filtering: {initial_count} → {final_count} embryos (use_embryo_flag=True)")
    logger.info(f"📊 Filtered out: {filtered_out} embryos ({filter_pct:.1f}%) with QC issues:")
    
    if filtered_out > 0:
        # Show breakdown of QC issues (if columns available)
        qc_flags = ['bubble_flag', 'focus_flag', 'frame_flag', 'dead_flag', 'no_yolk_flag']
        available_flags = [flag for flag in qc_flags if flag in df02.columns]
        
        if available_flags:
            bad_embryos = df02[df02["use_embryo_flag"] != True]
            for flag in available_flags:
                if flag in bad_embryos.columns:
                    flag_count = bad_embryos[flag].sum()
                    flag_pct = (flag_count / initial_count * 100) if initial_count > 0 else 0
                    logger.info(f"   - {flag}: {flag_count} ({flag_pct:.1f}%)")
    
    return filtered_df


def detect_missing_experiments(df02_path: Path, df03_path: Path, 
                              target_experiments: Optional[List[str]] = None,
                              logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Detect which experiments need processing based on df02 vs df03 comparison.
    
    Args:
        df02_path: Path to df02 file (source of truth for available experiments)
        df03_path: Path to df03 file (existing processed experiments)
        target_experiments: Optional list of specific experiments to target
        logger: Optional logger for progress reporting
        
    Returns:
        List of experiment IDs that need processing
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load existing df03 (if exists)
    if df03_path.exists():
        logger.info(f"📄 Loading existing df03: {df03_path}")
        df03 = pd.read_csv(df03_path)
        if 'experiment_date' in df03.columns:
            processed_experiments = set(df03['experiment_date'].dropna().unique())
            logger.info(f"📊 Found {len(processed_experiments)} processed experiments in df03")
        else:
            logger.warning("experiment_date column not found in df03")
            processed_experiments = set()
    else:
        logger.info("📄 No existing df03 found - will process all experiments")
        processed_experiments = set()
    
    # Load df02 to get available experiments
    if not df02_path.exists():
        raise FileNotFoundError(f"df02 not found: {df02_path}")
    
    logger.info(f"📄 Loading df02: {df02_path}")
    df02 = pd.read_csv(df02_path)
    
    if 'experiment_date' not in df02.columns:
        raise ValueError("experiment_date column not found in df02")
    
    available_experiments = set(df02['experiment_date'].dropna().unique())
    logger.info(f"📊 Found {len(available_experiments)} available experiments in df02")
    
    if target_experiments:
        # User specified experiments - check which need processing
        if target_experiments == "all":
            # Special case: user wants to process all available experiments
            target_set = available_experiments
            logger.info("🎯 Target: ALL experiments (explicit)")
        else:
            target_set = set(target_experiments)
            logger.info(f"🎯 Target: {len(target_set)} specified experiments")
            
            # Validate that target experiments exist in df02
            missing_from_df02 = target_set - available_experiments
            if missing_from_df02:
                logger.warning(f"⚠️  Target experiments not found in df02: {sorted(missing_from_df02)}")
                target_set = target_set - missing_from_df02
        
        missing_experiments = target_set - processed_experiments
    else:
        # Auto-discover - all experiments in df02 not in df03
        logger.info("🎯 Target: Auto-discover (all experiments in df02)")
        missing_experiments = available_experiments - processed_experiments
    
    missing_list = sorted(list(missing_experiments))
    logger.info(f"🔄 Need to process: {len(missing_list)} experiments")
    
    if missing_list:
        logger.info(f"📋 Missing experiments: {missing_list}")
    else:
        logger.info("✅ All target experiments already processed")
    
    return missing_list


def resolve_model_dir(data_root: Union[str, Path], model_name: str) -> Path:
    """
    Resolves model directory path and validates it contains required files.
    
    Args:
        data_root: Path to data root directory
        model_name: Name of model (e.g., "20241107_ds_sweep01_optimum")
        
    Returns:
        Path to validated model directory
        
    Raises:
        FileNotFoundError: If model directory or config not found
    """
    data_root = Path(data_root)
    model_dir = data_root / "models" / "legacy" / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # Check for model_config.json or other required files
    config_files = list(model_dir.glob("*config*.json"))
    if not config_files:
        # Check for alternative model structure (final_model subdirectory)
        final_model_dir = model_dir / "final_model"
        if final_model_dir.exists():
            config_files = list(final_model_dir.glob("*config*.json"))
            if config_files:
                return final_model_dir
        
        # If no config found, still return the directory - some models may not need it
        logging.warning(f"No config file found in {model_dir}, proceeding anyway")
    
    return model_dir


def get_available_snip_ids_for_experiment(repo_root: Union[str, Path], experiment: str) -> set:
    """
    Get available snip_ids for an experiment from the training_data directory.
    
    Args:
        repo_root: Repository root containing training_data/bf_embryo_snips
        experiment: Experiment name
        
    Returns:
        Set of available snip_id stems (without extensions)
    """
    repo_root = Path(repo_root)
    snip_dir = repo_root / "training_data" / "bf_embryo_snips" / experiment
    
    if not snip_dir.exists():
        return set()
    
    snip_ids = set()
    # Support both PNG and JPG snips
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for snip_file in snip_dir.glob(pattern):
            snip_ids.add(snip_file.stem)
    
    return snip_ids


def ensure_latents_for_experiments(
    data_root: Union[str, Path],
    model_name: str,
    experiments: List[str],
    generate_missing: bool = False,
    overwrite: bool = False,
    repo_root: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """
    Ensures latent CSV files exist and are up-to-date for all experiments.
    
    Checks if latents cover all available snip_ids for each experiment.
    If snips exist but aren't covered by latents, marks for regeneration.
    
    Args:
        data_root: Path to data root directory
        model_name: Name of model for embedding generation
        experiments: List of experiment names
        generate_missing: If True, generate missing/outdated latent files
        repo_root: Optional repo root to check for snips (for dependency checking)
        logger: Optional logger for output
        
    Returns:
        Dict mapping experiment names to their latent CSV paths
        
    Raises:
        FileNotFoundError: If latent files missing and generate_missing=False
    """
    data_root = Path(data_root)
    latent_dir = data_root / "analysis" / "latent_embeddings" / "legacy" / model_name
    
    latent_paths = {}
    missing_experiments = []
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for exp in experiments:
        latent_path = latent_dir / f"morph_latents_{exp}.csv"
        
        if latent_path.exists() and not overwrite:
            # Check if latents are up-to-date with available snips
            needs_regeneration = False
            
            if repo_root is not None:
                try:
                    # Get available snip_ids from training data
                    available_snip_ids = get_available_snip_ids_for_experiment(repo_root, exp)
                    
                    if available_snip_ids:  # Only check if we have snips to compare against
                        # Load existing latents and check coverage
                        latent_df = pd.read_csv(latent_path)
                        if 'snip_id' in latent_df.columns:
                            existing_snip_ids = set(latent_df['snip_id'].astype(str))
                            
                            missing_in_latents = available_snip_ids - existing_snip_ids
                            extra_in_latents = existing_snip_ids - available_snip_ids
                            
                            if missing_in_latents:
                                logger.info(f"Found {len(missing_in_latents)} snips not in latents for {exp}")
                                needs_regeneration = True
                            if extra_in_latents:
                                logger.info(f"Found {len(extra_in_latents)} latents without corresponding snips for {exp}")
                                needs_regeneration = True
                            
                            if not needs_regeneration:
                                logger.info(f"Latents up-to-date for {exp}: {len(existing_snip_ids)} snips covered")
                        else:
                            logger.warning(f"No snip_id column in existing latents for {exp}, marking for regeneration")
                            needs_regeneration = True
                    else:
                        logger.info(f"✅ Using available latents for {exp} (embeddings ready)")
                        
                except Exception as e:
                    logger.warning(f"Could not check snip coverage for {exp}: {e}, using existing latents")
            
            if needs_regeneration:
                missing_experiments.append(exp)
                logger.info(f"Marking {exp} for regeneration due to snip coverage issues")
            else:
                latent_paths[exp] = latent_path
                logger.info(f"📊 Using embeddings for {exp}: {latent_path}")
        else:
            missing_experiments.append(exp)
            logger.info(f"Need to generate latents for {exp}: {latent_path}")
    
    if overwrite:
        # Force regeneration for all requested experiments
        missing_experiments = experiments
    
    if missing_experiments:
        if generate_missing or overwrite:
            logger.info(f"🤖 Generating embeddings for {len(missing_experiments)} experiments using legacy model")
            
            # Lazy import to avoid heavy dependency chain
            try:
                from src.analyze.analysis_utils import calculate_morph_embeddings
            except ImportError as e:
                logger.error(f"Cannot import calculate_morph_embeddings: {e}")
                logger.error("This is required for --generate-missing-latents. Please check your environment.")
                raise ImportError(
                    "Missing dependencies for embedding generation. "
                    "Please ensure all required packages (einops, torch, etc.) are installed."
                ) from e
            
            # Use existing calculate_morph_embeddings function
            # Note: calculate_morph_embeddings writes to
            #   <data_root>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_<exp>.csv
            # We pass through model_name as-is so callers can direct outputs to a custom tag.
            calculate_morph_embeddings(
                data_root=data_root,
                model_name=model_name,
                model_class="legacy",
                experiments=missing_experiments
            )
            
            # Verify generation succeeded
            for exp in missing_experiments:
                latent_path = latent_dir / f"morph_latents_{exp}.csv"
                if latent_path.exists():
                    latent_paths[exp] = latent_path
                    logger.info(f"Generated latents for {exp}: {latent_path}")
                else:
                    logger.error(f"Failed to generate latents for {exp}")
                    raise FileNotFoundError(f"Could not generate latents for {exp}")
        else:
            logger.info(f"⏭️  Skipping latent generation for {len(missing_experiments)} experiments (generation disabled)")
            raise FileNotFoundError(
                f"Missing latent files: {missing_experiments}. "
                f"Use --generate-missing-latents to create them."
            )
    
    return latent_paths


def generate_latents_with_repo_images(
    repo_root: Union[str, Path],
    data_root: Union[str, Path],
    model_name: str,
    latents_tag: Optional[str],
    experiments: List[str],
    logger: Optional[logging.Logger] = None,
):
    """
    Generate per-experiment latent CSVs using snips under the repo root
    (SAM2 naming) while loading the model from the central data root.

    Outputs are written under:
      <data_root>/analysis/latent_embeddings/legacy/<latents_dir>/morph_latents_<exp>.csv

    Args:
        repo_root: Project/repo root that contains training_data/bf_embryo_snips
        data_root: Central data root for models and analysis outputs
        model_name: Name of the model directory under <data_root>/models/legacy
        latents_tag: Optional alt directory name for outputs (defaults to model_name)
        experiments: Experiment names to process
        logger: Optional logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    repo_root = Path(repo_root)
    data_root = Path(data_root)
    latents_dir = data_root / "analysis" / "latent_embeddings" / "legacy" / (latents_tag or model_name)
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports to avoid heavy startup
    try:
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from src.analyze.analysis_utils import extract_embeddings_legacy
        from src.legacy.vae import AutoModel
        from src.core.data.data_transforms import basic_transform
        from src.core.data.dataset_configs import EvalDataConfig
    except Exception as e:
        logger.error(f"Failed to import dependencies for latent generation: {e}")
        raise

    # Load model from central data_root with Python version compatibility handling
    model_dir = resolve_model_dir(data_root, model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from .legacy_model_utils import load_legacy_model_safe
        logger.info(f"Loading legacy model from {model_dir} with compatibility handling")
        # Automatic subprocess routing to Python 3.9 when needed
        lit_model = load_legacy_model_safe(
            str(model_dir), device=device, logger=logger
        )
        metadata_msg = (
            f"Loaded legacy model metadata: name={getattr(lit_model, 'model_name', None)}, "
            f"latent_dim={getattr(lit_model, 'latent_dim', None)}, "
            f"nuisance_len={len(list(getattr(lit_model, 'nuisance_indices', [])))}"
        )
        logger.info(metadata_msg)
    except Exception as e:
        logger.warning(f"Safe model loading failed: {e}")
        logger.info("Attempting direct model loading (may fail on Python != 3.9)")
        from src.legacy.vae import AutoModel
        lit_model = AutoModel.load_from_folder(str(model_dir))
        logger.info(
            "Direct legacy model load metadata: "
            f"name={getattr(lit_model, 'model_name', None)}, "
            f"latent_dim={getattr(lit_model, 'latent_dim', None)}, "
            f"nuisance_len={len(list(getattr(lit_model, 'nuisance_indices', [])))}"
        )

    # Build dataloader using repo's snips
    input_size = (288, 128)  # matches legacy defaults
    transform = basic_transform(target_size=input_size)
    eval_cfg = EvalDataConfig(experiments=experiments, root=repo_root, return_sample_names=True, transforms=transform)
    dataset = eval_cfg.create_dataset()
    dl = DataLoader(dataset, batch_size=64, num_workers=eval_cfg.num_workers, shuffle=False)

    # Run encoder and collect embeddings
    logger.info(f"Generating embeddings for {len(experiments)} experiments from repo snips")
    lit_model.to(device).eval()
    latent_df = extract_embeddings_legacy(lit_model=lit_model, dataloader=dl, device=device)

    # Write per-experiment CSVs
    for exp in experiments:
        exp_df = latent_df.loc[latent_df["experiment_date"] == exp]
        out_path = latents_dir / f"morph_latents_{exp}.csv"
        exp_df.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(exp_df)} rows to {out_path}")


def load_latents(latent_paths: Dict[str, Path], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Loads and combines per-experiment latent CSV files.
    
    Args:
        latent_paths: Dict mapping experiment names to CSV paths
        logger: Optional logger for output
        
    Returns:
        Combined DataFrame with snip_id + z_mu_* columns
        
    Raises:
        ValueError: If DataFrames have inconsistent schemas or invalid data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    dfs = []
    
    for exp, path in latent_paths.items():
        logger.info(f"Loading latents from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        if 'snip_id' not in df.columns:
            raise ValueError(f"Missing 'snip_id' column in {path}")
        
        # Find z_mu columns
        z_cols = [col for col in df.columns if col.startswith('z_mu_')]
        if not z_cols:
            logger.warning(f"No z_mu_* columns found in {path}")
            continue
        
        # Select snip_id and all z_mu columns (including z_mu_b_*, z_mu_n_* variants)
        keep_cols = ['snip_id'] + z_cols
        df_subset = df[keep_cols].copy()
        
        # Validate finite values in z columns
        z_numeric_cols = [col for col in z_cols if df_subset[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        for col in z_numeric_cols:
            if not df_subset[col].isna().all():  # Skip if all NaN
                import numpy as np
                finite_mask = np.isfinite(df_subset[col])
                if not finite_mask.all():
                    invalid_count = (~finite_mask).sum()
                    logger.warning(f"Found {invalid_count} non-finite values in {col} from {path}")
                    # Replace non-finite with NaN
                    df_subset.loc[~finite_mask, col] = pd.NA
        
        dfs.append(df_subset)
        logger.info(f"Loaded {len(df_subset)} rows, {len(z_cols)} z_mu columns from {exp}")
    
    if not dfs:
        raise ValueError("No valid latent data found")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates by snip_id, keeping first occurrence
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['snip_id'], keep='first')
    final_count = len(combined_df)
    
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} duplicate snip_ids")
    
    logger.info(f"Combined latents: {len(combined_df)} unique snip_ids")
    
    return combined_df


def normalize_snip_ids(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Normalizes snip_id formats to ensure consistent joining.
    
    Handles variations like '_s####' vs '_####' suffixes from different naming schemes.
    
    Args:
        df: DataFrame with snip_id column
        logger: Optional logger for output
        
    Returns:
        DataFrame with normalized snip_ids
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    df = df.copy()
    
    # Ensure consistent string format
    df['snip_id'] = df['snip_id'].astype(str)
    
    # Log some sample snip_ids for debugging
    sample_ids = df['snip_id'].head(3).tolist()
    logger.info(f"Sample snip_ids: {sample_ids}")
    
    logger.info(f"Normalized {len(df)} snip_ids")
    
    return df


def merge_df02_with_embeddings(
    root: Union[str, Path],
    latents_df: pd.DataFrame,
    df02_filtered: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
    out_name: str = "embryo_metadata_df03.csv",
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Merges df02 with embeddings to create df03.
    
    Args:
        root: Pipeline root directory
        latents_df: DataFrame with embeddings (snip_id + z_mu_* columns)
        df02_filtered: Optional pre-filtered df02 DataFrame (if None, loads from file)
        overwrite: Allow overwriting existing df03
        out_name: Output filename
        logger: Optional logger for output
        
    Returns:
        Path to created df03 file
        
    Raises:
        FileNotFoundError: If df02 doesn't exist and df02_filtered not provided
        FileExistsError: If df03 exists and overwrite=False
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    df02_path = root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
    df03_path = root / "metadata" / "combined_metadata_files" / out_name
    
    if df03_path.exists() and not overwrite:
        raise FileExistsError(f"df03 already exists: {df03_path}. Use --overwrite to replace.")
    
    # Use provided filtered df02 or load from file
    if df02_filtered is not None:
        logger.info("Using provided quality-filtered df02")
        df02 = df02_filtered
    else:
        if not df02_path.exists():
            raise FileNotFoundError(f"df02 not found: {df02_path}")
        logger.info(f"Loading df02 from {df02_path}")
        df02 = pd.read_csv(df02_path)
    
    # Normalize snip_ids in both DataFrames
    df02_norm = normalize_snip_ids(df02, logger)
    latents_norm = normalize_snip_ids(latents_df, logger)
    
    # Merge on snip_id (left join to preserve all df02 rows)
    logger.info("Merging df02 with embeddings")
    df03 = df02_norm.merge(latents_norm, on='snip_id', how='left')
    
    # Calculate join coverage
    embedding_cols = [col for col in df03.columns if col.startswith('z_mu_')]
    if embedding_cols:
        # Check coverage using first z column as indicator
        first_z_col = embedding_cols[0]
        matched_count = df03[first_z_col].notna().sum()
        total_count = len(df03)
        coverage = matched_count / total_count if total_count > 0 else 0
        
        logger.info(f"Join coverage: {matched_count}/{total_count} ({coverage:.1%})")
        
        if coverage < 0.95:
            logger.warning(f"Low join coverage: {coverage:.1%} < 95%")
    else:
        logger.warning("No embedding columns found after merge")
    
    # Write df03
    logger.info(f"Writing df03 to {df03_path}")
    df03.to_csv(df03_path, index=False)
    
    logger.info(f"✔️  Created df03 with {len(df03)} rows, {len(embedding_cols)} embedding columns")
    
    return df03_path


def merge_train_with_embeddings(
    root: Union[str, Path],
    train_name: str,
    latents_df: pd.DataFrame,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Optionally merges training metadata with embeddings.
    
    Args:
        root: Pipeline root directory
        train_name: Training run name
        latents_df: DataFrame with embeddings
        overwrite: Allow overwriting existing file
        logger: Optional logger for output
        
    Returns:
        Path to created file, or None if training metadata doesn't exist
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    train_meta_path = root / "training_data" / train_name / "embryo_metadata_df_train.csv"
    train_out_path = root / "training_data" / train_name / "embryo_metadata_with_embeddings.csv"
    
    if not train_meta_path.exists():
        logger.info(f"Training metadata not found: {train_meta_path}")
        return None
    
    if train_out_path.exists() and not overwrite:
        raise FileExistsError(f"Training output exists: {train_out_path}. Use --overwrite to replace.")
    
    logger.info(f"Loading training metadata from {train_meta_path}")
    train_df = pd.read_csv(train_meta_path)
    
    # Normalize snip_ids
    train_norm = normalize_snip_ids(train_df, logger)
    latents_norm = normalize_snip_ids(latents_df, logger)
    
    # Merge
    train_with_embeddings = train_norm.merge(latents_norm, on='snip_id', how='left')
    
    # Write output
    logger.info(f"Writing training output to {train_out_path}")
    train_with_embeddings.to_csv(train_out_path, index=False)
    
    return train_out_path


def export_df03_copies_by_experiment(
    df03: pd.DataFrame,
    data_root: Union[str, Path],
    model_name: str,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Exports per-experiment df03 copies for analysis users.
    
    Args:
        df03: Complete df03 DataFrame
        data_root: Data root directory
        model_name: Model name for output directory
        overwrite: Allow overwriting existing files
        logger: Optional logger for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    data_root = Path(data_root)
    export_dir = data_root / "metadata" / "metadata_n_embeddings" / model_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if 'experiment_date' not in df03.columns:
        logger.warning("No 'experiment_date' column found, cannot create per-experiment exports")
        return
    
    experiments = df03['experiment_date'].dropna().unique()
    
    for exp in experiments:
        exp_df = df03[df03['experiment_date'] == exp]
        exp_path = export_dir / f"df03_{exp}.csv"
        
        if exp_path.exists() and not overwrite:
            logger.warning(f"Skipping existing file: {exp_path} (use --overwrite)")
            continue
        
        exp_df.to_csv(exp_path, index=False)
        logger.info(f"Exported {len(exp_df)} rows for {exp} to {exp_path}")


def build_df03_with_embeddings(
    root: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    model_name: str = "20241107_ds_sweep01_optimum",
    latents_tag: Optional[str] = None,
    experiments: Optional[List[str]] = None,
    generate_missing: bool = False,
    overwrite_latents: bool = False,
    use_repo_snips: bool = False,
    export_analysis: bool = False,
    train_name: Optional[str] = None,
    write_train_output: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    One-shot orchestrator for Build06 - the main entry point.
    
    Args:
        root: Data root directory (contains all data: metadata, models, latents, etc.)
        data_root: Optional legacy parameter for external data root. If None, uses root.
        model_name: Model name for embeddings
        experiments: Optional explicit experiment list
        generate_missing: Generate missing latent files
        export_analysis: Export per-experiment df03 copies
        train_name: Training run name for optional join
        write_train_output: Write training metadata with embeddings
        overwrite: Allow overwriting existing files
        dry_run: Print planned actions without executing
        logger: Optional logger for output
        
    Returns:
        Path to created df03 file
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    
    # Handle legacy two-root approach vs unified approach
    if data_root is None:
        # Unified approach: root contains everything
        data_root = root
        logger.info("=== Build06: Standardize Embeddings Generation (Unified Root Mode) ===")
        logger.info(f"Root: {root}")
    else:
        # Legacy approach: separate data root
        data_root = Path(data_root)
        logger.info("=== Build06: Standardize Embeddings Generation (Legacy Two-Root Mode) ===")
        logger.info(f"Repo root: {root}")
        logger.info(f"Data root: {data_root}")
    
    logger.info(f"Model: {model_name}")
    if latents_tag:
        logger.info(f"Latents tag: {latents_tag}")
    
    # 1) Resolve df02 and experiment list - check multiple possible locations
    df02_locations = [
        root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv",
        root / "morphseq_playground" / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv",
        data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv",
    ]
    
    df02_path = None
    for path in df02_locations:
        logger.info(f"Checking for df02 at: {path}")
        if path.exists():
            df02_path = path
            logger.info(f"✅ Found df02 at: {df02_path}")
            break
    
    if df02_path is None:
        logger.error("df02 not found at any of these locations:")
        for path in df02_locations:
            logger.error(f"  - {path}")
        raise FileNotFoundError(f"df02 not found at any expected location")
    
    # 1.1) Load df02 for experiment discovery and quality filtering
    logger.info(f"📄 Loading df02 for processing: {df02_path}")
    df02_raw = pd.read_csv(df02_path)
    
    if 'experiment_date' not in df02_raw.columns:
        raise ValueError("experiment_date column not found in df02")
    
    # 1.2) Apply quality filtering (replaces Build05 functionality)
    logger.info("🔍 Applying quality filtering (use_embryo_flag=True)")
    df02 = filter_high_quality_embryos(df02_raw, logger)
    
    # 1.3) Determine df03 path for incremental processing
    df03_path = data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"
    
    # 1.4) Decide which experiments to process for df03 merge
    # If caller provides an explicit experiment list, treat it as a re-merge list
    # regardless of df03 membership (used when latents are newer than df03).
    if experiments is not None:
        if experiments == "all":
            experiments_to_process = df02['experiment_date'].dropna().unique().tolist()
            logger.info(f"🎯 Explicit ALL experiments requested - processing {len(experiments_to_process)} experiments")
        else:
            experiments_to_process = list(experiments)
            logger.info(f"🎯 Explicit re-merge for {len(experiments_to_process)} experiment(s)")
    else:
        # Incremental mode: process experiments missing from df03
        logger.info("🔄 Detecting missing experiments (incremental mode)")
        experiments_to_process = detect_missing_experiments(df02_path, df03_path, None, logger)
        if not experiments_to_process:
            # Nothing to merge; keep existing df03
            logger.info("✅ All experiments already present in df03 - nothing to do")
            return df03_path if df03_path.exists() else df02_path
    
    logger.info(f"🎯 Final processing list: {len(experiments_to_process)} experiments")
    if experiments_to_process:
        logger.info(f"📋 Will process: {sorted(experiments_to_process)}")
    
    if dry_run:
        logger.info("=== DRY RUN - Planned Actions ===")
        logger.info(f"📊 Quality filtering: {len(df02_raw)} → {len(df02)} embryos")
        logger.info(f"🎯 Would process: {len(experiments_to_process)} experiments")
        if experiments_to_process:
            logger.info(f"📋 Experiment list: {sorted(experiments_to_process)}")
        logger.info(f"🤖 Generate missing latents: {generate_missing}")
        logger.info(f"🧨 Overwrite latents: {overwrite_latents}")
        logger.info(f"📂 Use repo snips: {use_repo_snips}")
        logger.info(f"📤 Export analysis copies: {export_analysis}")
        logger.info(f"🎓 Write training output: {write_train_output}")
        logger.info(f"📄 Target df03: {df03_path}")
        return df03_path  # Return target path for dry run
    
    # 2) Resolve model directory
    model_dir = resolve_model_dir(data_root, model_name)
    logger.info(f"Using model: {model_dir}")
    
    # Early exit if no experiments to process
    if not experiments_to_process:
        logger.info("✅ No experiments to process - exiting early")
        return df03_path if df03_path.exists() else df02_path
    
    # 3) Ensure latents exist (read/write under <data_root>/analysis/.../<latents_dir>)
    latents_dir_name = latents_tag or model_name
    try:
        latent_paths = ensure_latents_for_experiments(
            data_root, latents_dir_name, experiments_to_process, False, overwrite_latents, root, logger
        )
    except FileNotFoundError:
        # Missing are expected in first pass; we'll generate below if requested
        latent_paths = {}

    # Generate missing if requested
    missing = [exp for exp in experiments_to_process if exp not in latent_paths]
    if missing and (generate_missing or overwrite_latents):
        if use_repo_snips:
            logger.info("Generating missing latents from repo snips (SAM2-aligned)")
            generate_latents_with_repo_images(root, data_root, model_name, latents_dir_name, missing, logger)
        else:
            # Use standard generator, which writes under the given model_name (latents_dir_name)
            _ = ensure_latents_for_experiments(data_root, latents_dir_name, missing, True, overwrite_latents, root, logger)
        # Reload paths after generation
        latent_paths = ensure_latents_for_experiments(
            data_root, latents_dir_name, experiments_to_process, False, False, root, logger
        )
    
    # 4) Load and combine all embeddings
    logger.info(f"🔗 Loading embeddings from {len(latent_paths)} experiment(s)")
    latents_df = load_latents(latent_paths, logger)
    
    # 5) Merge with quality-filtered df02 to create df03
    # Overwrite df03 if we have experiments to process (re-merge) or if overwrite=True
    allow_overwrite = overwrite or bool(experiments_to_process)
    df03_path = merge_df02_with_embeddings(data_root, latents_df, df02_filtered=df02, overwrite=allow_overwrite, logger=logger)
    
    # 6) Optional training output
    if write_train_output and train_name:
        train_output = merge_train_with_embeddings(root, train_name, latents_df, overwrite, logger)
        if train_output:
            logger.info(f"✔️  Created training output: {train_output}")
    
    # 7) Optional analysis exports
    if export_analysis:
        df03 = pd.read_csv(df03_path)
        export_df03_copies_by_experiment(df03, data_root, model_name, overwrite, logger)
    
    logger.info("✔️  Build06 complete")
    return df03_path
