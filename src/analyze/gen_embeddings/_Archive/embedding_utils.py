#!/usr/bin/env python3
"""
Simple embedding utilities for build06 integration.
Human-readable, no complex nesting.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

def generate_embeddings_for_experiments(
    data_root: str | Path,
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy", 
    py39_env_path: str = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
    overwrite: bool = False,
    process_missing: bool = False,
    verbose: bool = False
) -> bool:
    """
    Generate embeddings for experiments using Python 3.9 subprocess.
    
    Simple wrapper for build06 CLI integration.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name (default: 20241107_ds_sweep01_optimum)
        model_class: Model class (default: legacy)
        py39_env_path: Path to Python 3.9 environment
        overwrite: Overwrite existing embeddings
        process_missing: Only process missing embeddings
        verbose: Print verbose output
        
    Returns:
        True if all embeddings generated successfully, False otherwise
    """
    
    # Build command - use absolute path to the script
    script_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/generate_embeddings_cli.py")
    cmd = [
        "python", str(script_path),
        "--data-root", str(data_root),
        "--model-name", model_name,
        "--model-class", model_class,
        "--py39-env", py39_env_path,
        "--experiments"
    ] + experiments
    
    if overwrite:
        cmd.append("--overwrite")
    if process_missing:
        cmd.append("--process-missing")
    if verbose:
        cmd.append("--verbose")
    
    print(f"🔄 Generating embeddings for {len(experiments)} experiments...")
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the CLI script
        result = subprocess.run(cmd, check=False, capture_output=not verbose, text=True)
        
        if result.returncode == 0:
            print("✅ All embeddings generated successfully")
            return True
        else:
            print("❌ Some embeddings failed to generate")
            if not verbose and result.stderr:
                print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running embedding generation: {e}")
        return False


def check_existing_embeddings(
    data_root: str | Path, 
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy"
) -> tuple[List[str], List[str]]:
    """
    Check which embeddings already exist.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        
    Returns:
        (existing_experiments, missing_experiments)
    """
    data_root = Path(data_root)
    latents_dir = data_root / "analysis" / "latent_embeddings" / model_class / model_name
    
    existing = []
    missing = []
    
    for experiment in experiments:
        latents_file = latents_dir / f"morph_latents_{experiment}.csv"
        if latents_file.exists():
            existing.append(experiment)
        else:
            missing.append(experiment)
    
    return existing, missing


def ensure_embeddings_for_experiments(
    data_root: str | Path,
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum", 
    model_class: str = "legacy",
    py39_env_path: str = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
    overwrite: bool = False,
    process_missing: bool = False,
    verbose: bool = False
) -> bool:
    """
    Ensure embeddings exist for all experiments, generating missing ones.
    
    Simple function for build06 integration.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        py39_env_path: Path to Python 3.9 environment
        overwrite: Force regeneration even if embeddings exist
        process_missing: Only process missing embeddings
        verbose: Print verbose output
        
    Returns:
        True if all embeddings are available, False otherwise
    """
    
    if overwrite:
        # If caller requested overwrite, be explicit about the scope.
        if isinstance(experiments, str) and experiments.lower() == "all":
            print("🔄 Force regenerating ALL embeddings...")
        elif isinstance(experiments, (list, tuple)):
            print(f"🔄 Force regenerating embeddings for {len(experiments)} experiments...")
        else:
            print("🔄 Force regenerating embeddings (unknown scope)...")
        missing_experiments = experiments
    else:
        existing, missing = check_existing_embeddings(
            data_root, experiments, model_name, model_class
        )
        
        if existing:
            print(f"✅ Found existing embeddings for {len(existing)} experiments")
        
        if not missing:
            print("✅ All embeddings already exist!")
            return True
        
        print(f"⚙️  Need to generate embeddings for {len(missing)} experiments")
        missing_experiments = missing
    
    # Generate missing embeddings
    success = generate_embeddings_for_experiments(
        data_root=data_root,
        experiments=missing_experiments,
        model_name=model_name,
        model_class=model_class,
        py39_env_path=py39_env_path,
        overwrite=overwrite,
        process_missing=process_missing,
        verbose=verbose
    )
    
    return success