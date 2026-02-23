"""
Pipeline integration wrapper for build06.

Provides clean integration functions for embedding generation
within the morphseq pipeline, specifically for build06 usage.
"""

from pathlib import Path
from typing import List, Optional, Union

from .subprocess_runner import run_embedding_generation_subprocess
from .file_utils import check_existing_embeddings, validate_data_root


def prepare_experiment_list(
    experiments: Union[str, List[str]]
) -> List[str]:
    """
    Prepare experiment list, handling special cases like "all".
    
    Args:
        experiments: Single experiment, list of experiments, or "all"
        
    Returns:
        List of experiment names
    """
    if isinstance(experiments, str):
        if experiments.lower() == "all":
            # For "all", caller must handle experiment discovery
            return ["all"]
        else:
            return [experiments]
    elif isinstance(experiments, (list, tuple)):
        return list(experiments)
    else:
        raise ValueError(f"Invalid experiments type: {type(experiments)}")


def report_generation_results(
    total_experiments: int,
    successful_experiments: int,
    verbose: bool = False
) -> None:
    """
    Report embedding generation results in standardized format.
    
    Args:
        total_experiments: Total number of experiments processed
        successful_experiments: Number of successful experiments
        verbose: Whether to show detailed output
    """
    failed_experiments = total_experiments - successful_experiments
    
    print(f"🔄 Embedding generation complete:")
    print(f"   Total: {total_experiments}")
    print(f"   Successful: {successful_experiments}")
    print(f"   Failed: {failed_experiments}")
    
    if failed_experiments == 0:
        print("✅ All embeddings generated successfully")
    elif successful_experiments > 0:
        print(f"⚠️  {failed_experiments} experiments failed")
    else:
        print("❌ All experiments failed")


def ensure_embeddings_for_experiments(
    data_root: Union[str, Path],
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy",
    py39_env_path: str = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
    overwrite: bool = False,
    process_missing: bool = False,
    generate_missing: Optional[bool] = None,
    verbose: bool = False
) -> bool:
    """
    Ensure embeddings exist for all experiments, generating missing ones.
    
    This is the main integration function for build06 usage.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        py39_env_path: Path to Python 3.9 environment
        overwrite: Force regeneration even if embeddings exist
        process_missing: Only process missing embeddings
        generate_missing: Alias for process_missing (kept for CLI compatibility)
        verbose: Print verbose output
        
    Returns:
        True if all embeddings are available, False otherwise
    """
    # Validate environment
    if not validate_data_root(str(data_root)):
        return False
    
    # Prepare experiment list
    experiment_list = prepare_experiment_list(experiments)
    
    if overwrite:
        # If caller requested overwrite, be explicit about the scope
        if "all" in experiment_list:
            print("🔄 Force regenerating ALL embeddings...")
        else:
            print(f"🔄 Force regenerating embeddings for {len(experiment_list)} experiments...")
        missing_experiments = experiment_list
    else:
        existing, missing = check_existing_embeddings(
            data_root, experiment_list, model_name, model_class
        )
        
        if existing:
            print(f"✅ Found existing embeddings for {len(existing)} experiments")
            if verbose:
                print(f"   Existing: {existing}")
        
        if not missing:
            print("✅ All embeddings already exist!")
            return True
        
        print(f"⚙️  Need to generate embeddings for {len(missing)} experiments")
        if verbose:
            print(f"   Missing: {missing}")
        missing_experiments = missing
    
    # Support legacy callers that pass generate_missing (e.g., experiment manager)
    if generate_missing is not None:
        process_missing = generate_missing
    
    # Generate missing embeddings
    success = run_embedding_generation_subprocess(
        data_root=str(data_root),
        experiments=missing_experiments,
        model_name=model_name,
        model_class=model_class,
        py39_env_path=py39_env_path,
        overwrite=overwrite,
        process_missing=process_missing,
        verbose=verbose
    )
    
    return success


def generate_embeddings_for_build06(
    data_root: Union[str, Path],
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy", 
    py39_env_path: Optional[str] = None,
    overwrite: bool = False,
    process_missing: bool = True,
    verbose: bool = False
) -> bool:
    """
    Specific build06 integration wrapper with sensible defaults.
    
    This function provides build06-specific defaults and behavior.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        py39_env_path: Path to Python 3.9 environment (auto-detect if None)
        overwrite: Force regeneration even if embeddings exist
        process_missing: Only process missing embeddings (default: True)
        verbose: Print verbose output
        
    Returns:
        True if all embeddings are available, False otherwise
    """
    # Set default Python 3.9 environment if not specified
    if py39_env_path is None:
        py39_env_path = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster"
    
    print(f"🤖 Build06 embedding generation:")
    print(f"   Model: {model_name} ({model_class})")
    print(f"   Experiments: {len(experiments)}")
    print(f"   Mode: {'overwrite' if overwrite else 'missing-only' if process_missing else 'standard'}")
    
    # Call main integration function
    success = ensure_embeddings_for_experiments(
        data_root=data_root,
        experiments=experiments,
        model_name=model_name,
        model_class=model_class,
        py39_env_path=py39_env_path,
        overwrite=overwrite,
        process_missing=process_missing,
        verbose=verbose
    )
    
    if success:
        print("🎉 Build06 embedding generation successful!")
    else:
        print("❌ Build06 embedding generation failed")
    
    return success


def check_embedding_status(
    data_root: Union[str, Path],
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy",
    verbose: bool = False
) -> dict:
    """
    Check embedding status for experiments without generating anything.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        verbose: Whether to show detailed output
        
    Returns:
        Dictionary with status information
    """
    existing, missing = check_existing_embeddings(
        data_root, experiments, model_name, model_class
    )
    
    status = {
        'total': len(experiments),
        'existing': existing,
        'missing': missing,
        'existing_count': len(existing),
        'missing_count': len(missing)
    }
    
    if verbose:
        print(f"📊 Embedding status for {len(experiments)} experiments:")
        print(f"   Existing: {len(existing)} ({existing})")
        print(f"   Missing: {len(missing)} ({missing})")
    
    return status
