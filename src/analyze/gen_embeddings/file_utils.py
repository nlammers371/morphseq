"""
File utilities for embedding path management.

Handles file checking, path resolution, and directory validation 
for embedding generation operations.
"""

from pathlib import Path
from typing import List, Tuple, Optional


def get_embedding_file_path(
    data_root: Path,
    experiment: str,
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy"
) -> Path:
    """
    Get the standard path for embedding files.
    
    Args:
        data_root: Data root directory
        experiment: Experiment name
        model_name: Model name
        model_class: Model class
        
    Returns:
        Path to embedding file
    """
    latents_dir = data_root / "analysis" / "latent_embeddings" / model_class / model_name
    return latents_dir / f"morph_latents_{experiment}.csv"


def validate_data_root(data_root: str) -> bool:
    """
    Validate that data root directory exists and has expected structure.
    
    Args:
        data_root: Path to data root directory
        
    Returns:
        True if data root is valid, False otherwise
    """
    data_root_path = Path(data_root)
    
    if not data_root_path.exists():
        print(f"❌ Error: Data root directory not found: {data_root_path}")
        return False
    
    if not data_root_path.is_dir():
        print(f"❌ Error: Data root is not a directory: {data_root_path}")
        return False
    
    # Check for expected subdirectories
    expected_dirs = ["metadata", "built_image_data"]
    missing_dirs = []
    
    for expected_dir in expected_dirs:
        if not (data_root_path / expected_dir).exists():
            missing_dirs.append(expected_dir)
    
    if missing_dirs:
        print(f"⚠️  Warning: Missing expected directories in data root: {missing_dirs}")
        # Don't fail validation - these might be created later
    
    return True


def validate_python39_environment(py39_env_path: str) -> bool:
    """
    Validate that Python 3.9 environment exists and is accessible.
    
    Args:
        py39_env_path: Path to Python 3.9 environment
        
    Returns:
        True if environment is valid, False otherwise
    """
    py39_python = Path(py39_env_path) / "bin" / "python"
    
    if not py39_python.exists():
        print(f"❌ Error: Python 3.9 environment not found: {py39_python}")
        return False
    
    return True


def check_existing_embeddings(
    data_root: str | Path,
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy"
) -> Tuple[List[str], List[str]]:
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
    data_root_path = Path(data_root)
    existing = []
    missing = []
    
    for experiment in experiments:
        latents_file = get_embedding_file_path(
            data_root_path, experiment, model_name, model_class
        )
        
        if latents_file.exists():
            existing.append(experiment)
        else:
            missing.append(experiment)
    
    return existing, missing


def list_missing_experiments(
    data_root: str | Path,
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy"
) -> List[str]:
    """
    Identify experiments without embeddings.
    
    Args:
        data_root: Path to data root directory
        experiments: List of experiment names
        model_name: Model name
        model_class: Model class
        
    Returns:
        List of experiments missing embeddings
    """
    _, missing = check_existing_embeddings(
        data_root, experiments, model_name, model_class
    )
    return missing


def ensure_embedding_directory(
    data_root: str | Path,
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy"
) -> Path:
    """
    Ensure embedding directory exists, creating it if necessary.
    
    Args:
        data_root: Path to data root directory
        model_name: Model name
        model_class: Model class
        
    Returns:
        Path to embedding directory
    """
    data_root_path = Path(data_root)
    latents_dir = data_root_path / "analysis" / "latent_embeddings" / model_class / model_name
    
    latents_dir.mkdir(parents=True, exist_ok=True)
    return latents_dir


def validate_embedding_file(embedding_file: Path) -> bool:
    """
    Validate that an embedding file exists and is readable.
    
    Args:
        embedding_file: Path to embedding file
        
    Returns:
        True if file is valid, False otherwise
    """
    if not embedding_file.exists():
        return False
    
    if not embedding_file.is_file():
        return False
    
    try:
        # Check if file is readable and not empty
        with open(embedding_file, 'r') as f:
            first_line = f.readline()
            if not first_line.strip():
                return False
        return True
    except (IOError, PermissionError):
        return False