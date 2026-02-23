"""
Python 3.9 subprocess orchestration for embedding generation.

Handles the execution of embedding generation in a Python 3.9 subprocess
to maintain compatibility with legacy models that require specific Python versions.
"""

import subprocess
from pathlib import Path
from typing import List, Optional


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


def build_subprocess_command(
    data_root: str,
    experiment: str,
    model_name: str,
    model_class: str,
    repo_root: Optional[str] = None
) -> str:
    """
    Build Python script content for subprocess execution.
    
    Args:
        data_root: Data root directory
        experiment: Experiment name
        model_name: Model name
        model_class: Model class
        repo_root: Repository root path (auto-detected if None)
        
    Returns:
        Python script content as string
    """
    if repo_root is None:
        repo_root = str(Path(__file__).parent.parent.parent.parent)
    
    script_content = f'''
import sys
from pathlib import Path

# Add src/ to path
repo_root = Path("{repo_root}")
src_root = repo_root / "src"
sys.path.insert(0, str(src_root))

print("Python version:", sys.version_info)

from analyze.analysis_utils import calculate_morph_embeddings

print("Calling calculate_morph_embeddings...")

try:
    result = calculate_morph_embeddings(
        data_root="{data_root}",
        model_name="{model_name}",
        model_class="{model_class}",
        experiments=["{experiment}"]
    )
    print("✅ Success:", result)
except Exception as e:
    print("❌ Error:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    return script_content


def handle_subprocess_output(result: subprocess.CompletedProcess, 
                           experiment: str, verbose: bool) -> bool:
    """
    Process subprocess results and provide user feedback.
    
    Args:
        result: Completed subprocess result
        experiment: Experiment name being processed
        verbose: Whether to show verbose output
        
    Returns:
        True if subprocess succeeded, False otherwise
    """
    if verbose:
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"✅ Successfully generated embeddings for {experiment}")
        return True
    else:
        print(f"❌ Failed to generate embeddings for {experiment}")
        print("Error output:", result.stderr)
        return False


def run_embedding_generation_subprocess(
    data_root: str,
    experiments: List[str],
    model_name: str = "20241107_ds_sweep01_optimum",
    model_class: str = "legacy",
    py39_env_path: str = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
    overwrite: bool = False,
    process_missing: bool = False,
    verbose: bool = False
) -> bool:
    """
    Execute embedding generation using Python 3.9 subprocess.
    
    Args:
        data_root: Data root directory
        experiments: List of experiment names to process
        model_name: Model name
        model_class: Model class
        py39_env_path: Path to Python 3.9 environment
        overwrite: Whether to overwrite existing embeddings
        process_missing: Whether to skip existing embeddings
        verbose: Whether to show verbose output
        
    Returns:
        True if all experiments processed successfully, False otherwise
    """
    from .file_utils import get_embedding_file_path
    
    py39_python = Path(py39_env_path) / "bin" / "python"
    data_root_path = Path(data_root)
    
    # Validate environment
    if not validate_python39_environment(py39_env_path):
        return False
    
    success_count = 0
    
    for experiment in experiments:
        print(f"=== Processing {experiment} ===")
        
        # Check if embeddings already exist
        latents_file = get_embedding_file_path(
            data_root_path, experiment, model_name, model_class
        )
        
        if latents_file.exists():
            if overwrite:
                print(f"🔄 Overwriting existing: {latents_file}")
            elif process_missing:
                print(f"⏭️  Skipping existing: {latents_file}")
                success_count += 1
                continue
            else:
                print(f"✅ Embeddings already exist: {latents_file}")
                success_count += 1
                continue
        
        print(f"⚙️  Generating embeddings for {experiment}...")
        
        # Create Python 3.9 subprocess script
        script_content = build_subprocess_command(
            data_root=data_root,
            experiment=experiment,
            model_name=model_name,
            model_class=model_class
        )
        
        try:
            # Run subprocess
            result = subprocess.run(
                [str(py39_python), "-c", script_content],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes
            )
            
            if handle_subprocess_output(result, experiment, verbose):
                success_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout generating embeddings for {experiment}")
        except Exception as e:
            print(f"❌ Subprocess error for {experiment}: {e}")
    
    # Report summary
    print()
    print(f"=== Summary ===")
    print(f"Processed: {len(experiments)} experiments")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(experiments) - success_count}")
    
    return success_count == len(experiments)
