"""
Python 3.9 subprocess orchestration for embedding generation.

Handles the execution of embedding generation in a Python 3.9 subprocess
to maintain compatibility with legacy models that require specific Python versions.

Uses the standalone generate_embeddings_py39.py script which imports from
src.legacy.vae — matching the BaseDecoder hierarchy the pickled models expect.

REDUNDANCY NOTE (2026-03-23):
  This module is called by Step 6 of the e2e pipeline via:
    exp.generate_latents()
      → pipeline_integration.ensure_embeddings_for_experiments()
        → run_embedding_generation_subprocess()  [THIS FILE]

  Step 7 (run_build06) ALSO generates missing latents via a separate path:
    run_build06()
      → services.gen_embeddings.ensure_latents_for_experiments()
        → calculate_morph_embeddings()  [analysis_utils.py]
          → (detects Python != 3.9) → generate_embeddings_py39.py

  Both paths ultimately call generate_embeddings_py39.py for Py3.9 models.
  If Step 6 fails, Step 7 silently regenerates the same latents.

  TODO: Consolidate Steps 6 and 7 into a single embedding generation path.
  The recommended approach is to remove Step 6 entirely and let run_build06
  handle both latent generation and df03 merging.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_SCRIPT = REPO_ROOT / "src" / "run_morphseq_pipeline" / "services" / "generate_embeddings_py39.py"


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
        # Show concise error summary — full traceback only in verbose mode
        stderr = result.stderr or ""
        if "BadInheritanceError" in stderr:
            print(f"   Root cause: BadInheritanceError — pickle BaseDecoder mismatch")
            print(f"   The Py3.9 subprocess loaded the wrong AutoModel (src.vae instead of src.legacy.vae)")
            print(f"   This is a known bug in the inline -c script path. Step 7 will retry correctly.")
        elif verbose:
            print("Error output:", stderr)
        else:
            # Show just the last meaningful line
            lines = [l for l in stderr.strip().splitlines() if l.strip()]
            if lines:
                print(f"   Error: {lines[-1].strip()}")
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

        experiments_json = json.dumps([experiment])
        batch_size = 64

        try:
            result = subprocess.run(
                [
                    str(py39_python),
                    str(EMBEDDING_SCRIPT),
                    data_root,
                    model_name,
                    model_class,
                    experiments_json,
                    str(batch_size),
                ],
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes
                cwd=str(REPO_ROOT),
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
