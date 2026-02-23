"""
Utilities for loading legacy models with Python version compatibility handling.

Updated policy: Build06 no longer switches environments automatically. It must
be run from a Python 3.9 environment (`mseq_pipeline_py3.9`). These utilities
now validate the interpreter version and provide actionable errors instead of
attempting conda env switching.
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional
import torch


class ScriptedLegacyModelAdapter:
    """Lightweight adapter that restores metadata lost during TorchScript export."""

    def __init__(self, scripted_model: torch.jit.ScriptModule, metadata: Optional[dict] = None):
        self._scripted = scripted_model
        self._metadata = metadata or {}

        self.model_name = self._metadata.get('model_name') or getattr(scripted_model, 'model_name', None)

        latent_dim = self._metadata.get('latent_dim')
        if latent_dim is None and hasattr(scripted_model, 'latent_dim'):
            latent_dim = getattr(scripted_model, 'latent_dim')
        try:
            self.latent_dim = int(latent_dim) if latent_dim is not None else None
        except (TypeError, ValueError):
            self.latent_dim = None

        nuisance = self._metadata.get('nuisance_indices')
        if nuisance is None and hasattr(scripted_model, 'nuisance_indices'):
            nuisance = getattr(scripted_model, 'nuisance_indices')
            if hasattr(nuisance, 'tolist'):
                nuisance = nuisance.tolist()

        try:
            self.nuisance_indices = {int(idx) for idx in nuisance} if nuisance is not None else set()
        except (TypeError, ValueError):
            self.nuisance_indices = set()

    def to(self, device: str):
        self._scripted.to(device)
        return self

    def eval(self):
        self._scripted.eval()
        return self

    def __getattr__(self, item):
        return getattr(self._scripted, item)

def get_current_conda_env() -> Optional[str]:
    """
    Detect the currently active conda environment.
    
    Returns:
        Name of current conda environment, or None if not in conda
    """
    # Method 1: Check CONDA_PREFIX (most reliable for active env)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        return Path(conda_prefix).name
    
    # Method 2: Check CONDA_DEFAULT_ENV (fallback)
    conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_default_env:
        return conda_default_env
        
    # Method 3: Parse from conda info (most comprehensive but slower)
    try:
        result = subprocess.run(
            ['conda', 'info', '--envs'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '*' in line:  # Active environment marked with *
                    parts = line.split()
                    if len(parts) >= 1:
                        return parts[0].replace('*', '').strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None

def check_conda_env_exists(env_name: str) -> bool:
    """
    Check if a conda environment exists.
    
    Args:
        env_name: Name of conda environment to check
        
    Returns:
        True if environment exists, False otherwise
    """
    try:
        # Try multiple conda commands to find environments
        commands_to_try = [
            ['conda', 'env', 'list'],
            ['conda', 'info', '--envs'],
            ['conda', 'list', '--envs']
        ]
        
        for cmd in commands_to_try:
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=10,
                    env=os.environ.copy()  # Preserve environment variables
                )
                if result.returncode == 0:
                    # Check if env_name appears as a word boundary in the output
                    import re
                    pattern = r'\b' + re.escape(env_name) + r'\b'
                    if re.search(pattern, result.stdout):
                        return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue
                
    except Exception:
        pass
    
    return False

def load_legacy_model_safe(
    model_path: str,
    device: str = "cpu",
    target_python_env: str = "mseq_pipeline_py3.9",
    logger: Optional[logging.Logger] = None,
    enable_env_switch: bool = True,
) -> torch.nn.Module:
    """
    Safely load a legacy model, handling Python version compatibility issues.

    Policy: Automatic environment switching by default. If Python != 3.9,
    automatically route to subprocess using `mseq_pipeline_py3.9` environment.
    Set `enable_env_switch=False` to disable and get error instead.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        target_python_env: Conda environment with Python 3.9
        logger: Optional logger for output
        
    Returns:
        Loaded model ready for inference
        
    Raises:
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model loading fails
        EnvironmentError: If Python != 3.9 or environment switching fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    current_python = f"{sys.version_info[0]}.{sys.version_info[1]}"

    # If already Python 3.9, load directly
    if current_python == "3.9":
        logger.info("Already running Python 3.9, loading model directly")
        from src.legacy.vae import AutoModel
        lit_model = AutoModel.load_from_folder(str(model_path))
        lit_model.to(device).eval()
        return lit_model
    
    if not enable_env_switch:
        raise EnvironmentError(
            "Legacy model requires Python 3.9 for pickle deserialization. "
            f"Detected Python {current_python}. Please activate the 'mseq_pipeline_py3.9' "
            "conda environment and re-run Build06."
        )

    # Optional: switch environments (escape hatch)
    logger.info(
        f"Current Python {current_python} != 3.9. enable_env_switch=True so attempting "
        f"to run in conda env '{target_python_env}'."
    )

    if not check_conda_env_exists(target_python_env):
        raise EnvironmentError(f"Conda environment '{target_python_env}' not found")

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        temp_model_path = tmp_file.name

    try:
        subprocess_script = Path(__file__).parent.parent.parent.parent / "load_model_subprocess.py"

        cmd = [
            'conda', 'run', '-n', target_python_env, 'python', str(subprocess_script),
            '--model-path', str(model_path),
            '--output-path', temp_model_path,
            '--device', device
        ]

        logger.info(f"Running subprocess: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            error_msg = f"Model loading subprocess failed (exit code {result.returncode})"
            if result.stderr:
                error_msg += f":\n{result.stderr}"
            raise RuntimeError(error_msg)
        
        logger.info("Subprocess completed successfully")
        if result.stdout:
            logger.info(f"Subprocess output: {result.stdout.strip()}")
        
        # Load the saved model state
        logger.info(f"Loading saved model state from {temp_model_path}")
        model_data = torch.load(temp_model_path, map_location=device)
        
        # Handle different model save formats
        if model_data.get('model_type') == 'scripted':
            # Use scripted model directly - restore metadata via adapter
            logger.info("Loading scripted model (cross-version compatible)")
            metadata = model_data.get('metadata', {})
            scripted_model = model_data['scripted_model']
            lit_model = ScriptedLegacyModelAdapter(scripted_model, metadata)
            lit_model.to(device).eval()
        else:
            # Fallback to state dict approach - this still has the reconstruction problem
            logger.info("Loading from state dict (requires model reconstruction)")
            
            # Try to load model directly first (will likely fail)
            try:
                from src.legacy.vae import AutoModel
                lit_model = AutoModel.load_from_folder(str(model_path))
                lit_model.load_state_dict(model_data['model_state_dict'])
                lit_model.to(device).eval()
                logger.info("Successfully loaded model with direct approach")
            except Exception as e:
                logger.error(f"Failed to reconstruct model in current process: {e}")
                raise RuntimeError(
                    f"Could not load legacy model due to architecture reconstruction failure. "
                    f"Original error: {e}. "
                    f"This typically happens when the model uses custom encoder/decoder "
                    f"that cannot be pickled across Python versions. "
                    f"Consider using Python 3.9 directly or contact support."
                )
        
        logger.info("✅ Legacy model loaded successfully with environment switching")
        return lit_model
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Model loading subprocess timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Failed to load legacy model: {e}")
        raise
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_model_path)
        except OSError:
            pass
