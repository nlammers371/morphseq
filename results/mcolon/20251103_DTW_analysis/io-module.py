# io.py
"""Simple I/O helpers for saving/loading data and plots."""

import pickle
import numpy as np
from pathlib import Path
import json

# Step labels for organizing outputs
STEP_LABELS = {
    0: 'dtw',
    1: 'cluster',
    2: 'select_k',
    3: 'select_k',
    4: 'membership',
    5: 'fit_models',
    6: 'outputs'
}

# ============ CORE FUNCTIONS ============

def save_data(step: int, name: str, obj, output_dir: Path = Path("output")):
    """Save data object to file."""
    label = STEP_LABELS.get(step, f'step{step}')
    step_dir = output_dir / f"{step}_{label}"
    data_dir = step_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_data(step: int, name: str, output_dir: Path = Path("output")):
    """Load data object from file."""
    label = STEP_LABELS.get(step, f'step{step}')
    step_dir = output_dir / f"{step}_{label}"
    filepath = step_dir / "data" / f"{name}.pkl"

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_plot(step: int, name: str, fig, output_dir: Path = Path("output")):
    """Save matplotlib figure."""
    label = STEP_LABELS.get(step, f'step{step}')
    step_dir = output_dir / f"{step}_{label}"
    plot_dir = step_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    filepath = plot_dir / f"{name}.png"
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"Saved plot: {filepath}")


def save_results_json(step: int, name: str, results: dict, output_dir: Path = Path("output")):
    """Save results as JSON for inspection."""
    label = STEP_LABELS.get(step, f'step{step}')
    step_dir = output_dir / f"{step}_{label}"
    data_dir = step_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    filepath = data_dir / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved JSON: {filepath}")