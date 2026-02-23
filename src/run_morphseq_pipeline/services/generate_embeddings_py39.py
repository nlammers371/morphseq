#!/usr/bin/env python3
"""
Standalone Python 3.9 script for generating embeddings using legacy models.
This script is called as a subprocess from the main Build06 pipeline.
"""

import sys
import os
from pathlib import Path
import json

def main():
    # Parse command line arguments
    if len(sys.argv) != 6:
        print("Usage: python generate_embeddings_py39.py <data_root> <model_name> <model_class> <experiments_json> <batch_size>", file=sys.stderr)
        return 1
    
    data_root = Path(sys.argv[1])
    model_name = sys.argv[2] 
    model_class = sys.argv[3]
    experiments = json.loads(sys.argv[4])
    batch_size = int(sys.argv[5])
    
    # Verbosity via env var (default: quiet)
    verbose = os.environ.get("MORPHSEQ_EMBED_VERBOSE", "0") == "1"
    if verbose:
        print(f"Python 3.9 subprocess running...")
        print(f"Python version: {sys.version_info}")
        print(f"Data root: {data_root}")
        print(f"Model: {model_name}")
        print(f"Experiments: {experiments}")
    else:
        # Minimal header
        print(f"Generating embeddings for {', '.join(experiments)} using {model_name}...")
    
    # Validate Python version
    if sys.version_info[:2] != (3, 9):
        print(f"ERROR: Expected Python 3.9, got {sys.version_info[0]}.{sys.version_info[1]}", file=sys.stderr)
        return 1
    
    try:
        # Import the minimal dependencies needed
        import torch
        import numpy as np
        import pandas as pd
        from torch.utils.data import DataLoader
        
        # Add repo to Python path - assumes script is run from morphseq repo root
        # (subprocess caller sets cwd=repo_root)
        if verbose:
            print("🔍 Running from repo root, adding '.' to Python path")
        sys.path.insert(0, ".")
        
        # Now import the specific modules we need (without the problematic imports)
        from src.legacy.vae import AutoModel
        from src.core.data.dataset_configs import EvalDataConfig
        from src.core.data.data_transforms import basic_transform
        from src.analyze.analysis_utils import extract_embeddings_legacy
        
        if verbose:
            print("✅ Successfully imported dependencies in Python 3.9")
        
        # Core embedding extraction logic (simplified)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Using device: {device}")
        
        # Load model
        model_dir = data_root / "models" / model_class / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        if verbose:
            print(f"Loading model from: {model_dir}")
        lit_model = AutoModel.load_from_folder(str(model_dir))
        lit_model.to(device)
        lit_model.eval()
        
        input_size = (288, 128)
        transform = basic_transform(target_size=input_size)
        
        # Process each experiment
        for exp in experiments:
            if verbose:
                print(f"Processing experiment: {exp}")
            
            # Initialize data config
            eval_data_config = EvalDataConfig(
                experiments=[exp],
                root=data_root,
                return_sample_names=True, 
                transforms=transform
            )
            
            # Create dataset from config
            dataset = eval_data_config.create_dataset()
            
            # Create dataloader and call shared extraction helper (produces z_mu/z_sigma with b/n split)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            latent_df = extract_embeddings_legacy(lit_model=lit_model, dataloader=dl, device=device)

            # Save embeddings per experiment
            save_root = data_root / "analysis" / "latent_embeddings" / model_class / model_name
            save_root.mkdir(parents=True, exist_ok=True)

            exp_df = latent_df[latent_df["experiment_date"] == exp]
            output_path = save_root / f"morph_latents_{exp}.csv"
            exp_df.to_csv(output_path, index=False)
            print(f"Saved embeddings: {output_path}")
        
        if verbose:
            print("✅ Embedding generation completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error in Python 3.9 subprocess: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
