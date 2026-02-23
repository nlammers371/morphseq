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
    
    print(f"Python 3.9 subprocess running...")
    print(f"Python version: {sys.version_info}")
    print(f"Data root: {data_root}")
    print(f"Model: {model_name}")
    print(f"Experiments: {experiments}")
    
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
        
        # Add repo to path
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root))
        
        # Now import the specific modules we need (without the problematic imports)
        from vae.models.auto_model import AutoModel
        from data.dataset_configs import EvalDataConfig
        from data.data_transforms import basic_transform
        
        print("✅ Successfully imported dependencies in Python 3.9")
        
        # Core embedding extraction logic (simplified)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model
        model_dir = data_root / "models" / model_class / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        print(f"Loading model from: {model_dir}")
        lit_model = AutoModel.load_from_folder(str(model_dir))
        lit_model.to(device)
        lit_model.eval()
        
        input_size = (288, 128)
        transform = basic_transform(target_size=input_size)
        
        # Process each experiment
        all_embeddings = []
        
        for exp in experiments:
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
            
            # Create dataloader
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            print(f"Processing {len(dl)} batches...")
            
            exp_embeddings = []
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dl):
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(dl)}")
                    
                    # Handle DatasetOutput from BasicEvalDataset
                    if hasattr(batch_data, 'data'):
                        # It's a DatasetOutput object
                        x = batch_data.data
                        sample_names = [Path(label[0]).stem for label in batch_data.label] if hasattr(batch_data, 'label') else []
                    else:
                        # Fallback to tuple unpacking if it's not DatasetOutput
                        x, sample_names = batch_data
                    
                    x = x.to(device)
                    
                    # Extract embeddings using encoder
                    encoder_output = lit_model.encoder(x)
                    
                    # Handle different encoder output formats
                    if hasattr(encoder_output, 'embedding'):
                        z_mu = encoder_output.embedding
                    elif hasattr(encoder_output, 'mu'):
                        z_mu = encoder_output.mu
                    elif isinstance(encoder_output, (tuple, list)) and len(encoder_output) >= 1:
                        z_mu = encoder_output[0]  # First element is usually mu
                    else:
                        z_mu = encoder_output
                    
                    # Move to CPU and convert to numpy
                    z_mu_np = z_mu.cpu().numpy()
                    
                    # Create dataframe for this batch
                    for i, sample_name in enumerate(sample_names):
                        row = {"snip_id": sample_name, "experiment_date": exp}
                        
                        # Add embedding dimensions
                        for j in range(z_mu_np.shape[1]):
                            row[f"z_mu_{j:02d}"] = z_mu_np[i, j]
                        
                        exp_embeddings.append(row)
            
            print(f"✅ Extracted {len(exp_embeddings)} embeddings for {exp}")
            all_embeddings.extend(exp_embeddings)
        
        # Create final dataframe
        latent_df = pd.DataFrame(all_embeddings)
        print(f"✅ Total embeddings extracted: {len(latent_df)}")
        
        # Save embeddings per experiment
        save_root = data_root / "analysis" / "latent_embeddings" / model_class / model_name
        save_root.mkdir(parents=True, exist_ok=True)
        
        for exp in experiments:
            exp_df = latent_df[latent_df["experiment_date"] == exp]
            output_path = save_root / f"morph_latents_{exp}.csv"
            exp_df.to_csv(output_path, index=False)
            print(f"✅ Saved embeddings for {exp}: {output_path}")
        
        print("✅ Embedding generation completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error in Python 3.9 subprocess: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())