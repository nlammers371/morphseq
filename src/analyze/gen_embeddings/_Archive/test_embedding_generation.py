#!/usr/bin/env python3
"""
Simple MVP test script to verify embedding generation works.
No CLI complexity - just test the core functionality.
"""

import sys
import os
from pathlib import Path

def main():
    print("=== MVP Embedding Generation Test ===")
    print()
    
    # Configuration
    data_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    model_name = "20241107_ds_sweep01_optimum"
    model_class = "legacy"
    experiment = "20250529_30hpf_ctrl_atf6"
    
    print(f"Data root: {data_root}")
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment}")
    print()
    
    # ===== SECTION 1: Check if embeddings exist =====
    print("SECTION 1: Checking if embeddings already exist...")
    
    latents_dir = data_root / "analysis" / "latent_embeddings" / model_class / model_name
    latents_file = latents_dir / f"morph_latents_{experiment}.csv"
    
    print(f"Looking for: {latents_file}")
    
    if latents_file.exists():
        print("✅ Embeddings already exist!")
        print(f"File size: {latents_file.stat().st_size} bytes")
        return 0
    else:
        print("❌ Embeddings missing - need to generate")
    
    print()
    
    # ===== SECTION 2: Generate embeddings using subprocess with Python 3.9 =====
    print("SECTION 2: Generating embeddings via Python 3.9 subprocess...")
    
    try:
        import subprocess
        import json
        
        # TODO: Add check if current env is Python 3.9, for now always use subprocess
        py39_env_path = "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster"
        python_executable = f"{py39_env_path}/bin/python"
        
        print(f"Using Python 3.9 environment: {py39_env_path}")
        print(f"Python executable: {python_executable}")
        
        # Prepare command to call calculate_morph_embeddings via subprocess
        script_content = f'''
import sys
from pathlib import Path

# Add repo to path
repo_root = Path("{Path(__file__).parent}")
sys.path.insert(0, str(repo_root))

from analyze.analysis_utils import calculate_morph_embeddings

print("Python version:", sys.version_info)
print("Calling calculate_morph_embeddings from Python 3.9...")

result = calculate_morph_embeddings(
    data_root="{data_root}",
    model_name="{model_name}",
    model_class="{model_class}",
    experiments=["{experiment}"]
)

print("✅ Embedding generation completed in Python 3.9")
print("Result:", result)
'''
        
        print("Running calculate_morph_embeddings in Python 3.9 subprocess...")
        
        # Run the subprocess
        result = subprocess.run(
            [python_executable, "-c", script_content],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Subprocess failed with return code: {result.returncode}")
            return 1
        
        print(f"✅ Subprocess completed successfully")
        
    except Exception as e:
        print(f"❌ Error running subprocess: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # ===== SECTION 3: Verify output was created =====
    print("SECTION 3: Verifying output...")
    
    if latents_file.exists():
        file_size = latents_file.stat().st_size
        print(f"✅ SUCCESS! Embeddings file created: {latents_file}")
        print(f"File size: {file_size} bytes")
        
        # Try to read the first few lines to verify format
        try:
            import pandas as pd
            df = pd.read_csv(latents_file)
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Sample columns: {list(df.columns[:5])}")
            if 'snip_id' in df.columns:
                print(f"Sample snip_ids: {df['snip_id'].head(3).tolist()}")
        except Exception as e:
            print(f"Warning: Could not read CSV: {e}")
            
        return 0
    else:
        print(f"❌ FAILURE! Embeddings file not found: {latents_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())