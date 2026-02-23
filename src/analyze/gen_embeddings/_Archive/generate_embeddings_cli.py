#!/usr/bin/env python3
"""
CLI-compatible embedding generation script.
Simple, human-readable wrapper that calls calculate_morph_embeddings via Python 3.9 subprocess.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings using Python 3.9 subprocess")
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--model-name", default="20241107_ds_sweep01_optimum", help="Model name")
    parser.add_argument("--model-class", default="legacy", help="Model class")
    parser.add_argument("--experiments", nargs="+", required=True, help="List of experiments")
    parser.add_argument("--py39-env", default="/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster", 
                       help="Python 3.9 environment path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite embeddings for specified experiments")
    parser.add_argument("--process-missing", action="store_true", help="Only process missing embeddings (skip existing)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    py39_python = Path(args.py39_env) / "bin" / "python"
    
    print("=== CLI Embedding Generation ===")
    print(f"Data root: {data_root}")
    print(f"Model: {args.model_name}")
    print(f"Experiments: {args.experiments}")
    print(f"Python 3.9: {py39_python}")
    print()
    
    # Check if Python 3.9 environment exists
    if not py39_python.exists():
        print(f"❌ Error: Python 3.9 environment not found: {py39_python}")
        return 1
    
    # Check overwrite logic
    if args.overwrite:
        print(f"🔄 Will overwrite specified experiments: {args.experiments}")
    
    # Generate embeddings for each experiment
    success_count = 0
    
    for experiment in args.experiments:
        print(f"=== Processing {experiment} ===")
        
        # Check if embeddings already exist
        latents_dir = data_root / "analysis" / "latent_embeddings" / args.model_class / args.model_name
        latents_file = latents_dir / f"morph_latents_{experiment}.csv"
        
        if latents_file.exists():
            if args.overwrite:
                print(f"🔄 Overwriting existing: {latents_file}")
            elif args.process_missing:
                print(f"⏭️  Skipping existing: {latents_file}")
                success_count += 1
                continue
            else:
                print(f"✅ Embeddings already exist: {latents_file}")
                success_count += 1
                continue
        
        print(f"⚙️  Generating embeddings for {experiment}...")
        
        # Create Python 3.9 subprocess script
        script_content = f'''
import sys
from pathlib import Path

# Add repo to path
repo_root = Path("{Path(__file__).parent}")
sys.path.insert(0, str(repo_root))

print("Python version:", sys.version_info)

from analyze.analysis_utils import calculate_morph_embeddings

print("Calling calculate_morph_embeddings...")

try:
    result = calculate_morph_embeddings(
        data_root="{data_root}",
        model_name="{args.model_name}",
        model_class="{args.model_class}",
        experiments=["{experiment}"]
    )
    print("✅ Success:", result)
except Exception as e:
    print("❌ Error:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            # Run subprocess
            result = subprocess.run(
                [str(py39_python), "-c", script_content],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes
            )
            
            if args.verbose:
                print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print(f"✅ Successfully generated embeddings for {experiment}")
                success_count += 1
            else:
                print(f"❌ Failed to generate embeddings for {experiment}")
                print("Error output:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout generating embeddings for {experiment}")
        except Exception as e:
            print(f"❌ Subprocess error for {experiment}: {e}")
    
    print()
    print(f"=== Summary ===")
    print(f"Processed: {len(args.experiments)} experiments")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(args.experiments) - success_count}")
    
    if success_count == len(args.experiments):
        print("🎉 All embeddings generated successfully!")
        return 0
    else:
        print("⚠️  Some embeddings failed to generate")
        return 1

if __name__ == "__main__":
    sys.exit(main())