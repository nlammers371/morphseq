#!/usr/bin/env python3
"""
CLI entry point for embedding generation.

Provides standalone CLI interface for generating embeddings using 
Python 3.9 subprocess orchestration for legacy model compatibility.
"""

import sys
import argparse
from pathlib import Path
from typing import List

from .subprocess_runner import run_embedding_generation_subprocess
from .file_utils import validate_data_root, validate_python39_environment


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings using Python 3.9 subprocess",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--data-root", required=True, 
                       help="Data root directory")
    parser.add_argument("--experiments", nargs="+", required=True, 
                       help="List of experiments to process")
    
    # Model configuration
    parser.add_argument("--model-name", default="20241107_ds_sweep01_optimum",
                       help="Model name")
    parser.add_argument("--model-class", default="legacy", 
                       help="Model class")
    
    # Environment configuration
    parser.add_argument("--py39-env", 
                       default="/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
                       help="Python 3.9 environment path")
    
    # Processing options
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite embeddings for specified experiments")
    parser.add_argument("--process-missing", action="store_true",
                       help="Only process missing embeddings (skip existing)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    return parser.parse_args()


def validate_environment(args) -> bool:
    """Validate environment and arguments."""
    # Check data root
    if not validate_data_root(args.data_root):
        return False
    
    # Check Python 3.9 environment
    if not validate_python39_environment(args.py39_env):
        return False
    
    return True


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    print("=== CLI Embedding Generation ===")
    print(f"Data root: {args.data_root}")
    print(f"Model: {args.model_name}")
    print(f"Experiments: {args.experiments}")
    print(f"Python 3.9: {args.py39_env}")
    print()
    
    # Validate environment
    if not validate_environment(args):
        return 1
    
    # Check overwrite logic
    if args.overwrite:
        print(f"🔄 Will overwrite specified experiments: {args.experiments}")
    
    # Run embedding generation
    success = run_embedding_generation_subprocess(
        data_root=args.data_root,
        experiments=args.experiments,
        model_name=args.model_name,
        model_class=args.model_class,
        py39_env_path=args.py39_env,
        overwrite=args.overwrite,
        process_missing=args.process_missing,
        verbose=args.verbose
    )
    
    if success:
        print("🎉 All embeddings generated successfully!")
        return 0
    else:
        print("⚠️  Some embeddings failed to generate")
        return 1


if __name__ == "__main__":
    sys.exit(main())