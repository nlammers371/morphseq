"""
Centralized embedding generation module.

This module provides a clean, organized API for generating embeddings 
using Python 3.9 subprocess orchestration for legacy model compatibility.
"""

from .pipeline_integration import (
    ensure_embeddings_for_experiments,
    generate_embeddings_for_build06
)
from .cli import main as cli_main
from .file_utils import check_existing_embeddings

__all__ = [
    'ensure_embeddings_for_experiments',
    'generate_embeddings_for_build06', 
    'cli_main',
    'check_existing_embeddings'
]