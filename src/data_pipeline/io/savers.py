"""File saving utilities for the MorphSeq pipeline.

This module provides functions to save various file types produced by the pipeline:
- Images (TIFF, JPEG, PNG)
- Masks (PNG, RLE-encoded)
- CSVs (metadata, features, QC flags)
- JSON (manifests, configs)
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np


def save_image(image: np.ndarray, output_path: Union[str, Path], **kwargs):
    """
    Save an image to file.

    Args:
        image: Image as numpy array
        output_path: Path to save image (extension determines format)
        **kwargs: Additional arguments for image saving (e.g., compression)

    Raises:
        ValueError: If image format is not supported
    """
    # TODO: Implement image saving (use tifffile, PIL as needed)
    raise NotImplementedError("save_image() to be implemented in Wave 2+")


def save_csv(df: pd.DataFrame, output_path: Union[str, Path], **kwargs):
    """
    Save a DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path to save CSV file
        **kwargs: Additional arguments passed to df.to_csv()

    Note:
        Creates parent directories if they don't exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, **kwargs)


def save_json(data: Dict[str, Any], output_path: Union[str, Path], indent: int = 2):
    """
    Save a dictionary to JSON file.

    Args:
        data: Dictionary to save
        output_path: Path to save JSON file
        indent: JSON indentation level (default: 2)

    Note:
        Creates parent directories if they don't exist.
    """
    # TODO: Implement JSON saving
    raise NotImplementedError("save_json() to be implemented in Wave 2+")


def save_mask(mask: np.ndarray, output_path: Union[str, Path], **kwargs):
    """
    Save a mask to file (PNG or RLE-encoded).

    Args:
        mask: Mask as numpy array
        output_path: Path to save mask file
        **kwargs: Additional arguments for mask saving

    Note:
        Creates parent directories if they don't exist.
    """
    # TODO: Implement mask saving
    raise NotImplementedError("save_mask() to be implemented in Wave 2+")
