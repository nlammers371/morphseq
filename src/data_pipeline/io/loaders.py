"""File loading utilities for the MorphSeq pipeline.

This module provides functions to load various file types used in the pipeline:
- Images (TIFF, ND2, JPEG)
- Masks (PNG, RLE-encoded)
- CSVs (metadata, features, QC flags)
- JSON (manifests, configs)
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image file.

    Args:
        image_path: Path to image file (TIFF, ND2, JPEG, PNG)

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    # TODO: Implement image loading (use tifffile, nd2reader, PIL as needed)
    raise NotImplementedError("load_image() to be implemented in Wave 2+")


def load_csv(csv_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with CSV contents

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path, **kwargs)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with JSON contents

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    # TODO: Implement JSON loading
    raise NotImplementedError("load_json() to be implemented in Wave 2+")


def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load a mask file (PNG or RLE-encoded).

    Args:
        mask_path: Path to mask file

    Returns:
        Mask as numpy array

    Raises:
        FileNotFoundError: If mask file doesn't exist
    """
    # TODO: Implement mask loading
    raise NotImplementedError("load_mask() to be implemented in Wave 2+")
