"""
SAM2 Mask Propagation

Temporal mask propagation using SAM2's video predictor for embryo tracking.
Supports both forward-only and bidirectional propagation modes.

Key Functions:
    - load_sam2_model: Initialize SAM2 video predictor
    - propagate_forward: Propagate masks from seed frame to end
    - propagate_bidirectional: Propagate in both temporal directions
    - decode_sam2_masks: Convert SAM2 outputs to structured format

Example Usage:
    ```python
    from pathlib import Path
    import numpy as np

    # Load model
    predictor = load_sam2_model(
        config_path="sam2_hiera_l.yaml",
        checkpoint_path="sam2_hiera_large.pt",
        device="cuda"
    )

    # Prepare seed boxes (from GroundingDINO)
    seed_boxes = np.array([
        [100, 100, 300, 300],  # embryo 1
        [400, 400, 600, 600],  # embryo 2
    ])

    # Propagate forward
    results = propagate_forward(
        predictor=predictor,
        frame_dir=Path("/tmp/frames"),
        seed_boxes=seed_boxes,
        seed_frame_idx=5
    )
    ```

Output Format:
    Results dict mapping frame index to frame results:
    {
        5: {  # frame index
            "embryo_0": {
                "mask": np.ndarray,  # binary mask
                "bbox": [x, y, x, y],
                "area": float,
                "confidence": float
            },
            "embryo_1": {...}
        },
        6: {...}
    }
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch


def load_sam2_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> "torch.nn.Module":
    """
    Load SAM2 video predictor model.

    Args:
        config_path: Path to SAM2 config YAML
        checkpoint_path: Path to SAM2 checkpoint
        device: Device to load model on

    Returns:
        SAM2 video predictor instance

    Raises:
        ImportError: If SAM2 is not installed
        FileNotFoundError: If config or checkpoint not found

    Example:
        >>> predictor = load_sam2_model(
        ...     "configs/sam2_hiera_l.yaml",
        ...     "checkpoints/sam2_hiera_large.pt"
        ... )
    """
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        raise ImportError(
            "SAM2 not installed. "
            "Install from: https://github.com/facebookresearch/segment-anything-2"
        )

    predictor = build_sam2_video_predictor(
        str(config_path),
        str(checkpoint_path),
        device=device
    )

    return predictor


def propagate_forward(
    predictor: "torch.nn.Module",
    frame_dir: Path,
    seed_boxes: np.ndarray,
    seed_frame_idx: int = 0,
    embryo_ids: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[int, Dict[str, Any]]:
    """
    Propagate masks forward from seed frame to end of video.

    Args:
        predictor: SAM2 video predictor
        frame_dir: Directory with sequentially numbered frames
        seed_boxes: Numpy array of shape (N, 4) with seed bounding boxes
        seed_frame_idx: Original index of seed frame (for result mapping)
        embryo_ids: Optional list of embryo IDs (default: embryo_0, embryo_1, ...)
        verbose: Enable verbose output

    Returns:
        Dict mapping original frame indices to per-embryo results

    Example:
        >>> seed_boxes = np.array([[100, 100, 300, 300]])
        >>> results = propagate_forward(
        ...     predictor,
        ...     Path("/tmp/frames"),
        ...     seed_boxes,
        ...     seed_frame_idx=5
        ... )
        >>> results[5]["embryo_0"]["mask"].shape
        (512, 512)
    """
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    # Initialize video state
    inference_state = predictor.init_state(video_path=str(frame_dir))

    # Generate embryo IDs if not provided
    if embryo_ids is None:
        embryo_ids = [f"embryo_{i}" for i in range(len(seed_boxes))]

    # Add seed boxes to first frame (index 0 in SAM2's numbering)
    for embryo_idx, (box, embryo_id) in enumerate(zip(seed_boxes, embryo_ids)):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,  # SAM2 always starts at 0
            obj_id=embryo_idx,
            box=box
        )

    # Propagate through video
    results = {}

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # SAM2 frame_idx starts at 0, convert to original indices
        original_frame_idx = seed_frame_idx + frame_idx

        # Decode masks for this frame
        frame_results = decode_sam2_masks(
            obj_ids=obj_ids,
            mask_logits=mask_logits,
            embryo_ids=embryo_ids
        )

        results[original_frame_idx] = frame_results

        if verbose and frame_idx % 10 == 0:
            print(f"Processed frame {original_frame_idx} ({len(frame_results)} embryos)")

    return results


def propagate_bidirectional(
    predictor: "torch.nn.Module",
    forward_dir: Path,
    backward_dir: Optional[Path],
    seed_boxes: np.ndarray,
    seed_frame_idx: int,
    embryo_ids: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[int, Dict[str, Any]]:
    """
    Propagate masks in both temporal directions from seed frame.

    Args:
        predictor: SAM2 video predictor
        forward_dir: Directory with frames from seed to end
        backward_dir: Directory with reversed frames from 0 to seed (or None)
        seed_boxes: Numpy array of seed bounding boxes
        seed_frame_idx: Original index of seed frame
        embryo_ids: Optional list of embryo IDs
        verbose: Enable verbose output

    Returns:
        Dict mapping frame indices to results (merged forward + backward)

    Example:
        >>> results = propagate_bidirectional(
        ...     predictor,
        ...     Path("/tmp/fwd"),
        ...     Path("/tmp/bwd"),
        ...     seed_boxes,
        ...     seed_frame_idx=10
        ... )
        >>> list(results.keys())  # Contains frames 0-20 if 20 total frames
        [0, 1, 2, ..., 19, 20]
    """
    # Forward propagation (seed to end)
    forward_results = propagate_forward(
        predictor=predictor,
        frame_dir=forward_dir,
        seed_boxes=seed_boxes,
        seed_frame_idx=seed_frame_idx,
        embryo_ids=embryo_ids,
        verbose=verbose
    )

    # If no backward directory, return forward only
    if backward_dir is None or seed_frame_idx == 0:
        return forward_results

    # Backward propagation (seed to start, reversed)
    backward_raw = propagate_forward(
        predictor=predictor,
        frame_dir=backward_dir,
        seed_boxes=seed_boxes,
        seed_frame_idx=0,  # Temporary starting point
        embryo_ids=embryo_ids,
        verbose=verbose
    )

    # Remap backward indices (seed_idx - offset)
    backward_results = {}
    for sam2_idx, frame_data in backward_raw.items():
        original_idx = seed_frame_idx - sam2_idx
        backward_results[original_idx] = frame_data

    # Merge: prefer forward results for overlapping frames
    merged = {**backward_results, **forward_results}

    return merged


def decode_sam2_masks(
    obj_ids: torch.Tensor,
    mask_logits: torch.Tensor,
    embryo_ids: List[str]
) -> Dict[str, Any]:
    """
    Decode SAM2 mask outputs into structured format.

    Args:
        obj_ids: Tensor of object IDs for this frame
        mask_logits: Tensor of mask logits (C, H, W)
        embryo_ids: List of embryo ID strings

    Returns:
        Dict mapping embryo_id to mask data:
        {
            "embryo_0": {
                "mask": binary mask array,
                "bbox": [x, y, x, y],
                "area": pixel count,
                "confidence": mean logit value
            }
        }

    Example:
        >>> obj_ids = torch.tensor([0, 1])
        >>> mask_logits = torch.randn(2, 512, 512)
        >>> embryo_ids = ["embryo_0", "embryo_1"]
        >>> results = decode_sam2_masks(obj_ids, mask_logits, embryo_ids)
        >>> "embryo_0" in results
        True
    """
    results = {}

    # Convert to numpy for processing
    obj_ids_np = obj_ids.cpu().numpy()
    mask_logits_np = mask_logits.cpu().numpy()

    for obj_idx, obj_id in enumerate(obj_ids_np):
        # Get embryo ID
        embryo_id = embryo_ids[obj_id] if obj_id < len(embryo_ids) else f"embryo_{obj_id}"

        # Get mask logits for this object
        mask_logit = mask_logits_np[obj_idx]

        # Threshold to binary mask (> 0)
        binary_mask = (mask_logit > 0).astype(np.uint8)

        # Calculate bounding box
        bbox = calculate_bbox_from_mask(binary_mask)

        # Calculate area
        area = float(np.sum(binary_mask))

        # Confidence (mean of positive logits)
        positive_logits = mask_logit[binary_mask > 0]
        confidence = float(np.mean(positive_logits)) if len(positive_logits) > 0 else 0.0

        results[embryo_id] = {
            "mask": binary_mask,
            "bbox": bbox,
            "area": area,
            "confidence": confidence
        }

    return results


def calculate_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """
    Calculate bounding box from binary mask.

    Args:
        mask: Binary mask array (H, W)

    Returns:
        Bounding box as [x_min, y_min, x_max, y_max]

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 30:50] = 1
        >>> bbox = calculate_bbox_from_mask(mask)
        >>> bbox
        [30, 20, 50, 40]
    """
    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty mask
        return [0, 0, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode binary mask to RLE format (pycocotools compatible).

    Args:
        mask: Binary mask array (H, W)

    Returns:
        RLE dict with 'counts' and 'size' keys

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 30:50] = 1
        >>> rle = encode_mask_to_rle(mask)
        >>> rle.keys()
        dict_keys(['counts', 'size'])
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools required for RLE encoding. Install with: pip install pycocotools")

    # Ensure correct format
    mask = np.asfortranarray(mask.astype(np.uint8))

    # Encode
    rle = mask_utils.encode(mask)

    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')

    return rle


def save_propagation_results(
    results: Dict[int, Dict[str, Any]],
    output_path: Path,
    format: str = "json"
) -> None:
    """
    Save propagation results to disk.

    Args:
        results: Propagation results dict
        output_path: Path to save results
        format: Output format ("json" or "npz")

    Example:
        >>> save_propagation_results(
        ...     results,
        ...     Path("results.json"),
        ...     format="json"
        ... )
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Convert masks to RLE for JSON serialization
        serializable = {}
        for frame_idx, frame_data in results.items():
            serializable[frame_idx] = {}
            for embryo_id, embryo_data in frame_data.items():
                serializable[frame_idx][embryo_id] = {
                    "bbox": embryo_data["bbox"],
                    "area": embryo_data["area"],
                    "confidence": embryo_data["confidence"],
                    "mask_rle": encode_mask_to_rle(embryo_data["mask"])
                }

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    elif format == "npz":
        # Save as numpy archive
        np.savez_compressed(output_path, **{
            f"frame_{frame_idx}": {
                embryo_id: embryo_data["mask"]
                for embryo_id, embryo_data in frame_data.items()
            }
            for frame_idx, frame_data in results.items()
        })

    else:
        raise ValueError(f"Unsupported format: {format}")
