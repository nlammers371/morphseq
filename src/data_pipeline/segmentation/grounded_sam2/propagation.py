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

from data_pipeline.models.sam2 import load_sam2_video_predictor


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

    # Support loading from a repo checkout under the pipeline models root.
    # We infer the models root from the config path's typical location under `sam2/configs/...`.
    sam2_models_root = config_path.parent.parent.parent
    return load_sam2_video_predictor(
        sam2_models_root=sam2_models_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )


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
    frame_dir: Path,
    seed_boxes: np.ndarray,
    seed_frame_idx: int,
    embryo_ids: Optional[List[str]] = None,
    verbose: bool = False,
    max_frame_num_to_track: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Propagate in both directions from one seed frame using one SAM2 inference state.

    This uses SAM2's native API:
    `propagate_in_video(inference_state, start_frame_idx=..., reverse=...)`
    """
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if seed_frame_idx < 0:
        raise ValueError(f"seed_frame_idx must be >= 0, got {seed_frame_idx}")

    inference_state = predictor.init_state(video_path=str(frame_dir))

    if embryo_ids is None:
        embryo_ids = [f"embryo_{i}" for i in range(len(seed_boxes))]

    # Seed once on the true seed frame in the full frame sequence.
    for embryo_idx, box in enumerate(seed_boxes):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=embryo_idx,
            box=box,
        )

    forward_results: Dict[int, Dict[str, Any]] = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=seed_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=False,
    ):
        frame_results = decode_sam2_masks(
            obj_ids=obj_ids,
            mask_logits=mask_logits,
            embryo_ids=embryo_ids,
        )
        forward_results[frame_idx] = frame_results
        if verbose and frame_idx % 10 == 0:
            print(f"[fwd] Processed frame {frame_idx} ({len(frame_results)} embryos)")

    if seed_frame_idx == 0:
        return forward_results

    backward_results: Dict[int, Dict[str, Any]] = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=seed_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=True,
    ):
        frame_results = decode_sam2_masks(
            obj_ids=obj_ids,
            mask_logits=mask_logits,
            embryo_ids=embryo_ids,
        )
        backward_results[frame_idx] = frame_results
        if verbose and frame_idx % 10 == 0:
            print(f"[bwd] Processed frame {frame_idx} ({len(frame_results)} embryos)")

    # Prefer forward when both passes produce the same frame index.
    return {**backward_results, **forward_results}


def decode_sam2_masks(
    obj_ids: Any,
    mask_logits: Any,
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

    # Convert to numpy for processing (SAM2 may return lists or tensors depending on version).
    if hasattr(obj_ids, "cpu"):
        obj_ids_np = obj_ids.cpu().numpy()
    else:
        obj_ids_np = np.asarray(obj_ids)

    if hasattr(mask_logits, "cpu"):
        mask_logits_np = mask_logits.cpu().numpy()
    else:
        mask_logits_np = np.asarray(mask_logits)

    for obj_idx, obj_id in enumerate(obj_ids_np):
        # Get embryo ID
        embryo_id = embryo_ids[obj_id] if obj_id < len(embryo_ids) else f"embryo_{obj_id}"

        # Get mask logits for this object.
        # SAM2 versions vary: (N, H, W) or (N, 1, H, W). Normalize to (H, W).
        mask_logit = np.asarray(mask_logits_np[obj_idx])
        if mask_logit.ndim == 3 and mask_logit.shape[0] == 1:
            mask_logit = mask_logit[0]
        elif mask_logit.ndim == 3 and mask_logit.shape[-1] == 1:
            mask_logit = mask_logit[..., 0]
        else:
            mask_logit = np.squeeze(mask_logit)
        if mask_logit.ndim != 2:
            raise ValueError(f"Unexpected SAM2 mask_logit shape for obj={obj_id}: {mask_logit.shape}")

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

    mask_arr = np.asarray(mask)
    # SAM2 may hand us (1, H, W) or (H, W, 1). Canonicalize to (H, W).
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    elif mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    else:
        mask_arr = np.squeeze(mask_arr)
    if mask_arr.ndim != 2:
        raise ValueError(f"Mask must be 2D for RLE encoding, got shape={mask_arr.shape}")

    # Ensure binary uint8 + Fortran order for pycocotools.
    mask_arr = (mask_arr > 0).astype(np.uint8)
    mask_arr = np.asfortranarray(mask_arr)

    rle = mask_utils.encode(mask_arr)
    # pycocotools returns a list when given an HxWxN array; be defensive anyway.
    if isinstance(rle, list):
        if len(rle) != 1:
            raise ValueError(f"Expected single RLE dict, got list of len={len(rle)}")
        rle = rle[0]

    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts_str = counts.decode("utf-8")
    elif isinstance(counts, str):
        counts_str = counts
    else:
        raise TypeError(f"Unexpected RLE counts type: {type(counts)!r}")

    size = rle.get("size")
    if size is None or len(size) != 2:
        raise ValueError(f"Unexpected RLE size: {size!r}")

    return {
        "counts": counts_str,
        "size": [int(size[0]), int(size[1])],
    }


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
