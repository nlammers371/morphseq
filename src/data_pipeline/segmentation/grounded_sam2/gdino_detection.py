"""
GroundingDINO Embryo Detection

Zero-shot object detection using GroundingDINO to identify embryos in microscopy images.
Provides seed detections for SAM2 mask propagation.

Key Functions:
    - load_groundingdino_model: Initialize model from config
    - detect_embryos: Run detection on a single image
    - filter_detections: Apply confidence and IoU thresholding
    - select_seed_frame: Choose best frame for SAM2 initialization

Example Usage:
    ```python
    import torch
    from pathlib import Path

    # Load model
    model = load_groundingdino_model(
        config_path="GroundingDINO_SwinT_OGC.py",
        weights_path="groundingdino_swint_ogc.pth",
        device="cuda"
    )

    # Detect embryos
    detections = detect_embryos(
        model=model,
        image_path=Path("embryo_image.jpg"),
        text_prompt="individual embryo",
        box_threshold=0.35,
        text_threshold=0.25
    )

    # Filter high-quality detections
    filtered = filter_detections(
        detections,
        confidence_threshold=0.45,
        iou_threshold=0.5
    )
    ```

Detection Format:
    Each detection is a dict with:
    - box_xyxy: [x_min, y_min, x_max, y_max] in normalized coords [0, 1]
    - confidence: float confidence score
    - phrase: str matched text phrase
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")


def load_groundingdino_model(
    config_path: str,
    weights_path: str,
    device: str = "cuda"
) -> "torch.nn.Module":
    """
    Load GroundingDINO model.

    Args:
        config_path: Path to model config file (e.g., "GroundingDINO_SwinT_OGC.py")
        weights_path: Path to model weights (e.g., "groundingdino_swint_ogc.pth")
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Loaded GroundingDINO model

    Raises:
        ImportError: If GroundingDINO is not installed
        FileNotFoundError: If config or weights files don't exist

    Example:
        >>> model = load_groundingdino_model(
        ...     "models/GroundingDINO_SwinT_OGC.py",
        ...     "weights/groundingdino_swint_ogc.pth"
        ... )
    """
    config_path = Path(config_path)
    weights_path = Path(weights_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "GroundingDINO not installed. "
            "Install from: https://github.com/IDEA-Research/GroundingDINO"
        )

    model = load_model(str(config_path), str(weights_path), device=device)
    return model


def detect_embryos(
    model: "torch.nn.Module",
    image_path: Path,
    text_prompt: str = "individual embryo",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25
) -> List[Dict]:
    """
    Detect embryos in an image using GroundingDINO.

    Args:
        model: Loaded GroundingDINO model
        image_path: Path to input image
        text_prompt: Text prompt for detection
        box_threshold: Box confidence threshold
        text_threshold: Text matching threshold

    Returns:
        List of detection dicts, each containing:
        - box_xyxy: [x_min, y_min, x_max, y_max] normalized [0, 1]
        - confidence: detection confidence score
        - phrase: matched text phrase

    Raises:
        FileNotFoundError: If image file doesn't exist

    Example:
        >>> detections = detect_embryos(
        ...     model,
        ...     Path("image.jpg"),
        ...     text_prompt="individual embryo"
        ... )
        >>> len(detections)
        3
        >>> detections[0]["confidence"]
        0.87
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Import required GroundingDINO utilities
    try:
        from groundingdino.util.inference import predict
        from groundingdino.util.utils import get_phrases_from_posmap
        import groundingdino.datasets.transforms as T
    except ImportError:
        raise ImportError("GroundingDINO utilities not available")

    # Load and transform image
    image_source = cv2.imread(str(image_path))
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image_source, None)

    # Run detection
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # Convert to standard format
    detections = []
    for box, confidence, phrase in zip(boxes, logits, phrases):
        detections.append({
            "box_xyxy": box.cpu().numpy().tolist(),
            "confidence": float(confidence.cpu().numpy()),
            "phrase": phrase
        })

    return detections


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [x_min, y_min, x_max, y_max] in normalized coords
        box2: [x_min, y_min, x_max, y_max] in normalized coords

    Returns:
        IoU value between 0 and 1

    Example:
        >>> box1 = [0.2, 0.2, 0.6, 0.6]
        >>> box2 = [0.4, 0.4, 0.8, 0.8]
        >>> iou = calculate_iou(box1, box2)
        >>> 0.0 < iou < 1.0
        True
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def filter_detections(
    detections: List[Dict],
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter detections by confidence and remove duplicates via NMS.

    Applies:
    1. Confidence thresholding
    2. Non-Maximum Suppression (NMS) based on IoU

    Args:
        detections: List of detection dicts
        confidence_threshold: Minimum confidence to keep
        iou_threshold: IoU threshold for duplicate removal

    Returns:
        Filtered list of detections

    Example:
        >>> detections = [
        ...     {"box_xyxy": [0.2, 0.2, 0.4, 0.4], "confidence": 0.9, "phrase": "embryo"},
        ...     {"box_xyxy": [0.21, 0.21, 0.41, 0.41], "confidence": 0.8, "phrase": "embryo"},
        ...     {"box_xyxy": [0.6, 0.6, 0.8, 0.8], "confidence": 0.85, "phrase": "embryo"},
        ... ]
        >>> filtered = filter_detections(detections, confidence_threshold=0.7, iou_threshold=0.5)
        >>> len(filtered)  # Should remove the duplicate
        2
    """
    # Filter by confidence
    filtered = [d for d in detections if d["confidence"] >= confidence_threshold]

    if not filtered:
        return []

    # Sort by confidence (descending)
    filtered = sorted(filtered, key=lambda x: x["confidence"], reverse=True)

    # Non-Maximum Suppression
    keep = []
    while filtered:
        # Keep highest confidence detection
        current = filtered.pop(0)
        keep.append(current)

        # Remove detections with high IoU to current
        filtered = [
            d for d in filtered
            if calculate_iou(current["box_xyxy"], d["box_xyxy"]) < iou_threshold
        ]

    return keep


def select_seed_frame(
    frame_detections: Dict[str, List[Dict]],
    min_detections: int = 1
) -> Optional[str]:
    """
    Select best seed frame for SAM2 initialization.

    Chooses frame with:
    1. At least min_detections embryos
    2. Highest average detection confidence

    Args:
        frame_detections: Dict mapping frame_id to list of detections
        min_detections: Minimum number of detections required

    Returns:
        Frame ID of selected seed frame, or None if no suitable frame

    Example:
        >>> frame_detections = {
        ...     "frame_0000": [{"confidence": 0.8}, {"confidence": 0.7}],
        ...     "frame_0001": [{"confidence": 0.9}, {"confidence": 0.85}],
        ...     "frame_0002": [{"confidence": 0.6}],
        ... }
        >>> seed = select_seed_frame(frame_detections, min_detections=2)
        >>> seed
        'frame_0001'
    """
    candidates = {}

    for frame_id, detections in frame_detections.items():
        if len(detections) >= min_detections:
            avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
            candidates[frame_id] = avg_confidence

    if not candidates:
        return None

    # Return frame with highest average confidence
    return max(candidates.items(), key=lambda x: x[1])[0]


def convert_boxes_to_sam2_format(
    detections: List[Dict],
    image_height: int,
    image_width: int
) -> np.ndarray:
    """
    Convert GroundingDINO boxes to SAM2 prompt format.

    SAM2 expects boxes in absolute pixel coordinates as numpy array.

    Args:
        detections: List of detection dicts with normalized box_xyxy
        image_height: Image height in pixels
        image_width: Image width in pixels

    Returns:
        Numpy array of shape (N, 4) with absolute pixel coordinates

    Example:
        >>> detections = [{"box_xyxy": [0.2, 0.3, 0.6, 0.7], "confidence": 0.9}]
        >>> boxes = convert_boxes_to_sam2_format(detections, 1000, 1000)
        >>> boxes.shape
        (1, 4)
        >>> boxes[0]
        array([200., 300., 600., 700.])
    """
    if not detections:
        return np.array([]).reshape(0, 4)

    boxes = []
    for det in detections:
        box_norm = det["box_xyxy"]
        # Convert normalized [0, 1] to absolute pixels
        box_abs = [
            box_norm[0] * image_width,
            box_norm[1] * image_height,
            box_norm[2] * image_width,
            box_norm[3] * image_height,
        ]
        boxes.append(box_abs)

    return np.array(boxes, dtype=np.float32)
