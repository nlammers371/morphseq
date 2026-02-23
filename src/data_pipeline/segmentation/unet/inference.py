"""
UNet Auxiliary Mask Inference (STUB - MVP)

Generate auxiliary segmentation masks for quality control.
This is a STUB implementation for MVP - full UNet pipeline to be implemented later.

Auxiliary masks include:
    - Viability (via): Dead/dying tissue regions
    - Yolk: Yolk sac segmentation
    - Focus: Out-of-focus regions
    - Bubble: Air bubble detection

Key Functions (STUB):
    - load_unet_models: Load all 5 UNet model checkpoints
    - run_unet_inference: Generate auxiliary masks for images
    - export_auxiliary_masks: Save masks to disk

NOTE: This is a placeholder for Wave 4 MVP.
Full implementation will be added in a future wave when UNet integration is prioritized.

For MVP, downstream QC steps can check for auxiliary mask availability and
gracefully skip UNet-based QC flags if masks are not present.

Example Usage (Future):
    ```python
    # Load models (all share same architecture, different checkpoints)
    models = load_unet_models(
        mask_checkpoint="weights/mask_v0_0100.pth",
        via_checkpoint="weights/via_v1_0100.pth",
        yolk_checkpoint="weights/yolk_v1_0050.pth",
        focus_checkpoint="weights/focus_v0_0100.pth",
        bubble_checkpoint="weights/bubble_v0_0100.pth",
        device="cuda"
    )

    # Run inference
    auxiliary_masks = run_unet_inference(
        models=models,
        image_path=Path("image.jpg")
    )

    # Export masks
    export_auxiliary_masks(
        masks=auxiliary_masks,
        output_dir=Path("unet_masks"),
        image_id="exp_A01_t0000"
    )
    ```

Output Structure:
    ```
    unet_masks/
    ├── via/{image_id}_via.png
    ├── yolk/{image_id}_yolk.png
    ├── focus/{image_id}_focus.png
    ├── bubble/{image_id}_bubble.png
    └── mask/{image_id}_mask.png  # UNet embryo (validation only)
    ```
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class UNetNotImplementedError(NotImplementedError):
    """Raised when UNet functionality is not yet implemented."""

    def __init__(self):
        super().__init__(
            "UNet auxiliary mask inference is not yet implemented. "
            "This is a stub for MVP. Full implementation coming in future wave. "
            "Downstream QC should check for auxiliary mask availability and "
            "gracefully skip UNet-based QC flags if not present."
        )


def load_unet_models(
    mask_checkpoint: str,
    via_checkpoint: str,
    yolk_checkpoint: str,
    focus_checkpoint: str,
    bubble_checkpoint: str,
    device: str = "cuda"
) -> Dict[str, "torch.nn.Module"]:
    """
    Load all UNet model checkpoints.

    STUB: Not yet implemented for MVP.

    Args:
        mask_checkpoint: Path to embryo mask model weights
        via_checkpoint: Path to viability model weights
        yolk_checkpoint: Path to yolk model weights
        focus_checkpoint: Path to focus model weights
        bubble_checkpoint: Path to bubble model weights
        device: Device to load models on

    Returns:
        Dict mapping model name to loaded model

    Raises:
        UNetNotImplementedError: Always (stub implementation)
    """
    raise UNetNotImplementedError()


def run_unet_inference(
    models: Dict[str, "torch.nn.Module"],
    image_path: Path
) -> Dict[str, np.ndarray]:
    """
    Generate auxiliary masks for an image.

    STUB: Not yet implemented for MVP.

    Args:
        models: Dict of loaded UNet models
        image_path: Path to input image

    Returns:
        Dict mapping mask type to binary mask array

    Raises:
        UNetNotImplementedError: Always (stub implementation)
    """
    raise UNetNotImplementedError()


def export_auxiliary_masks(
    masks: Dict[str, np.ndarray],
    output_dir: Path,
    image_id: str
) -> Dict[str, Path]:
    """
    Export auxiliary masks to disk.

    STUB: Not yet implemented for MVP.

    Args:
        masks: Dict mapping mask type to mask array
        output_dir: Base output directory
        image_id: Image identifier for filename

    Returns:
        Dict mapping mask type to saved file path

    Raises:
        UNetNotImplementedError: Always (stub implementation)
    """
    raise UNetNotImplementedError()


# Placeholder functions for future implementation

def check_unet_available() -> bool:
    """
    Check if UNet models and dependencies are available.

    Returns:
        False (always, for MVP stub)
    """
    return False


def get_unet_model_info() -> Dict[str, str]:
    """
    Get information about available UNet models.

    Returns:
        Empty dict (stub implementation)
    """
    return {
        "status": "not_implemented",
        "message": "UNet auxiliary masks not yet implemented for MVP",
        "planned_models": [
            "mask_v0_0100 - Embryo segmentation (validation)",
            "via_v1_0100 - Viability/dead tissue",
            "yolk_v1_0050 - Yolk sac",
            "focus_v0_0100 - Out-of-focus regions",
            "bubble_v0_0100 - Air bubbles",
        ]
    }


# Future implementation notes:
#
# When implementing UNet inference:
# 1. Extract model architecture from build02B_segment_bf_main.py
# 2. Confirm all 5 models use identical post-processing (as assumed)
# 3. Implement shared inference pipeline with checkpoint swapping
# 4. Add preprocessing transforms (normalization, resizing)
# 5. Add postprocessing (thresholding, morphological ops)
# 6. Integrate with existing mask export utilities
# 7. Update QC modules to handle optional auxiliary masks
#
# Key files to reference:
# - src/build/build02B_segment_bf_main.py (UNet architecture & inference)
# - src/build/build04_perform_embryo_qc.py (QC logic using UNet masks)
# - src/build/qc_utils.py (Auxiliary mask QC utilities)
