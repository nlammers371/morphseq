"""
UNet Auxiliary Masks (STUB - MVP)

Auxiliary segmentation masks for quality control.
Full implementation to be added in future wave.

For MVP, this module provides stub functions that raise
UNetNotImplementedError with helpful messages.
"""

from .inference import (
    load_unet_models,
    run_unet_inference,
    export_auxiliary_masks,
    check_unet_available,
    get_unet_model_info,
    UNetNotImplementedError,
)

__all__ = [
    "load_unet_models",
    "run_unet_inference",
    "export_auxiliary_masks",
    "check_unet_available",
    "get_unet_model_info",
    "UNetNotImplementedError",
]
