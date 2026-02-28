"""Model integration helpers.

This package exists so `src/data_pipeline` can load external model repos (GDINO, SAM2, ...)
from a single, controlled integration layer rather than scattering sys.path hacks throughout
the pipeline code.
"""

from .groundingdino import load_groundingdino_model  # noqa: F401
from .sam2 import load_sam2_video_predictor  # noqa: F401

