"""
Video generation configuration and constants.
"""

import cv2
from typing import Dict, Tuple

class VideoConfig:
    """Video generation configuration constants."""
    
    # Video settings
    FPS = 5
    CODEC = 'mp4v'
    JPEG_QUALITY = 90
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2  # Increased from 1.0 for bigger image_id overlay
    FONT_THICKNESS = 3
    
    # Default text settings
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BACKGROUND_ALPHA = 0.7
    TEXT_BACKGROUND_COLOR = (0, 0, 0)  # Black
    TEXT_MARGIN = 10
    
    # Overlay positioning
    # Image ID position: 10% down from top, right-aligned with margin (like original)
    IMAGE_ID_Y_PERCENTAGE = 0.1  # 10% down from top
    IMAGE_ID_MARGIN_RIGHT = 10   # pixels from right edge
    
    TOP_RIGHT_OFFSET = (10, 30)  # (margin_from_right, margin_from_top)
    TOP_LEFT_OFFSET = (10, 30)
    BOTTOM_LEFT_OFFSET = (10, 30)  # from bottom
    BOTTOM_RIGHT_OFFSET = (10, 30)
    
    # Bounding box settings
    BBOX_THICKNESS = 3
    BBOX_ALPHA = 0.4
    
    # Mask settings
    MASK_ALPHA = 0.4
    MASK_OUTLINE_THICKNESS = 2
    
    @classmethod
    def fast_generation(cls):
        """Return a config optimized for fast generation."""
        return cls()
        
    @classmethod
    def high_quality(cls):
        """Return a config for higher quality (slower) generation."""
        config = cls()
        config.FPS = 10
        config.FONT_THICKNESS = 3
        return config

# Color-blind friendly pastel palette
# Based on Tol's color schemes - distinguishable for all color vision types
COLORBLIND_PALETTE = {
    'light_blue': (173, 216, 230),     # Light blue - very accessible
    'light_green': (144, 238, 144),    # Light green 
    'light_coral': (240, 128, 128),    # Light coral/pink
    'light_yellow': (255, 215, 0),     # Gold yellow - more visible
    'light_purple': (221, 160, 221),   # Light purple/plum
    'light_orange': (255, 218, 185),   # Light orange/peach
    'light_cyan': (0, 255, 255),       # Proper cyan - more visible
    'light_rose': (255, 182, 193),     # Light rose
    'light_mint': (152, 255, 152),     # Proper mint - more visible
    'light_lavender': (186, 135, 186), # Proper lavender - more visible
}

# Specific color assignments for different overlay types
OVERLAY_COLORS = {
    'gdino_detection': COLORBLIND_PALETTE['light_blue'],
    'sam2_mask': COLORBLIND_PALETTE['light_green'], 
    'embryo_metadata': COLORBLIND_PALETTE['light_coral'],
    'qc_flag_good': COLORBLIND_PALETTE['light_green'],
    'qc_flag_warning': COLORBLIND_PALETTE['light_yellow'],
    'qc_flag_error': COLORBLIND_PALETTE['light_coral'],
    'frame_number': (255, 255, 255),  # White - always visible
}

def get_color_cycle():
    """Returns a cycling iterator of colors for multiple detections."""
    colors = list(COLORBLIND_PALETTE.values())
    while True:
        for color in colors:
            yield color
