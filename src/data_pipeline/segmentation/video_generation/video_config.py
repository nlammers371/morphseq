"""Video rendering configuration and color constants."""

from __future__ import annotations

import cv2


class VideoConfig:
    """Video generation configuration constants."""

    # Video settings
    FPS = 5
    CODEC = "mp4v"
    JPEG_QUALITY = 90

    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2
    FONT_THICKNESS = 3

    # Default text settings
    TEXT_COLOR = (255, 255, 255)
    TEXT_BACKGROUND_ALPHA = 0.7
    TEXT_BACKGROUND_COLOR = (0, 0, 0)
    TEXT_MARGIN = 10

    # Overlay positioning
    IMAGE_ID_Y_PERCENTAGE = 0.1
    IMAGE_ID_MARGIN_RIGHT = 10

    TOP_RIGHT_OFFSET = (10, 30)
    TOP_LEFT_OFFSET = (10, 30)
    BOTTOM_LEFT_OFFSET = (10, 30)
    BOTTOM_RIGHT_OFFSET = (10, 30)

    # Bounding box settings
    BBOX_THICKNESS = 3
    BBOX_ALPHA = 0.4

    # Mask settings
    MASK_ALPHA = 0.4
    MASK_OUTLINE_THICKNESS = 2

    @classmethod
    def fast_generation(cls) -> "VideoConfig":
        return cls()

    @classmethod
    def high_quality(cls) -> "VideoConfig":
        config = cls()
        config.FPS = 10
        config.FONT_THICKNESS = 3
        return config


COLORBLIND_PALETTE = {
    "light_blue": (173, 216, 230),
    "light_green": (144, 238, 144),
    "light_coral": (240, 128, 128),
    "light_yellow": (255, 215, 0),
    "light_purple": (221, 160, 221),
    "light_orange": (255, 218, 185),
    "light_cyan": (0, 255, 255),
    "light_rose": (255, 182, 193),
    "light_mint": (152, 255, 152),
    "light_lavender": (186, 135, 186),
}


OVERLAY_COLORS = {
    "detection": COLORBLIND_PALETTE["light_blue"],
    "mask": COLORBLIND_PALETTE["light_green"],
    "metadata": COLORBLIND_PALETTE["light_coral"],
    "qc_good": COLORBLIND_PALETTE["light_green"],
    "qc_warning": COLORBLIND_PALETTE["light_yellow"],
    "qc_error": COLORBLIND_PALETTE["light_coral"],
    "frame_number": (255, 255, 255),
}


def get_color_cycle():
    colors = list(COLORBLIND_PALETTE.values())
    while True:
        for color in colors:
            yield color
