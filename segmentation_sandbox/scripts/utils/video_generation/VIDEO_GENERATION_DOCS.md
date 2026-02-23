so # Video Generation System Documentation

This document explains the video generation system used in the MorphSeq pipeline for creating annotated videos from microscopy image sequences.

## Overview

The video generation system creates MP4 videos from sequences of microscopy images with various overlay annotations (detections, masks, metadata, etc.). It's designed to be modular, configurable, and colorblind-friendly.

## Core Components

### 1. VideoConfig Class (`video_config.py`)

The main configuration class that controls all aspects of video generation:

#### Video Settings
- **FPS**: 5 frames per second (optimized for biological data viewing)
- **CODEC**: 'mp4v' (widely compatible MP4 codec)
- **JPEG_QUALITY**: 90% (high quality for scientific visualization)

#### Font Settings
- **FONT**: OpenCV's Hershey Simplex (clean, readable)
- **FONT_SCALE**: 2.0 (increased for better visibility of image IDs)
- **FONT_THICKNESS**: 2 pixels (balanced readability)

#### Text Appearance
- **TEXT_COLOR**: White (255, 255, 255) - high contrast
- **TEXT_BACKGROUND**: Semi-transparent black overlay (70% opacity)
- **TEXT_MARGIN**: 10 pixels from edges

#### Overlay Positioning System
The system uses percentage-based positioning for consistent placement across different image sizes:

- **IMAGE_ID_Y_PERCENTAGE**: 0.1 (10% down from top)
- **IMAGE_ID_MARGIN_RIGHT**: 10 pixels from right edge
- **Corner offsets**: Standardized 10px margin, 30px from edges

#### Visual Elements
- **BBOX_THICKNESS**: 3 pixels (prominent but not overwhelming)
- **BBOX_ALPHA**: 0.8 (80% opacity for clear visibility)
- **MASK_ALPHA**: 0.4 (40% opacity to show underlying image)
- **MASK_OUTLINE_THICKNESS**: 2 pixels

### 2. Color System

#### Colorblind-Friendly Palette
Based on Tol's scientific color schemes, designed to be distinguishable for all types of color vision:

```python
COLORBLIND_PALETTE = {
    'light_blue': (173, 216, 230),     # Primary detection color
    'light_green': (144, 238, 144),    # Success/good quality
    'light_coral': (240, 128, 128),    # Warnings/metadata
    'light_yellow': (255, 255, 224),   # Caution states
    'light_purple': (221, 160, 221),   # Alternative annotations
    # ... additional colors for variety
}
```

#### Overlay Color Assignments
Specific colors are assigned to different annotation types:

- **gdino_detection**: Light blue (primary detection visualization)
- **sam2_mask**: Light green (segmentation results)
- **embryo_metadata**: Light coral (biological metadata)
- **qc_flag_good**: Light green (quality control - passed)
- **qc_flag_warning**: Light yellow (quality control - caution)
- **qc_flag_error**: Light coral (quality control - failed)
- **frame_number**: White (always visible, high contrast)

### 3. Configuration Presets

#### Fast Generation
```python
config = VideoConfig.fast_generation()
```
- Standard settings optimized for quick processing
- 5 FPS, standard quality

#### High Quality
```python
config = VideoConfig.high_quality()
```
- Enhanced settings for publication/presentation
- 10 FPS, thicker fonts (thickness=3)
- Better for detailed analysis

## Usage Patterns

### Basic Video Creation
```python
from video_config import VideoConfig

config = VideoConfig()
# Use config.FONT_SCALE, config.OVERLAY_COLORS, etc.
```

### Color Cycling for Multiple Objects
```python
from video_config import get_color_cycle

color_gen = get_color_cycle()
for detection in detections:
    color = next(color_gen)  # Get next color in cycle
```

### Overlay Positioning
```python
# Image ID placement (top-right)
y_pos = int(image_height * config.IMAGE_ID_Y_PERCENTAGE)
x_pos = image_width - config.IMAGE_ID_MARGIN_RIGHT

# Other overlays use offset tuples
top_right_x = image_width - config.TOP_RIGHT_OFFSET[0]
top_right_y = config.TOP_RIGHT_OFFSET[1]
```

## Integration with Pipeline

### Step 1: Video Preparation (`01_prepare_videos.py`)
- Uses VideoConfig for creating base videos from image sequences
- Applies image_id overlays using the font settings
- Generates MP4 files organized by experiment/video structure

### Annotation Overlays (Future Steps)
- Step 3+ will use the color assignments for detection overlays
- SAM2 masks will use the mask alpha settings
- Quality control flags will use the QC color scheme

## Design Principles

### 1. Accessibility
- Colorblind-friendly palette ensures scientific accessibility
- High contrast text with background overlays
- Consistent sizing relative to image dimensions

### 2. Scientific Standards
- Conservative color choices appropriate for publications
- Clear visual hierarchy (text > bounding boxes > masks)
- Standardized positioning for consistent viewing

### 3. Configurability
- Easy to adjust settings without code changes
- Preset configurations for different use cases
- Modular color system for different annotation types

### 4. Performance
- Optimized settings balance quality and generation speed
- Reasonable defaults for biological imaging workflows
- Scalable to different image sizes

## Customization Guide

### Changing Font Size
```python
# Make image IDs even bigger
VideoConfig.FONT_SCALE = 3.0
```

### Adding New Overlay Types
```python
# Add to OVERLAY_COLORS
OVERLAY_COLORS['my_new_annotation'] = COLORBLIND_PALETTE['light_purple']
```

### Custom Color Palette
```python
# Extend or modify the palette
COLORBLIND_PALETTE['my_custom_color'] = (R, G, B)
```

### Positioning Adjustments
```python
# Move image ID lower on screen
VideoConfig.IMAGE_ID_Y_PERCENTAGE = 0.15  # 15% from top
```

## Future Enhancements

1. **Dynamic Scaling**: Automatic font scaling based on image resolution
2. **Theme Support**: Dark/light themes for different viewing conditions
3. **Export Presets**: Specialized configurations for different output formats
4. **Interactive Overlays**: Support for clickable annotations in web players
5. **Batch Customization**: Per-experiment color/layout customization

## Troubleshooting

### Common Issues

**Text too small/large**: Adjust `FONT_SCALE` and `FONT_THICKNESS`
**Colors not distinguishable**: Check monitor calibration, consider high-contrast preset
**Overlays overlap**: Adjust margin and offset values
**Performance issues**: Use `fast_generation()` preset or reduce FPS

### Color Vision Testing
The palette has been tested with online colorblind simulators, but always verify with actual users when possible.

## File Structure
```
scripts/utils/video_generation/
├── video_config.py          # Main configuration (this file)
├── VIDEO_GENERATION_DOCS.md # This documentation
└── [other video utilities]  # Additional video generation tools
```
