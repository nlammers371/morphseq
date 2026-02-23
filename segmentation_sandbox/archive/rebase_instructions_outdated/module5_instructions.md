# Module 5: Visualization System

## Overview
Create comprehensive visualization system that automatically overlays annotations, confidence scores, QC flags, and metadata. Simple API - just specify what to show.

## Dependencies
- Modules 1-4 completed
- QC system must be in place for flag display
- OpenCV, matplotlib for rendering

## Files to Create/Modify

```
utils/
└── visualization/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── layout_manager.py      # Zone-based layouts
    │   ├── overlay_base.py        # Base overlay classes
    │   └── color_schemes.py       # Visual themes
    ├── components/
    │   ├── __init__.py
    │   ├── info_panels.py         # Corner information
    │   ├── annotation_layers.py   # Detection/segmentation
    │   └── comparison_views.py    # Side-by-side
    └── renderers/
        ├── __init__.py
        ├── video_renderer.py      # MP4 generation
        ├── frame_renderer.py      # Single frames
        └── grid_renderer.py       # Multi-video grids
```

## Implementation Steps (Pseudocode)

### Step 1: Create Enhanced Zone System

```python
# utils/visualization/core/layout_manager.py
"""8-zone modular layout system."""

class Zone:
    """Information zone definition."""
    def __init__(self, position, size=(0.25, 0.15)):
        self.position = position  # 'top-left', 'top-middle', etc.
        self.size = size
        self.content = None
        self.data_source = None

class ModularLayout:
    """8-zone plug-and-play layout."""
    
    POSITIONS = {
        'top-left': (0, 0),
        'top-middle': (0.375, 0),
        'top-right': (0.75, 0),
        'middle-left': (0, 0.425),
        'middle-right': (0.75, 0.425),
        'bottom-left': (0, 0.85),
        'bottom-middle': (0.375, 0.85),
        'bottom-right': (0.75, 0.85)
    }
    
    def __init__(self):
        self.zones = {pos: Zone(pos) for pos in self.POSITIONS}
        self.data_registry = {}  # Register data sources
        
    def assign_content(self, position, content_type, data_source=None):
        """Assign content to a zone."""
        self.zones[position].content = content_type
        self.zones[position].data_source = data_source
        
    def register_data_provider(self, name, provider_func):
        """Register function that provides data for a zone."""
        self.data_registry[name] = provider_func
```

### Step 2: Create Extensible Data Provider System

```python
# utils/visualization/core/data_providers.py
"""Modular data providers for zones."""

class DataProviderRegistry:
    """Registry for all data providers."""
    
    def __init__(self):
        self.providers = {}
        self._register_default_providers()
        
    def _register_default_providers(self):
        """Register built-in providers."""
        self.providers['frame_info'] = self._provide_frame_info
        self.providers['embryo_metadata'] = self._provide_embryo_metadata
        self.providers['qc_status'] = self._provide_qc_status
        self.providers['experiment_info'] = self._provide_experiment_info
        self.providers['confidence_scores'] = self._provide_confidence
        self.providers['tracking_stats'] = self._provide_tracking_stats
        # Future: custom calculations can be added here
        
    def register_custom_provider(self, name, func):
        """Add custom data provider."""
        self.providers[name] = func
        
    def get_data(self, provider_name, image_id, context):
        """Get data for specific image."""
        if provider_name in self.providers:
            return self.providers[provider_name](image_id, context)
        return None
        
    def _provide_frame_info(self, image_id, context):
        """Provide frame information."""
        parsed = parse_image_id(image_id)
        return {
            'frame': parsed['frame_number'],
            'time': float(parsed['frame_number']) / context.get('fps', 5),
            'image_id': image_id
        }
    
    # [Other provider methods...]
```

### Step 3: Create Image-Level Data Aggregator

```python
# utils/visualization/core/frame_data.py
"""Aggregate all data at image level."""

class FrameDataAggregator:
    """Collect all data for a single frame."""
    
    def __init__(self, data_sources):
        self.data_sources = data_sources
        self.cache = {}
        
    def get_frame_data(self, image_id):
        """Get all data for one image."""
        if image_id in self.cache:
            return self.cache[image_id]
            
        frame_data = {
            'image_id': image_id,
            'zones': {}
        }
        
        # Parse once
        parsed = parse_image_id(image_id)
        video_id = parsed['video_id']
        
        # Collect from all sources
        if 'experiment_metadata' in self.data_sources:
            exp_data = self._get_experiment_data(image_id)
            frame_data['zones']['experiment'] = exp_data
            
        if 'gdino_annotations' in self.data_sources:
            detections = self._get_detections(image_id)
            frame_data['zones']['detections'] = detections
            
        if 'embryo_metadata' in self.data_sources:
            embryo_data = self._get_embryo_data(image_id)
            frame_data['zones']['embryo'] = embryo_data
            
        # [Collect from other sources...]
        
        self.cache[image_id] = frame_data
        return frame_data
```

### Step 4: Enhanced Main Visualizer

```python
# utils/visualization/__init__.py
class PipelineVisualizer:
    """Enhanced modular visualization."""
    
    def __init__(self):
        self.layout = ModularLayout()
        self.data_registry = DataProviderRegistry()
        self.data_sources = {}
        self.aggregator = None
        
    def configure_layout(self, config):
        """
        Configure zones with content.
        
        Example config:
        {
            'top-left': 'frame_info',
            'top-middle': 'confidence_scores',
            'top-right': 'embryo_metadata',
            'middle-left': None,  # Empty
            'middle-right': 'custom_metric',  # Future custom
            'bottom-left': 'experiment_info',
            'bottom-middle': 'tracking_stats',
            'bottom-right': 'qc_status'
        }
        """
        for position, content in config.items():
            if content:
                self.layout.assign_content(position, content)
                
    def register_custom_metric(self, name, calculation_func):
        """Add custom calculated metric."""
        self.data_registry.register_custom_provider(name, calculation_func)
        
    def create_video(self, video_id, zone_config=None, overlays=None, output=None):
        """Create video with modular zones."""
        # Use provided config or default
        if zone_config:
            self.configure_layout(zone_config)
            
        # Initialize aggregator
        self.aggregator = FrameDataAggregator(self.data_sources)
        
        # Get video info
        video_data = self._get_video_data(video_id)
        image_ids = sorted(video_data['images'].keys())
        
        # Process each frame
        for image_id in image_ids:
            # Get all data for this frame
            frame_data = self.aggregator.get_frame_data(image_id)
            
            # Render frame with zones
            frame = self._render_frame(image_id, frame_data, overlays)
            
            # [Write to video]
```

### Step 5: Zone Rendering System

```python
# utils/visualization/core/zone_renderer.py
"""Render individual zones."""

class ZoneRenderer:
    """Render content in zones."""
    
    def render_zone(self, frame, zone, data, theme):
        """Render single zone."""
        # Get pixel coordinates
        x, y, w, h = self._get_pixel_coords(zone, frame.shape)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), theme['panel_bg'], -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Render content based on type
        if zone.content == 'frame_info':
            self._render_frame_info(frame, x, y, w, h, data)
        elif zone.content == 'qc_status':
            self._render_qc_status(frame, x, y, w, h, data)
        elif zone.content == 'custom_metric':
            self._render_custom(frame, x, y, w, h, data)
        # [Other content types...]
        
        return frame
```

## Usage Examples

```python
# Standard usage
viz = PipelineVisualizer()
viz.load_all_data_sources()

# Default 8-zone layout
viz.create_video("video_id", output="analysis.mp4")

# Custom zone configuration
custom_layout = {
    'top-left': 'frame_info',
    'top-middle': 'velocity_metric',  # Custom
    'top-right': 'embryo_count',
    'middle-left': None,
    'middle-right': None,
    'bottom-left': 'temperature',     # Custom
    'bottom-middle': 'ph_level',      # Custom  
    'bottom-right': 'qc_status'
}

# Register custom metrics
viz.register_custom_metric('velocity_metric', calculate_velocity)
viz.register_custom_metric('temperature', get_temperature_data)
viz.register_custom_metric('ph_level', get_ph_data)

# Create with custom layout
viz.create_video("video_id", zone_config=custom_layout)

# Add new calculated metric later
viz.register_custom_metric('morphology_score', calculate_morphology)
```

## Key Design Points

1. **8 Zones**: Full flexibility in placement
2. **Plug-and-Play**: Assign any content to any zone
3. **Extensible**: Easy to add new metrics/calculations
4. **Image-Level Cache**: All data aggregated per frame
5. **Custom Providers**: Register any calculation function
6. **Future-Proof**: New analyses just register as providers

## Backup & Compatibility

- Zone configs saved with videos for reproducibility  
- Default zones if none specified
- Graceful handling of missing data
- Backward compatible with 4-zone layout

### Step 2: Create main visualizer

```python
# utils/visualization/__init__.py
class PipelineVisualizer:
    """Main visualization interface - simple API."""
    
    def __init__(self):
        self.data_sources = {}
        self.layout = 'standard'
        
    def load_experiment_metadata(self, path):
        """Load experiment metadata for display."""
        # [Load and store]
        pass
        
    def load_gdino_annotations(self, path):
        """Load detection annotations."""
        # [Load and store]
        pass
        
    def load_sam2_annotations(self, path):
        """Load segmentation annotations."""
        # [Load and store]
        pass
        
    def load_embryo_metadata(self, path):
        """Load embryo metadata."""
        # [Load and store]
        pass
    
    def create_video(self, video_id, overlays, output):
        """
        Simple API - just specify what to show.
        
        Args:
            video_id: Which video to process
            overlays: List of what to show
                     ['detections', 'masks', 'embryo_ids', 
                      'confidence', 'qc_flags', 'tracks']
            output: Output video path
        """
        # [Pseudocode]
        # 1. Get base video from experiment metadata
        # 2. For each frame:
        #    - Load base frame
        #    - Add requested overlays
        #    - Add info panels
        #    - Write frame
        # 3. Save video
        pass
    
    def create_comparison(self, videos, overlays, layout, output):
        """Create comparison video."""
        # [Pseudocode - handle multiple videos]
        pass
```

### Step 3: Create components (pseudocode only)

```python
# utils/visualization/components/info_panels.py
class InfoPanelBase:
    """Base class for info panels."""
    def render(self, data, zone, frame):
        # [Add semi-transparent background]
        # [Render text/graphics]
        pass

class FrameInfoPanel(InfoPanelBase):
    """Top-left: frame number, timestamp."""
    pass

class EmbryoMetadataPanel(InfoPanelBase):
    """Top-right: genotype, phenotype, treatment."""
    pass

class ExperimentInfoPanel(InfoPanelBase):
    """Bottom-left: experiment ID, conditions."""
    pass

class QCStatusPanel(InfoPanelBase):
    """Bottom-right: QC flags with severity colors."""
    pass

# utils/visualization/components/annotation_layers.py
class DetectionLayer:
    """Overlay detection boxes."""
    def apply(self, frame, detections):
        # [Draw boxes with confidence]
        pass

class SegmentationLayer:
    """Overlay segmentation masks."""
    def apply(self, frame, masks):
        # [Draw semi-transparent masks]
        pass

class TrackingLayer:
    """Show embryo tracks over time."""
    def apply(self, frame, tracks):
        # [Draw motion trails]
        pass

# utils/visualization/core/color_schemes.py
THEMES = {
    'default': {
        'embryo_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F'],
        'qc_severity': {
            'info': '#3498db',
            'warning': '#f39c12', 
            'error': '#e74c3c',
            'critical': '#c0392b'
        },
        'confidence': {
            'high': '#27ae60',
            'medium': '#f39c12',
            'low': '#e74c3c'
        },
        'panel_bg': (0, 0, 0, 180),  # RGBA
        'text': (255, 255, 255)
    }
}
```

## Usage Examples

```python
# Simple usage - just load and specify overlays
viz = PipelineVisualizer()
viz.load_experiment_metadata("exp_metadata.json")
viz.load_gdino_annotations("gdino.json")
viz.load_sam2_annotations("sam2.json") 
viz.load_embryo_metadata("embryo.json")

# Create analysis video
viz.create_video(
    video_id="20250622_chem_35C_T01_1605_H09",
    overlays=['detections', 'masks', 'embryo_ids', 'qc_flags'],
    output="analysis.mp4"
)

# Create comparison
viz.create_comparison(
    videos=["..._H09", "..._H10"],
    overlays=['masks', 'qc_flags'],
    layout='side_by_side',
    output="comparison.mp4"
)
```

## Key Design Points

1. **Simple API**: Just load files and specify what to show
2. **Automatic layout**: Zones handle information placement
3. **QC-aware**: Displays flags with severity colors
4. **Flexible overlays**: Mix and match what to display
5. **Reuses base videos**: No need to recreate videos

## Testing Checklist

- [ ] Test zone calculations
- [ ] Test overlay rendering
- [ ] Test info panel display
- [ ] Test QC flag visualization
- [ ] Test comparison layouts
- [ ] Test video generation
- [ ] Test with missing data sources