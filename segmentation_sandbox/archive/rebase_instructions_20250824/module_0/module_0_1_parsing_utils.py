"""
Core ID parsing utilities for MorphSeq pipeline.
Parses backwards from end patterns to handle complex experiment IDs.
"""

import re
from typing import Dict, Optional


def parse_entity_id(entity_id: str) -> Dict[str, str]:
    """Auto-detect type and parse to components."""
    entity_type = get_entity_type(entity_id)
    
    if entity_type == "snip":
        return _parse_backwards_snip(entity_id)
    elif entity_type == "embryo":
        return _parse_backwards_embryo(entity_id)
    elif entity_type == "image":
        return _parse_backwards_image(entity_id)
    elif entity_type == "video":
        return _parse_backwards_video(entity_id)
    else:
        return {"experiment_id": entity_id, "entity_type": "experiment"}


def get_entity_type(entity_id: str) -> str:
    """Return: experiment/video/image/embryo/snip."""
    if re.search(r'_s?\d{3,4}$', entity_id) and '_e' in entity_id:
        return "snip"
    elif re.search(r'_e\d+$', entity_id):
        return "embryo"
    elif re.search(r'_t\d{3,4}$', entity_id):
        return "image"
    elif re.search(r'_[A-H]\d{2}$', entity_id):
        return "video"
    else:
        return "experiment"


def extract_frame_number(entity_id: str) -> Optional[int]:
    """Extract frame from snip_id/image_id."""
    # Try image pattern first
    match = re.search(r'_t(\d{3,4})$', entity_id)
    if match:
        return int(match.group(1))
    
    # Try snip pattern (ensure has _e)
    match = re.search(r'_s?(\d{3,4})$', entity_id)
    if match and '_e' in entity_id:
        return int(match.group(1))
    
    return None


def extract_experiment_id(entity_id: str) -> str:
    """Get experiment_id from any entity."""
    components = parse_entity_id(entity_id)
    return components.get("experiment_id", entity_id)


def extract_video_id(entity_id: str) -> Optional[str]:
    """Get video_id from embryo/snip."""
    components = parse_entity_id(entity_id)
    return components.get("video_id")


def extract_image_id(entity_id: str) -> Optional[str]:
    """Get image_id from snip (convert frame)."""
    if get_entity_type(entity_id) == "snip":
        components = _parse_backwards_snip(entity_id)
        video_id = components.get("video_id")
        frame = components.get("frame_number")
        if video_id and frame:
            return f"{video_id}_t{frame}"
    return None


def extract_embryo_id(entity_id: str) -> Optional[str]:
    """Get embryo_id from snip."""
    components = parse_entity_id(entity_id)
    return components.get("embryo_id")


def validate_id_format(entity_id: str, expected_type: str) -> bool:
    """Check if ID matches expected format."""
    try:
        detected_type = get_entity_type(entity_id)
        return detected_type == expected_type
    except:
        return False


def get_parent_id(entity_id: str, parent_type: str) -> Optional[str]:
    """Get specific parent (video_id from snip_id, etc.)."""
    components = parse_entity_id(entity_id)
    
    if parent_type == "experiment":
        return components.get("experiment_id")
    elif parent_type == "video":
        return components.get("video_id")
    elif parent_type == "embryo":
        return components.get("embryo_id")
    
    return None


def _parse_backwards_video(video_id: str) -> Dict[str, str]:
    """Parse video_id: {experiment_id}_{WELL}."""
    match = re.search(r'_([A-H]\d{2})$', video_id)
    if not match:
        raise ValueError(f"Invalid video_id: {video_id}")
    
    well_id = match.group(1)
    experiment_id = video_id[:match.start()]
    
    return {
        "experiment_id": experiment_id,
        "well_id": well_id,
        "video_id": video_id,
        "entity_type": "video"
    }


def _parse_backwards_image(image_id: str) -> Dict[str, str]:
    """Parse image_id: {video_id}_t{FRAME}."""
    match = re.search(r'_t(\d{3,4})$', image_id)
    if not match:
        raise ValueError(f"Invalid image_id: {image_id}")
    
    frame_number = match.group(1)
    video_id = image_id[:match.start()]
    
    # Parse video_id recursively
    video_components = _parse_backwards_video(video_id)
    
    return {
        **video_components,
        "frame_number": frame_number,
        "image_id": image_id,
        "entity_type": "image"
    }


def _parse_backwards_embryo(embryo_id: str) -> Dict[str, str]:
    """Parse embryo_id: {video_id}_e{NN}."""
    match = re.search(r'_e(\d+)$', embryo_id)
    if not match:
        raise ValueError(f"Invalid embryo_id: {embryo_id}")
    
    embryo_number = match.group(1)
    video_id = embryo_id[:match.start()]
    
    # Parse video_id recursively
    video_components = _parse_backwards_video(video_id)
    
    return {
        **video_components,
        "embryo_number": embryo_number,
        "embryo_id": embryo_id,
        "entity_type": "embryo"
    }


def _parse_backwards_snip(snip_id: str) -> Dict[str, str]:
    """Parse snip_id: {embryo_id}_{FRAME} or {embryo_id}_s{FRAME}."""
    match = re.search(r'_s?(\d{3,4})$', snip_id)
    if not match:
        raise ValueError(f"Invalid snip_id: {snip_id}")
    
    frame_number = match.group(1)
    embryo_id = snip_id[:match.start()]
    
    # Parse embryo_id recursively
    embryo_components = _parse_backwards_embryo(embryo_id)
    
    return {
        **embryo_components,
        "frame_number": frame_number,
        "snip_id": snip_id,
        "entity_type": "snip"
    }