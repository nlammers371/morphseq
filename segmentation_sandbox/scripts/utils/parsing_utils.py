"""
Core ID parsing utilities for MorphSeq pipeline.

Parses backwards from end patterns to handle complex experiment IDs.
All ID parsing works backwards from the end to handle variable experiment names.

ENTITY HIERARCHY:
    experiment → video → image/embryo → snip

ID FORMATS:
    experiment_id: "20250624_chem02_28C_T00_1356"
    video_id:      "20250624_chem02_28C_T00_1356_H01"
    image_id:      "20250624_chem02_28C_T00_1356_H01_ch00_t0042"  (with channel + 't' prefix)
    embryo_id:     "20250624_chem02_28C_T00_1356_H01_e01"
    snip_id:       "20250624_chem02_28C_T00_1356_H01_e01_s0034"

MAIN FUNCTIONS:
    parse_entity_id()     - Auto-detect and parse any ID type
        Output: Dict with keys for all components, e.g. {'experiment_id': ..., 'video_id': ..., 'entity_type': ...}
    get_entity_type()     - Determine ID type from format
        Output: String, one of 'experiment', 'video', 'image', 'embryo', 'snip'
    build_image_id()      - Create image ID with channel + 't' prefix
        Output: e.g. '20250624_chem02_28C_T00_1356_H01_ch00_t0042'
    build_video_id()      - Create video ID from experiment + well
        Output: e.g. '20250624_chem02_28C_T00_1356_H01'
    build_embryo_id()     - Create embryo ID from video + number
        Output: e.g. '20250624_chem02_28C_T00_1356_H01_e01'
    build_snip_id()       - Create snip ID from embryo + frame
        Output: e.g. '20250624_chem02_28C_T00_1356_H01_e01_s0034'
    
EXTRACTION FUNCTIONS:
    extract_frame_number()    - Get frame from image/snip ID
    extract_experiment_id()   - Get experiment ID from any child ID
    extract_video_id()        - Get video ID from image/embryo/snip ID
    extract_embryo_id()       - Get embryo ID from snip ID
    
VALIDATION FUNCTIONS:
    validate_id_format()      - Check ID matches expected type
    get_parent_id()          - Get parent ID of specified type
    
NORMALIZATION FUNCTIONS:
    normalize_frame_number()  - Convert frame string to integer
    normalize_well_id()       - Standardize well format (A01, B12, etc.)
    
PATH UTILITIES:
    get_image_filename_from_id()     - Convert image_id to NNNN.jpg filename
    get_relative_image_path()        - Get relative path: images/video_id/NNNN.jpg  
    get_relative_video_path()        - Get relative path: images/video_id/
    build_image_path_from_base()     - Build full path (backward compatibility)
    build_video_path_from_base()     - Build full path (backward compatibility)
    
    NOTE: For robust path resolution, use ExperimentMetadata.get_image_path() 
          which can be configured with base paths and handles missing files.
"""

import re
from typing import Dict, Optional, Union
from pathlib import Path


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
    elif re.search(r'_ch\d+_t\d{3,4}$', entity_id) or re.search(r'_t\d{3,4}$', entity_id):
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
    """Parse image_id: {video_id}_ch{CHANNEL}_t{FRAME} or legacy {video_id}_t{FRAME}."""
    # Try new format with channel first
    match = re.search(r'_ch(\d+)_t(\d{3,4})$', image_id)
    if match:
        channel = match.group(1)
        frame_number = match.group(2)
        video_id = image_id[:match.start()]
    else:
        # Fallback to legacy format without channel
        match = re.search(r'_t(\d{3,4})$', image_id)
        if not match:
            raise ValueError(f"Invalid image_id: {image_id}")
        channel = "0"  # Default channel for legacy format
        frame_number = match.group(1)
        video_id = image_id[:match.start()]
    
    # Parse video_id recursively
    video_components = _parse_backwards_video(video_id)
    
    return {
        **video_components,
        "channel": channel,
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


# ========== BUILDER FUNCTIONS ==========

def build_image_id(video_id: str, frame_number: int, channel: int = 0) -> str:
    """Build image ID with channel + 't' prefix from video_id, frame number, and channel."""
    return f"{video_id}_ch{channel:02d}_t{frame_number:04d}"


def build_video_id(experiment_id: str, well_id: str) -> str:
    """Build video ID from experiment_id and well_id."""
    well_normalized = normalize_well_id(well_id)
    return f"{experiment_id}_{well_normalized}"


def build_embryo_id(video_id: str, embryo_number: int) -> str:
    """Build embryo ID from video_id and embryo number."""
    return f"{video_id}_e{embryo_number:02d}"


def build_snip_id(embryo_id: str, frame_number: int, use_s_prefix: bool = True) -> str:
    """Build snip ID from embryo_id and frame number."""
    prefix = "_s" if use_s_prefix else "_"
    return f"{embryo_id}{prefix}{frame_number:04d}"


# ========== NORMALIZATION FUNCTIONS ==========

def normalize_frame_number(frame_str: str) -> int:
    """Convert frame string to integer."""
    return int(frame_str)


def normalize_well_id(well_id: str) -> str:
    """Standardize well format to A01, B12, etc."""
    # Remove any existing prefix/suffix and extract letter + number
    match = re.search(r'([A-H])(\d{1,2})', well_id.upper())
    if not match:
        raise ValueError(f"Invalid well_id format: {well_id}")
    
    letter = match.group(1)
    number = int(match.group(2))
    return f"{letter}{number:02d}"


def normalize_embryo_number(embryo_str: str) -> int:
    """Convert embryo string to integer."""
    return int(embryo_str)


# ========== PATH UTILITIES ==========

def get_image_filename_from_id(image_id: str, extension: str = "jpg") -> str:
    """Convert image_id to disk filename.

    Current (simplest) behavior: always use the full `image_id` as the
    filename on disk: ``{image_id}.{extension}``.
    """
    return f"{image_id}.{extension}"


def get_image_id_path(processed_jpg_images_dir: Union[str, Path], image_id: str, extension: str = "jpg") -> Path:
    """
    Construct full image path from processed directory and image_id.
    
    Args:
        processed_jpg_images_dir: Directory containing processed images (from metadata)
        image_id: Full image ID (e.g., "20250612_30hpf_ctrl_atf6_F11_ch00_t0000")
        extension: File extension (default: "jpg")
    
    Returns:
        Full path: processed_jpg_images_dir / {image_id}.{extension}
    
    Example:
        >>> get_image_id_path("/data/exp/images/video_A01", "20240411_A01_ch00_t0042")
        Path('/data/exp/images/video_A01/20240411_A01_ch00_t0042.jpg')
    """
    return Path(processed_jpg_images_dir) / get_image_filename_from_id(image_id, extension)


def get_relative_image_path(image_id: str, extension: str = "jpg") -> str:
    """Get relative path from experiment root: images/video_id/{image_id}.jpg"""
    parsed = parse_entity_id(image_id)
    video_id = parsed["video_id"]
    filename = get_image_filename_from_id(image_id, extension)
    return f"images/{video_id}/{filename}"


def get_relative_video_path(video_id: str) -> str:
    """Get relative video directory path: images/video_id/"""
    return f"images/{video_id}"


def build_image_path_from_base(image_id: str, base_path: str, extension: str = "jpg") -> str:
    """Build full image path from base directory (uses `{image_id}.jpg`)."""
    parsed = parse_entity_id(image_id)
    experiment_id = parsed["experiment_id"]
    relative_path = get_relative_image_path(image_id, extension)
    return f"{base_path}/{experiment_id}/{relative_path}"


def build_video_path_from_base(video_id: str, base_path: str) -> str:
    """Build full video directory path from base directory (backward compatibility)."""
    experiment_id = extract_experiment_id(video_id)
    relative_path = get_relative_video_path(video_id)
    return f"{base_path}/{experiment_id}/{relative_path}"


def get_experiment_relative_path(experiment_id: str) -> str:
    """Get relative experiment directory path."""
    return experiment_id


# ========== CONVERSION UTILITIES ==========

def disk_filename_to_image_id(video_id: str, filename: str) -> str:
        """Convert an on-disk filename to an image_id.

        With the new convention filenames are expected to be full image IDs
        (``{image_id}.jpg``). This helper does a best-effort conversion:
        - If the stripped filename starts with the provided ``video_id`` and
            matches the image pattern (with or without channel), it is returned unchanged.
        - Otherwise, if the filename is numeric (NNNN) it will be converted to
            ``{video_id}_ch00_tNNNN`` for backwards compatibility.
        - Otherwise the stripped filename is returned (caller should validate).
        """
        name = filename.replace('.jpg', '')
        if name.startswith(video_id) and (re.search(r'_ch\d+_t\d{3,4}$', name) or re.search(r'_t\d{3,4}$', name)):
                return name
        # numeric fallback - use default channel 0
        if re.fullmatch(r'\d{1,4}', name):
                return build_image_id(video_id, int(name), channel=0)
        return name


def image_id_to_disk_filename(image_id: str) -> str:
    """Convert image_id to disk filename: ``{image_id}.jpg``."""
    return f"{image_id}.jpg"


def get_image_id_from_filename(video_id: str, filename: str, extension: str = "jpg") -> str:
    """Return the image_id represented by an on-disk filename.

    With the new naming convention the filename should be the full image_id
    (``{image_id}.jpg``) and this function will simply strip the extension
    and return the name. For robustness it will still accept legacy numeric
    filenames and convert them to ``{video_id}_ch00_tNNNN`` (default channel 0).
    """
    name = filename.replace(f'.{extension}', '')
    # Check for channel-inclusive format or legacy format
    if name.startswith(video_id) and (re.search(r'_ch\d+_t\d{3,4}$', name) or re.search(r'_t\d{3,4}$', name)):
        return name
    # numeric fallback - use default channel 0
    if re.fullmatch(r'\d{1,4}', name):
        return build_image_id(video_id, int(name), channel=0)
    return name


# ========== VALIDATION HELPERS ==========

def is_valid_experiment_id(experiment_id: str) -> bool:
    """Check if string could be a valid experiment ID."""
    # Basic check: should not end with well/frame/embryo patterns
    return not re.search(r'_([A-H]\d{2}|[te]\d+|s?\d{3,4})$', experiment_id)


def is_valid_well_id(well_id: str) -> bool:
    """Check if string is a valid well ID format."""
    return bool(re.match(r'^[A-H](0?[1-9]|1[0-2])$', well_id.upper()))


def is_valid_frame_number(frame_number: int) -> bool:
    """Check if frame number is in valid range."""
    return 0 <= frame_number <= 9999


# ========== BATCH OPERATIONS ==========

def parse_multiple_ids(entity_ids: list) -> Dict[str, list]:
    """Parse multiple IDs and group by type."""
    results = {
        "experiments": [],
        "videos": [],
        "images": [],
        "embryos": [],
        "snips": [],
        "invalid": []
    }
    
    for entity_id in entity_ids:
        try:
            parsed = parse_entity_id(entity_id)
            entity_type = parsed["entity_type"]
            if entity_type == "experiment":
                results["experiments"].append(parsed)
            elif entity_type == "video":
                results["videos"].append(parsed)
            elif entity_type == "image":
                results["images"].append(parsed)
            elif entity_type == "embryo":
                results["embryos"].append(parsed)
            elif entity_type == "snip":
                results["snips"].append(parsed)
        except ValueError:
            results["invalid"].append(entity_id)
    
    return results


def extract_unique_experiments(entity_ids: list) -> list:
    """Extract unique experiment IDs from a list of any entity IDs."""
    experiments = set()
    for entity_id in entity_ids:
        try:
            exp_id = extract_experiment_id(entity_id)
            experiments.add(exp_id)
        except ValueError:
            continue
    return sorted(list(experiments))


def extract_unique_videos(entity_ids: list) -> list:
    """Extract unique video IDs from a list of any entity IDs."""
    videos = set()
    for entity_id in entity_ids:
        try:
            video_id = extract_video_id(entity_id)
            if video_id:
                videos.add(video_id)
        except ValueError:
            continue
    return sorted(list(videos))


# ========== ENTITY GROUPING UTILITIES ==========

def group_by(entity_ids: list, group_type: str) -> dict:
    """
    Group entity IDs by hierarchical context.
    
    Args:
        entity_ids: List of entity IDs to group
        group_type: Type to group by ("experiment", "video", "embryo", etc.)
        
    Returns:
        Dict mapping group_id to list of entity_ids
        
    Examples:
        # Group images by video
        group_by(image_ids, "video")
        # {"20240411_A01": ["20240411_A01_t0000", "20240411_A01_t0001"], ...}
        
        # Group snips by embryo  
        group_by(snip_ids, "embryo")
        # {"20240411_A01_e01": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001"], ...}
    """
    grouped = {}
    group_key = f"{group_type}_id"
    
    for entity_id in entity_ids:
        try:
            parsed = parse_entity_id(entity_id)
            parent_id = parsed.get(group_key)
            
            if parent_id:
                if parent_id not in grouped:
                    grouped[parent_id] = []
                grouped[parent_id].append(entity_id)
        except ValueError:
            # Skip invalid entity IDs
            continue
    
    return grouped


def group_entities_by_hierarchy(entity_ids: list) -> dict:
    """
    Group entities by all hierarchy levels simultaneously.
    
    Returns:
        Dict with keys: experiments, videos, images, embryos, snips
    """
    return {
        "experiments": group_by(entity_ids, "experiment"),
        "videos": group_by(entity_ids, "video"),
        "images": group_by(entity_ids, "image"),  # Will be empty for most cases
        "embryos": group_by(entity_ids, "embryo"),
        "snips": {}  # Snips are leaf nodes, no grouping needed
    }


def get_entities_by_type(entity_ids: list) -> dict:
    """
    Categorize entity IDs by their entity type.
    
    Returns:
        Dict with lists: experiments, videos, images, embryos, snips
    """
    categorized = {
        "experiments": [],
        "videos": [], 
        "images": [],
        "embryos": [],
        "snips": []
    }
    
    for entity_id in entity_ids:
        try:
            entity_type = get_entity_type(entity_id)
            if entity_type in categorized:
                categorized[entity_type].append(entity_id)
        except ValueError:
            continue
    
    return categorized


# ========== BACKWARD COMPATIBILITY ==========

def get_image_path_from_id(image_id: str, base_path: str = "raw_data_organized") -> str:
    """
    DEPRECATED: Use ExperimentMetadata.get_image_path() or build_image_path_from_base().
    Convert image_id to disk file path (no 't' prefix on disk).
    """
    return build_image_path_from_base(image_id, base_path, "jpg")


def get_video_directory_path(video_id: str, base_path: str = "raw_data_organized") -> str:
    """
    DEPRECATED: Use ExperimentMetadata.get_video_directory_path() or build_video_path_from_base().
    Get video directory path from video_id.
    """
    return build_video_path_from_base(video_id, base_path)
