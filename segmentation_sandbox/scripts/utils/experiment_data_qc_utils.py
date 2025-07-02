#!/usr/bin/env python
"""
experiment_data_qc_utils.py

Utilities for managing hierarchical experiment data quality control flags for the MorphSeq pipeline.
Uses a JSON structure inspired by COCO and embryo_metadata.json for better organization
and maintenance of QC flags across experiment, video, image, and embryo levels.

IMPORTANT DESIGN PHILOSOPHY:
===========================
Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
This ensures:
1. Consistency across all tools and users
2. Manual control over what flags are allowed
3. No hardcoded categories in Python code
4. Easy addition of new categories without code changes

To add new QC flag categories:
1. Edit the "valid_qc_flag_categories" section in experiment_data_qc.json
2. Add new flags with descriptions at the appropriate level
3. The system will automatically validate against these categories

Flag Integrity:
- By default, all operations check flag integrity against JSON-defined categories
- Invalid flags will cause errors and suggest cleanup procedures
- Use clean_invalid_flags() to remove invalid flags automatically

Expected Directory Structure:
- raw_data_organized/: Contains experiment_metadata.json
- quality_control/: Contains experiment_data_qc.json (created automatically)

Usage Example:
    # Initialize QC structure from experiment metadata
    initialize_qc_structure_from_metadata(
        quality_control_dir="/path/to/quality_control",
        experiment_metadata_path="/path/to/raw_data_organized/experiment_metadata.json"
    )
    
    # Flag an image with blur
    flag_image(
        quality_control_dir="/path/to/quality_control",
        image_id="20241215_A01_t001",
        qc_flag="BLUR",
        author="automatic",
        notes="Low variance of Laplacian: 45"
    )

QC JSON Structure:
{
    "valid_qc_flag_categories": {
        "experiment_level": {
            "PROTOCOL_DEVIATION": "Deviation from standard imaging protocol"
        },
        "video_level": {},
        "image_level": {
            "BLUR": "Image is blurry (low variance of Laplacian)",
            "DRY_WELL": "Well dried out during imaging",
            "CORRUPT": "Cannot read/process image"
        },
        "embryo_level": {
            "DEAD_EMBRYO": "Embryo appears dead",
            "EMBRYO_NOT_DETECTED": "No embryo detected in expected location"
        }
    },
    "experiments": {
        "20241215": {
            "flags": ["PROTOCOL_DEVIATION"],
            "authors": ["mcolon"],
            "notes": ["Manual review - protocol issues throughout"],
            "videos": {
                "20241215_A01": {
                    "flags": [],
                    "authors": [],
                    "notes": [],
                    "images": {
                        "20241215_A01_t001": {
                            "flags": ["BLUR"],
                            "authors": ["automatic"],
                            "notes": ["Automatic: blur_score=45 < threshold=100"]
                        }
                    },
                    "embryos": {
                        "20241215_A01_t001_e01": {
                            "flags": ["DEAD_EMBRYO"],
                            "authors": ["expert_reviewer"],
                            "notes": ["Manual inspection - no movement detected"]
                        }
                    }
                }
            }
        }
    }
}

DESIGN PHILOSOPHY:
Valid QC flag categories are ONLY stored in the "valid_qc_flag_categories" section of the JSON file.
The Python code does NOT define or hardcode any valid flag categories - it only uses what's in the JSON.
To add new flag categories, manually edit the JSON file's "valid_qc_flag_categories" section.
Flag integrity is checked by default and will raise errors if invalid flags are found.
Use clean_invalid_flags() utility to remove invalid flags or manually edit the JSON.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
# Default QC flag categories - ONLY used when creating new JSON files
# IMPORTANT: Valid QC flag categories are stored in the JSON file itself as the single source of truth
# These defaults are only used to bootstrap new JSON files and should NOT be used for validation
DEFAULT_QC_FLAG_CATEGORIES = {
    "experiment_level": {
        "PROTOCOL_DEVIATION": "Deviation from standard imaging protocol"
    },
    "video_level": {
    },
    "image_level": {
        "BLUR": "Image is blurry (low variance of Laplacian)",
        "DRY_WELL": "Well dried out during imaging",
        "CORRUPT": "Cannot read/process image"
    },
    "embryo_level": {
        "DEAD_EMBRYO": "Embryo appears dead",
        "EMBRYO_NOT_DETECTED": "No embryo detected in expected location"
    }
}

def get_qc_json_path(quality_control_dir: Union[str, Path]) -> Path:
    """Get the path to the experiment data QC JSON file."""
    return Path(quality_control_dir) / "experiment_data_qc.json"

def load_qc_data(quality_control_dir: Union[str, Path]) -> Dict:
    """Load QC data from JSON file."""
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if not qc_json_path.exists():
        # Create empty structure with default categories
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }
    
    try:
        with open(qc_json_path, 'r') as f:
            qc_data = json.load(f)
        
        # Ensure structure is complete
        if "valid_qc_flag_categories" not in qc_data:
            qc_data["valid_qc_flag_categories"] = DEFAULT_QC_FLAG_CATEGORIES.copy()
        if "experiments" not in qc_data:
            qc_data["experiments"] = {}
            
        return qc_data
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load QC data from {qc_json_path}: {e}")
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }

def save_qc_data(qc_data: Dict, quality_control_dir: Union[str, Path]) -> None:
    """Save QC data to JSON file."""
    qc_json_path = get_qc_json_path(quality_control_dir)
    qc_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(qc_json_path, 'w') as f:
        json.dump(qc_data, f, indent=2)

def get_valid_qc_flag_categories(quality_control_dir: Union[str, Path]) -> Dict:
    """
    Get valid QC flag categories from the JSON file.
    
    Args:
        quality_control_dir: Directory containing the QC JSON file
        
    Returns:
        Dictionary of valid flag categories by level
        
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This function retrieves them for display or validation purposes.
    """
    qc_data = load_qc_data(quality_control_dir)
    return qc_data.get("valid_qc_flag_categories", {})

def validate_qc_flag(qc_flag: str, level: str, qc_data: Dict) -> bool:
    """Validate that a QC flag is valid for the given level."""
    valid_flags = qc_data.get("valid_qc_flag_categories", {}).get(f"{level}_level", {})
    return qc_flag in valid_flags

def check_flag_integrity(qc_data: Dict, fix_invalid: bool = False, verbose: bool = True) -> Dict:
    """
    Check integrity of QC flags against valid categories stored in the JSON.
    
    Args:
        qc_data: QC data dictionary
        fix_invalid: If True, remove invalid flags. If False, raise error on invalid flags.
        verbose: Print detailed information about found issues
        
    Returns:
        Dict with integrity check results and optionally cleaned data
        
    Raises:
        ValueError: If invalid flags found and fix_invalid=False
        
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This function enforces that only flags defined in "valid_qc_flag_categories" are allowed.
        Categories can only be added manually by editing the JSON file directly.
    """
    valid_categories = qc_data.get("valid_qc_flag_categories", {})
    issues = []
    fixed_count = 0
    
    # Check each level
    for exp_id, exp_data in qc_data.get("experiments", {}).items():
        # Check experiment-level flags
        for flag in exp_data.get("flags", []):
            if flag not in valid_categories.get("experiment_level", {}):
                issue = f"Invalid experiment flag '{flag}' in experiment {exp_id}"
                issues.append(issue)
                if fix_invalid:
                    exp_data["flags"].remove(flag)
                    # Also remove corresponding author/notes
                    try:
                        flag_idx = exp_data["flags"].index(flag)
                        if len(exp_data.get("authors", [])) > flag_idx:
                            exp_data["authors"].pop(flag_idx)
                        if len(exp_data.get("notes", [])) > flag_idx:
                            exp_data["notes"].pop(flag_idx)
                    except (ValueError, IndexError):
                        pass
                    fixed_count += 1
        
        # Check video-level flags
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            for flag in vid_data.get("flags", []):
                if flag not in valid_categories.get("video_level", {}):
                    issue = f"Invalid video flag '{flag}' in video {vid_id}"
                    issues.append(issue)
                    if fix_invalid:
                        vid_data["flags"].remove(flag)
                        fixed_count += 1
            
            # Check image-level flags
            for img_id, img_data in vid_data.get("images", {}).items():
                for flag in img_data.get("flags", []):
                    if flag not in valid_categories.get("image_level", {}):
                        issue = f"Invalid image flag '{flag}' in image {img_id}"
                        issues.append(issue)
                        if fix_invalid:
                            img_data["flags"].remove(flag)
                            fixed_count += 1
            
            # Check embryo-level flags
            for emb_id, emb_data in vid_data.get("embryos", {}).items():
                for flag in emb_data.get("flags", []):
                    if flag not in valid_categories.get("embryo_level", {}):
                        issue = f"Invalid embryo flag '{flag}' in embryo {emb_id}"
                        issues.append(issue)
                        if fix_invalid:
                            emb_data["flags"].remove(flag)
                            fixed_count += 1
    
    # Report results
    if verbose:
        if issues:
            print(f"🚨 Flag integrity check found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more issues")
            
            if fix_invalid:
                print(f"✅ Fixed {fixed_count} invalid flags")
            else:
                print(f"💡 Run with fix_invalid=True to automatically remove invalid flags")
        else:
            print("✅ All flags are valid!")
    
    # Raise error if issues found and not fixing
    if issues and not fix_invalid:
        raise ValueError(
            f"Found {len(issues)} invalid QC flags. "
            f"Valid categories are defined in the JSON file under 'valid_qc_flag_categories'. "
            f"Either fix these manually or run check_flag_integrity() with fix_invalid=True"
        )
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "fixed_count": fixed_count if fix_invalid else 0,
        "qc_data": qc_data if fix_invalid else None
    }

def clean_invalid_flags(quality_control_dir: Union[str, Path], backup: bool = True) -> Dict:
    """
    Clean invalid flags from QC data file.
    
    Args:
        quality_control_dir: Directory containing QC JSON file
        backup: Create backup before cleaning
        
    Returns:
        Cleaning results
    """
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if backup and qc_json_path.exists():
        backup_path = qc_json_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        import shutil
        shutil.copy2(qc_json_path, backup_path)
        print(f"📋 Created backup: {backup_path}")
    
    qc_data = load_qc_data(quality_control_dir)
    result = check_flag_integrity(qc_data, fix_invalid=True, verbose=True)
    
    if result["fixed_count"] > 0:
        save_qc_data(qc_data, quality_control_dir)
        print(f"💾 Saved cleaned QC data")
    
    return result

def list_valid_flags(quality_control_dir: Union[str, Path]) -> None:
    """Display all valid QC flag categories from the JSON file."""
    qc_data = load_qc_data(quality_control_dir)
    valid_categories = qc_data.get("valid_qc_flag_categories", {})
    
    print("📋 Valid QC Flag Categories (from JSON file):")
    print("=" * 60)
    
    for level, flags in valid_categories.items():
        print(f"\n🏷️  {level.replace('_', ' ').title()}:")
        if flags:
            for flag, description in flags.items():
                print(f"   • {flag}: {description}")
        else:
            print("   (No flags defined for this level)")
    
    print(f"\n💡 To add new flags, edit the 'valid_qc_flag_categories' section in:")
    print(f"   {get_qc_json_path(quality_control_dir)}")

def show_valid_qc_flag_categories(quality_control_dir: Union[str, Path]) -> None:
    """
    Display all valid QC flag categories from the JSON file.
    
    Args:
        quality_control_dir: Directory containing the QC JSON file
        
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This function displays them for user reference when adding flags.
    """
    valid_categories = get_valid_qc_flag_categories(quality_control_dir)
    
    print("📋 Valid QC Flag Categories (from JSON file):")
    print("=" * 60)
    
    for level, flags in valid_categories.items():
        print(f"\n🏷️  {level.replace('_', ' ').title()}:")
        if flags:
            for flag, description in flags.items():
                print(f"   • {flag}: {description}")
        else:
            print("   • (No flags defined at this level)")
    
    print("\n💡 To add new flag categories, edit the 'valid_qc_flag_categories'")
    print("   section in the experiment_data_qc.json file manually.")

def initialize_qc_structure_from_metadata(
    quality_control_dir: Union[str, Path],
    experiment_metadata_path: Union[str, Path],
    overwrite: bool = False,
    validate_flags: bool = True
) -> Dict:
    """
    Initialize QC structure from experiment metadata.
    Creates entries for all experiments, videos, and images found in metadata.
    
    Args:
        quality_control_dir: Directory where the QC JSON file will be stored
        experiment_metadata_path: Path to the experiment_metadata.json file
        overwrite: Whether to overwrite existing QC data
        validate_flags: Whether to validate existing flags against valid categories
    
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This ensures consistency and allows manual management of flag categories.
        By default, flag integrity is checked to prevent invalid flags from persisting.
    """
    
    # Load existing QC data
    qc_data = load_qc_data(quality_control_dir) if not overwrite else {
        "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
        "experiments": {}
    }
    
    # Load experiment metadata
    if not Path(experiment_metadata_path).exists():
        print(f"Warning: Experiment metadata not found at {experiment_metadata_path}")
        return qc_data
    
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    experiments_data = metadata.get('experiments', {})
    added_count = {"experiments": 0, "videos": 0, "images": 0}
    
    for experiment_id, experiment_data in experiments_data.items():
        # Initialize experiment if not exists
        if experiment_id not in qc_data["experiments"]:
            qc_data["experiments"][experiment_id] = {
                "flags": [],
                "authors": [],
                "notes": [],
                "videos": {}
            }
            added_count["experiments"] += 1
        
        experiment_qc = qc_data["experiments"][experiment_id]
        
        # Process videos in this experiment
        videos_data = experiment_data.get('videos', {})
        for video_id, video_data in videos_data.items():
            # Initialize video if not exists
            if video_id not in experiment_qc["videos"]:
                experiment_qc["videos"][video_id] = {
                    "flags": [],
                    "authors": [],
                    "notes": [],
                    "images": {},
                    "embryos": {}
                }
                added_count["videos"] += 1
            
            video_qc = experiment_qc["videos"][video_id]
            
            # Process images in this video
            image_ids = video_data.get('image_ids', [])
            for image_id in image_ids:
                if image_id not in video_qc["images"]:
                    video_qc["images"][image_id] = {
                        "flags": [],
                        "authors": [],
                        "notes": []
                    }
                    added_count["images"] += 1
    
    # Check flag integrity if requested
    if validate_flags:
        print("🔍 Checking flag integrity...")
        try:
            check_flag_integrity(qc_data, fix_invalid=False, verbose=True)
        except ValueError as e:
            print(f"\n🚨 FLAG INTEGRITY ERROR: {e}")
            print("\n🛠️  To fix this, you can:")
            print("   1. Run clean_invalid_flags() to automatically remove invalid flags")
            print("   2. Manually edit the QC JSON file to remove/rename invalid flags")
            print("   3. Add missing categories to 'valid_qc_flag_categories' in the JSON")
            raise
    
    # Save updated QC data
    save_qc_data(qc_data, quality_control_dir)
    
    print(f"QC structure initialized: {added_count['experiments']} experiments, "
          f"{added_count['videos']} videos, {added_count['images']} images added")
    
    return qc_data

def add_qc_flag(
    quality_control_dir: Union[str, Path],
    level: str,  # "experiment", "video", "image", or "embryo"
    entity_id: str,  # experiment_id, video_id, image_id, or embryo_id
    qc_flag: str,
    author: str,
    notes: str = "",
    parent_ids: Optional[Dict[str, str]] = None,  # For nested entities
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """
    Add a QC flag to an entity at the specified level.
    
    Args:
        quality_control_dir: Directory containing the QC JSON file
        level: "experiment", "video", "image", or "embryo"
        entity_id: ID of the entity to flag
        qc_flag: QC flag to add
        author: Who added the flag
        notes: Optional notes
        parent_ids: For nested entities (e.g., {"experiment_id": "20241215", "video_id": "20241215_A01"})
        overwrite: Whether to replace existing flags
        write_directly: If True, save data immediately. If False (default), defer saving for better performance with large files.
    """
    qc_data = load_qc_data(quality_control_dir)
    
    # Validate flag
    if not validate_qc_flag(qc_flag, level, qc_data):
        valid_flags = list(qc_data["valid_qc_flag_categories"].get(f"{level}_level", {}).keys())
        raise ValueError(f"Invalid {level}_level flag '{qc_flag}'. Valid flags: {valid_flags}")
    
    # Navigate to the correct entity
    if level == "experiment":
        if entity_id not in qc_data["experiments"]:
            qc_data["experiments"][entity_id] = {
                "flags": [], "authors": [], "notes": [], "videos": {}
            }
        entity = qc_data["experiments"][entity_id]
        
    elif level == "video":
        if not parent_ids or "experiment_id" not in parent_ids:
            raise ValueError("video level flags require parent_ids with experiment_id")
        exp_id = parent_ids["experiment_id"]
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {
                "flags": [], "authors": [], "notes": [], "videos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][entity_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        entity = qc_data["experiments"][exp_id]["videos"][entity_id]
        
    elif level == "image":
        if not parent_ids or "experiment_id" not in parent_ids or "video_id" not in parent_ids:
            raise ValueError("image level flags require parent_ids with experiment_id and video_id")
        exp_id = parent_ids["experiment_id"]
        vid_id = parent_ids["video_id"]
        
        # Ensure structure exists
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {"flags": [], "authors": [], "notes": [], "videos": {}}
        if vid_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"][vid_id]["images"]:
            qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id] = {
                "flags": [], "authors": [], "notes": []
            }
        entity = qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id]
        
    elif level == "embryo":
        if not parent_ids or "experiment_id" not in parent_ids or "video_id" not in parent_ids:
            raise ValueError("embryo level flags require parent_ids with experiment_id and video_id")
        exp_id = parent_ids["experiment_id"]
        vid_id = parent_ids["video_id"]
        
        # Ensure structure exists
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {"flags": [], "authors": [], "notes": [], "videos": {}}
        if vid_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id] = {
                "flags": [], "authors": [], "notes": []
            }
        entity = qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id]
    
    else:
        raise ValueError(f"Invalid level '{level}'. Must be experiment, video, image, or embryo")
    
    # Add or update flag
    if overwrite:
        entity["flags"] = [qc_flag]
        entity["authors"] = [author]
        entity["notes"] = [notes]
    else:
        if qc_flag not in entity["flags"]:
            entity["flags"].append(qc_flag)
            entity["authors"].append(author)
            entity["notes"].append(notes)
    
    # Save data only if write_directly is True
    if write_directly:
        save_qc_data(qc_data, quality_control_dir)
    return qc_data

def get_qc_flags(
    quality_control_dir: Union[str, Path],
    level: str,
    entity_id: str,
    parent_ids: Optional[Dict[str, str]] = None
) -> Dict[str, List]:
    """Get QC flags for a specific entity."""
    qc_data = load_qc_data(quality_control_dir)
    
    try:
        if level == "experiment":
            entity = qc_data["experiments"][entity_id]
        elif level == "video":
            exp_id = parent_ids["experiment_id"]
            entity = qc_data["experiments"][exp_id]["videos"][entity_id]
        elif level == "image":
            exp_id = parent_ids["experiment_id"]
            vid_id = parent_ids["video_id"]
            entity = qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id]
        elif level == "embryo":
            exp_id = parent_ids["experiment_id"]
            vid_id = parent_ids["video_id"]
            entity = qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id]
        else:
            raise ValueError(f"Invalid level: {level}")
            
        return {
            "flags": entity.get("flags", []),
            "authors": entity.get("authors", []),
            "notes": entity.get("notes", [])
        }
    except KeyError:
        return {"flags": [], "authors": [], "notes": []}

def get_qc_summary(quality_control_dir: Union[str, Path]) -> Dict:
    """Generate a summary of QC flags across all levels."""
    qc_data = load_qc_data(quality_control_dir)
    
    summary = {
        "experiment_level": {},
        "video_level": {},
        "image_level": {},
        "embryo_level": {}
    }
    
    for exp_id, exp_data in qc_data.get("experiments", {}).items():
        # Count experiment flags
        for flag in exp_data.get("flags", []):
            summary["experiment_level"][flag] = summary["experiment_level"].get(flag, 0) + 1
        
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            # Count video flags
            for flag in vid_data.get("flags", []):
                summary["video_level"][flag] = summary["video_level"].get(flag, 0) + 1
            
            # Count image flags
            for img_id, img_data in vid_data.get("images", {}).items():
                for flag in img_data.get("flags", []):
                    summary["image_level"][flag] = summary["image_level"].get(flag, 0) + 1
            
            # Count embryo flags
            for emb_id, emb_data in vid_data.get("embryos", {}).items():
                for flag in emb_data.get("flags", []):
                    summary["embryo_level"][flag] = summary["embryo_level"].get(flag, 0) + 1
    
    return summary

def parse_image_id(image_id: str) -> Dict[str, str]:
    """Parse image ID to extract experiment_id and video_id."""
    # Expected format: experiment_id_well_id_timepoint (e.g., "20241215_A01_t001")
    parts = image_id.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id format: {image_id}")
    
    experiment_id = parts[0]
    well_id = parts[1]
    video_id = f"{experiment_id}_{well_id}"
    
    return {
        "experiment_id": experiment_id,
        "video_id": video_id,
        "well_id": well_id
    }

def parse_embryo_id(embryo_id: str) -> Dict[str, str]:
    """Parse embryo ID to extract parent IDs."""
    # Expected format: experiment_id_well_id_timepoint_embryo_num (e.g., "20241215_A01_t001_e01")
    parts = embryo_id.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid embryo_id format: {embryo_id}")
    
    experiment_id = parts[0]
    well_id = parts[1]
    video_id = f"{experiment_id}_{well_id}"
    image_id = f"{experiment_id}_{well_id}_{parts[2]}"
    
    return {
        "experiment_id": experiment_id,
        "video_id": video_id,
        "image_id": image_id,
        "well_id": well_id
    }

# Convenience functions for common QC operations
def flag_image(
    quality_control_dir: Union[str, Path],
    image_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an image."""
    parent_ids = parse_image_id(image_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="image",
        entity_id=image_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_video(
    quality_control_dir: Union[str, Path],
    video_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag a video."""
    # Extract experiment_id from video_id (format: experiment_id_well_id)
    parts = video_id.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid video_id format: {video_id}")
    
    experiment_id = parts[0]
    parent_ids = {"experiment_id": experiment_id}
    
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="video",
        entity_id=video_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_experiment(
    quality_control_dir: Union[str, Path],
    experiment_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an experiment."""
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="experiment",
        entity_id=experiment_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_embryo(
    quality_control_dir: Union[str, Path],
    embryo_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an embryo."""
    parent_ids = parse_embryo_id(embryo_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="embryo",
        entity_id=embryo_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def save_qc_batch(qc_data: Dict, quality_control_dir: Union[str, Path]) -> None:
    """
    Save QC data after batch operations.
    
    Use this function when you've been adding flags with write_directly=False
    and want to save all changes at once for better performance.
    
    Args:
        qc_data: The QC data dictionary to save
        quality_control_dir: Directory where the QC JSON file will be stored
    """
    save_qc_data(qc_data, quality_control_dir)
    print(f"💾 Saved QC data to {get_qc_json_path(quality_control_dir)}")

# Legacy compatibility functions (for backward compatibility with CSV-based system)
def flag_qc(
    quality_control_dir: Union[str, Path],
    image_ids: List[str],
    qc_flag: str,
    annotator: str = 'manual',
    notes: str = '',
    overwrite: bool = False
) -> None:
    """Legacy compatibility function for flagging images."""
    for image_id in image_ids:
        flag_image(
            quality_control_dir=quality_control_dir,
            image_id=image_id,
            qc_flag=qc_flag,
            author=annotator,
            notes=notes,
            overwrite=overwrite
        )

def check_existing_qc(
    quality_control_dir: Union[str, Path],
    image_ids: List[str]
) -> Dict[str, List[str]]:
    """Legacy compatibility function to check existing QC flags for images."""
    result = {}
    for image_id in image_ids:
        try:
            parent_ids = parse_image_id(image_id)
            qc_info = get_qc_flags(quality_control_dir, "image", image_id, parent_ids)
            result[image_id] = qc_info["flags"]
        except (ValueError, KeyError):
            result[image_id] = []
    return result
