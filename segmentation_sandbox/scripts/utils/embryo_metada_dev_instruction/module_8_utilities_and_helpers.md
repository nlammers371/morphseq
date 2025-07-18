# Module 8: Utilities and Helpers

## Overview
This module contains utility functions, data converters, helper methods, and common operations used throughout the EmbryoMetadata system.

## Path and File Utilities

```python
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import re

def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and convert path to Path object.
    
    Args:
        path: Path string or Path object
        must_exist: Whether path must exist
    
    Returns:
        Path object
    
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return path

def create_timestamped_backup(file_path: Path, 
                            backup_dir: Optional[Path] = None) -> Path:
    """
    Create timestamped backup of a file.
    
    Args:
        file_path: File to backup
        backup_dir: Directory for backup (default: same as file)
    
    Returns:
        Path to backup file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}.backup.{timestamp}{file_path.suffix}"
    
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / backup_name
    else:
        backup_path = file_path.parent / backup_name
    
    shutil.copy2(file_path, backup_path)
    return backup_path

def load_json(file_path: Path, 
              create_if_missing: bool = False,
              default_content: Optional[Dict] = None) -> Dict:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        create_if_missing: Create file if it doesn't exist
        default_content: Default content for new file
    
    Returns:
        Loaded JSON data
    """
    if not file_path.exists():
        if create_if_missing:
            content = default_content or {}
            save_json(content, file_path)
            return content
        else:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json(data: Dict, file_path: Path, 
              create_backup: bool = True,
              indent: int = 2) -> None:
    """
    Save data to JSON file with atomic write.
    
    Args:
        data: Data to save
        file_path: Target file path
        create_backup: Create backup of existing file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if requested
    if create_backup and file_path.exists():
        create_timestamped_backup(file_path)
    
    # Atomic write
    temp_path = file_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=indent)
        temp_path.replace(file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
```

## ID Parsing and Validation

```python
class IdParser:
    """Utilities for parsing and validating IDs."""
    
    # ID format patterns
    PATTERNS = {
        "experiment": re.compile(r'^(\d{8})$'),
        "video": re.compile(r'^(\d{8})_([A-H]\d{2})$'),
        "image": re.compile(r'^(\d{8})_([A-H]\d{2})_(\d{4})$'),
        "embryo": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})$'),
        "snip": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})_(\d{4})$')
    }
    
    @staticmethod
    def parse_id(entity_id: str) -> Dict[str, str]:
        """
        Parse any ID type and extract components.
        
        Args:
            entity_id: ID to parse
        
        Returns:
            Dict with parsed components and type
        """
        # Try each pattern
        for id_type, pattern in IdParser.PATTERNS.items():
            match = pattern.match(entity_id)
            if match:
                groups = match.groups()
                
                if id_type == "experiment":
                    return {
                        "type": "experiment",
                        "experiment_id": groups[0]
                    }
                
                elif id_type == "video":
                    return {
                        "type": "video",
                        "experiment_id": groups[0],
                        "well_id": groups[1],
                        "video_id": entity_id
                    }
                
                elif id_type == "image":
                    return {
                        "type": "image",
                        "experiment_id": groups[0],
                        "well_id": groups[1],
                        "frame": groups[2],
                        "video_id": f"{groups[0]}_{groups[1]}",
                        "image_id": entity_id
                    }
                
                elif id_type == "embryo":
                    return {
                        "type": "embryo",
                        "experiment_id": groups[0],
                        "well_id": groups[1],
                        "embryo_num": groups[2],
                        "video_id": f"{groups[0]}_{groups[1]}",
                        "embryo_id": entity_id
                    }
                
                elif id_type == "snip":
                    return {
                        "type": "snip",
                        "experiment_id": groups[0],
                        "well_id": groups[1],
                        "embryo_num": groups[2],
                        "frame": groups[3],
                        "video_id": f"{groups[0]}_{groups[1]}",
                        "embryo_id": f"{groups[0]}_{groups[1]}_e{groups[2]}",
                        "image_id": f"{groups[0]}_{groups[1]}_{groups[3]}",
                        "snip_id": entity_id
                    }
        
        # No match
        return {"type": "unknown", "id": entity_id}
    
    @staticmethod
    def extract_frame_number(id_str: str) -> int:
        """Extract frame number from image or snip ID."""
        parsed = IdParser.parse_id(id_str)
        
        if parsed["type"] in ["image", "snip"]:
            return int(parsed["frame"])
        
        return -1
    
    @staticmethod
    def get_parent_id(child_id: str, parent_type: str) -> Optional[str]:
        """
        Get parent ID from child ID.
        
        Examples:
            get_parent_id("20240411_A01_e01_0001", "embryo") -> "20240411_A01_e01"
            get_parent_id("20240411_A01_e01", "video") -> "20240411_A01"
        """
        parsed = IdParser.parse_id(child_id)
        
        if parsed["type"] == "unknown":
            return None
        
        return parsed.get(f"{parent_type}_id")
```

## Data Conversion Utilities

```python
class DataConverter:
    """Utilities for converting between data formats."""
    
    @staticmethod
    def phenotype_to_numeric(phenotype: str, 
                           mapping: Optional[Dict[str, int]] = None) -> int:
        """
        Convert phenotype to numeric value for analysis.
        
        Args:
            phenotype: Phenotype string
            mapping: Custom mapping (default: severity-based)
        
        Returns:
            Numeric value
        """
        if mapping:
            return mapping.get(phenotype, -1)
        
        # Default severity-based mapping
        default_mapping = {
            "NONE": 0,
            "EDEMA": 1,
            "BODY_AXIS": 2,
            "CONVERGENCE_EXTENSION": 2,
            "DEAD": 3
        }
        
        return default_mapping.get(phenotype, -1)
    
    @staticmethod
    def flags_to_bitmask(flags: List[str], 
                        flag_definitions: Dict[str, int]) -> int:
        """
        Convert list of flags to bitmask for efficient storage.
        
        Args:
            flags: List of flag values
            flag_definitions: Mapping of flag to bit position
        
        Returns:
            Bitmask integer
        """
        bitmask = 0
        
        for flag in flags:
            if flag in flag_definitions:
                bit_position = flag_definitions[flag]
                bitmask |= (1 << bit_position)
        
        return bitmask
    
    @staticmethod
    def bitmask_to_flags(bitmask: int, 
                        flag_definitions: Dict[str, int]) -> List[str]:
        """Convert bitmask back to list of flags."""
        flags = []
        
        for flag, bit_position in flag_definitions.items():
            if bitmask & (1 << bit_position):
                flags.append(flag)
        
        return flags
    
    @staticmethod
    def export_to_long_format(metadata: 'EmbryoMetadata') -> List[Dict]:
        """
        Convert metadata to long format for analysis.
        
        Each row represents one snip with all associated data.
        """
        rows = []
        
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            # Base embryo info
            base_row = {
                "embryo_id": embryo_id,
                "experiment_id": embryo_data["source"]["experiment_id"],
                "video_id": embryo_data["source"]["video_id"]
            }
            
            # Add genotype
            if embryo_data.get("genotype"):
                base_row["genotype"] = embryo_data["genotype"]["value"]
                base_row["genotype_confirmed"] = embryo_data["genotype"].get("confirmed", False)
            else:
                base_row["genotype"] = None
                base_row["genotype_confirmed"] = False
            
            # Add each snip
            for snip_id, snip_data in embryo_data["snips"].items():
                snip_row = base_row.copy()
                snip_row["snip_id"] = snip_id
                snip_row["frame"] = IdParser.extract_frame_number(snip_id)
                
                # Phenotype
                phenotype_info = snip_data["phenotype"]
                snip_row["phenotype"] = phenotype_info["value"]
                snip_row["phenotype_author"] = phenotype_info["author"]
                snip_row["phenotype_confidence"] = phenotype_info.get("confidence")
                snip_row["phenotype_numeric"] = DataConverter.phenotype_to_numeric(
                    phenotype_info["value"]
                )
                
                # Flags
                flags = snip_data.get("flags", [])
                snip_row["num_flags"] = len(flags)
                snip_row["has_flags"] = len(flags) > 0
                snip_row["flag_list"] = ",".join([f["value"] for f in flags])
                
                rows.append(snip_row)
        
        return rows
```

## Statistical Utilities

```python
class StatsUtils:
    """Statistical utilities for metadata analysis."""
    
    @staticmethod
    def calculate_phenotype_transitions(timeline: List[Dict]) -> Dict:
        """
        Calculate phenotype transition statistics.
        
        Args:
            timeline: Phenotype timeline from get_phenotype_timeline()
        
        Returns:
            Transition statistics
        """
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(timeline)):
            prev = timeline[i-1]["phenotype"]
            curr = timeline[i]["phenotype"]
            
            if prev != curr:
                transitions[prev][curr] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for from_phenotype, to_counts in transitions.items():
            total = sum(to_counts.values())
            transition_probs[from_phenotype] = {
                to_phenotype: count / total
                for to_phenotype, count in to_counts.items()
            }
        
        return {
            "counts": dict(transitions),
            "probabilities": transition_probs
        }
    
    @staticmethod
    def calculate_phenotype_onset(metadata: 'EmbryoMetadata',
                                phenotype: str) -> Dict:
        """
        Calculate onset timing statistics for a phenotype.
        
        Returns:
            Dict with onset statistics
        """
        onset_frames = []
        
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            timeline = metadata.get_phenotype_timeline(embryo_id)
            
            # Find first occurrence
            for entry in timeline:
                if entry["phenotype"] == phenotype:
                    onset_frames.append(entry["frame"])
                    break
        
        if not onset_frames:
            return {"n": 0, "phenotype": phenotype}
        
        import numpy as np
        
        return {
            "phenotype": phenotype,
            "n": len(onset_frames),
            "mean_onset": np.mean(onset_frames),
            "median_onset": np.median(onset_frames),
            "std_onset": np.std(onset_frames),
            "min_onset": np.min(onset_frames),
            "max_onset": np.max(onset_frames),
            "q25_onset": np.percentile(onset_frames, 25),
            "q75_onset": np.percentile(onset_frames, 75)
        }
```

## Validation Utilities

```python
class ValidationUtils:
    """Additional validation utilities."""
    
    @staticmethod
    def validate_data_integrity(metadata: 'EmbryoMetadata') -> List[str]:
        """
        Comprehensive data integrity check.
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Check for orphaned snips
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            for snip_id in embryo_data["snips"].keys():
                parsed = IdParser.parse_id(snip_id)
                if parsed.get("embryo_id") != embryo_id:
                    issues.append(f"Snip {snip_id} in wrong embryo {embryo_id}")
        
        # Check for duplicate entries
        seen_snips = set()
        for embryo_data in metadata.data["embryos"].values():
            for snip_id in embryo_data["snips"].keys():
                if snip_id in seen_snips:
                    issues.append(f"Duplicate snip {snip_id}")
                seen_snips.add(snip_id)
        
        # Check temporal consistency
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            death_frame = metadata._get_death_frame(embryo_id)
            if death_frame is not None:
                # Check no phenotypes after death
                for snip_id, snip_data in embryo_data["snips"].items():
                    frame = IdParser.extract_frame_number(snip_id)
                    phenotype = snip_data["phenotype"]["value"]
                    
                    if frame > death_frame and phenotype not in ["NONE", "DEAD"]:
                        issues.append(
                            f"Non-DEAD phenotype '{phenotype}' after death "
                            f"in {embryo_id} at frame {frame}"
                        )
        
        return issues
    
    @staticmethod
    def validate_against_schema(data: Dict, schema: Dict) -> List[str]:
        """
        Validate data structure against schema.
        
        Simple schema validation for nested dictionaries.
        """
        def check_structure(data, schema, path=""):
            errors = []
            
            for key, expected_type in schema.items():
                if key not in data:
                    errors.append(f"Missing key at {path}.{key}")
                    continue
                
                if isinstance(expected_type, dict):
                    # Nested structure
                    if not isinstance(data[key], dict):
                        errors.append(f"Expected dict at {path}.{key}")
                    else:
                        errors.extend(
                            check_structure(data[key], expected_type, f"{path}.{key}")
                        )
                else:
                    # Type check
                    if not isinstance(data[key], expected_type):
                        errors.append(
                            f"Wrong type at {path}.{key}: "
                            f"expected {expected_type}, got {type(data[key])}"
                        )
            
            return errors
        
        return check_structure(data, schema)
```

## Logging and Change Tracking

```python
class ChangeLogger:
    """Change tracking and audit logging."""
    
    def __init__(self):
        self.changes = []
    
    def log_change(self, operation: str, details: Dict) -> None:
        """Log a change operation."""
        self.changes.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
    
    def get_changes_since(self, timestamp: datetime) -> List[Dict]:
        """Get changes since a timestamp."""
        return [
            change for change in self.changes
            if datetime.fromisoformat(change["timestamp"]) > timestamp
        ]
    
    def export_audit_log(self, output_path: Path) -> None:
        """Export audit log to file."""
        with open(output_path, 'w') as f:
            json.dump(self.changes, f, indent=2)
    
    def clear(self) -> None:
        """Clear change log."""
        self.changes.clear()
```

## Performance Utilities

```python
class PerformanceUtils:
    """Performance monitoring and optimization utilities."""
    
    @staticmethod
    def profile_operation(func: callable, *args, **kwargs) -> Tuple[any, float]:
        """
        Profile execution time of an operation.
        
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        import time
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        return result, end - start
    
    @staticmethod
    def estimate_memory_usage(obj: any) -> int:
        """
        Estimate memory usage of an object in bytes.
        
        Note: This is an approximation.
        """
        import sys
        
        def get_size(obj, seen=None):
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            
            return size
        
        return get_size(obj)
```

## Common Helper Functions

```python
def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()

def generate_unique_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix."""
    import uuid
    unique = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique}" if prefix else unique

def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return numerator / denominator if denominator != 0 else default

def chunks(lst: List, size: int) -> List[List]:
    """Yield successive chunks from list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary.
    
    Example:
        {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """
    Unflatten dictionary.
    
    Example:
        {"a.b": 1} -> {"a": {"b": 1}}
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

def calculate_checksum(data: Union[Dict, str]) -> str:
    """Calculate SHA256 checksum of data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    
    return hashlib.sha256(data.encode()).hexdigest()

def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem compatibility."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename
```