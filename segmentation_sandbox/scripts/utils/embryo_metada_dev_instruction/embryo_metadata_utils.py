"""
Utilities and Helper Functions for EmbryoMetadata System
Module 8: Supporting utilities for file operations, data conversion, and validation
"""

import json
import shutil
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from functools import wraps
import re

# -------------------------------------------------------------------------
# File Operations
# -------------------------------------------------------------------------

def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and convert path to Path object.
    
    Args:
        path: String or Path object
        must_exist: Whether path must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or doesn't exist when required
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path_obj}")
    
    return path_obj

def load_json(file_path: Path, create_if_missing: bool = False, 
              default_content: Optional[Dict] = None) -> Dict:
    """
    Load JSON file with options for missing files.
    
    Args:
        file_path: Path to JSON file
        create_if_missing: Create file if it doesn't exist
        default_content: Default content for new files
        
    Returns:
        Loaded JSON data
    """
    file_path = validate_path(file_path)
    
    if not file_path.exists():
        if create_if_missing:
            content = default_content or {}
            save_json(file_path, content)
            return content
        else:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json(file_path: Path, data: Dict, create_backup: bool = True,
              indent: int = 2) -> None:
    """
    Save JSON with atomic write and optional backup.
    
    Args:
        file_path: Path to save JSON
        data: Data to save
        create_backup: Whether to create backup
        indent: JSON indentation
    """
    file_path = validate_path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if file exists and backup requested
    if create_backup and file_path.exists():
        create_backup_file(file_path)
    
    # Atomic write using temporary file
    temp_path = file_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=indent)
        temp_path.replace(file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

def create_backup_file(file_path: Path, max_backups: int = 5) -> Path:
    """
    Create timestamped backup and manage backup count.
    
    Args:
        file_path: File to backup
        max_backups: Maximum number of backups to keep
        
    Returns:
        Path to created backup
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f'.backup.{timestamp}{file_path.suffix}')
    
    shutil.copy2(file_path, backup_path)
    
    # Clean old backups
    cleanup_old_backups(file_path, max_backups)
    
    return backup_path

def cleanup_old_backups(file_path: Path, max_backups: int) -> None:
    """Remove old backup files, keeping only the most recent."""
    pattern = f"{file_path.stem}.backup.*{file_path.suffix}"
    backups = sorted(file_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
    
    # Remove oldest backups if we exceed the limit
    while len(backups) > max_backups:
        oldest = backups.pop(0)
        oldest.unlink()

def calculate_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """Calculate hash of file contents."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# -------------------------------------------------------------------------
# ID Utilities
# -------------------------------------------------------------------------

class IdParser:
    """Enhanced ID parsing utilities."""
    
    @staticmethod
    def parse_id(entity_id: str) -> Dict[str, str]:
        """Parse ID and return components - delegated to BaseAnnotationParser."""
        # This would typically use the BaseAnnotationParser method
        # For now, implementing basic parsing
        patterns = {
            "experiment": r'^(\d{8})$',
            "video": r'^(\d{8})_([A-H]\d{2})$',
            "image": r'^(\d{8})_([A-H]\d{2})_(\d{4})$',
            "embryo": r'^(\d{8})_([A-H]\d{2})_e(\d{2})$',
            "snip": r'^(\d{8})_([A-H]\d{2})_e(\d{2})_(\d{4})$'
        }
        
        for id_type, pattern in patterns.items():
            if re.match(pattern, entity_id):
                return {"type": id_type, "id": entity_id}
        
        return {"type": "unknown", "id": entity_id}
    
    @staticmethod
    def generate_sequential_ids(base_id: str, count: int, 
                               id_type: str = "snip") -> List[str]:
        """Generate sequential IDs for batch operations."""
        ids = []
        if id_type == "snip":
            # Extract base parts and generate frame sequence
            parts = base_id.split('_')
            if len(parts) >= 4:
                base = '_'.join(parts[:-1])
                start_frame = int(parts[-1])
                for i in range(count):
                    frame = str(start_frame + i).zfill(4)
                    ids.append(f"{base}_{frame}")
        
        return ids
    
    @staticmethod
    def validate_id_consistency(ids: List[str]) -> Dict[str, Any]:
        """Validate that a list of IDs are consistent (same experiment, etc.)."""
        if not ids:
            return {"valid": True, "issues": []}
        
        parsed_ids = [IdParser.parse_id(id_str) for id_str in ids]
        
        # Group by experiment
        experiments = set()
        for parsed in parsed_ids:
            if parsed["type"] != "unknown":
                # Extract experiment ID from the ID
                exp_match = re.match(r'^(\d{8})', parsed["id"])
                if exp_match:
                    experiments.add(exp_match.group(1))
        
        issues = []
        if len(experiments) > 1:
            issues.append(f"Multiple experiments found: {experiments}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "experiments": list(experiments),
            "id_types": [p["type"] for p in parsed_ids]
        }

# -------------------------------------------------------------------------
# Data Conversion Utilities
# -------------------------------------------------------------------------

def convert_timestamps(data: Dict, target_format: str = "iso") -> Dict:
    """Convert timestamps in data structure to target format."""
    converted = data.copy()
    
    def convert_value(value):
        if isinstance(value, str) and is_timestamp(value):
            if target_format == "iso":
                return normalize_timestamp(value)
            elif target_format == "unix":
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return int(dt.timestamp())
        elif isinstance(value, dict):
            return convert_timestamps(value, target_format)
        elif isinstance(value, list):
            return [convert_value(item) for item in value]
        return value
    
    for key, value in converted.items():
        converted[key] = convert_value(value)
    
    return converted

def is_timestamp(value: str) -> bool:
    """Check if string looks like a timestamp."""
    timestamp_patterns = [
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',   # Space separated
    ]
    
    return any(re.match(pattern, value) for pattern in timestamp_patterns)

def normalize_timestamp(timestamp_str: str) -> str:
    """Normalize timestamp to ISO format."""
    try:
        # Handle different input formats
        if 'T' not in timestamp_str and ' ' in timestamp_str:
            timestamp_str = timestamp_str.replace(' ', 'T')
        
        # Parse and reformat
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.isoformat()
    except ValueError:
        return timestamp_str  # Return original if can't parse

def flatten_nested_dict(data: Dict, sep: str = '.') -> Dict:
    """Flatten nested dictionary with dot notation keys."""
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data)

def unflatten_dict(data: Dict, sep: str = '.') -> Dict:
    """Unflatten dictionary with dot notation keys."""
    result = {}
    for key, value in data.items():
        keys = key.split(sep)
        d = result
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return result

# -------------------------------------------------------------------------
# Performance Utilities
# -------------------------------------------------------------------------

class PerformanceUtils:
    """Performance monitoring and optimization utilities."""
    
    @staticmethod
    def profile_operation(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Profile function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def time_it(func: Callable = None, *, iterations: int = 1):
        """Decorator to time function execution."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                times = []
                result = None
                
                for _ in range(iterations):
                    start = time.time()
                    result = f(*args, **kwargs)
                    end = time.time()
                    times.append(end - start)
                
                avg_time = sum(times) / len(times)
                print(f"⏱️  {f.__name__}: {avg_time:.4f}s avg over {iterations} iterations")
                
                return result
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)

def batch_process_with_progress(items: List, process_func: Callable,
                               batch_size: int = 100, 
                               progress_callback: Callable = None) -> List:
    """Process items in batches with optional progress reporting."""
    results = []
    total = len(items)
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
        
        if progress_callback:
            progress = min(i + batch_size, total) / total
            progress_callback(progress, i + len(batch), total)
    
    return results

# -------------------------------------------------------------------------
# Validation Helpers
# -------------------------------------------------------------------------

def validate_data_structure(data: Dict, schema: Dict) -> Dict[str, List[str]]:
    """Validate data structure against schema."""
    errors = {}
    
    def check_required(obj: Dict, schema_obj: Dict, path: str = ""):
        required = schema_obj.get("required", [])
        for field in required:
            if field not in obj:
                if path not in errors:
                    errors[path] = []
                errors[path].append(f"Missing required field: {field}")
    
    def check_types(obj: Dict, schema_obj: Dict, path: str = ""):
        properties = schema_obj.get("properties", {})
        for field, field_schema in properties.items():
            if field in obj:
                expected_type = field_schema.get("type")
                actual_value = obj[field]
                
                if expected_type == "string" and not isinstance(actual_value, str):
                    if path not in errors:
                        errors[path] = []
                    errors[path].append(f"Field {field} should be string, got {type(actual_value)}")
    
    check_required(data, schema)
    check_types(data, schema)
    
    return errors

def validate_embryo_metadata_structure(data: Dict) -> List[str]:
    """Validate EmbryoMetadata specific structure."""
    issues = []
    
    required_top_level = ["file_info", "embryos", "flags"]
    for key in required_top_level:
        if key not in data:
            issues.append(f"Missing top-level key: {key}")
    
    if "embryos" in data:
        for embryo_id, embryo_data in data["embryos"].items():
            # Validate embryo ID format
            if not re.match(r'^\d{8}_[A-H]\d{2}_e\d{2}$', embryo_id):
                issues.append(f"Invalid embryo ID format: {embryo_id}")
            
            # Check required embryo fields
            required_embryo_fields = ["snips"]
            for field in required_embryo_fields:
                if field not in embryo_data:
                    issues.append(f"Missing field {field} in embryo {embryo_id}")
    
    return issues

# -------------------------------------------------------------------------
# Logging and Debug Utilities
# -------------------------------------------------------------------------

def setup_debug_logging(verbose: bool = True) -> Callable:
    """Setup debug logging function."""
    def debug_log(message: str, level: str = "INFO"):
        if verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    return debug_log

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_memory_usage() -> Dict[str, str]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": format_file_size(memory_info.rss),
            "vms": format_file_size(memory_info.vms),
            "percent": f"{process.memory_percent():.1f}%"
        }
    except ImportError:
        return {"error": "psutil not available"}

# -------------------------------------------------------------------------
# Export Utilities
# -------------------------------------------------------------------------

def export_to_csv(data: List[Dict], output_path: Path, 
                  columns: Optional[List[str]] = None) -> None:
    """Export data to CSV format."""
    import csv
    
    if not data:
        return
    
    if columns is None:
        columns = list(data[0].keys())
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            # Only include specified columns
            filtered_row = {col: row.get(col, '') for col in columns}
            writer.writerow(filtered_row)

def export_to_excel(data: List[Dict], output_path: Path,
                   sheet_name: str = "Data") -> None:
    """Export data to Excel format."""
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_excel(output_path, sheet_name=sheet_name, index=False)
    except ImportError:
        raise ImportError("pandas required for Excel export")

# -------------------------------------------------------------------------
# Configuration Utilities
# -------------------------------------------------------------------------

def load_config(config_path: Path, defaults: Dict = None) -> Dict:
    """Load configuration with defaults."""
    config = defaults.copy() if defaults else {}
    
    if config_path.exists():
        user_config = load_json(config_path)
        config.update(user_config)
    
    return config

def save_config(config: Dict, config_path: Path) -> None:
    """Save configuration to file."""
    save_json(config_path, config)

# Default configuration for EmbryoMetadata
DEFAULT_EMBRYO_METADATA_CONFIG = {
    "auto_backup": True,
    "max_backups": 5,
    "verbose": True,
    "batch_size": 100,
    "validation": {
        "strict_id_format": True,
        "require_author": True,
        "allow_empty_values": False
    }
}
