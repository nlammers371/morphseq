"""
Base Annotation Parser Class for MorphSeq Pipeline
Foundational class providing common ID parsing, file operations, and utility functions
used across all annotation classes (EmbryoMetadata, GroundedDinoAnnotations, etc.)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import shutil

class BaseAnnotationParser:
    """
    Base class for all annotation-related classes in the MorphSeq pipeline.
    
    Provides common functionality for:
    - ID parsing and validation
    - File I/O operations
    - Change tracking
    - Common utilities
    """
    
    # ID format patterns shared across all classes
    ID_PATTERNS = {
        "experiment": re.compile(r'^(\d{8})$'),
        "video": re.compile(r'^(\d{8})_([A-H]\d{2})$'),
        "image": re.compile(r'^(\d{8})_([A-H]\d{2})_(\d{4})$'),
        "embryo": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})$'),
        "snip": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})_(\d{4})$')
    }
    
    def __init__(self, filepath: Union[str, Path], verbose: bool = True):
        """Initialize base parser with common attributes."""
        self.filepath = Path(filepath)
        self.verbose = verbose
        self._unsaved_changes = False
        self._change_log = []
    
    # -------------------------------------------------------------------------
    # File I/O Operations
    # -------------------------------------------------------------------------
    
    def load_json(self, file_path: Path = None) -> Dict:
        """Load JSON file with error handling."""
        if file_path is None:
            file_path = self.filepath
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def save_json(self, data: Dict, file_path: Path = None, 
                  create_backup: bool = True) -> None:
        """Save JSON with atomic write and optional backup."""
        if file_path is None:
            file_path = self.filepath
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested
        if create_backup and file_path.exists():
            self._create_backup(file_path)
        
        # Atomic write
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(file_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create timestamped backup of file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup.{timestamp}{file_path.suffix}')
        shutil.copy2(file_path, backup_path)
        if self.verbose:
            print(f"📦 Created backup: {backup_path.name}")
        return backup_path
    
    # -------------------------------------------------------------------------
    # ID Parsing and Validation
    # -------------------------------------------------------------------------
    
    def parse_id(self, entity_id: str) -> Dict[str, str]:
        """Parse any ID type and extract components."""
        for id_type, pattern in self.ID_PATTERNS.items():
            match = pattern.match(entity_id)
            if match:
                return self._extract_id_components(id_type, match.groups(), entity_id)
        
        return {"type": "unknown", "id": entity_id}
    
    def _extract_id_components(self, id_type: str, groups: Tuple, 
                              entity_id: str) -> Dict[str, str]:
        """Extract components based on ID type."""
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
        
        return {"type": "unknown", "id": entity_id}
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Extract embryo ID from snip ID."""
        parsed = self.parse_id(snip_id)
        if parsed["type"] == "snip":
            return parsed["embryo_id"]
        return None
    
    def extract_frame_number(self, id_str: str) -> int:
        """Extract frame number from image or snip ID."""
        parsed = self.parse_id(id_str)
        if parsed["type"] in ["image", "snip"] and "frame" in parsed:
            return int(parsed["frame"])
        return -1
    
    def validate_id_format(self, entity_id: str, expected_type: str) -> bool:
        """Validate ID matches expected type."""
        parsed = self.parse_id(entity_id)
        return parsed["type"] == expected_type
    
    def get_video_id_from_entity(self, entity_id: str) -> Optional[str]:
        """Extract video ID from any entity ID."""
        parsed = self.parse_id(entity_id)
        return parsed.get("video_id")
    
    def get_experiment_id_from_entity(self, entity_id: str) -> Optional[str]:
        """Extract experiment ID from any entity ID."""
        parsed = self.parse_id(entity_id)
        return parsed.get("experiment_id")
    
    # -------------------------------------------------------------------------
    # Change Tracking
    # -------------------------------------------------------------------------
    
    def _add_change_log(self, operation: str, details: Dict) -> None:
        """Add entry to change log."""
        self._change_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
        self._unsaved_changes = True
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent changes."""
        return self._change_log[-limit:]
    
    def clear_change_log(self) -> None:
        """Clear change log after save."""
        self._change_log.clear()
    
    # -------------------------------------------------------------------------
    # Common Utilities
    # -------------------------------------------------------------------------
    
    def get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now().isoformat()
    
    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def mark_saved(self) -> None:
        """Mark as saved and clear change tracking."""
        self._unsaved_changes = False
        self.clear_change_log()
    
    def get_file_stats(self) -> Dict:
        """Get file statistics."""
        if not self.filepath.exists():
            return {"exists": False}
        
        stat = self.filepath.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
    
    def validate_json_structure(self, data: Dict, required_keys: List[str]) -> bool:
        """Validate that JSON data has required top-level keys."""
        return all(key in data for key in required_keys)
    
    def log_operation(self, operation: str, entity_id: str = None, **kwargs) -> None:
        """Log an operation with optional entity ID and additional details."""
        details = {"entity_id": entity_id} if entity_id else {}
        details.update(kwargs)
        self._add_change_log(operation, details)
        
        if self.verbose:
            print(f"📝 {operation}: {entity_id or 'N/A'}")

    # -------------------------------------------------------------------------
    # Entity Level Detection and Hierarchical Management
    # -------------------------------------------------------------------------
    
    def detect_entity_level(self, entity_id: str) -> str:
        """
        Automatically detect the hierarchical level of an entity ID.
        
        Args:
            entity_id: ID to analyze
            
        Returns:
            str: Entity level ('experiment', 'video', 'image', 'embryo', 'snip', 'unknown')
        """
        parsed = self.parse_id(entity_id)
        return parsed["type"]
    
    def get_parent_entities(self, entity_id: str) -> Dict[str, str]:
        """
        Get all parent entities in the hierarchy for a given entity.
        
        Args:
            entity_id: ID to get parents for
            
        Returns:
            Dict with parent IDs at each level
        """
        parsed = self.parse_id(entity_id)
        parents = {}
        
        if parsed["type"] == "unknown":
            return parents
        
        # Build parent hierarchy
        if "date" in parsed:
            parents["experiment"] = parsed["date"]
        
        if "well" in parsed:
            parents["video"] = f"{parsed['date']}_{parsed['well']}"
        
        if "frame" in parsed and parsed["type"] in ["image", "embryo", "snip"]:
            if parsed["type"] == "image":
                parents["image"] = entity_id
            else:
                # For embryo/snip, construct image ID
                if "embryo_num" in parsed:
                    # Extract frame from snip or use embryo context
                    frame_part = parsed.get("snip_frame", "0001")  # Default frame
                    parents["image"] = f"{parsed['date']}_{parsed['well']}_{frame_part}"
        
        if "embryo_num" in parsed:
            parents["embryo"] = f"{parsed['date']}_{parsed['well']}_e{parsed['embryo_num']:02d}"
        
        return parents
    
    def get_child_entity_pattern(self, entity_id: str, child_type: str) -> str:
        """
        Generate the pattern for child entities of a given type.
        
        Args:
            entity_id: Parent entity ID
            child_type: Type of children to get pattern for
            
        Returns:
            str: Regex pattern for matching children
        """
        parsed = self.parse_id(entity_id)
        
        if parsed["type"] == "experiment" and child_type == "video":
            return f"{parsed['date']}_[A-H]\\d{{2}}$"
        elif parsed["type"] == "video" and child_type == "image":
            return f"{entity_id}_\\d{{4}}$"
        elif parsed["type"] == "video" and child_type == "embryo":
            return f"{entity_id}_e\\d{{2}}$"
        elif parsed["type"] == "embryo" and child_type == "snip":
            return f"{entity_id}_\\d{{4}}$"
        
        return ""
    
    def validate_entity_hierarchy(self, parent_id: str, child_id: str) -> bool:
        """
        Validate that a child entity belongs to the specified parent.
        
        Args:
            parent_id: Parent entity ID
            child_id: Child entity ID to validate
            
        Returns:
            bool: True if valid hierarchy relationship
        """
        parent_parsed = self.parse_id(parent_id)
        child_parsed = self.parse_id(child_id)
        
        if parent_parsed["type"] == "unknown" or child_parsed["type"] == "unknown":
            return False
        
        # Check hierarchical relationships
        valid_relationships = {
            "experiment": ["video"],
            "video": ["image", "embryo"], 
            "image": ["snip"],  # snips belong to images
            "embryo": ["snip"]  # but also to embryos
        }
        
        if child_parsed["type"] not in valid_relationships.get(parent_parsed["type"], []):
            return False
        
        # Validate ID components match
        if "date" in parent_parsed and parent_parsed["date"] != child_parsed.get("date"):
            return False
        if "well" in parent_parsed and parent_parsed["well"] != child_parsed.get("well"):
            return False
        if "embryo_num" in parent_parsed and parent_parsed["embryo_num"] != child_parsed.get("embryo_num"):
            return False
        
        return True
