"""
EntityIDTracker - Entity container and tracking utility for MorphSeq pipeline.

DESIGN PHILOSOPHY:
EntityIDTracker is a PURE STATIC UTILITY for tracking entities across pipeline steps.
It embeds tracking data directly into pipeline JSON files (not separate files).

CORE RESPONSIBILITIES:
1. Extract entity IDs from JSON data structures
2. Embed entity tracking sections into pipeline JSON files
3. Compare entity states between pipeline steps

WHAT IT DOESN'T DO:
- No file I/O beyond simple JSON loading for comparison
- No instance state management (all static methods)
- No complex history tracking (pipeline step is implicit from file)
"""

from typing import Dict, Set, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from .parsing_utils import parse_entity_id, extract_experiment_id, get_entity_type


class EntityIDTracker:
    """Static utility for entity ID tracking in MorphSeq pipeline."""
    
    @staticmethod
    def extract_entities(json_data: Dict) -> Dict[str, Set[str]]:
        """
        Extract all entity IDs from JSON structure.
        Returns dict with sets of IDs for each entity type.
        """
        entities = {
            "experiments": set(),
            "videos": set(), 
            "images": set(),
            "embryos": set(),
            "snips": set()
        }
        
        def scan_recursive(obj):
            if isinstance(obj, str):
                if EntityIDTracker._looks_like_entity_id(obj):
                    entity_type = EntityIDTracker._get_entity_type(obj)
                    if entity_type:
                        entities[entity_type].add(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    # Also check keys, as they can be IDs (e.g., in experiment dicts)
                    if isinstance(key, str) and EntityIDTracker._looks_like_entity_id(key):
                        entity_type = EntityIDTracker._get_entity_type(key)
                        if entity_type:
                            entities[entity_type].add(key)
                    scan_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    scan_recursive(item)
        
        scan_recursive(json_data)
        return entities
    
    @staticmethod
    def _get_entity_type(entity_id: str) -> Optional[str]:
        """Determine entity type from its ID string using parsing_utils with fallback."""
        try:
            # Try to use the robust function from parsing_utils
            entity_type = get_entity_type(entity_id)
            # Convert to plural form used by EntityIDTracker
            type_mapping = {
                "experiment": "experiments",
                "video": "videos", 
                "image": "images",
                "embryo": "embryos",
                "snip": "snips"
            }
            return type_mapping.get(entity_type)
        except:
            # Fallback to local implementation if parsing_utils fails
            import re
            
            # Order is critical: from most specific to least specific
            # A snip ID also matches embryo, video, and experiment patterns.
            
            # Snip: ..._eXX_sXXXX or ..._eXX_XXXX
            if re.search(r'_e\d+_s?\d{3,4}$', entity_id):
                return "snips"
            # Embryo: ..._eXX
            if re.search(r'_e\d+$', entity_id):
                return "embryos"
            # Image: ..._tXXXX
            if re.search(r'_t\d{3,4}$', entity_id):
                return "images"
            # Video: ..._HXX
            if re.search(r'_[A-H]\d{2}$', entity_id):
                return "videos"
            # Experiment: If it doesn't match others but is a valid ID format
            if re.match(r'^20\d{6}', entity_id):
                return "experiments"
                
            return None
    
    @staticmethod
    def _add_ids(value, target_set):
        """Helper to add IDs to set, using proper ID validation."""
        if isinstance(value, str):
            # Use proper ID validation instead of simple filtering
            if EntityIDTracker._looks_like_entity_id(value):
                target_set.add(value)
        elif isinstance(value, list):
            for item in value:
                EntityIDTracker._add_ids(item, target_set)
    
    @staticmethod
    def _looks_like_entity_id(value: str) -> bool:
        """
        Determine if a string looks like a valid entity ID using proper criteria.
        
        Entity IDs have specific patterns:
        - experiment_id: "20250624_chem02_28C_T00_1356"
        - video_id: "20250624_chem02_28C_T00_1356_H01"  
        - image_id: "20250624_chem02_28C_T00_1356_H01_t0042"
        - embryo_id: "20250624_chem02_28C_T00_1356_H01_e01"
        - snip_id: "20250624_chem02_28C_T00_1356_H01_e01_s0034"
        """
        import re
        
        # Reject obvious non-IDs
        if not value or len(value) < 8:  # Allow simple date IDs (YYYYMMDD = 8 chars)
            return False
        
        # Reject file paths
        if '/' in value or '\\' in value or value.startswith('.'):
            return False
            
        # Reject URLs or other schemes
        if '://' in value or value.startswith(('http', 'ftp', 'file')):
            return False
        
        # Check for date pattern at start (YYYYMMDD)
        if not re.match(r'^20\d{6}', value):  # Remove underscore requirement
            return False
            
        # Check for valid entity patterns
        patterns = [
            # Experiment patterns (flexible)
            r'^20\d{6}$',  # Simple date-only: 20240506
            r'^20\d{6}_\w+$',  # Date + name: 20240506_ctrl
            r'^20\d{6}_\w+_\d+C_T\d+_\d+$',  # Full: 20250703_chem3_28C_T00_1325
            
            # Video patterns  
            r'^20\d{6}_\w+_\d+C_T\d+_\d+_[A-H]\d{2}$',  # Full video
            r'^20\d{6}_\w+_[A-H]\d{2}$',  # Simple video: date_name_well
            r'^20\d{6}_[A-H]\d{2}$',  # Minimal video: date_well
            
            # Image patterns
            r'^20\d{6}_\w+_\d+C_T\d+_\d+_[A-H]\d{2}_t\d{3,4}$',  # Full image
            r'^20\d{6}_\w+_[A-H]\d{2}_t\d{3,4}$',  # Simple image
            r'^20\d{6}_[A-H]\d{2}_t\d{3,4}$',  # Minimal image
            
            # Embryo patterns
            r'^20\d{6}_\w+_\d+C_T\d+_\d+_[A-H]\d{2}_e\d+$',  # Full embryo
            r'^20\d{6}_\w+_[A-H]\d{2}_e\d+$',  # Simple embryo
            r'^20\d{6}_[A-H]\d{2}_e\d+$',  # Minimal embryo
            
            # Snip patterns  
            r'^20\d{6}_\w+_\d+C_T\d+_\d+_[A-H]\d{2}_e\d+_s?\d{3,4}$',  # Full snip
            r'^20\d{6}_\w+_[A-H]\d{2}_e\d+_s?\d{3,4}$',  # Simple snip
            r'^20\d{6}_[A-H]\d{2}_e\d+_s?\d{3,4}$'  # Minimal snip
        ]
        
        return any(re.match(pattern, value) for pattern in patterns)
    
    @staticmethod
    def add_entity_tracker(data: Dict, pipeline_step: str = None) -> Dict:
        """Add entity_tracker section to JSON data."""
        entities = EntityIDTracker.extract_entities(data)
        
        data["entity_tracker"] = {
            "creation_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "pipeline_step": pipeline_step,  # Optional, for documentation only
            "entities": {k: sorted(list(v)) for k, v in entities.items()}
        }
        return data
    
    @staticmethod
    def update_entity_tracker(data: Dict, pipeline_step: str = None) -> Dict:
        """Update or create entity_tracker in JSON data."""
        entities = EntityIDTracker.extract_entities(data)
        
        if "entity_tracker" not in data:
            return EntityIDTracker.add_entity_tracker(data, pipeline_step)
        
        data["entity_tracker"]["last_updated"] = datetime.now().isoformat()
        if pipeline_step:
            data["entity_tracker"]["pipeline_step"] = pipeline_step
        data["entity_tracker"]["entities"] = {k: sorted(list(v)) for k, v in entities.items()}
        
        return data
    
    @staticmethod
    def compare_files(file1_path: str, file2_path: str) -> Dict:
        """Compare entity trackers in two JSON files."""
        try:
            with open(file1_path, 'r') as f:
                data1 = json.load(f)
            with open(file2_path, 'r') as f:
                data2 = json.load(f)
            
            return EntityIDTracker.compare_data(data1, data2)
        except Exception as e:
            return {"error": f"Failed to compare files: {e}"}
    
    @staticmethod
    def compare_data(data1: Dict, data2: Dict) -> Dict:
        """Compare entity trackers in two JSON objects."""
        # Extract entities from both data objects
        entities1 = data1.get("entity_tracker", {}).get("entities", {})
        entities2 = data2.get("entity_tracker", {}).get("entities", {})
        
        result = {}
        for entity_type in ["experiments", "videos", "images", "embryos", "snips"]:
            set1 = set(entities1.get(entity_type, []))
            set2 = set(entities2.get(entity_type, []))
            
            result[entity_type] = {
                "in_first_only": sorted(list(set1 - set2)),
                "in_second_only": sorted(list(set2 - set1)),
                "in_both": sorted(list(set1 & set2))
            }
        
        return result
    
    @staticmethod
    def compare_trackers(tracker1: Dict, tracker2: Dict) -> Dict:
        """Compare two in-memory entity trackers (already extracted)."""
        result = {}
        for entity_type in ["experiments", "videos", "images", "embryos", "snips"]:
            set1 = set(tracker1.get(entity_type, []))
            set2 = set(tracker2.get(entity_type, []))
            
            result[entity_type] = {
                "in_first_only": sorted(list(set1 - set2)),
                "in_second_only": sorted(list(set2 - set1)),
                "in_both": sorted(list(set1 & set2)),
                "count_first": len(set1),
                "count_second": len(set2)
            }
        
        return result
    
    @staticmethod
    def get_counts(entities: Dict[str, Set[str]]) -> Dict[str, int]:
        """Get count of entities for each type."""
        return {
            entity_type: len(entity_set) 
            for entity_type, entity_set in entities.items()
        }
    
    @staticmethod
    def validate_hierarchy(entities: Dict[str, Set[str]], check_hierarchy: bool = False) -> Dict:
        """
        Validate parent-child ID relationships.
        
        Args:
            entities: Dictionary of entity sets to validate
            check_hierarchy: If True, perform full hierarchy validation. If False, skip validation.
                           Default False since hierarchy validation is often not needed and can
                           cause issues when processing partial datasets.
        """
        if not check_hierarchy:
            return {"valid": True, "violations": [], "skipped": "Hierarchy validation disabled"}
        
        violations = []
        
        # Check embryos reference valid experiments
        for embryo_id in entities["embryos"]:
            try:
                parsed = parse_entity_id(embryo_id)
                exp_id = parsed.get("experiment_id")
                if exp_id and exp_id not in entities["experiments"]:
                    violations.append(f"Embryo {embryo_id} -> missing experiment {exp_id}")
            except ValueError:
                violations.append(f"Invalid embryo ID format: {embryo_id}")
        
        # Check snips reference valid embryos  
        for snip_id in entities["snips"]:
            try:
                parsed = parse_entity_id(snip_id)
                embryo_id = parsed.get("embryo_id")
                if embryo_id and embryo_id not in entities["embryos"]:
                    violations.append(f"Snip {snip_id} -> missing embryo {embryo_id}")
            except ValueError:
                violations.append(f"Invalid snip ID format: {snip_id}")
        
        # Videos reference valid experiments
        for video_id in entities["videos"]:
            try:
                exp_id = extract_experiment_id(video_id)
                if exp_id not in entities["experiments"]:
                    violations.append(f"Video {video_id} -> missing experiment {exp_id}")
            except ValueError:
                violations.append(f"Invalid video ID format: {video_id}")
        
        # Images reference valid videos
        for image_id in entities["images"]:
            try:
                parsed = parse_entity_id(image_id)
                video_id = parsed.get("video_id")
                if video_id and video_id not in entities["videos"]:
                    violations.append(f"Image {image_id} -> missing video {video_id}")
            except ValueError:
                violations.append(f"Invalid image ID format: {image_id}")
        
        return {"valid": len(violations) == 0, "violations": violations}
