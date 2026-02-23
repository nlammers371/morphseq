"""
EntityIDTracker - Simple utility for extracting and comparing entity IDs.
Works with any MorphSeq pipeline JSON that has ["entity_tracker"] section.

Key functionality:
- extract_entities(): Scan JSON and extract all entity IDs by type
- diff_entities(): Compare two entity states (added/removed/unchanged)
- get_new_entities(): Get only newly added entities since last state
- validate_hierarchy(): Check parent-child ID relationships are valid
- compare_entities(): Find entities missing from target vs source
- remove_orphaned_entities(): Clean up entities without valid parents
"""

from typing import Dict, Set
from parsing_utils import parse_entity_id, extract_experiment_id


class EntityIDTracker:
    """Extract and compare entity IDs from JSON structures."""
    
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
            if isinstance(obj, dict):
                # Scan for ID lists/values
                for key, value in obj.items():
                    if "experiment" in key.lower():
                        EntityIDTracker._add_ids(value, entities["experiments"])
                    elif "video" in key.lower():
                        EntityIDTracker._add_ids(value, entities["videos"])
                    elif "image" in key.lower():
                        EntityIDTracker._add_ids(value, entities["images"])
                    elif "embryo" in key.lower():
                        EntityIDTracker._add_ids(value, entities["embryos"])
                    elif "snip" in key.lower():
                        EntityIDTracker._add_ids(value, entities["snips"])
                    
                    scan_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    scan_recursive(item)
        
        scan_recursive(json_data)
        return entities
    
    @staticmethod
    def _add_ids(value, target_set):
        """Helper to add IDs to set."""
        if isinstance(value, str):
            target_set.add(value)
        elif isinstance(value, list):
            target_set.update(value)
    
    @staticmethod
    def compare_entities(source: Dict[str, Set[str]], target: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Compare two entity dictionaries, return missing from target."""
        missing = {}
        for entity_type in source:
            missing[entity_type] = source[entity_type] - target.get(entity_type, set())
        return missing
    
    @staticmethod
    def diff_entities(current: Dict[str, Set[str]], previous: Dict[str, Set[str]]) -> Dict[str, Dict[str, Set[str]]]:
        """
        Cross-reference two entity dictionaries and return differences at all levels.
        
        Args:
            current: Current entity state
            previous: Previous entity state
            
        Returns:
            Dict with 'added', 'removed', 'unchanged' for each entity type
        """
        diff = {}
        
        for entity_type in ["experiments", "videos", "images", "embryos", "snips"]:
            current_set = current.get(entity_type, set())
            previous_set = previous.get(entity_type, set())
            
            diff[entity_type] = {
                "added": current_set - previous_set,
                "removed": previous_set - current_set,
                "unchanged": current_set & previous_set
            }
        
        return diff
    
    @staticmethod
    def get_new_entities(current: Dict[str, Set[str]], previous: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Get only new entities (added since previous state).
        
        Returns:
            Dict with sets of new IDs for each entity type
        """
        new_entities = {}
        for entity_type in ["experiments", "videos", "images", "embryos", "snips"]:
            current_set = current.get(entity_type, set())
            previous_set = previous.get(entity_type, set())
            new_entities[entity_type] = current_set - previous_set
        
        return new_entities
    
    @staticmethod
    def summarize_diff(diff: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Dict[str, int]]:
        """
        Summarize diff results with counts.
        
        Returns:
            Dict with counts for added/removed/unchanged per entity type
        """
        summary = {}
        for entity_type, changes in diff.items():
            summary[entity_type] = {
                "added": len(changes["added"]),
                "removed": len(changes["removed"]), 
                "unchanged": len(changes["unchanged"])
            }
        return summary
    
    @staticmethod
    def get_counts(entities: Dict[str, Set[str]]) -> Dict[str, int]:
        """Get count of each entity type."""
        return {entity_type: len(ids) for entity_type, ids in entities.items()}
    
    @staticmethod
    def validate_hierarchy(entities: Dict[str, Set[str]], raise_on_violations: bool = True) -> Dict:
        """
        Validate parent-child ID relationships.
        
        Args:
            entities: Entity ID sets to validate
            raise_on_violations: If True, raises ValueError on violations (default)
        
        Raises:
            ValueError: If violations found and raise_on_violations=True
        """
        violations = []
        
        # Check embryos reference valid experiments
        for embryo_id in entities["embryos"]:
            try:
                exp_id = extract_experiment_id(embryo_id)
                if exp_id not in entities["experiments"]:
                    violations.append(f"Embryo {embryo_id} -> missing experiment {exp_id}")
            except ValueError:
                violations.append(f"Invalid embryo ID format: {embryo_id}")
        
        # Check snips reference valid embryos  
        for snip_id in entities["snips"]:
            try:
                embryo_id = parse_entity_id(snip_id, "embryo")
                if embryo_id not in entities["embryos"]:
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
                video_id = parse_entity_id(image_id, "video")
                if video_id not in entities["videos"]:
                    violations.append(f"Image {image_id} -> missing video {video_id}")
            except ValueError:
                violations.append(f"Invalid image ID format: {image_id}")
        
        if violations and raise_on_violations:
            raise ValueError(f"Entity hierarchy violations found:\n" + "\n".join(violations))
        
        return {"valid": len(violations) == 0, "violations": violations}
    
    @staticmethod
    def remove_orphaned_entities(entities: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Remove entities that don't have valid parent references.
        Returns cleaned entity sets.
        """
        cleaned = {key: set(ids) for key, ids in entities.items()}
        
        # Remove orphaned embryos (no valid experiment)
        valid_embryos = set()
        for embryo_id in cleaned["embryos"]:
            try:
                exp_id = extract_experiment_id(embryo_id)
                if exp_id in cleaned["experiments"]:
                    valid_embryos.add(embryo_id)
            except ValueError:
                pass  # Invalid format, remove
        cleaned["embryos"] = valid_embryos
        
        # Remove orphaned snips (no valid embryo)
        valid_snips = set()
        for snip_id in cleaned["snips"]:
            try:
                embryo_id = parse_entity_id(snip_id, "embryo")
                if embryo_id in cleaned["embryos"]:
                    valid_snips.add(snip_id)
            except ValueError:
                pass  # Invalid format, remove
        cleaned["snips"] = valid_snips
        
        # Remove orphaned videos (no valid experiment)
        valid_videos = set()
        for video_id in cleaned["videos"]:
            try:
                exp_id = extract_experiment_id(video_id)
                if exp_id in cleaned["experiments"]:
                    valid_videos.add(video_id)
            except ValueError:
                pass
        cleaned["videos"] = valid_videos
        
        # Remove orphaned images (no valid video)
        valid_images = set()
        for image_id in cleaned["images"]:
            try:
                video_id = parse_entity_id(image_id, "video")
                if video_id in cleaned["videos"]:
                    valid_images.add(image_id)
            except ValueError:
                pass
        cleaned["images"] = valid_images
        
        return cleaned
    
    @staticmethod
    def validate_id_format(entity_id: str) -> str:
        """
        Simple ID validation - detect entity type for debugging.
        Uses parsing_utils.detect_entity_level() if available.
        """
        try:
            from parsing_utils import detect_entity_level
            return detect_entity_level(entity_id)
        except (ImportError, ValueError):
            # Fallback simple detection
            if "_e" in entity_id and "_s" in entity_id:
                return "snip"
            elif "_e" in entity_id:
                return "embryo"
            elif "_" in entity_id:
                return "experiment_or_video_or_image"
            else:
                return "unknown"