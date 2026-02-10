"""
Module 3 Simplified: EmbryoMetadata MVP Implementation

Core class for managing biological annotations layered on top of SAM2 segmentation data.
Uses composition with BaseFileHandler for atomic file operations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .annotation_batch import AnnotationBatch

from ..utils.base_file_handler import BaseFileHandler


class EmbryoMetadata:
    """
    MVP: Biological annotation system with composition approach.
    
    Features:
    - BaseFileHandler composition for atomic file operations
    - SAM2 import and auto-update functionality
    - Basic phenotype annotation with embryo_id + target='all'
    - Hardcoded validation lists (no config files in MVP)
    """
    
    # Default validation lists (fallback if config not found)
    VALID_PHENOTYPES: List[str] = ["NORMAL", "EDEMA", "DEAD", "CONVERGENCE_EXTENSION", "BLUR", "CORRUPT"]
    VALID_GENES: List[str] = ["WT", "tmem67", "lmx1b", "sox9a", "cep290", "b9d2", "rpgrip1l"]
    VALID_ZYGOSITY: List[str] = ["homozygous", "heterozygous", "compound_heterozygous", "crispant", "morpholino"]
    VALID_TREATMENTS: List[str] = ["control", "DMSO", "PTU", "BIO", "SB431542", "DAPT", "heat_shock", "cold_shock"]
    VALID_FLAGS: List[str] = ["MOTION_BLUR", "OUT_OF_FOCUS", "DARK", "CORRUPT"]
    
    def __init__(self, sam2_path: str, annotations_path: Optional[str] = None):
        """
        Initialize with SAM2 file and optional existing annotations.
        
        Args:
            sam2_path: Path to SAM2 annotations JSON file
            annotations_path: Optional path to existing biology annotations
            
        Raises:
            FileNotFoundError: If SAM2 file doesn't exist
            ValueError: If SAM2 file contains invalid JSON or structure
        """
        self.sam2_path = Path(sam2_path)
        
        # Validate SAM2 file exists and is readable
        if not self.sam2_path.exists():
            raise FileNotFoundError(f"SAM2 file not found: {self.sam2_path}")
        
        if not self.sam2_path.is_file():
            raise ValueError(f"SAM2 path is not a file: {self.sam2_path}")
        
        # Validate SAM2 JSON structure early
        try:
            sam2_data = self._load_sam2_data()
            if "experiments" not in sam2_data:
                raise ValueError(f"Invalid SAM2 format: missing 'experiments' key in {self.sam2_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SAM2 file {self.sam2_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading SAM2 file {self.sam2_path}: {e}")
        
        # Determine annotations path
        if annotations_path is None:
            # Default: sam2_annotations.json -> sam2_annotations_biology.json
            self.annotations_path = self.sam2_path.parent / f"{self.sam2_path.stem}_biology.json"
        else:
            self.annotations_path = Path(annotations_path)
        
        # Validate annotations path directory exists
        if not self.annotations_path.parent.exists():
            raise FileNotFoundError(f"Directory for annotations file does not exist: {self.annotations_path.parent}")
        
        # Initialize BaseFileHandler for atomic operations
        self.file_handler = BaseFileHandler(self.annotations_path)
        
        # Enable validation by default
        self.validate = True
        
        # Load configuration (with fallback to hardcoded defaults)
        self.__class__._load_config()
        
        # Load or create data structure
        if self.annotations_path.exists():
            print(f"Loading existing annotations from: {self.annotations_path}")
            try:
                self.data = self.file_handler.load_json()
                # Validate loaded annotation structure
                if not isinstance(self.data, dict) or "embryos" not in self.data:
                    raise ValueError(f"Invalid annotation file structure in {self.annotations_path}: missing 'embryos' key")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in annotations file {self.annotations_path}: {e}")
            except Exception as e:
                raise ValueError(f"Error loading annotations file {self.annotations_path}: {e}")
            
            # Auto-update with any new embryos from SAM2
            self._update_from_sam2()
        else:
            print(f"Creating new annotations from SAM2: {self.sam2_path}")
            self.data = self._create_from_sam2()
    
    def _load_sam2_data(self) -> Dict:
        """Load SAM2 data from file with error handling.
        
        Returns:
            Dict containing SAM2 data
            
        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        try:
            with open(self.sam2_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Re-raise with file context
            raise
        except FileNotFoundError:
            # Re-raise with file context
            raise
        except PermissionError:
            # Re-raise with file context
            raise
    
    def _extract_frame_number(self, image_id: str) -> int:
        """Extract frame number from image_id like '20240418_A01_t0100' -> 100."""
        if '_t' in image_id:
            frame_part = image_id.split('_t')[-1]
            return int(frame_part)
        else:
            raise ValueError(f"Cannot extract frame number from image_id: {image_id}")
    
    def _create_embryo_structure(self, embryo_id: str) -> Dict:
        """Create empty embryo structure for annotations."""
        # Extract experiment and video IDs from embryo_id
        parts = embryo_id.split('_')
        if len(parts) >= 3:
            experiment_id = parts[0]
            video_id = '_'.join(parts[:2])
        else:
            experiment_id = "unknown"
            video_id = "unknown"
        
        return {
            "embryo_id": embryo_id,
            "experiment_id": experiment_id,
            "video_id": video_id,
            "genotype": None,
            "treatments": [],
            "snips": {}
        }
    
    def _create_snip_structure(self, snip_id: str, frame_number: int) -> Dict:
        """Create empty snip structure for frame-level annotations."""
        return {
            "snip_id": snip_id,
            "frame_number": frame_number,
            "phenotypes": [],
            "flags": []
        }
    
    def _extract_embryos_from_sam2(self) -> Dict[str, Dict]:
        """
        Extract embryo structure from SAM2 data.
        
        Returns:
            Dict mapping embryo_id to embryo structure with all snips
        """
        sam2_data = self._load_sam2_data()
        embryos = {}
        
        # Scan through experiments -> videos -> images -> embryos
        for exp_id, exp_data in sam2_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    try:
                        frame_num = self._extract_frame_number(image_id)
                    except ValueError:
                        print(f"Warning: Skipping image with invalid frame format: {image_id}")
                        continue
                    
                    for embryo_id in image_data.get("embryos", {}):
                        # Create embryo structure if first time seeing this embryo
                        if embryo_id not in embryos:
                            embryos[embryo_id] = self._create_embryo_structure(embryo_id)
                        
                        # Add snip for this frame
                        snip_id = f"{embryo_id}_s{frame_num:04d}"
                        embryos[embryo_id]["snips"][snip_id] = self._create_snip_structure(snip_id, frame_num)
        
        return embryos
    
    def _create_from_sam2(self) -> Dict:
        """Create new annotation structure from SAM2 data."""
        embryos = self._extract_embryos_from_sam2()
        
        return {
            "metadata": {
                "source_sam2": str(self.sam2_path),
                "created": datetime.now().isoformat(),
                "version": "simplified_v1"
            },
            "embryos": embryos
        }
    
    def _update_from_sam2(self) -> None:
        """
        Auto-detect new embryos from SAM2 and merge without overwriting existing annotations.
        """
        sam2_embryos = self._extract_embryos_from_sam2()
        
        # Update metadata timestamp
        self.data["metadata"]["updated"] = datetime.now().isoformat()
        
        # Merge embryos
        new_embryo_count = 0
        new_snip_count = 0
        
        for embryo_id, embryo_structure in sam2_embryos.items():
            if embryo_id not in self.data["embryos"]:
                # New embryo - add complete structure
                self.data["embryos"][embryo_id] = embryo_structure
                new_embryo_count += 1
                new_snip_count += len(embryo_structure["snips"])
            else:
                # Existing embryo - only add new snips, preserve annotations
                existing_snips = self.data["embryos"][embryo_id].get("snips", {})
                new_snips = embryo_structure.get("snips", {})
                
                for snip_id, snip_data in new_snips.items():
                    if snip_id not in existing_snips:
                        existing_snips[snip_id] = snip_data
                        new_snip_count += 1
        
        if new_embryo_count > 0:
            print(f"Auto-detected {new_embryo_count} new embryos")
        if new_snip_count > 0:
            print(f"Auto-detected {new_snip_count} new snips")
    
    def _select_mode(self, embryo_id: Optional[str] = None, target: Optional[str] = None, snip_ids: Optional[List[str]] = None) -> str:
        """
        Prevent ambiguous parameter combinations.
        
        Args:
            embryo_id: Embryo ID for embryo-based approach
            target: Target specification for embryo-based approach  
            snip_ids: List of snip IDs for direct snip approach
        
        Returns:
            "embryo" if embryo-based approach, "snips" if snip-based approach
            
        Raises:
            ValueError: If parameters are ambiguous or missing
        """
        by_embryo = embryo_id is not None or target is not None
        by_snips = snip_ids is not None and len(snip_ids) > 0
        
        if by_embryo and by_snips:
            print(f"❌ ERROR: add_phenotype() called with both approaches:")
            print(f"   Embryo approach: embryo_id='{embryo_id}', target='{target}'")
            print(f"   Snip approach: snip_ids={snip_ids}")
            print(f"   SOLUTION: Use either (embryo_id + target) OR snip_ids, not both")
            raise ValueError("Ambiguous parameters: cannot use both embryo and snip approaches")
        
        if not by_embryo and not by_snips:
            print(f"❌ ERROR: add_phenotype() called without specifying target:")
            print(f"   Missing parameters: embryo_id={embryo_id}, target={target}, snip_ids={snip_ids}")
            print(f"   SOLUTION: Provide either (embryo_id='embryo_e01', target='all') OR snip_ids=['snip1', 'snip2']")
            raise ValueError("Missing parameters: must specify either embryo or snip approach")
        
        return "embryo" if by_embryo else "snips"
    
    def _resolve_target_to_snips(self, embryo_id: str, target: str) -> List[str]:
        """
        Resolve target specification to list of snip IDs.
        
        Args:
            embryo_id: Target embryo ID
            target: Target specification ('all', '30:50', '200:', etc.)
            
        Returns:
            List of snip IDs
        """
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found. Available: {list(self.data['embryos'].keys())[:5]}...")
        
        embryo_data = self.data["embryos"][embryo_id]
        available_snips = list(embryo_data["snips"].keys())
        
        if not available_snips:
            raise ValueError(f"No snips available for embryo: {embryo_id}")
        
        if target == "all":
            return available_snips
        
        # For Phase 2, add basic range parsing
        if ":" in target:
            return self._parse_frame_range(embryo_id, target, available_snips)
        
        # Single frame number
        if target.isdigit():
            frame_num = int(target)
            snip_id = f"{embryo_id}_s{frame_num:04d}"
            if snip_id in available_snips:
                return [snip_id]
            else:
                raise ValueError(f"Frame {frame_num} not found for embryo {embryo_id}")
        
        raise ValueError(f"Invalid target format: '{target}'. Use 'all', frame number, or 'start:end' range")
    
    def _parse_frame_range(self, embryo_id: str, target: str, available_snips: List[str]) -> List[str]:
        """
        Parse frame range like '30:50' or '200:' into snip IDs.
        
        Args:
            embryo_id: Embryo ID for generating snip IDs
            target: Range specification like '30:50' or '200:'
            available_snips: List of available snip IDs to filter against
            
        Returns:
            List of snip IDs that exist and fall within range
        """
        try:
            if target.endswith(":"):
                # Open-ended range like '200:'
                start_frame = int(target[:-1])
                end_frame = None
            elif target.startswith(":"):
                # Range from beginning like ':100'  
                start_frame = None
                end_frame = int(target[1:])
            else:
                # Closed range like '30:50'
                start_str, end_str = target.split(":", 1)
                start_frame = int(start_str) if start_str else None
                end_frame = int(end_str) if end_str else None
        except ValueError:
            raise ValueError(f"Invalid range format: '{target}'. Use 'start:end', 'start:', or ':end'")
        
        matching_snips = []
        for snip_id in available_snips:
            # Extract frame number from snip_id like "embryo_e01_s0100"
            try:
                frame_part = snip_id.split("_s")[-1]
                frame_num = int(frame_part)
                
                # Check if frame is in range
                if start_frame is not None and frame_num < start_frame:
                    continue
                if end_frame is not None and frame_num >= end_frame:
                    continue
                    
                matching_snips.append(snip_id)
            except (ValueError, IndexError):
                # Skip malformed snip IDs
                continue
        
        if not matching_snips:
            raise ValueError(f"No snips found in range '{target}' for embryo {embryo_id}")
        
        return matching_snips
    
    def _add_phenotype_to_snip(self, snip_id: str, phenotype: str, author: str) -> None:
        """
        Add phenotype to a specific snip.
        
        Args:
            snip_id: Target snip ID
            phenotype: Phenotype value
            author: Author of annotation
        """
        # Find the embryo for this snip
        embryo_id = "_".join(snip_id.split("_")[:-1])  # Remove the _sXXXX part
        
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo for snip '{snip_id}' not found: {embryo_id}")
        
        embryo_data = self.data["embryos"][embryo_id]
        if snip_id not in embryo_data["snips"]:
            raise ValueError(f"Snip '{snip_id}' not found in embryo {embryo_id}")
        
        snip_data = embryo_data["snips"][snip_id]
        if "phenotypes" not in snip_data:
            snip_data["phenotypes"] = []
        
        # Create phenotype record
        phenotype_record = {
            "value": phenotype,
            "author": author,
            "timestamp": datetime.now().isoformat()
        }
        
        snip_data["phenotypes"].append(phenotype_record)
    
    def _should_skip_dead_frame(self, snip_id: str, phenotype: str, overwrite_dead: bool = False) -> bool:
        """
        Check if frame should be skipped due to DEAD status.
        
        Args:
            snip_id: Target snip ID
            phenotype: Phenotype being added
            overwrite_dead: Whether to allow overwriting DEAD frames
            
        Returns:
            True if frame should be skipped, False otherwise
        """
        # Find embryo for this snip
        embryo_id = "_".join(snip_id.split("_")[:-1])
        embryo_data = self.data["embryos"][embryo_id]
        snip_data = embryo_data["snips"][snip_id]
        
        existing_phenotypes = [p["value"] for p in snip_data.get("phenotypes", [])]
        
        # Skip if trying to add non-DEAD to DEAD frame
        if "DEAD" in existing_phenotypes and phenotype != "DEAD" and not overwrite_dead:
            return True
        
        return False
    
    def _validate_dead_exclusivity(self, snip_id: str, phenotype: str, overwrite_dead: bool = False, strict_mode: bool = False) -> None:
        """
        Validate DEAD exclusivity: DEAD cannot coexist with other phenotypes at same snip.
        
        Args:
            snip_id: Target snip ID
            phenotype: Phenotype being added
            overwrite_dead: Whether to allow overwriting DEAD frames
            strict_mode: If True, raise errors instead of silent skipping
            
        Raises:
            ValueError: If DEAD exclusivity would be violated
        """
        # Find embryo for this snip
        embryo_id = "_".join(snip_id.split("_")[:-1])
        embryo_data = self.data["embryos"][embryo_id]
        snip_data = embryo_data["snips"][snip_id]
        
        existing_phenotypes = [p["value"] for p in snip_data.get("phenotypes", [])]
        
        # Check if snip already has DEAD
        if "DEAD" in existing_phenotypes and phenotype != "DEAD" and not overwrite_dead:
            if strict_mode:
                # Strict mode - raise error for direct snip operations
                print(f"❌ ERROR: Cannot add {phenotype} to DEAD snip {snip_id}")
                print(f"   SOLUTION: Use overwrite_dead=True to override DEAD status")
                raise ValueError(f"DEAD exclusivity violation: snip {snip_id} already has DEAD phenotype")
            # For non-strict mode, we use the skip method instead of raising
        
        # Check if adding DEAD to snip with other phenotypes
        if phenotype == "DEAD" and existing_phenotypes and not overwrite_dead:
            non_dead_phenotypes = [p for p in existing_phenotypes if p != "DEAD"]
            if non_dead_phenotypes:
                print(f"❌ ERROR: Cannot add DEAD to snip {snip_id}")
                print(f"   Existing phenotypes: {non_dead_phenotypes}")
                print(f"   SOLUTION: Use overwrite_dead=True to override, or remove existing phenotypes first")
                raise ValueError(f"DEAD exclusivity violation: snip {snip_id} has existing phenotypes {non_dead_phenotypes}")
    
    def _validate_dead_permanence(self, embryo_id: str, target_frame: int, phenotype: str, overwrite_dead: bool = False) -> None:
        """
        Validate DEAD permanence: Once dead at frame N, all frames >= N must be DEAD.
        
        Args:
            embryo_id: Target embryo ID
            target_frame: Frame number being annotated
            phenotype: Phenotype being added
            overwrite_dead: Whether to allow overriding DEAD permanence
            
        Raises:
            ValueError: If DEAD permanence would be violated
        """
        if phenotype != "DEAD" and not overwrite_dead:
            # Check if embryo is already dead at an earlier frame
            embryo_data = self.data["embryos"][embryo_id]
            
            for snip_id, snip_data in embryo_data["snips"].items():
                # Extract frame number from snip
                try:
                    frame_part = snip_id.split("_s")[-1]
                    frame_num = int(frame_part)
                except (ValueError, IndexError):
                    continue
                
                # Check if this frame has DEAD phenotype
                existing_phenotypes = [p["value"] for p in snip_data.get("phenotypes", [])]
                if "DEAD" in existing_phenotypes and frame_num <= target_frame:
                    print(f"❌ ERROR: Cannot add {phenotype} to frame {target_frame}")
                    print(f"   Embryo {embryo_id} is already DEAD at frame {frame_num}")
                    print(f"   SOLUTION: Use overwrite_dead=True to change death timeline")
                    raise ValueError(f"DEAD permanence violation: embryo {embryo_id} already dead at frame {frame_num}")
    
    def _get_embryo_death_frame(self, embryo_id: str) -> Optional[int]:
        """
        Find the earliest frame where embryo is marked DEAD.
        
        Args:
            embryo_id: Target embryo ID
            
        Returns:
            Frame number of earliest DEAD annotation, or None if not dead
        """
        embryo_data = self.data["embryos"][embryo_id]
        death_frames = []
        
        for snip_id, snip_data in embryo_data["snips"].items():
            # Extract frame number
            try:
                frame_part = snip_id.split("_s")[-1]
                frame_num = int(frame_part)
            except (ValueError, IndexError):
                continue
            
            # Check if this frame has DEAD
            existing_phenotypes = [p["value"] for p in snip_data.get("phenotypes", [])]
            if "DEAD" in existing_phenotypes:
                death_frames.append(frame_num)
        
        return min(death_frames) if death_frames else None
    
    def add_phenotype(self, phenotype: str, author: str, embryo_id: Optional[str] = None, 
                     target: Optional[str] = None, snip_ids: Optional[List[str]] = None, 
                     overwrite_dead: bool = False) -> Dict[str, Any]:
        """
        Enhanced: Add phenotype annotation using either embryo or snip approach.
        
        Args:
            phenotype: Phenotype value (must be in VALID_PHENOTYPES)
            author: Author of the annotation
            embryo_id: Target embryo ID (for embryo approach)
            target: Target specification - 'all', '30:50', '200:', frame number (for embryo approach)
            snip_ids: List of snip IDs (for direct snip approach)
        
        Returns:
            Dict with operation details
            
        Raises:
            ValueError: If parameters are invalid or ambiguous
        """
        # Validate phenotype
        if phenotype not in self.VALID_PHENOTYPES:
            print(f"❌ ERROR: Invalid phenotype '{phenotype}'")
            print(f"   Valid options: {self.VALID_PHENOTYPES}")
            raise ValueError(f"Invalid phenotype '{phenotype}'. Valid options: {self.VALID_PHENOTYPES}")
        
        # Validate parameter combination
        mode = self._select_mode(embryo_id, target, snip_ids)
        
        # Resolve to snip IDs based on approach
        if mode == "embryo":
            if target is None:
                target = "all"  # Default target
            resolved_snips = self._resolve_target_to_snips(embryo_id, target)
        else:  # mode == "snips"
            # Validate that all snip IDs exist
            resolved_snips = []
            for snip_id in snip_ids:
                # Find embryo for this snip
                embryo_id_for_snip = "_".join(snip_id.split("_")[:-1])
                if embryo_id_for_snip not in self.data["embryos"]:
                    raise ValueError(f"Embryo for snip '{snip_id}' not found: {embryo_id_for_snip}")
                if snip_id not in self.data["embryos"][embryo_id_for_snip]["snips"]:
                    raise ValueError(f"Snip '{snip_id}' not found")
                resolved_snips.append(snip_id)
        
        # Apply phenotype to all resolved snips with DEAD validation
        applied_snips = []
        skipped_snips = []
        
        # Determine strict mode once for all operations
        strict_mode = (mode == "snips")  # Direct snip operations are strict
        
        for snip_id in resolved_snips:
            try:
                # Extract frame number for validation
                frame_part = snip_id.split("_s")[-1]
                frame_num = int(frame_part)
                
                # Get embryo ID for this snip
                snip_embryo_id = "_".join(snip_id.split("_")[:-1])
                
                # Check DEAD safety first (using boolean method for silent skipping)
                if not strict_mode and self._should_skip_dead_frame(snip_id, phenotype, overwrite_dead):
                    # Silent skip for DEAD safety
                    skipped_snips.append(snip_id)
                    continue
                
                # For strict mode, validate and let errors bubble up
                if strict_mode:
                    self._validate_dead_exclusivity(snip_id, phenotype, overwrite_dead, strict_mode)
                
                # Validate DEAD permanence with appropriate behavior
                # For direct snip operations, we want strict validation
                # For range operations, we want silent skipping
                
                if phenotype != "DEAD" and not overwrite_dead:
                    death_frame = self._get_embryo_death_frame(snip_embryo_id)
                    if death_frame is not None and frame_num >= death_frame:
                        if strict_mode:
                            # Strict validation - raise error
                            print(f"❌ ERROR: Cannot add {phenotype} to frame {frame_num}")
                            print(f"   Embryo {snip_embryo_id} is already DEAD at frame {death_frame}")
                            print(f"   SOLUTION: Use overwrite_dead=True to change death timeline")
                            raise ValueError(f"DEAD permanence violation: embryo {snip_embryo_id} already dead at frame {death_frame}. Use overwrite_dead=True to change death timeline")
                        else:
                            # Safety mode - silent skip
                            skipped_snips.append(snip_id)
                            continue
                
                # Apply phenotype
                self._add_phenotype_to_snip(snip_id, phenotype, author)
                applied_snips.append(snip_id)
                
            except ValueError as e:
                # Re-raise validation errors (non-safety issues)
                raise
        
        # Return operation details with DEAD safety information
        if mode == "embryo":
            result = {
                "operation": "add_phenotype",
                "approach": "embryo",
                "phenotype": phenotype,
                "embryo_id": embryo_id,
                "target": target,
                "applied_to": applied_snips,
                "count": len(applied_snips)
            }
        else:
            result = {
                "operation": "add_phenotype", 
                "approach": "snips",
                "phenotype": phenotype,
                "snip_ids": snip_ids,
                "applied_to": applied_snips,
                "count": len(applied_snips)
            }
        
        # Add DEAD safety information if frames were skipped
        if skipped_snips:
            result["skipped_dead_frames"] = skipped_snips
            result["skipped_count"] = len(skipped_snips)
            
        return result
    
    def add_genotype(self, gene: str, author: str, embryo_id: str, 
                    allele: Optional[str] = None, zygosity: str = "unknown", 
                    overwrite: bool = False) -> Dict[str, Any]:
        """
        Add genotype annotation to embryo with validation.
        
        Args:
            gene: Gene name (must be in VALID_GENES)
            author: Author of the annotation
            embryo_id: Target embryo ID
            allele: Optional allele specification
            zygosity: Zygosity (must be in VALID_ZYGOSITY)
            overwrite: Whether to overwrite existing genotype
        
        Returns:
            Dict with operation details
            
        Raises:
            ValueError: If validation fails or genotype exists without overwrite
        """
        # Validate gene
        if gene not in self.VALID_GENES:
            print(f"❌ ERROR: Invalid gene '{gene}'")
            print(f"   Valid options: {self.VALID_GENES}")
            raise ValueError(f"Invalid gene '{gene}'. Valid options: {self.VALID_GENES}")
        
        # Validate zygosity
        if zygosity not in self.VALID_ZYGOSITY:
            print(f"❌ ERROR: Invalid zygosity '{zygosity}'")
            print(f"   Valid options: {self.VALID_ZYGOSITY}")
            raise ValueError(f"Invalid zygosity '{zygosity}'. Valid options: {self.VALID_ZYGOSITY}")
        
        # Check embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found")
        
        # Check for existing genotype
        existing_genotype = self.data["embryos"][embryo_id].get("genotype")
        if existing_genotype and not overwrite:
            print(f"❌ ERROR: Embryo {embryo_id} already has genotype")
            print(f"   Existing: {existing_genotype['gene']} ({existing_genotype['zygosity']})")
            print(f"   SOLUTION: Use overwrite=True to replace existing genotype")
            raise ValueError(f"Embryo {embryo_id} already has genotype. Use overwrite=True to replace.")
        
        # Create genotype record
        genotype_record = {
            "gene": gene,
            "allele": allele,
            "zygosity": zygosity,
            "author": author,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to embryo
        self.data["embryos"][embryo_id]["genotype"] = genotype_record
        
        return {
            "operation": "add_genotype",
            "gene": gene,
            "embryo_id": embryo_id,
            "zygosity": zygosity,
            "overwrite": overwrite,
            "previous_genotype": existing_genotype
        }
    
    def add_treatment(self, treatment: str, author: str, embryo_id: str, 
                     temperature_celsius: Optional[float] = None,
                     concentration: Optional[str] = None,
                     notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Add treatment annotation to embryo.
        
        Args:
            treatment: Treatment name (must be in VALID_TREATMENTS)
            author: Author of annotation
            embryo_id: Target embryo ID
            temperature_celsius: Optional temperature
            concentration: Optional concentration
            notes: Optional notes
            
        Returns:
            Dict with operation details
        """
        # Validate treatment
        if treatment not in self.VALID_TREATMENTS:
            print(f"❌ ERROR: Invalid treatment '{treatment}'")
            print(f"   Valid options: {self.VALID_TREATMENTS}")
            raise ValueError(f"Invalid treatment '{treatment}'. Valid options: {self.VALID_TREATMENTS}")
        
        # Check embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found")
        
        # Create treatment record
        treatment_record = {
            "value": treatment,
            "temperature_celsius": temperature_celsius,
            "concentration": concentration,
            "notes": notes,
            "author": author,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to embryo treatments list
        embryo_data = self.data["embryos"][embryo_id]
        if "treatments" not in embryo_data:
            embryo_data["treatments"] = []
        
        embryo_data["treatments"].append(treatment_record)
        
        return {
            "operation": "add_treatment",
            "treatment": treatment,
            "embryo_id": embryo_id,
            "temperature_celsius": temperature_celsius,
            "concentration": concentration
        }
    
    def get_embryo_summary(self, embryo_id: str) -> Dict[str, Any]:
        """Get summary of annotations for an embryo."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found")
        
        embryo_data = self.data["embryos"][embryo_id]
        
        # Count phenotypes by type
        phenotype_counts = {}
        total_snips = len(embryo_data.get("snips", {}))
        
        for snip_data in embryo_data.get("snips", {}).values():
            for phenotype in snip_data.get("phenotypes", []):
                pheno_value = phenotype["value"]
                phenotype_counts[pheno_value] = phenotype_counts.get(pheno_value, 0) + 1
        
        return {
            "embryo_id": embryo_id,
            "experiment_id": embryo_data.get("experiment_id"),
            "video_id": embryo_data.get("video_id"),
            "genotype": embryo_data.get("genotype"),
            "treatment_count": len(embryo_data.get("treatments", [])),
            "total_snips": total_snips,
            "phenotype_counts": phenotype_counts
        }
    
    def list_embryos(self) -> List[str]:
        """Get list of all embryo IDs."""
        return list(self.data["embryos"].keys())
    
    def save(self) -> None:
        """Save annotations to file using BaseFileHandler atomic operations."""
        self.file_handler.save_json(self.data)
        print(f"Saved annotations to: {self.annotations_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the annotation dataset."""
        embryo_count = len(self.data["embryos"])
        total_snips = sum(len(embryo["snips"]) for embryo in self.data["embryos"].values())
        
        # Count annotations
        total_phenotypes = 0
        genotyped_embryos = 0
        
        for embryo_data in self.data["embryos"].values():
            if embryo_data.get("genotype"):
                genotyped_embryos += 1
            
            for snip_data in embryo_data.get("snips", {}).values():
                total_phenotypes += len(snip_data.get("phenotypes", []))
        
        return {
            "embryo_count": embryo_count,
            "total_snips": total_snips,
            "total_phenotypes": total_phenotypes,
            "genotyped_embryos": genotyped_embryos,
            "source_sam2": self.data["metadata"].get("source_sam2"),
            "created": self.data["metadata"].get("created"),
            "updated": self.data["metadata"].get("updated")
        }
    
    def initialize_batch(self, mode: str = "skeleton", author: str = None) -> 'AnnotationBatch':
        """
        Create an AnnotationBatch from current metadata.
        
        Args:
            mode: Initialization mode
                - "skeleton": Empty annotations, preserve embryo/snip structure
                - "copy": Full copy of current annotations
            author: Required batch author
            
        Returns:
            AnnotationBatch instance
        """
        if not author:
            raise ValueError("Author required for batch initialization")
        
        # Import here to avoid circular imports
        from .annotation_batch import AnnotationBatch
        
        if mode == "skeleton":
            # Create empty structure preserving embryo/snip organization
            batch_data = {
                "metadata": {
                    "source_sam2": self.data["metadata"].get("source_sam2"),
                    "created": self.data["metadata"].get("created"),
                    "version": "batch_skeleton",
                    "batch_mode": mode
                },
                "embryos": {}
            }
            
            # Copy embryo structure but clear annotations
            for embryo_id, embryo_data in self.data["embryos"].items():
                batch_data["embryos"][embryo_id] = {
                    "embryo_id": embryo_id,
                    "experiment_id": embryo_data.get("experiment_id"),
                    "video_id": embryo_data.get("video_id"),
                    "genotype": None,
                    "treatments": [],
                    "snips": {}
                }
                
                # Copy snip structure but clear annotations
                for snip_id, snip_data in embryo_data.get("snips", {}).items():
                    batch_data["embryos"][embryo_id]["snips"][snip_id] = {
                        "snip_id": snip_id,
                        "frame_number": snip_data.get("frame_number"),
                        "phenotypes": [],
                        "flags": []
                    }
        
        elif mode == "copy":
            # Full copy of current data
            import copy
            batch_data = copy.deepcopy(self.data)
            batch_data["metadata"]["version"] = "batch_copy"
            batch_data["metadata"]["batch_mode"] = mode
        
        else:
            raise ValueError(f"Invalid batch mode: {mode}. Use 'skeleton' or 'copy'")
        
        return AnnotationBatch(batch_data, author, validate=self.validate)
    
    def apply_batch(self, batch: 'AnnotationBatch', on_conflict: str = "error", dry_run: bool = False) -> Dict:
        """
        Apply batch changes to metadata with conflict resolution.
        
        Args:
            batch: AnnotationBatch instance
            on_conflict: Conflict resolution strategy
                - "error": Fail on any conflict
                - "skip": Keep existing data, skip conflicts
                - "overwrite": Replace existing data completely
                - "merge": Intelligently combine annotations
            dry_run: If True, validate without applying changes
            
        Returns:
            Report with applied count, conflicts, errors
        """
        # Import here to avoid circular imports
        from .annotation_batch import AnnotationBatch
        
        report = {
            "operation": "apply_batch",
            "dry_run": dry_run,
            "on_conflict": on_conflict,
            "applied_count": 0,
            "skipped_count": 0,
            "conflicts": [],
            "errors": []
        }
        
        if not isinstance(batch, AnnotationBatch):
            raise ValueError("Must provide AnnotationBatch instance")
        
        # Apply changes
        for embryo_id, batch_embryo in batch.data["embryos"].items():
            try:
                # Ensure embryo exists in metadata
                if embryo_id not in self.data["embryos"]:
                    report["errors"].append(f"Embryo {embryo_id} not found in metadata")
                    continue
                
                # Apply genotype
                if batch_embryo.get("genotype"):
                    existing_genotype = self.data["embryos"][embryo_id].get("genotype")
                    if existing_genotype and on_conflict == "error":
                        report["conflicts"].append(f"Genotype conflict for {embryo_id}")
                    elif existing_genotype and on_conflict == "skip":
                        report["skipped_count"] += 1
                    elif not dry_run:
                        self.data["embryos"][embryo_id]["genotype"] = batch_embryo["genotype"]
                        report["applied_count"] += 1
                
                # Apply treatments
                if batch_embryo.get("treatments"):
                    if not dry_run:
                        existing_treatments = self.data["embryos"][embryo_id].get("treatments", [])
                        if on_conflict == "overwrite":
                            self.data["embryos"][embryo_id]["treatments"] = batch_embryo["treatments"]
                        else:
                            # Merge treatments
                            self.data["embryos"][embryo_id]["treatments"] = existing_treatments + batch_embryo["treatments"]
                        report["applied_count"] += len(batch_embryo["treatments"])
                
                # Apply phenotypes
                for snip_id, batch_snip in batch_embryo.get("snips", {}).items():
                    if batch_snip.get("phenotypes"):
                        if snip_id not in self.data["embryos"][embryo_id]["snips"]:
                            report["errors"].append(f"Snip {snip_id} not found")
                            continue
                        
                        if not dry_run:
                            existing_phenotypes = self.data["embryos"][embryo_id]["snips"][snip_id].get("phenotypes", [])
                            if on_conflict == "overwrite":
                                self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = batch_snip["phenotypes"]
                            else:
                                # Merge phenotypes
                                self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = existing_phenotypes + batch_snip["phenotypes"]
                            report["applied_count"] += len(batch_snip["phenotypes"])
            
            except Exception as e:
                report["errors"].append(f"Error processing {embryo_id}: {str(e)}")
        
        return report
    
    @classmethod
    def _load_config(cls, config_path: Optional[Path] = None) -> None:
        """
        Load validation lists from config file.
        
        Args:
            config_path: Optional path to config file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update class attributes from config
                if "phenotypes" in config:
                    cls.VALID_PHENOTYPES = config["phenotypes"]
                if "genes" in config:
                    cls.VALID_GENES = config["genes"]
                if "zygosity" in config:
                    cls.VALID_ZYGOSITY = config["zygosity"]
                if "treatments" in config:
                    cls.VALID_TREATMENTS = config["treatments"]
                if "flags" in config:
                    cls.VALID_FLAGS = config["flags"]
                
                print(f"✅ Loaded configuration from: {config_path}")
            else:
                print(f"⚠️ Config file not found: {config_path}, using defaults")
        
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON in config file {config_path}: {e}, using defaults")
        except Exception as e:
            print(f"⚠️ Error loading config file {config_path}: {e}, using defaults")