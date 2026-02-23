"""
 AnnotationBatch: Composition-based batch annotation system

Provides a temporary workspace for safely manipulating embryo annotations
before applying them to the persistent EmbryoMetadata store.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from pathlib import Path


class AnnotationBatch:
    """
    Temporary workspace using composition to reuse validation logic.
    
    Uses composition instead of inheritance for cleaner separation of concerns
    and better maintainability. Shares configuration system with EmbryoMetadata.
    """
    
    # Default validation lists (loaded from config in __init__)
    VALID_PHENOTYPES: List[str] = ["NORMAL", "EDEMA", "DEAD", "CONVERGENCE_EXTENSION", "BLUR", "CORRUPT"]
    VALID_GENES: List[str] = ["WT", "tmem67", "lmx1b", "sox9a", "cep290", "b9d2", "rpgrip1l"]
    VALID_ZYGOSITY: List[str] = ["homozygous", "heterozygous", "compound_heterozygous", "crispant", "morpholino"]
    VALID_TREATMENTS: List[str] = ["control", "DMSO", "PTU", "BIO", "SB431542", "DAPT", "heat_shock", "cold_shock"]
    VALID_FLAGS: List[str] = ["MOTION_BLUR", "OUT_OF_FOCUS", "DARK", "CORRUPT"]
    
    def __init__(self, data_structure: Dict, author: str, validate: bool = True):
        """
        Initialize batch workspace with composition.
        
        Args:
            data_structure: Skeleton structure from metadata.initialize_batch()
            author: Default author for all operations
            validate: Enable/disable validation
        """
        if not author:
            raise ValueError("Author required for AnnotationBatch")
        
        self.data = data_structure
        self.validate = validate
        self.author = author
        
        # Load configuration to keep in sync with EmbryoMetadata
        self._load_config()
    
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
        
        # For basic range parsing
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
    
    def _validate_snip_ids(self, snip_ids: List[str]) -> List[str]:
        """Validate that all snip IDs exist."""
        validated_snips = []
        for snip_id in snip_ids:
            # Find embryo for this snip
            embryo_id_for_snip = "_".join(snip_id.split("_")[:-1])
            if embryo_id_for_snip not in self.data["embryos"]:
                raise ValueError(f"Embryo for snip '{snip_id}' not found: {embryo_id_for_snip}")
            if snip_id not in self.data["embryos"][embryo_id_for_snip]["snips"]:
                raise ValueError(f"Snip '{snip_id}' not found")
            validated_snips.append(snip_id)
        return validated_snips
    
    def _should_skip_dead_frame(self, snip_id: str, phenotype: str) -> bool:
        """
        Check if frame should be skipped due to DEAD status.
        
        Args:
            snip_id: Target snip ID
            phenotype: Phenotype being added
            
        Returns:
            True if frame should be skipped, False otherwise
        """
        # Find embryo for this snip
        embryo_id = "_".join(snip_id.split("_")[:-1])
        embryo_data = self.data["embryos"][embryo_id]
        snip_data = embryo_data["snips"][snip_id]
        
        existing_phenotypes = [p["value"] for p in snip_data.get("phenotypes", [])]
        
        # Skip if trying to add non-DEAD to DEAD frame
        if "DEAD" in existing_phenotypes and phenotype != "DEAD":
            return True
        
        return False
    
    def _add_phenotype_to_snip(self, snip_id: str, phenotype: str, author: str) -> None:
        """
        Add phenotype to a specific snip.
        
        Args:
            snip_id: Target snip ID
            phenotype: Phenotype value
            author: Author of annotation
        """
        # Find the embryo for this snip
        embryo_id = "_".join(snip_id.split("_")[:-1])
        
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
    
    def add_phenotype(self, phenotype: str, author: Optional[str] = None, embryo_id: Optional[str] = None, 
                     target: Optional[str] = None, snip_ids: Optional[List[str]] = None, 
                     overwrite_dead: bool = False) -> Dict[str, Any]:
        """
        Add phenotype annotation with batch author default.
        
        Args:
            phenotype: Phenotype value
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID (embryo approach)
            target: Target specification (embryo approach)
            snip_ids: List of snip IDs (snip approach)
            overwrite_dead: Whether to overwrite DEAD frames
            
        Returns:
            Dict with operation details
        """
        # Validate phenotype
        if self.validate and phenotype not in self.VALID_PHENOTYPES:
            print(f"❌ ERROR: Invalid phenotype '{phenotype}'")
            print(f"   Valid options: {self.VALID_PHENOTYPES}")
            raise ValueError(f"Invalid phenotype '{phenotype}'. Valid options: {self.VALID_PHENOTYPES}")
        
        author = author or self.author
        
        # Validate parameter combination
        mode = self._select_mode(embryo_id, target, snip_ids)
        
        # Resolve to snip IDs based on approach
        if mode == "embryo":
            if target is None:
                target = "all"
            resolved_snips = self._resolve_target_to_snips(embryo_id, target)
        else:
            resolved_snips = self._validate_snip_ids(snip_ids)
        
        # Apply phenotype to all resolved snips
        applied_snips = []
        skipped_snips = []
        
        for snip_id in resolved_snips:
            if not overwrite_dead and self._should_skip_dead_frame(snip_id, phenotype):
                skipped_snips.append(snip_id)
                continue
            
            self._add_phenotype_to_snip(snip_id, phenotype, author)
            applied_snips.append(snip_id)
        
        # Return operation details
        result = {
            "operation": "add_phenotype",
            "approach": mode,
            "phenotype": phenotype,
            "applied_to": applied_snips,
            "count": len(applied_snips)
        }
        
        if mode == "embryo":
            result["embryo_id"] = embryo_id
            result["target"] = target
        else:
            result["snip_ids"] = snip_ids
        
        if skipped_snips:
            result["skipped_dead_frames"] = skipped_snips
            result["skipped_count"] = len(skipped_snips)
            
        return result
    
    def add_genotype(self, gene: str, author: Optional[str] = None, embryo_id: Optional[str] = None,
                    allele: Optional[str] = None, zygosity: str = "unknown",
                    overwrite: bool = False) -> Dict[str, Any]:
        """
        Add genotype annotation with batch author default.
        
        Args:
            gene: Gene name
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID
            allele: Optional allele specification
            zygosity: Zygosity specification
            overwrite: Whether to overwrite existing genotype
            
        Returns:
            Dict with operation details
        """
        # Validate inputs
        if self.validate:
            if gene not in self.VALID_GENES:
                print(f"❌ ERROR: Invalid gene '{gene}'")
                print(f"   Valid options: {self.VALID_GENES}")
                raise ValueError(f"Invalid gene '{gene}'. Valid options: {self.VALID_GENES}")
            
            if zygosity not in self.VALID_ZYGOSITY:
                print(f"❌ ERROR: Invalid zygosity '{zygosity}'")
                print(f"   Valid options: {self.VALID_ZYGOSITY}")
                raise ValueError(f"Invalid zygosity '{zygosity}'. Valid options: {self.VALID_ZYGOSITY}")
        
        # Check embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found")
        
        author = author or self.author
        
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
    
    def add_treatment(self, treatment: str, author: Optional[str] = None, embryo_id: Optional[str] = None,
                     temperature_celsius: Optional[float] = None,
                     concentration: Optional[str] = None,
                     notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Add treatment annotation with batch author default.
        
        Args:
            treatment: Treatment name
            author: Author (defaults to batch author if not provided)
            embryo_id: Target embryo ID
            temperature_celsius: Optional temperature
            concentration: Optional concentration
            notes: Optional notes
            
        Returns:
            Dict with operation details
        """
        # Validate treatment
        if self.validate and treatment not in self.VALID_TREATMENTS:
            print(f"❌ ERROR: Invalid treatment '{treatment}'")
            print(f"   Valid options: {self.VALID_TREATMENTS}")
            raise ValueError(f"Invalid treatment '{treatment}'. Valid options: {self.VALID_TREATMENTS}")
        
        # Check embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo '{embryo_id}' not found")
        
        author = author or self.author
        
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
    
    def preview(self, limit: int = 10) -> str:
        """
        Generate human-readable summary of batch contents.
        
        Args:
            limit: Maximum number of embryos to show in detail
            
        Returns:
            Formatted preview string
        """
        lines = [f"AnnotationBatch (Author: {self.author})", ""]
        
        embryo_count = 0
        total_phenotypes = 0
        total_genotypes = 0
        total_treatments = 0
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if embryo_count >= limit:
                lines.append(f"... and {len(self.data['embryos']) - limit} more embryos")
                break
            
            # Collect statistics for this embryo
            stats_parts = []
            
            # Genotype info
            if embryo_data.get("genotype"):
                g = embryo_data["genotype"]
                stats_parts.append(f"🧬 {g['gene']} ({g['zygosity']})")
                total_genotypes += 1
            
            # Phenotype info
            phenotype_counts = {}
            for snip_data in embryo_data.get("snips", {}).values():
                for phenotype in snip_data.get("phenotypes", []):
                    pheno_value = phenotype["value"]
                    phenotype_counts[pheno_value] = phenotype_counts.get(pheno_value, 0) + 1
                    total_phenotypes += 1
            
            if phenotype_counts:
                # Show top 3 phenotypes with counts
                top_phenotypes = sorted(phenotype_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                pheno_summary = ", ".join([f"{pheno}:{count}" for pheno, count in top_phenotypes])
                total_snip_count = sum(phenotype_counts.values())
                stats_parts.append(f"🔬 {total_snip_count} phenotypes ({pheno_summary})")
            
            # Treatment info
            treatment_count = len(embryo_data.get("treatments", []))
            if treatment_count > 0:
                stats_parts.append(f"💊 {treatment_count} treatments")
                total_treatments += treatment_count
            
            # Display embryo summary
            if stats_parts:
                lines.append(f"📋 {embryo_id}: {' | '.join(stats_parts)}")
            else:
                lines.append(f"📋 {embryo_id}: (no annotations)")
            
            embryo_count += 1
        
        # Add overall summary
        lines.append("")
        lines.append(f"Summary: {len(self.data['embryos'])} embryos, {total_genotypes} genotypes, "
                    f"{total_phenotypes} phenotypes, {total_treatments} treatments")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about batch contents."""
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
            "batch_author": self.author,
            "is_batch": True,
            "source_sam2": self.data["metadata"].get("source_sam2"),
            "created": self.data["metadata"].get("created"),
            "updated": self.data["metadata"].get("updated")
        }
    
    def save(self):
        """Override save to prevent accidental saves of batch data."""
        raise NotImplementedError("AnnotationBatch cannot be saved directly. Use metadata.apply_batch() instead.")
    
    def _load_config(self, config_path: Optional[Path] = None) -> None:
        """
        Load validation lists from config file (same as EmbryoMetadata).
        
        Args:
            config_path: Optional path to config file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update instance attributes from config (not class attributes)
                if "phenotypes" in config:
                    self.VALID_PHENOTYPES = config["phenotypes"]
                if "genes" in config:
                    self.VALID_GENES = config["genes"]
                if "zygosity" in config:
                    self.VALID_ZYGOSITY = config["zygosity"]
                if "treatments" in config:
                    self.VALID_TREATMENTS = config["treatments"]
                if "flags" in config:
                    self.VALID_FLAGS = config["flags"]
        
        except (json.JSONDecodeError, Exception):
            # Silently fall back to defaults for batch operations
            pass