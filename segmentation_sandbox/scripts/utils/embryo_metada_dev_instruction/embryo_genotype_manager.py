"""
Embryo Genotype Manager
Handles all genotype-related operations for EmbryoMetadata.
Enforces single genotype per embryo rule (experimental design requirement).
Designed as a mixin class to keep embryo_metadata.py manageable.
"""

from typing import Dict, List, Optional
from embryo_metadata_models import Genotype, ValidationError


class EmbryoGenotypeManager:
    """
    Mixin class for genotype management operations.
    
    CRITICAL: Enforces single genotype per embryo rule.
    
    Required attributes from parent class:
    - self.data: Dict containing embryo metadata
    - self.schema_manager: PermittedValuesManager instance
    - self.verbose: bool for logging
    - self.get_timestamp(): method for timestamps
    - self.log_operation(): method for operation logging
    - self.validate_id_format(): method for ID validation
    """
    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str,
                    zygosity: str = "heterozygous", confidence: float = 1.0,
                    notes: str = "", overwrite: bool = False) -> bool:
        """
        Add a genotype to an embryo. 
        
        CRITICAL: Only ONE genotype per embryo is allowed (experimental design requirement).
        If embryo already has a genotype, this will fail unless overwrite=True.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Gene name (must be in permitted values)
            allele: Allele designation
            zygosity: Zygosity (homozygous, heterozygous, hemizygous)
            confidence: Confidence score (0.0-1.0)
            notes: Optional notes
            overwrite: Whether to overwrite existing genotype
            
        Returns:
            bool: True if added successfully
            
        Raises:
            ValueError: If validation fails or multiple genotypes attempted
        """
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate gene name
        if not self.schema_manager.is_valid_genotype(gene_name):
            available = list(self.schema_manager.get_genotypes().keys())
            raise ValueError(f"Invalid gene '{gene_name}'. Available: {available}")
        
        # Validate zygosity
        zygosity_types = ["homozygous", "heterozygous", "hemizygous"]
        if zygosity not in zygosity_types:
            raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {zygosity_types}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Initialize embryo data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "phenotypes": {},
                "genotypes": {},
                "flags": {},
                "treatments": {},
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                }
            }
        
        # 🚨 CRITICAL: ENFORCE SINGLE GENOTYPE PER EMBRYO 🚨
        existing_genotypes = self.data["embryos"][embryo_id]["genotypes"]
        
        # Case 1: No existing genotypes - OK to add
        if not existing_genotypes:
            pass  # Good to go
            
        # Case 2: Overwriting the same gene - OK
        elif gene_name in existing_genotypes and overwrite:
            if self.verbose:
                print(f"🔄 Overwriting existing genotype for gene '{gene_name}' in {embryo_id}")
            
        # Case 3: Adding same gene without overwrite - Error
        elif gene_name in existing_genotypes and not overwrite:
            existing = existing_genotypes[gene_name]
            raise ValueError(f"Genotype for '{gene_name}' already exists for {embryo_id}. "
                           f"Current: {existing}. Use overwrite=True to replace.")
            
        # Case 4: Different gene when genotype already exists - FORBIDDEN
        elif existing_genotypes and gene_name not in existing_genotypes:
            existing_genes = list(existing_genotypes.keys())
            raise ValueError(f"❌ SINGLE GENOTYPE RULE VIOLATION: Embryo {embryo_id} already has "
                           f"genotype for {existing_genes}. Only ONE genotype per embryo is allowed. "
                           f"Cannot add additional gene '{gene_name}'. This is an experimental design constraint.")
        
        # Create genotype using our model
        genotype = Genotype(
            value=gene_name,
            author=self.config.get("default_author", "unknown"),
            allele=allele,
            zygosity=zygosity,
            confidence=confidence,
            notes=notes
        )
        
        # Add to data (clear existing genotypes to ensure single genotype)
        if not overwrite and existing_genotypes:
            # This should never happen due to above checks, but safety net
            raise ValueError(f"Internal error: attempting to add genotype when {len(existing_genotypes)} already exist")
        
        # If overwriting, clear all genotypes and add the new one (single genotype rule)
        self.data["embryos"][embryo_id]["genotypes"] = {gene_name: genotype.to_dict()}
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        self.log_operation("add_genotype", embryo_id,
                         gene=gene_name, allele=allele, zygosity=zygosity, confidence=confidence,
                         enforced_single_genotype=True)
        
        if self.verbose:
            print(f"🧬 Added genotype '{gene_name}:{allele}' to {embryo_id} (single genotype enforced)")
        
        return True
    
    def edit_genotype(self, embryo_id: str, gene_name: str,
                     allele: str = None, zygosity: str = None,
                     confidence: float = None, notes: str = None) -> bool:
        """
        Edit an existing genotype.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Name of existing gene
            allele: New allele (optional)
            zygosity: New zygosity (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if genotype exists
        if gene_name not in self.data["embryos"][embryo_id]["genotypes"]:
            available = list(self.data["embryos"][embryo_id]["genotypes"].keys())
            raise ValueError(f"Genotype for '{gene_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current genotype
        current = self.data["embryos"][embryo_id]["genotypes"][gene_name]
        
        # Update fields if provided
        if allele is not None:
            current["allele"] = allele
        
        if zygosity is not None:
            if not self.schema_manager.validate_value("zygosity_types", zygosity):
                available = self.schema_manager.get_values("zygosity_types")
                raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {available}")
            current["zygosity"] = zygosity
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if allele is not None:
            changes["allele"] = allele
        if zygosity is not None:
            changes["zygosity"] = zygosity
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_genotype", embryo_id,
                         gene=gene_name, changes=changes)
        
        if self.verbose:
            print(f"🧬 Updated genotype for '{gene_name}' in {embryo_id}")
        
        return True
    
    def remove_genotype(self, embryo_id: str, gene_name: str) -> bool:
        """
        Remove a genotype from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Name of gene to remove
            
        Returns:
            bool: True if removed successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if genotype exists
        if gene_name not in self.data["embryos"][embryo_id]["genotypes"]:
            available = list(self.data["embryos"][embryo_id]["genotypes"].keys())
            raise ValueError(f"Genotype for '{gene_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove genotype
        removed = self.data["embryos"][embryo_id]["genotypes"].pop(gene_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        self.log_operation("remove_genotype", embryo_id,
                         gene=gene_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed genotype for '{gene_name}' from {embryo_id}")
        
        return True
    
    def get_genotypes(self, embryo_id: str) -> dict:
        """
        Get all genotypes for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Genotypes data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id]["genotypes"].copy()
    
    def list_genotypes_by_gene(self, gene_name: str) -> list:
        """
        Find all embryos with genotypes for a specific gene.
        
        Args:
            gene_name: Name of gene to search for
            
        Returns:
            list: List of embryo IDs with this gene
        """
        embryos_with_gene = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if gene_name in embryo_data["genotypes"]:
                embryos_with_gene.append(embryo_id)
        
        return embryos_with_gene
    
    def get_genotype_statistics(self) -> Dict:
        """
        Get comprehensive genotype statistics.
        
        Returns:
            Dict containing genotype counts, completion rates, etc.
        """
        stats = {
            "total_embryos": len(self.data["embryos"]),
            "genotyped_embryos": 0,
            "completion_rate": 0.0,
            "gene_counts": {},
            "allele_counts": {},
            "zygosity_counts": {},
            "single_genotype_compliance": True  # Always true due to enforcement
        }
        
        for embryo_data in self.data["embryos"].values():
            genotypes = embryo_data.get("genotypes", {})
            if genotypes:
                stats["genotyped_embryos"] += 1
                
                # Should only be one genotype per embryo due to enforcement
                for gene_name, genotype_data in genotypes.items():
                    # Count genes
                    stats["gene_counts"][gene_name] = stats["gene_counts"].get(gene_name, 0) + 1
                    
                    # Count alleles
                    allele = genotype_data.get("allele", "unknown")
                    stats["allele_counts"][allele] = stats["allele_counts"].get(allele, 0) + 1
                    
                    # Count zygosity
                    zygosity = genotype_data.get("zygosity", "unknown")
                    stats["zygosity_counts"][zygosity] = stats["zygosity_counts"].get(zygosity, 0) + 1
        
        # Calculate completion rate
        if stats["total_embryos"] > 0:
            stats["completion_rate"] = stats["genotyped_embryos"] / stats["total_embryos"]
        
        return stats
    
    def validate_single_genotype_compliance(self) -> Dict:
        """
        Validate that all embryos comply with single genotype rule.
        
        Returns:
            Dict with compliance status and any violations found
        """
        compliance = {
            "compliant": True,
            "violations": [],
            "total_checked": 0,
            "compliant_count": 0
        }
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            compliance["total_checked"] += 1
            genotypes = embryo_data.get("genotypes", {})
            
            if len(genotypes) <= 1:
                compliance["compliant_count"] += 1
            else:
                compliance["compliant"] = False
                compliance["violations"].append({
                    "embryo_id": embryo_id,
                    "genotype_count": len(genotypes),
                    "genes": list(genotypes.keys())
                })
        
        return compliance
