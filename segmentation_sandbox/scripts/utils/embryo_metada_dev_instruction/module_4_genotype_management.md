# Module 4: Genotype Management

## Overview
This module manages genotype information at the embryo level, including assignment, validation, overwrite protection, and missing genotype warnings.

## Core Genotype Methods

### Add/Update Genotype

```python
def add_genotype(self, embryo_id: str, genotype: str, author: str,
                notes: str = None, confirmed: bool = False, 
                method: str = None, overwrite_genotype: bool = False) -> bool:
    """
    Add or update genotype for an embryo.
    
    Args:
        embryo_id: Embryo identifier
        genotype: Genotype value (user-defined)
        author: Who is setting the genotype
        notes: Optional notes about genotype
        confirmed: Whether genotype is lab-confirmed
        method: Genotyping method (e.g., "PCR", "sequencing")
        overwrite_genotype: Allow overwriting existing genotype
    
    Returns:
        bool: True if genotype was set, False if skipped
    
    Raises:
        ValidationError: If overwrite protection triggered
    """
    # Validate embryo exists
    if embryo_id not in self.data["embryos"]:
        raise ValueError(f"Embryo {embryo_id} not found")
    
    embryo_data = self.data["embryos"][embryo_id]
    existing_genotype = embryo_data.get("genotype")
    
    # Check overwrite protection
    if existing_genotype and not overwrite_genotype:
        existing_value = existing_genotype.get("value", "unknown")
        if self.verbose:
            print(f"⚠️ Embryo {embryo_id} already has genotype '{existing_value}'")
            print(f"   Use overwrite_genotype=True to change")
        raise ValidationError(
            f"Genotype already set to '{existing_value}'. "
            "Use overwrite_genotype=True to change."
        )
    
    # Validate genotype format
    if not genotype or not genotype.strip():
        raise ValidationError("Genotype cannot be empty")
    
    # Create genotype object
    genotype_obj = Genotype(
        value=genotype.strip(),
        author=author,
        notes=notes,
        confirmed=confirmed,
        method=method
    )
    
    # Store genotype
    old_genotype = existing_genotype.get("value") if existing_genotype else None
    embryo_data["genotype"] = genotype_obj.to_dict()
    
    # Track change
    self._unsaved_changes = True
    self._add_change_log("add_genotype", {
        "embryo_id": embryo_id,
        "old_value": old_genotype,
        "new_value": genotype,
        "author": author,
        "overwrite": old_genotype is not None
    })
    
    if self.verbose:
        action = "Updated" if old_genotype else "Added"
        print(f"✅ {action} genotype '{genotype}' for {embryo_id}")
        if confirmed:
            print(f"   Confirmed by: {method or 'unspecified method'}")
    
    # Update missing genotype cache
    self._update_genotype_tracking()
    
    return True
```

### Remove Genotype

```python
def remove_genotype(self, embryo_id: str, author: str, 
                   reason: str = None) -> bool:
    """
    Remove genotype from embryo (rare operation).
    
    Args:
        embryo_id: Embryo identifier
        author: Who is removing the genotype
        reason: Reason for removal (required)
    
    Returns:
        bool: True if removed
    """
    if not reason:
        raise ValueError("Reason required for genotype removal")
    
    if embryo_id not in self.data["embryos"]:
        raise ValueError(f"Embryo {embryo_id} not found")
    
    embryo_data = self.data["embryos"][embryo_id]
    existing_genotype = embryo_data.get("genotype")
    
    if not existing_genotype:
        if self.verbose:
            print(f"⚠️ Embryo {embryo_id} has no genotype to remove")
        return False
    
    # Archive the old genotype
    archived_genotype = existing_genotype.copy()
    archived_genotype["removal_reason"] = reason
    archived_genotype["removed_by"] = author
    archived_genotype["removed_at"] = datetime.now().isoformat()
    
    # Store in archive if exists
    if "archived_genotypes" not in embryo_data:
        embryo_data["archived_genotypes"] = []
    embryo_data["archived_genotypes"].append(archived_genotype)
    
    # Remove genotype
    embryo_data["genotype"] = None
    
    self._unsaved_changes = True
    self._add_change_log("remove_genotype", {
        "embryo_id": embryo_id,
        "removed_value": existing_genotype["value"],
        "author": author,
        "reason": reason
    })
    
    if self.verbose:
        print(f"✅ Removed genotype from {embryo_id}")
        print(f"   Reason: {reason}")
    
    # Update tracking
    self._update_genotype_tracking()
    
    return True
```

## Batch Genotype Operations

### Batch Assignment

```python
def batch_add_genotypes(self, genotype_map: Dict[str, Union[str, Dict]], 
                       author: str, overwrite_all: bool = False) -> Dict:
    """
    Batch assign genotypes to multiple embryos.
    
    Args:
        genotype_map: Mapping of embryo_id to genotype info
            Simple: {"embryo1": "wildtype", "embryo2": "mutant"}
            Detailed: {"embryo1": {"value": "wildtype", "confirmed": True}}
        author: Author for all assignments
        overwrite_all: Allow overwriting all existing genotypes
    
    Returns:
        Results dictionary with added/skipped/failed
    """
    results = {
        "added": [],
        "updated": [],
        "skipped": [],
        "failed": []
    }
    
    total = len(genotype_map)
    if self.verbose:
        print(f"🔄 Batch genotyping {total} embryos...")
    
    for i, (embryo_id, genotype_info) in enumerate(genotype_map.items()):
        try:
            # Parse genotype info
            if isinstance(genotype_info, str):
                genotype = genotype_info
                confirmed = False
                method = None
                notes = None
            else:
                genotype = genotype_info["value"]
                confirmed = genotype_info.get("confirmed", False)
                method = genotype_info.get("method")
                notes = genotype_info.get("notes")
            
            # Check if update or new
            is_update = self._has_genotype(embryo_id)
            
            # Add genotype
            success = self.add_genotype(
                embryo_id, genotype, author,
                notes=notes, confirmed=confirmed, 
                method=method, overwrite_genotype=overwrite_all
            )
            
            if success:
                if is_update:
                    results["updated"].append(embryo_id)
                else:
                    results["added"].append(embryo_id)
                    
        except ValidationError as e:
            if "already set" in str(e):
                results["skipped"].append((embryo_id, "Has genotype"))
            else:
                results["failed"].append((embryo_id, str(e)))
        except Exception as e:
            results["failed"].append((embryo_id, str(e)))
        
        # Progress indicator
        if self.verbose and (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{total}...")
    
    # Summary
    if self.verbose:
        print(f"✅ Results:")
        print(f"   Added: {len(results['added'])}")
        print(f"   Updated: {len(results['updated'])}")
        print(f"   Skipped: {len(results['skipped'])}")
        print(f"   Failed: {len(results['failed'])}")
    
    return results
```

### Import from External Source

```python
def import_genotypes_from_csv(self, csv_path: Path, author: str,
                             embryo_col: str = "embryo_id",
                             genotype_col: str = "genotype",
                             confirmed_col: str = None,
                             method_col: str = None,
                             overwrite: bool = False) -> Dict:
    """
    Import genotypes from CSV file.
    
    Args:
        csv_path: Path to CSV file
        author: Author for all imports
        embryo_col: Column name for embryo IDs
        genotype_col: Column name for genotypes
        confirmed_col: Optional column for confirmation status
        method_col: Optional column for genotyping method
        overwrite: Allow overwriting existing genotypes
    
    Returns:
        Import results
    """
    import csv
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    genotype_map = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Validate columns
        if embryo_col not in reader.fieldnames:
            raise ValueError(f"Column '{embryo_col}' not found in CSV")
        if genotype_col not in reader.fieldnames:
            raise ValueError(f"Column '{genotype_col}' not found in CSV")
        
        for row in reader:
            embryo_id = row[embryo_col].strip()
            genotype = row[genotype_col].strip()
            
            if not embryo_id or not genotype:
                continue
            
            # Build genotype info
            genotype_info = {"value": genotype}
            
            if confirmed_col and confirmed_col in row:
                confirmed_str = row[confirmed_col].strip().lower()
                genotype_info["confirmed"] = confirmed_str in ['true', 'yes', '1']
            
            if method_col and method_col in row:
                genotype_info["method"] = row[method_col].strip()
            
            genotype_map[embryo_id] = genotype_info
    
    if self.verbose:
        print(f"📁 Loaded {len(genotype_map)} genotypes from {csv_path.name}")
    
    # Batch import
    return self.batch_add_genotypes(genotype_map, author, overwrite_all=overwrite)
```

## Missing Genotype Management

### Check Missing Genotypes

```python
def get_missing_genotypes(self, by_experiment: bool = True) -> Dict:
    """
    Get embryos missing genotype information.
    
    Args:
        by_experiment: Group results by experiment
    
    Returns:
        Missing genotype information
    """
    missing = {
        "total_missing": 0,
        "total_embryos": len(self.data["embryos"]),
        "missing_rate": 0.0
    }
    
    if by_experiment:
        missing["by_experiment"] = defaultdict(list)
    else:
        missing["embryo_ids"] = []
    
    # Check each embryo
    for embryo_id, embryo_data in self.data["embryos"].items():
        if not embryo_data.get("genotype"):
            missing["total_missing"] += 1
            
            if by_experiment:
                exp_id = embryo_data["source"]["experiment_id"]
                missing["by_experiment"][exp_id].append(embryo_id)
            else:
                missing["embryo_ids"].append(embryo_id)
    
    # Calculate rate
    if missing["total_embryos"] > 0:
        missing["missing_rate"] = missing["total_missing"] / missing["total_embryos"]
    
    # Convert defaultdict to regular dict
    if by_experiment:
        missing["by_experiment"] = dict(missing["by_experiment"])
    
    return missing

def show_missing_genotypes_warning(self) -> None:
    """Display warning about missing genotypes."""
    missing_info = self.get_missing_genotypes(by_experiment=True)
    
    if missing_info["total_missing"] == 0:
        return
    
    print("\n⚠️  MISSING GENOTYPE WARNING")
    print("=" * 40)
    print(f"Total embryos: {missing_info['total_embryos']}")
    print(f"Missing genotypes: {missing_info['total_missing']} ({missing_info['missing_rate']:.1%})")
    
    if missing_info.get("by_experiment"):
        print("\nBy experiment:")
        for exp_id, embryos in sorted(missing_info["by_experiment"].items()):
            print(f"  {exp_id}: {len(embryos)} embryos")
            if len(embryos) <= 5:
                print(f"    {', '.join(embryos)}")
            else:
                print(f"    {', '.join(embryos[:3])}, ... ({len(embryos)-3} more)")
```

### Auto-check on Load

```python
def _check_genotypes_on_load(self) -> None:
    """Automatically check for missing genotypes when loading."""
    missing_info = self.get_missing_genotypes()
    
    if missing_info["total_missing"] > 0:
        self.show_missing_genotypes_warning()
        
        # Set flag for tracking
        self._has_missing_genotypes = True
        self._missing_genotype_count = missing_info["total_missing"]
```

## Genotype Validation

### Validate Genotype Format

```python
def validate_genotype_format(self, genotype: str, 
                            strict_mode: bool = False) -> Tuple[bool, str]:
    """
    Validate genotype format/nomenclature.
    
    Args:
        genotype: Genotype string to validate
        strict_mode: Use strict nomenclature rules
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic validation
    if not genotype or not genotype.strip():
        return False, "Genotype cannot be empty"
    
    genotype = genotype.strip()
    
    # Check length
    if len(genotype) > 100:
        return False, "Genotype too long (max 100 characters)"
    
    # Strict mode validation
    if strict_mode:
        # Example: Check zebrafish nomenclature
        import re
        
        # Wild type
        if genotype.lower() in ["wildtype", "wt", "wild-type", "+"]:
            return True, ""
        
        # Allele format: gene^allele (e.g., tp53^M214K)
        allele_pattern = r'^[a-zA-Z][a-zA-Z0-9]*\^[a-zA-Z0-9]+$'
        if re.match(allele_pattern, genotype):
            return True, ""
        
        # Transgenic format: Tg(promoter:gene)
        tg_pattern = r'^Tg\([a-zA-Z0-9]+:[a-zA-Z0-9]+\)$'
        if re.match(tg_pattern, genotype):
            return True, ""
        
        return False, "Genotype does not match standard nomenclature"
    
    # Basic validation passed
    return True, ""

def add_permitted_genotype(self, genotype: str, description: str = "") -> None:
    """
    Add a genotype to permitted values.
    
    Args:
        genotype: Genotype to add
        description: Optional description
    """
    if "genotypes" not in self.data["permitted_values"]:
        self.data["permitted_values"]["genotypes"] = {}
    
    self.data["permitted_values"]["genotypes"][genotype] = {
        "description": description,
        "added_date": datetime.now().isoformat()
    }
    
    self._unsaved_changes = True
    
    if self.verbose:
        print(f"✅ Added '{genotype}' to permitted genotypes")
```

## Genotype Statistics

```python
def get_genotype_statistics(self, include_phenotype_correlation: bool = True) -> Dict:
    """
    Calculate genotype statistics.
    
    Args:
        include_phenotype_correlation: Include phenotype breakdown by genotype
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_embryos": len(self.data["embryos"]),
        "genotyped_embryos": 0,
        "confirmed_genotypes": 0,
        "genotype_counts": defaultdict(int),
        "genotype_methods": defaultdict(int),
        "missing_by_experiment": defaultdict(int)
    }
    
    # Collect genotype data
    for embryo_id, embryo_data in self.data["embryos"].items():
        genotype_info = embryo_data.get("genotype")
        
        if genotype_info:
            stats["genotyped_embryos"] += 1
            genotype_value = genotype_info["value"]
            stats["genotype_counts"][genotype_value] += 1
            
            if genotype_info.get("confirmed"):
                stats["confirmed_genotypes"] += 1
            
            method = genotype_info.get("method", "unspecified")
            stats["genotype_methods"][method] += 1
        else:
            exp_id = embryo_data["source"]["experiment_id"]
            stats["missing_by_experiment"][exp_id] += 1
    
    # Calculate rates
    if stats["total_embryos"] > 0:
        stats["genotyping_rate"] = stats["genotyped_embryos"] / stats["total_embryos"]
        stats["confirmation_rate"] = stats["confirmed_genotypes"] / max(1, stats["genotyped_embryos"])
    else:
        stats["genotyping_rate"] = 0
        stats["confirmation_rate"] = 0
    
    # Phenotype correlation
    if include_phenotype_correlation:
        stats["phenotype_by_genotype"] = self._calculate_phenotype_genotype_correlation()
    
    # Convert defaultdicts
    stats["genotype_counts"] = dict(stats["genotype_counts"])
    stats["genotype_methods"] = dict(stats["genotype_methods"])
    stats["missing_by_experiment"] = dict(stats["missing_by_experiment"])
    
    return stats

def _calculate_phenotype_genotype_correlation(self) -> Dict:
    """Calculate phenotype distribution by genotype."""
    correlation = defaultdict(lambda: defaultdict(int))
    
    for embryo_id, embryo_data in self.data["embryos"].items():
        genotype_info = embryo_data.get("genotype")
        if not genotype_info:
            continue
        
        genotype = genotype_info["value"]
        
        # Count phenotypes for this embryo
        phenotypes = set()
        for snip_data in embryo_data["snips"].values():
            phenotype = snip_data["phenotype"]["value"]
            if phenotype != "NONE":
                phenotypes.add(phenotype)
        
        if phenotypes:
            for phenotype in phenotypes:
                correlation[genotype][phenotype] += 1
        else:
            correlation[genotype]["NONE"] += 1
    
    # Convert to regular dict
    return {
        genotype: dict(phenotypes) 
        for genotype, phenotypes in correlation.items()
    }
```

## Export Functions

```python
def export_genotypes(self, output_path: Path, include_metadata: bool = True) -> None:
    """
    Export genotype data.
    
    Args:
        output_path: Output file path
        include_metadata: Include additional metadata
    """
    import csv
    
    rows = []
    for embryo_id, embryo_data in self.data["embryos"].items():
        row = {
            "embryo_id": embryo_id,
            "experiment_id": embryo_data["source"]["experiment_id"],
            "video_id": embryo_data["source"]["video_id"]
        }
        
        genotype_info = embryo_data.get("genotype")
        if genotype_info:
            row.update({
                "genotype": genotype_info["value"],
                "author": genotype_info["author"],
                "timestamp": genotype_info["timestamp"],
                "confirmed": genotype_info.get("confirmed", False),
                "method": genotype_info.get("method", ""),
                "notes": genotype_info.get("notes", "")
            })
        else:
            row.update({
                "genotype": "",
                "author": "",
                "timestamp": "",
                "confirmed": "",
                "method": "",
                "notes": "MISSING"
            })
        
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    if self.verbose:
        print(f"📊 Exported {len(rows)} genotype records to {output_path}")
```

## Helper Methods

```python
def _has_genotype(self, embryo_id: str) -> bool:
    """Check if embryo has genotype assigned."""
    embryo_data = self.data["embryos"].get(embryo_id, {})
    return embryo_data.get("genotype") is not None

def _update_genotype_tracking(self) -> None:
    """Update internal genotype tracking after changes."""
    missing_info = self.get_missing_genotypes()
    self._has_missing_genotypes = missing_info["total_missing"] > 0
    self._missing_genotype_count = missing_info["total_missing"]

def _get_genotype_history(self, embryo_id: str) -> List[Dict]:
    """Get genotype change history for an embryo."""
    history = []
    
    # Current genotype
    embryo_data = self.data["embryos"].get(embryo_id, {})
    current = embryo_data.get("genotype")
    if current:
        history.append({
            "status": "current",
            "genotype": current["value"],
            "author": current["author"],
            "timestamp": current["timestamp"]
        })
    
    # Archived genotypes
    archived = embryo_data.get("archived_genotypes", [])
    for arch in archived:
        history.append({
            "status": "archived",
            "genotype": arch["value"],
            "author": arch["author"],
            "timestamp": arch["timestamp"],
            "removed_by": arch.get("removed_by"),
            "removal_reason": arch.get("removal_reason")
        })
    
    # Sort by timestamp
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return history
```