from typing import List

# Module 3: Phenotype Management

## Overview
This module handles all phenotype-related operations including adding, updating, removing phenotypes with special handling for the DEAD phenotype and temporal tracking.

## Core Phenotype Methods

### Add Single Phenotype

```python
def add_phenotype(self, snip_id: str, phenotype: str, author: str, 
                  notes: str = None, confidence: float = None,
                  force_dead: bool = False) -> bool:
    """
    Add phenotype to a specific snip.
    
    Args:
        snip_id: Snip identifier
        phenotype: Phenotype value (must be in permitted values)
        author: Who is adding the phenotype
        notes: Optional notes
        confidence: Optional confidence score (0-1) for ML predictions
        force_dead: Override DEAD phenotype protection
    
    Returns:
        bool: True if phenotype was added, False if skipped
    
    Raises:
        ValidationError: If phenotype invalid or rules violated
    """
    
    # Find embryo and snip
    embryo_id = self._get_embryo_id_from_snip(snip_id)
    if not embryo_id:
        raise ValueError(f"Snip {snip_id} not found")
    
    embryo_data = self.data["embryos"][embryo_id]
    snip_data = embryo_data["snips"][snip_id]

    # Get frame of death if embryo is dead
    death_frame = self._get_death_frame(embryo_id)
    snip_frame = self._extract_frame_number(snip_id)

    # Disallow adding non-DEAD phenotype after embryo death frame
    if death_frame is not None and snip_frame >= death_frame:
        if phenotype != "DEAD" and not force_dead:
            if self.verbose:
                print(f"⚠️ Skipping {snip_id} — embryo died at frame {death_frame}")
            return False
    
    # Validate phenotype
    permitted = self.data["permitted_values"]["phenotypes"]
    validator = Validator()
    
    
    # Get existing phenotypes for validation
    existing_phenotypes = self._get_embryo_phenotypes(embryo_id)
    validator.validate_phenotype(phenotype, permitted, existing_phenotypes)
    
    # Create phenotype object
    phenotype_obj = Phenotype(
        value=phenotype,
        author=author,
        notes=notes,
        confidence=confidence
    )
    
    # Store phenotype
    old_phenotype = snip_data["phenotype"]["value"]
    snip_data["phenotype"] = phenotype_obj.to_dict()
    
    # Mark as changed
    self._unsaved_changes = True
    self._add_change_log("add_phenotype", {
        "snip_id": snip_id,
        "old_value": old_phenotype,
        "new_value": phenotype,
        "author": author
    })
    
    if self.verbose:
        print(f"✅ Added phenotype '{phenotype}' to {snip_id}")
    
    # Handle DEAD phenotype propagation
    if phenotype == "DEAD":
        self._handle_dead_phenotype(embryo_id, snip_id, author)
    
    return True
```

### Remove Phenotype

```python
def remove_phenotype(self, snip_id: str, author: str, 
                    allow_dead_removal: bool = False) -> bool:
    """
    Remove phenotype (set to NONE).
    
    Args:
        snip_id: Snip identifier
        author: Who is removing the phenotype
        allow_dead_removal: Allow removing DEAD phenotype
    
    Returns:
        bool: True if removed, False if skipped
    """
    embryo_id = self._get_embryo_id_from_snip(snip_id)
    if not embryo_id:
        raise ValueError(f"Snip {snip_id} not found")
    
    snip_data = self.data["embryos"][embryo_id]["snips"][snip_id]
    current_phenotype = snip_data["phenotype"]["value"]
    
    # Check DEAD phenotype protection
    if current_phenotype == "DEAD" and not allow_dead_removal:
        if self.verbose:
            print(f"⚠️ Cannot remove DEAD phenotype without allow_dead_removal=True")
        return False
    
    # Set to NONE
    snip_data["phenotype"] = Phenotype(
        value="NONE",
        author=author,
        notes=f"Removed {current_phenotype}"
    ).to_dict()
    
    self._unsaved_changes = True
    self._add_change_log("remove_phenotype", {
        "snip_id": snip_id,
        "old_value": current_phenotype,
        "author": author
    })
    
    if self.verbose:
        print(f"✅ Removed phenotype from {snip_id} (was '{current_phenotype}')")
    
    return True
```

## Batch Phenotype Operations

### Batch Add with Range Support

```python
def batch_add_phenotype(self, embryo_id: str, phenotype: str, 
                       snip_range: Union[str, List[int], List[str]], 
                       author: str, notes: str = None,
                       skip_existing: bool = True) -> Dict:
    """
    Add phenotype to multiple snips using range syntax.
    
    Args:
        embryo_id: Embryo identifier
        phenotype: Phenotype to add
        snip_range: Range specification:
            - String: "[5:10]", "[5::]", etc.
            - List of indices: [5, 6, 7, 8, 9]
            - List of snip_ids: ["..._e01_0005", "..._e01_0006"]
        author: Author of the changes
        notes: Optional notes
        skip_existing: Skip snips that already have phenotypes
    
    Returns:
        Dict with results: {
            "added": [...],
            "skipped": [...],
            "failed": [...]
        }
    """
    results = {
        "added": [],
        "skipped": [],
        "failed": []
    }
    
    # Get all snips for embryo
    if embryo_id not in self.data["embryos"]:
        raise ValueError(f"Embryo {embryo_id} not found")
    
    embryo_data = self.data["embryos"][embryo_id]
    all_snips = sorted(embryo_data["snips"].keys())
    
    # Parse range to get target snips
    target_snips = self._parse_snip_range(snip_range, all_snips)
    
    if self.verbose:
        print(f"🔄 Batch adding '{phenotype}' to {len(target_snips)} snips")
    
    # Process each snip
    for snip_id in target_snips:
        try:
            # Check if should skip
            current = embryo_data["snips"][snip_id]["phenotype"]["value"]
            if skip_existing and current != "NONE":
                results["skipped"].append((snip_id, f"Has '{current}'"))
                continue
            
            # Add phenotype
            success = self.add_phenotype(
                snip_id, phenotype, author, notes,
                force_dead=False
            )
            
            if success:
                results["added"].append(snip_id)
            else:
                results["skipped"].append((snip_id, "Dead embryo"))
                
        except Exception as e:
            results["failed"].append((snip_id, str(e)))
    
    # Summary
    if self.verbose:
        print(f"✅ Added: {len(results['added'])}")
        print(f"⏭️  Skipped: {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")
    
    return results
```

### Multi-Embryo Batch

```python
def batch_add_phenotypes(self, batch_data: Dict[str, Union[str, List]], 
                        author: str, level: str = "snip") -> Dict:
    """
    Add phenotypes to multiple entities.
    
    Args:
        batch_data: Dict mapping IDs to phenotypes
            - For snips: {"snip_id": "phenotype", ...}
            - For embryos: {"embryo_id": ["phenotype", "range"], ...}
        author: Author for all changes
        level: "snip" or "embryo"
    
    Returns:
        Aggregated results from all operations
    """
    all_results = {
        "added": [],
        "skipped": [],
        "failed": []
    }
    
    for entity_id, phenotype_spec in batch_data.items():
        try:
            if level == "snip":
                # Direct snip assignment
                if isinstance(phenotype_spec, str):
                    success = self.add_phenotype(entity_id, phenotype_spec, author)
                    if success:
                        all_results["added"].append(entity_id)
                    else:
                        all_results["skipped"].append(entity_id)
                        
            elif level == "embryo":
                # Embryo with range
                if isinstance(phenotype_spec, list) and len(phenotype_spec) == 2:
                    phenotype, range_spec = phenotype_spec
                    results = self.batch_add_phenotype(
                        entity_id, phenotype, range_spec, author
                    )
                    all_results["added"].extend(results["added"])
                    all_results["skipped"].extend(results["skipped"])
                    all_results["failed"].extend(results["failed"])
                    
        except Exception as e:
            all_results["failed"].append((entity_id, str(e)))
    
    return all_results
```

## Temporal Phenotype Analysis

### Get Phenotype Timeline

```python
def get_phenotype_timeline(self, embryo_id: str) -> List[Dict]:
    """
    Get temporal progression of phenotypes for an embryo.
    
    Returns:
        List of dicts with frame info and phenotype
    """
    if embryo_id not in self.data["embryos"]:
        raise ValueError(f"Embryo {embryo_id} not found")
    
    embryo_data = self.data["embryos"][embryo_id]
    timeline = []
    
    # Sort snips by frame number
    sorted_snips = sorted(embryo_data["snips"].items(), 
                         key=lambda x: self._extract_frame_number(x[0]))
    
    for snip_id, snip_data in sorted_snips:
        frame_num = self._extract_frame_number(snip_id)
        phenotype_data = snip_data["phenotype"]
        
        timeline.append({
            "snip_id": snip_id,
            "frame": frame_num,
            "phenotype": phenotype_data["value"],
            "author": phenotype_data["author"],
            "timestamp": phenotype_data["timestamp"],
            "confidence": phenotype_data.get("confidence")
        })
    
    return timeline

def find_phenotype_transitions(self, embryo_id: str) -> List[Dict]:
    """
    Find points where phenotype changes.
    
    Returns:
        List of transition events
    """
    timeline = self.get_phenotype_timeline(embryo_id)
    transitions = []
    
    for i in range(1, len(timeline)):
        prev = timeline[i-1]
        curr = timeline[i]
        
        if prev["phenotype"] != curr["phenotype"]:
            transitions.append({
                "frame": curr["frame"],
                "snip_id": curr["snip_id"],
                "from_phenotype": prev["phenotype"],
                "to_phenotype": curr["phenotype"],
                "author": curr["author"],
                "timestamp": curr["timestamp"]
            })
    
    return transitions
```

## Special Phenotype Handlers

### DEAD Phenotype Handler

```python

def _handle_dead_phenotype(self, embryo_id: str, death_snip_id: str, author: str) -> None:
    death_frame = self._extract_frame_number(death_snip_id)
    embryo_data = self.data["embryos"][embryo_id]

    # Clear phenotypes before death (optional)
    for snip_id, snip_data in embryo_data["snips"].items():
        frame = self._extract_frame_number(snip_id)
        current_val = snip_data["phenotype"]["value"]
        if frame < death_frame and current_val not in ["NONE", "DEAD"]:
            snip_data["phenotype"] = Phenotype(
                value="NONE",
                author="system",
                notes=f"Cleared pre-death"
            ).to_dict()

    # Propagate DEAD to all snips from death frame onward
    for snip_id in self._get_snips_after(embryo_id, death_frame):
        snip_data = embryo_data["snips"][snip_id]
        snip_data["phenotype"] = Phenotype(
            value="DEAD",
            author=author,
            notes=f"Propagated from {death_snip_id}"
        ).to_dict()

    self._add_embryo_flag(embryo_id, "DEATH_FRAME", author, 
                          notes=f"Death at snip {death_snip_id}")

# Utility function to list all future snips
def _get_snips_after(self, embryo_id: str, frame_threshold: int) -> List[str]:
    """Return list of snip_ids at or after a given frame number."""
    embryo_data = self.data["embryos"][embryo_id]
    return [
        snip_id for snip_id in embryo_data["snips"]
        if self._extract_frame_number(snip_id) >= frame_threshold
    ]

def _is_embryo_dead(self, embryo_id: str) -> bool:
    """Check if embryo has DEAD phenotype."""
    embryo_data = self.data["embryos"].get(embryo_id, {})
    
    for snip_data in embryo_data.get("snips", {}).values():
        if snip_data["phenotype"]["value"] == "DEAD":
            return True
    
    return False

def _get_death_frame(self, embryo_id: str) -> Optional[int]:
    """Get frame number where embryo died."""
    embryo_data = self.data["embryos"].get(embryo_id, {})
    
    for snip_id, snip_data in embryo_data.get("snips", {}).items():
        if snip_data["phenotype"]["value"] == "DEAD":
            return self._extract_frame_number(snip_id)
    
    return None
```

## Phenotype Statistics

```python
def get_phenotype_statistics(self, experiment_id: str = None) -> Dict:
    """
    Calculate phenotype statistics.
    
    Args:
        experiment_id: Optional filter by experiment
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_embryos": 0,
        "phenotyped_embryos": 0,
        "dead_embryos": 0,
        "phenotype_counts": defaultdict(int),
        "phenotype_by_experiment": defaultdict(lambda: defaultdict(int)),
        "death_timeline": []
    }
    
    for embryo_id, embryo_data in self.data["embryos"].items():
        # Apply experiment filter
        if experiment_id:
            if embryo_data["source"]["experiment_id"] != experiment_id:
                continue
        
        stats["total_embryos"] += 1
        
        # Check if phenotyped
        has_phenotype = False
        is_dead = False
        exp_id = embryo_data["source"]["experiment_id"]
        
        for snip_data in embryo_data["snips"].values():
            phenotype = snip_data["phenotype"]["value"]
            
            if phenotype != "NONE":
                has_phenotype = True
                stats["phenotype_counts"][phenotype] += 1
                stats["phenotype_by_experiment"][exp_id][phenotype] += 1
                
                if phenotype == "DEAD":
                    is_dead = True
        
        if has_phenotype:
            stats["phenotyped_embryos"] += 1
        
        if is_dead:
            stats["dead_embryos"] += 1
            death_frame = self._get_death_frame(embryo_id)
            if death_frame is not None:
                stats["death_timeline"].append({
                    "embryo_id": embryo_id,
                    "frame": death_frame
                })
    
    # Sort death timeline
    stats["death_timeline"].sort(key=lambda x: x["frame"])
    
    # Calculate rates
    if stats["total_embryos"] > 0:
        stats["phenotyping_rate"] = stats["phenotyped_embryos"] / stats["total_embryos"]
        stats["mortality_rate"] = stats["dead_embryos"] / stats["total_embryos"]
    else:
        stats["phenotyping_rate"] = 0
        stats["mortality_rate"] = 0
    
    return dict(stats)
```

## Phenotype Export/Import

```python
def export_phenotypes(self, output_path: Path, format: str = "csv") -> None:
    """
    Export phenotype data for analysis.
    
    Formats:
    - csv: Flat format for statistics
    - json: Full structured data
    - tsv: Tab-separated for R/Python
    """
    if format == "csv":
        self._export_phenotypes_csv(output_path)
    elif format == "json":
        self._export_phenotypes_json(output_path)
    elif format == "tsv":
        self._export_phenotypes_tsv(output_path)
    else:
        raise ValueError(f"Unknown export format: {format}")

def _export_phenotypes_csv(self, output_path: Path) -> None:
    """Export to CSV format."""
    import csv
    
    rows = []
    for embryo_id, embryo_data in self.data["embryos"].items():
        exp_id = embryo_data["source"]["experiment_id"]
        video_id = embryo_data["source"]["video_id"]
        
        for snip_id, snip_data in embryo_data["snips"].items():
            frame = self._extract_frame_number(snip_id)
            phenotype_info = snip_data["phenotype"]
            
            rows.append({
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "experiment_id": exp_id,
                "video_id": video_id,
                "frame": frame,
                "phenotype": phenotype_info["value"],
                "author": phenotype_info["author"],
                "timestamp": phenotype_info["timestamp"],
                "confidence": phenotype_info.get("confidence", "")
            })
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    if self.verbose:
        print(f"📊 Exported {len(rows)} phenotype records to {output_path}")
```

## Helper Methods

```python
def _get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
    """Find embryo ID that contains the snip."""
    # Extract embryo_id from snip_id format
    # snip_id: YYYYMMDD_WELL_eNN_FRAME
    parts = snip_id.split('_')
    if len(parts) >= 3 and parts[2].startswith('e'):
        embryo_id = '_'.join(parts[:3])
        if embryo_id in self.data["embryos"]:
            return embryo_id
    
    # Fallback: search all embryos
    for embryo_id, embryo_data in self.data["embryos"].items():
        if snip_id in embryo_data["snips"]:
            return embryo_id
    
    return None

def _extract_frame_number(self, snip_id: str) -> int:
    """Extract frame number from snip_id."""
    # Format: YYYYMMDD_WELL_eNN_FRAME
    parts = snip_id.split('_')
    if len(parts) >= 4:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return -1

def _get_embryo_phenotypes(self, embryo_id: str) -> List[str]:
    """Get all phenotypes for an embryo."""
    embryo_data = self.data["embryos"].get(embryo_id, {})
    phenotypes = []
    
    for snip_data in embryo_data.get("snips", {}).values():
        phenotype = snip_data["phenotype"]["value"]
        if phenotype not in phenotypes:
            phenotypes.append(phenotype)
    
    return phenotypes
```