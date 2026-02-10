# Module 6: Batch Processing Engine

## Overview
This module provides efficient batch processing capabilities for large-scale operations, including advanced range syntax parsing and parallel processing support.

## Range Syntax Parser

### Core Range Parser

```python
class RangeParser:
    """Advanced range syntax parser for temporal specifications."""
    
    @staticmethod
    def parse_range(range_spec: Union[str, List[int], List[str]], 
                   available_items: List[str]) -> List[str]:
        """
        Parse various range specifications into list of items.
        
        Supported formats:
        - String ranges: "[5]", "[5:10]", "[5::]", "[::2]", "[5:10:2]"
        - List of indices: [5, 6, 7, 8, 9]
        - List of IDs: ["id1", "id2", "id3"]
        - Special syntax: "[23::]" means from index 23 to end
        
        Args:
            range_spec: Range specification
            available_items: List of available items (e.g., snip_ids)
        
        Returns:
            List of selected items
        """
        if isinstance(range_spec, list):
            # Handle list inputs
            if not range_spec:
                return []
            
            if isinstance(range_spec[0], int):
                # List of indices
                selected = []
                for idx in range_spec:
                    if 0 <= idx < len(available_items):
                        selected.append(available_items[idx])
                return selected
            
            elif isinstance(range_spec[0], str):
                # List of IDs - validate they exist
                selected = []
                for item_id in range_spec:
                    if item_id in available_items:
                        selected.append(item_id)
                return selected
        
        elif isinstance(range_spec, str):
            # Parse string range syntax
            return RangeParser._parse_string_range(range_spec, available_items)
        
        else:
            raise ValueError(f"Invalid range specification type: {type(range_spec)}")
    
    @staticmethod
    def _parse_string_range(range_str: str, available_items: List[str]) -> List[str]:
        """Parse string range syntax."""
        # Remove brackets and whitespace
        range_str = range_str.strip().strip("[]")
        
        # Empty range
        if not range_str:
            return []
        
        # Single index
        if ":" not in range_str:
            try:
                idx = int(range_str)
                if 0 <= idx < len(available_items):
                    return [available_items[idx]]
                return []
            except ValueError:
                # Maybe it's an ID directly
                if range_str in available_items:
                    return [range_str]
                return []
        
        # Range with colons
        parts = range_str.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid range syntax: too many colons in '{range_str}'")
        
        # Parse start, stop, step
        max_idx = len(available_items)
        
        # Start index
        start = 0
        if parts[0]:
            start = int(parts[0])
            if start < 0:  # Handle negative indexing
                start = max_idx + start
        
        # Stop index
        stop = max_idx
        if len(parts) > 1 and parts[1]:
            stop = int(parts[1])
            if stop < 0:
                stop = max_idx + stop
        
        # Step
        step = 1
        if len(parts) > 2 and parts[2]:
            step = int(parts[2])
            if step == 0:
                raise ValueError("Step cannot be zero")
        
        # Generate indices
        indices = list(range(start, stop, step))
        
        # Convert to items
        selected = []
        for idx in indices:
            if 0 <= idx < len(available_items):
                selected.append(available_items[idx])
        
        return selected
    
    @staticmethod
    def expand_range_to_indices(range_spec: str, max_value: int) -> List[int]:
        """
        Expand range specification to list of indices.
        
        This is useful for numeric operations without actual items.
        """
        # Create dummy items
        dummy_items = list(range(max_value))
        
        # Parse and extract indices
        selected_items = RangeParser._parse_string_range(
            range_spec, 
            [str(i) for i in dummy_items]
        )
        
        return [int(item) for item in selected_items]
```

### Temporal Range Extensions

```python
class TemporalRangeParser(RangeParser):
    """Extended parser for temporal/sequential data."""
    
    @staticmethod
    def parse_frame_range(embryo_id: str, frame_spec: str, 
                         metadata: 'EmbryoMetadata') -> List[str]:
        """
        Parse frame range for an embryo into snip_ids.
        
        Special syntax:
        - "all": All frames
        - "first": First frame only
        - "last": Last frame only
        - "death:": From death frame onwards (if applicable)
        - "death-5:death+5": 5 frames before and after death
        
        Args:
            embryo_id: Embryo identifier
            frame_spec: Frame specification
            metadata: EmbryoMetadata instance for context
        
        Returns:
            List of snip_ids
        """
        # Get all snips for embryo
        embryo_data = metadata.data["embryos"].get(embryo_id, {})
        all_snips = sorted(embryo_data.get("snips", {}).keys())
        
        if not all_snips:
            return []
        
        # Handle special keywords
        if frame_spec == "all":
            return all_snips
        
        elif frame_spec == "first":
            return [all_snips[0]]
        
        elif frame_spec == "last":
            return [all_snips[-1]]
        
        elif frame_spec.startswith("death"):
            # Find death frame
            death_frame = metadata._get_death_frame(embryo_id)
            if death_frame is None:
                raise ValueError(f"No death frame found for {embryo_id}")
            
            # Find corresponding snip
            death_snip_idx = None
            for i, snip_id in enumerate(all_snips):
                if metadata._extract_frame_number(snip_id) == death_frame:
                    death_snip_idx = i
                    break
            
            if death_snip_idx is None:
                raise ValueError(f"Death frame {death_frame} not found in snips")
            
            # Parse relative to death
            if frame_spec == "death:":
                # From death onwards
                return all_snips[death_snip_idx:]
            
            elif ":" in frame_spec:
                # Range relative to death
                parts = frame_spec.split(":")
                start_str = parts[0].replace("death", str(death_snip_idx))
                end_str = parts[1].replace("death", str(death_snip_idx)) if parts[1] else ""
                
                # Evaluate expressions like "death-5" -> "10-5" -> 5
                start_idx = eval(start_str)
                end_idx = eval(end_str) if end_str else len(all_snips)
                
                return all_snips[max(0, start_idx):min(len(all_snips), end_idx)]
            
            else:
                # Just the death frame
                return [all_snips[death_snip_idx]]
        
        else:
            # Standard range syntax
            return RangeParser.parse_range(frame_spec, all_snips)
```

## Batch Processing Engine

### Core Batch Processor

```python
class BatchProcessor:
    """High-performance batch processing engine."""
    
    def __init__(self, metadata: 'EmbryoMetadata', 
                 parallel: bool = False, 
                 num_workers: int = 4,
                 progress_callback: callable = None):
        """
        Initialize batch processor.
        
        Args:
            metadata: EmbryoMetadata instance
            parallel: Enable parallel processing
            num_workers: Number of parallel workers
            progress_callback: Callback for progress updates
        """
        self.metadata = metadata
        self.parallel = parallel
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        self._processed_count = 0
        self._total_count = 0
    
    def process_batch(self, 
                     items: List[Tuple], 
                     operation: callable,
                     chunk_size: int = 100,
                     auto_save_interval: int = None) -> Dict:
        """
        Process items in batches with optional auto-save.
        
        Args:
            items: List of (item_id, data) tuples
            operation: Function to apply to each item
            chunk_size: Size of processing chunks
            auto_save_interval: Save after N items
        
        Returns:
            Processing results
        """
        self._total_count = len(items)
        self._processed_count = 0
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        # Process in chunks
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            if self.parallel:
                chunk_results = self._process_chunk_parallel(chunk, operation)
            else:
                chunk_results = self._process_chunk_sequential(chunk, operation)
            
            # Aggregate results
            results["successful"].extend(chunk_results["successful"])
            results["failed"].extend(chunk_results["failed"])
            results["skipped"].extend(chunk_results["skipped"])
            
            self._processed_count += len(chunk)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback(self._processed_count, self._total_count)
            
            # Auto-save check
            if auto_save_interval and self._processed_count % auto_save_interval == 0:
                if self.metadata.verbose:
                    print(f"💾 Auto-saving after {self._processed_count} items...")
                self.metadata.save()
        
        return results
    
    def _process_chunk_sequential(self, chunk: List[Tuple], 
                                 operation: callable) -> Dict:
        """Process chunk sequentially."""
        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        for item_id, item_data in chunk:
            try:
                result = operation(item_id, item_data)
                
                if result is None or result is False:
                    results["skipped"].append(item_id)
                else:
                    results["successful"].append((item_id, result))
                    
            except Exception as e:
                results["failed"].append((item_id, str(e)))
        
        return results
    
    def _process_chunk_parallel(self, chunk: List[Tuple], 
                               operation: callable) -> Dict:
        """Process chunk in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            future_to_item = {
                executor.submit(operation, item_id, item_data): (item_id, item_data)
                for item_id, item_data in chunk
            }
            
            # Collect results
            for future in as_completed(future_to_item):
                item_id, item_data = future_to_item[future]
                
                try:
                    result = future.result()
                    
                    if result is None or result is False:
                        results["skipped"].append(item_id)
                    else:
                        results["successful"].append((item_id, result))
                        
                except Exception as e:
                    results["failed"].append((item_id, str(e)))
        
        return results
```

### Specialized Batch Operations

```python
class BatchOperations:
    """Collection of specialized batch operations."""
    
    @staticmethod
    def batch_phenotype_assignment(metadata: 'EmbryoMetadata',
                                  assignments: List[Dict],
                                  author: str,
                                  validate_ranges: bool = True) -> Dict:
        """
        Batch assign phenotypes with advanced range support.
        
        Assignment format:
        [
            {
                "embryo_id": "20240411_A01_e01",
                "phenotype": "EDEMA",
                "frames": "[10:20]",  # or "all", "death:", etc.
                "confidence": 0.95    # optional
            },
            ...
        ]
        """
        processor = BatchProcessor(metadata)
        results = {"assigned": 0, "failed": []}
        
        # Prepare items
        items = []
        for assignment in assignments:
            embryo_id = assignment["embryo_id"]
            phenotype = assignment["phenotype"]
            frame_spec = assignment.get("frames", "all")
            
            try:
                # Parse frame range
                snip_ids = TemporalRangeParser.parse_frame_range(
                    embryo_id, frame_spec, metadata
                )
                
                if validate_ranges and not snip_ids:
                    results["failed"].append((embryo_id, "No valid frames in range"))
                    continue
                
                # Add to items
                for snip_id in snip_ids:
                    items.append((snip_id, {
                        "phenotype": phenotype,
                        "author": author,
                        "confidence": assignment.get("confidence"),
                        "notes": assignment.get("notes")
                    }))
                    
            except Exception as e:
                results["failed"].append((embryo_id, str(e)))
        
        # Define operation
        def assign_phenotype(snip_id, data):
            return metadata.add_phenotype(
                snip_id,
                data["phenotype"],
                data["author"],
                notes=data.get("notes"),
                confidence=data.get("confidence")
            )
        
        # Process batch
        batch_results = processor.process_batch(items, assign_phenotype)
        results["assigned"] = len(batch_results["successful"])
        results["failed"].extend(batch_results["failed"])
        
        return results
    
    @staticmethod
    def batch_flag_detection(metadata: 'EmbryoMetadata',
                           detectors: List[Dict],
                           entities: List[str] = None,
                           parallel: bool = True) -> Dict:
        """
        Run batch flag detection with custom detectors.
        
        Detector format:
        {
            "name": "motion_blur_detector",
            "level": "snip",
            "function": detect_motion_blur_func,
            "params": {"threshold": 0.1}
        }
        """
        processor = BatchProcessor(metadata, parallel=parallel)
        all_results = {}
        
        for detector in detectors:
            level = detector["level"]
            detect_func = detector["function"]
            params = detector.get("params", {})
            
            # Get entities to check
            if entities:
                check_entities = entities
            else:
                check_entities = metadata._get_all_entities_at_level(level)
            
            # Prepare items
            items = []
            for entity_id in check_entities:
                entity_data = metadata._get_entity_data(entity_id, level)
                if entity_data:
                    items.append((entity_id, entity_data))
            
            # Define operation
            def run_detector(entity_id, entity_data):
                result = detect_func(entity_id, entity_data, **params)
                
                if result:
                    # Add flag
                    metadata.add_flag(
                        entity_id,
                        result["flag"],
                        level,
                        author="auto_detector",
                        notes=result.get("notes"),
                        severity=result.get("severity", "warning"),
                        auto_generated=True
                    )
                    return result
                
                return None
            
            # Process
            detector_results = processor.process_batch(items, run_detector)
            all_results[detector["name"]] = {
                "detected": len(detector_results["successful"]),
                "failed": len(detector_results["failed"]),
                "entities_checked": len(items)
            }
        
        return all_results
```

## Progress Tracking

```python
class ProgressTracker:
    """Progress tracking for batch operations."""
    
    def __init__(self, total: int, description: str = "Processing",
                 update_interval: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of operation
            update_interval: Update frequency
        """
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.processed = 0
        self.start_time = datetime.now()
        self.last_update = 0
    
    def update(self, count: int = 1) -> None:
        """Update progress."""
        self.processed += count
        
        if self.processed - self.last_update >= self.update_interval:
            self._print_progress()
            self.last_update = self.processed
    
    def _print_progress(self) -> None:
        """Print progress bar."""
        percent = self.processed / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.processed > 0:
            rate = self.processed / elapsed
            eta = (self.total - self.processed) / rate
        else:
            rate = 0
            eta = 0
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * self.processed / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% "
              f"({self.processed}/{self.total}) "
              f"Rate: {rate:.1f}/s ETA: {eta:.1f}s", 
              end="", flush=True)
    
    def finish(self) -> None:
        """Mark as complete."""
        self.processed = self.total
        self._print_progress()
        print()  # New line
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"✅ Completed in {elapsed:.1f}s")
```

## Batch Import/Export

```python
class BatchImportExport:
    """Batch import/export operations."""
    
    @staticmethod
    def import_from_spreadsheet(metadata: 'EmbryoMetadata',
                               file_path: Path,
                               mapping: Dict[str, str],
                               sheet_name: str = None) -> Dict:
        """
        Import data from Excel/CSV spreadsheet.
        
        Args:
            metadata: EmbryoMetadata instance
            file_path: Path to spreadsheet
            mapping: Column mapping
                {
                    "embryo_id": "Embryo ID",
                    "genotype": "Genotype",
                    "phenotype": "Phenotype",
                    "frame_range": "Frames",
                    "notes": "Notes"
                }
            sheet_name: Excel sheet name (if applicable)
        
        Returns:
            Import results
        """
        import pandas as pd
        
        # Read file
        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Process rows
        processor = BatchProcessor(metadata)
        items = []
        
        for idx, row in df.iterrows():
            try:
                item = {}
                for field, column in mapping.items():
                    if column in row and pd.notna(row[column]):
                        item[field] = str(row[column]).strip()
                
                if item:
                    items.append((idx, item))
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        # Define import operation
        def import_row(idx, data):
            results = []
            
            # Import genotype
            if "genotype" in data and "embryo_id" in data:
                try:
                    metadata.add_genotype(
                        data["embryo_id"],
                        data["genotype"],
                        author="import",
                        notes=data.get("notes")
                    )
                    results.append("genotype")
                except Exception as e:
                    results.append(f"genotype_error: {e}")
            
            # Import phenotype
            if "phenotype" in data and "embryo_id" in data:
                frame_range = data.get("frame_range", "all")
                try:
                    snips = TemporalRangeParser.parse_frame_range(
                        data["embryo_id"], frame_range, metadata
                    )
                    for snip_id in snips:
                        metadata.add_phenotype(
                            snip_id,
                            data["phenotype"],
                            author="import",
                            notes=data.get("notes")
                        )
                    results.append(f"phenotype_{len(snips)}_snips")
                except Exception as e:
                    results.append(f"phenotype_error: {e}")
            
            return results
        
        # Process import
        import_results = processor.process_batch(
            items, 
            import_row,
            auto_save_interval=100
        )
        
        return {
            "total_rows": len(df),
            "processed": len(items),
            "successful": len(import_results["successful"]),
            "failed": len(import_results["failed"])
        }
    
    @staticmethod
    def export_for_analysis(metadata: 'EmbryoMetadata',
                          output_path: Path,
                          include_fields: List[str] = None,
                          format: str = "parquet") -> None:
        """
        Export data in analysis-ready format.
        
        Args:
            metadata: EmbryoMetadata instance
            output_path: Output file path
            include_fields: Fields to include (default: all)
            format: Output format ("parquet", "feather", "hdf5")
        """
        import pandas as pd
        
        # Collect data
        records = []
        
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            embryo_record = {
                "embryo_id": embryo_id,
                "experiment_id": embryo_data["source"]["experiment_id"],
                "video_id": embryo_data["source"]["video_id"]
            }
            
            # Add genotype
            if embryo_data.get("genotype"):
                embryo_record["genotype"] = embryo_data["genotype"]["value"]
                embryo_record["genotype_confirmed"] = embryo_data["genotype"].get("confirmed", False)
            else:
                embryo_record["genotype"] = None
                embryo_record["genotype_confirmed"] = False
            
            # Add snip-level data
            for snip_id, snip_data in embryo_data["snips"].items():
                snip_record = embryo_record.copy()
                snip_record["snip_id"] = snip_id
                snip_record["frame"] = metadata._extract_frame_number(snip_id)
                
                # Phenotype
                phenotype_info = snip_data["phenotype"]
                snip_record["phenotype"] = phenotype_info["value"]
                snip_record["phenotype_confidence"] = phenotype_info.get("confidence")
                
                # Flags
                flags = snip_data.get("flags", [])
                snip_record["num_flags"] = len(flags)
                snip_record["flag_list"] = ",".join([f["value"] for f in flags])
                
                records.append(snip_record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Filter fields if specified
        if include_fields:
            df = df[include_fields]
        
        # Export
        if format == "parquet":
            df.to_parquet(output_path, engine='pyarrow')
        elif format == "feather":
            df.to_feather(output_path)
        elif format == "hdf5":
            df.to_hdf(output_path, key='embryo_metadata', mode='w')
        else:
            raise ValueError(f"Unknown format: {format}")
        
        if metadata.verbose:
            print(f"📊 Exported {len(df)} records to {output_path}")
```

## Optimization Utilities

```python
class BatchOptimizer:
    """Optimization utilities for batch operations."""
    
    @staticmethod
    def optimize_batch_order(items: List[Tuple], 
                           locality_key: callable = None) -> List[Tuple]:
        """
        Optimize batch processing order for better cache locality.
        
        Args:
            items: List of (id, data) tuples
            locality_key: Function to extract locality key
        
        Returns:
            Reordered items
        """
        if not locality_key:
            # Default: group by experiment/video
            def locality_key(item):
                item_id = item[0]
                parts = item_id.split('_')
                if len(parts) >= 2:
                    return f"{parts[0]}_{parts[1]}"  # experiment_video
                return parts[0]
        
        # Sort by locality
        return sorted(items, key=lambda x: locality_key(x))
    
    @staticmethod
    def estimate_memory_usage(metadata: 'EmbryoMetadata', 
                            operation: str) -> Dict:
        """
        Estimate memory usage for batch operation.
        
        Returns:
            Memory estimates in MB
        """
        import sys
        
        estimates = {
            "current_metadata_size": sys.getsizeof(metadata.data) / 1024 / 1024,
            "estimated_operation_overhead": 0
        }
        
        # Operation-specific estimates
        if operation == "phenotype_assignment":
            # ~100 bytes per snip update
            total_snips = sum(
                len(emb["snips"]) 
                for emb in metadata.data["embryos"].values()
            )
            estimates["estimated_operation_overhead"] = total_snips * 100 / 1024 / 1024
        
        elif operation == "flag_detection":
            # ~200 bytes per entity checked
            total_entities = metadata._count_all_entities()
            estimates["estimated_operation_overhead"] = total_entities * 200 / 1024 / 1024
        
        estimates["estimated_total"] = (
            estimates["current_metadata_size"] + 
            estimates["estimated_operation_overhead"]
        )
        
        return estimates
```