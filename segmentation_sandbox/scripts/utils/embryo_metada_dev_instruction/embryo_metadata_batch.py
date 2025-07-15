"""
Module 6: Batch Processing Engine for EmbryoMetadata

This module provides efficient batch processing capabilities for large-scale operations,
including advanced range syntax parsing and parallel processing support.

Author: EmbryoMetadata Development Team
Date: July 15, 2025
"""

from typing import Union, List, Dict, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


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
            
        Raises:
            ValueError: For invalid range specifications
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
        
        Args:
            range_spec: Range specification string
            max_value: Maximum index value
            
        Returns:
            List of indices
        """
        # Create dummy items
        dummy_items = [str(i) for i in range(max_value)]
        
        # Parse and extract indices
        selected_items = RangeParser._parse_string_range(range_spec, dummy_items)
        
        return [int(item) for item in selected_items]


class TemporalRangeParser(RangeParser):
    """Extended parser for temporal/sequential data with biological context."""
    
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
            
        Raises:
            ValueError: For invalid specifications or missing data
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
            death_frame = TemporalRangeParser._get_death_frame(embryo_id, metadata)
            if death_frame is None:
                raise ValueError(f"No death frame found for {embryo_id}")
            
            # Find corresponding snip
            death_snip_idx = None
            for i, snip_id in enumerate(all_snips):
                if TemporalRangeParser._extract_frame_number(snip_id) == death_frame:
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
    
    @staticmethod
    def _get_death_frame(embryo_id: str, metadata: 'EmbryoMetadata') -> Optional[int]:
        """Get death frame for embryo if it has DEAD phenotype."""
        embryo_data = metadata.data["embryos"].get(embryo_id, {})
        
        # Look for DEAD phenotype
        for snip_id, snip_data in embryo_data.get("snips", {}).items():
            phenotypes = snip_data.get("phenotypes", {})
            if any(p.get("value") == "DEAD" for p in phenotypes.values()):
                return TemporalRangeParser._extract_frame_number(snip_id)
        
        return None
    
    @staticmethod
    def _extract_frame_number(snip_id: str) -> int:
        """Extract frame number from snip_id."""
        # Assumes format like "20240411_A01_e01_0023"
        frame_match = re.search(r'_(\d{4})$', snip_id)
        if frame_match:
            return int(frame_match.group(1))
        
        raise ValueError(f"Cannot extract frame number from {snip_id}")


class BatchProcessor:
    """High-performance batch processing engine with parallel support."""
    
    def __init__(self, metadata: 'EmbryoMetadata', 
                 parallel: bool = False, 
                 num_workers: int = 4,
                 progress_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize batch processor.
        
        Args:
            metadata: EmbryoMetadata instance
            parallel: Enable parallel processing
            num_workers: Number of parallel workers
            progress_callback: Callback for progress updates
            verbose: Enable verbose output
        """
        self.metadata = metadata
        self.parallel = parallel
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        self.verbose = verbose
        self._processed_count = 0
        self._total_count = 0
        self._start_time = None
    
    def process_batch(self, 
                     items: List[Tuple], 
                     operation: Callable,
                     chunk_size: int = 100,
                     auto_save_interval: Optional[int] = None) -> Dict:
        """
        Process items in batches with optional auto-save.
        
        Args:
            items: List of (item_id, data) tuples
            operation: Function to apply to each item
            chunk_size: Size of processing chunks
            auto_save_interval: Save after N items (None = no auto-save)
        
        Returns:
            Processing results dictionary
        """
        self._total_count = len(items)
        self._processed_count = 0
        self._start_time = time.time()
        
        if self.verbose:
            print(f"🚀 Starting batch processing: {self._total_count} items")
            print(f"📊 Mode: {'Parallel' if self.parallel else 'Sequential'}")
            if self.parallel:
                print(f"👥 Workers: {self.num_workers}")
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "start_time": datetime.now().isoformat(),
            "processing_mode": "parallel" if self.parallel else "sequential"
        }
        
        # Process in chunks
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            if self.verbose and len(chunks) > 1:
                print(f"📦 Processing chunk {chunk_idx + 1}/{len(chunks)}")
            
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
                if self.verbose:
                    print(f"💾 Auto-saving after {self._processed_count} items...")
                self.metadata.save()
        
        # Final statistics
        elapsed_time = time.time() - self._start_time
        results["end_time"] = datetime.now().isoformat()
        results["elapsed_seconds"] = elapsed_time
        results["items_per_second"] = self._total_count / elapsed_time if elapsed_time > 0 else 0
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _process_chunk_sequential(self, chunk: List[Tuple], 
                                 operation: Callable) -> Dict:
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
                               operation: Callable) -> Dict:
        """Process chunk in parallel."""
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
    
    def _print_summary(self, results: Dict):
        """Print processing summary."""
        total = len(results["successful"]) + len(results["failed"]) + len(results["skipped"])
        print(f"\n📊 Batch Processing Summary:")
        print(f"   ✅ Successful: {len(results['successful'])}")
        print(f"   ❌ Failed: {len(results['failed'])}")
        print(f"   ⏭️  Skipped: {len(results['skipped'])}")
        print(f"   📈 Total: {total}")
        print(f"   ⏱️  Time: {results['elapsed_seconds']:.2f}s")
        print(f"   🚀 Speed: {results['items_per_second']:.1f} items/sec")


class BatchOperations:
    """Collection of specialized batch operations for common tasks."""
    
    @staticmethod
    def batch_phenotype_assignment(metadata: 'EmbryoMetadata',
                                  assignments: List[Dict],
                                  author: str,
                                  validate_ranges: bool = True,
                                  parallel: bool = False) -> Dict:
        """
        Batch assign phenotypes with advanced range support.
        
        Assignment format:
        [
            {
                "embryo_id": "20240411_A01_e01",
                "phenotype": "EDEMA",
                "frames": "[10:20]",  # or "all", "death:", etc.
                "confidence": 0.95,    # optional
                "notes": "Manual annotation"  # optional
            },
            ...
        ]
        
        Args:
            metadata: EmbryoMetadata instance
            assignments: List of assignment dictionaries
            author: Author of annotations
            validate_ranges: Validate frame ranges exist
            parallel: Use parallel processing
            
        Returns:
            Assignment results
        """
        processor = BatchProcessor(metadata, parallel=parallel)
        results = {"assigned": 0, "failed": [], "embryos_processed": set()}
        
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
                
                results["embryos_processed"].add(embryo_id)
                
                # Add to items
                for snip_id in snip_ids:
                    items.append((snip_id, {
                        "embryo_id": embryo_id,
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
        results["embryos_processed"] = len(results["embryos_processed"])
        
        return results
    
    @staticmethod
    def batch_genotype_assignment(metadata: 'EmbryoMetadata',
                                 assignments: List[Dict],
                                 author: str,
                                 overwrite: bool = False,
                                 parallel: bool = False) -> Dict:
        """
        Batch assign genotypes to embryos.
        
        Assignment format:
        [
            {
                "embryo_id": "20240411_A01_e01",
                "genotype": "WT",
                "gene": "lmx1b",  # optional, for specific gene
                "notes": "PCR confirmed"  # optional
            },
            ...
        ]
        """
        processor = BatchProcessor(metadata, parallel=parallel)
        
        # Prepare items
        items = []
        for assignment in assignments:
            items.append((assignment["embryo_id"], {
                "genotype": assignment["genotype"],
                "gene": assignment.get("gene"),
                "author": author,
                "notes": assignment.get("notes"),
                "overwrite": overwrite
            }))
        
        # Define operation
        def assign_genotype(embryo_id, data):
            # Map to current method signature: (embryo_id, gene_name, allele, zygosity, confidence, notes, overwrite)
            gene_name = data.get("gene") or "WT"  # Use WT as default if no gene specified
            allele = data["genotype"]  # Use genotype value as allele
            
            return metadata.add_genotype(
                embryo_id,
                gene_name,
                allele,
                zygosity="heterozygous",  # default
                confidence=1.0,  # default
                notes=data.get("notes", ""),
                overwrite=data["overwrite"]
            )
        
        # Process batch
        results = processor.process_batch(items, assign_genotype)
        return {
            "assigned": len(results["successful"]),
            "failed": results["failed"],
            "processing_time": results["elapsed_seconds"]
        }
    
    @staticmethod
    def batch_flag_detection(metadata: 'EmbryoMetadata',
                           detectors: List[Dict],
                           entities: Optional[List[str]] = None,
                           parallel: bool = True) -> Dict:
        """
        Run batch flag detection with custom detectors.
        
        Detector format:
        {
            "name": "motion_blur_detector",
            "level": "snip",
            "function": detect_motion_blur_func,
            "params": {"threshold": 0.1},
            "severity": "warning"  # optional
        }
        
        Args:
            metadata: EmbryoMetadata instance
            detectors: List of detector configurations
            entities: Specific entities to check (None = all)
            parallel: Use parallel processing
            
        Returns:
            Detection results
        """
        processor = BatchProcessor(metadata, parallel=parallel)
        all_results = {}
        
        for detector in detectors:
            detector_name = detector["name"]
            level = detector["level"]
            detect_func = detector["function"]
            params = detector.get("params", {})
            severity = detector.get("severity", "warning")
            
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
                    items.append((entity_id, {
                        "entity_data": entity_data,
                        "detector_name": detector_name,
                        "params": params,
                        "severity": severity
                    }))
            
            # Define operation
            def run_detector(entity_id, data):
                result = detect_func(entity_id, data["entity_data"], **data["params"])
                
                if result:
                    # Add flag
                    flag_result = metadata.add_flag(
                        entity_id,
                        result["flag"],
                        level,
                        author="auto_detector",
                        notes=result.get("notes"),
                        severity=data["severity"],
                        auto_generated=True
                    )
                    return result
                
                return None
            
            # Process
            detector_results = processor.process_batch(items, run_detector)
            all_results[detector_name] = {
                "flags_added": len(detector_results["successful"]),
                "entities_checked": len(items),
                "failed": detector_results["failed"]
            }
        
        return all_results


# Helper functions for common batch operations

def create_progress_callback(verbose: bool = True) -> Callable:
    """Create a simple progress callback function."""
    def progress_callback(current: int, total: int):
        if verbose:
            percentage = (current / total) * 100
            print(f"📊 Progress: {current}/{total} ({percentage:.1f}%)")
    
    return progress_callback


def estimate_batch_time(item_count: int, 
                       items_per_second: float = 100.0) -> str:
    """
    Estimate batch processing time.
    
    Args:
        item_count: Number of items to process
        items_per_second: Estimated processing speed
        
    Returns:
        Human-readable time estimate
    """
    seconds = item_count / items_per_second
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"
