    def merge_annotations(self, 
                         other: Union['GroundedDinoAnnotations', str, Path],
                         conflict_strategy: str = "skip",
                         merge_high_quality: bool = True,
                         dry_run: bool = False) -> Dict:
        """
        Merge annotations from another GroundedDinoAnnotations instance or file.
        
        Args:
            other: Another GroundedDinoAnnotations instance or path to JSON file
            conflict_strategy: How to handle conflicts ("skip", "overwrite", "merge")
                - "skip": Keep existing annotations, skip conflicting ones from other
                - "overwrite": Replace existing annotations with ones from other  
                - "merge": Combine annotations for same image (different annotation_ids)
            merge_high_quality: Whether to merge high-quality annotations
            dry_run: If True, show what would be merged without actually doing it
            
        Returns:
            Dictionary with merge statistics
        """
        # Load other annotations
        if isinstance(other, (str, Path)):
            other_path = Path(other)
            if not other_path.exists():
                raise FileNotFoundError(f"File not found: {other_path}")
            
            with open(other_path, 'r') as f:
                other_data = json.load(f)
            other_annotations = other_data
        elif isinstance(other, GroundedDinoAnnotations):
            other_annotations = other.annotations
        else:
            raise ValueError("Other must be GroundedDinoAnnotations instance or file path")
        
        if self.verbose:
            print(f"🔄 Merging annotations (strategy: {conflict_strategy}, dry_run: {dry_run})")
            if isinstance(other, (str, Path)):
                print(f"   Source: {other}")
        
        # Initialize merge statistics
        stats = {
            "images_added": 0,
            "images_merged": 0,
            "annotations_added": 0,
            "annotations_skipped": 0,
            "annotations_overwritten": 0,
            "high_quality_experiments_added": 0,
            "high_quality_experiments_skipped": 0,
            "conflicts_found": [],
            "dry_run": dry_run
        }
        
        # Merge regular annotations
        other_images = other_annotations.get("images", {})
        
        for image_id, other_image_data in other_images.items():
            other_annotations_list = other_image_data.get("annotations", [])
            
            if image_id not in self.annotations["images"]:
                # New image - add all annotations
                if not dry_run:
                    self.annotations["images"][image_id] = other_image_data.copy()
                stats["images_added"] += 1
                stats["annotations_added"] += len(other_annotations_list)
                
                if self.verbose:
                    print(f"   ✅ Added new image '{image_id}' with {len(other_annotations_list)} annotations")
            
            else:
                # Existing image - handle conflicts
                stats["images_merged"] += 1
                existing_annotations = self.annotations["images"][image_id]["annotations"]
                
                # Create lookup for existing annotations by prompt
                existing_by_prompt = {}
                for ann in existing_annotations:
                    prompt = ann.get("prompt", "")
                    if prompt not in existing_by_prompt:
                        existing_by_prompt[prompt] = []
                    existing_by_prompt[prompt].append(ann)
                
                annotations_to_add = []
                
                for other_ann in other_annotations_list:
                    other_prompt = other_ann.get("prompt", "")
                    
                    if other_prompt not in existing_by_prompt:
                        # No conflict - add annotation
                        annotations_to_add.append(other_ann)
                        stats["annotations_added"] += 1
                        
                        if self.verbose:
                            print(f"   ✅ Adding annotation '{image_id}' + '{other_prompt}'")
                    
                    else:
                        # Conflict detected
                        conflict_info = {
                            "image_id": image_id,
                            "prompt": other_prompt,
                            "existing_count": len(existing_by_prompt[other_prompt]),
                            "strategy_applied": conflict_strategy
                        }
                        stats["conflicts_found"].append(conflict_info)
                        
                        if conflict_strategy == "skip":
                            stats["annotations_skipped"] += 1
                            if self.verbose:
                                print(f"   ⚠️  Skipping conflicting annotation '{image_id}' + '{other_prompt}'")
                        
                        elif conflict_strategy == "overwrite":
                            # Remove existing annotations with same prompt
                            if not dry_run:
                                self.annotations["images"][image_id]["annotations"] = [
                                    ann for ann in existing_annotations 
                                    if ann.get("prompt") != other_prompt
                                ]
                            annotations_to_add.append(other_ann)
                            stats["annotations_overwritten"] += 1
                            
                            if self.verbose:
                                print(f"   🔄 Overwriting annotation '{image_id}' + '{other_prompt}'")
                        
                        elif conflict_strategy == "merge":
                            # Add with new annotation_id to avoid conflicts
                            merged_ann = other_ann.copy()
                            merged_ann["annotation_id"] = f"merged_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                            annotations_to_add.append(merged_ann)
                            stats["annotations_added"] += 1
                            
                            if self.verbose:
                                print(f"   🔀 Merging annotation '{image_id}' + '{other_prompt}' (new ID)")
                
                # Apply changes if not dry run
                if not dry_run and annotations_to_add:
                    self.annotations["images"][image_id]["annotations"].extend(annotations_to_add)
        
        # Merge high-quality annotations
        if merge_high_quality:
            other_hq = other_annotations.get("high_quality_annotations", {})
            
            if other_hq:
                if "high_quality_annotations" not in self.annotations:
                    self.annotations["high_quality_annotations"] = {}
                
                for exp_id, other_exp_data in other_hq.items():
                    if exp_id not in self.annotations["high_quality_annotations"]:
                        # New experiment - add it
                        if not dry_run:
                            self.annotations["high_quality_annotations"][exp_id] = other_exp_data.copy()
                        stats["high_quality_experiments_added"] += 1
                        
                        if self.verbose:
                            filtered_images = len(other_exp_data.get("filtered", {}))
                            print(f"   ⭐ Added high-quality experiment '{exp_id}' ({filtered_images} images)")
                    
                    else:
                        # Existing experiment - check if parameters match
                        existing_exp = self.annotations["high_quality_annotations"][exp_id]
                        
                        params_match = (
                            existing_exp.get("prompt") == other_exp_data.get("prompt") and
                            existing_exp.get("confidence_threshold") == other_exp_data.get("confidence_threshold") and
                            existing_exp.get("iou_threshold") == other_exp_data.get("iou_threshold")
                        )
                        
                        if params_match:
                            # Same parameters - merge filtered annotations
                            if not dry_run:
                                existing_filtered = existing_exp.get("filtered", {})
                                other_filtered = other_exp_data.get("filtered", {})
                                
                                for img_id, detections in other_filtered.items():
                                    if img_id not in existing_filtered:
                                        existing_filtered[img_id] = detections
                            
                            if self.verbose:
                                new_images = len([img for img in other_exp_data.get("filtered", {}) 
                                               if img not in existing_exp.get("filtered", {})])
                                print(f"   ⭐ Merged {new_images} new images into experiment '{exp_id}'")
                        
                        else:
                            # Different parameters - skip
                            stats["high_quality_experiments_skipped"] += 1
                            if self.verbose:
                                print(f"   ⚠️  Skipping high-quality experiment '{exp_id}' (parameter mismatch)")
        
        # Update file metadata
        if not dry_run and (stats["annotations_added"] > 0 or stats["annotations_overwritten"] > 0 or 
                           stats["high_quality_experiments_added"] > 0):
            self.annotations["file_info"]["last_updated"] = datetime.now().isoformat()
            if "merge_history" not in self.annotations["file_info"]:
                self.annotations["file_info"]["merge_history"] = []
            
            merge_record = {
                "timestamp": datetime.now().isoformat(),
                "source": str(other) if isinstance(other, (str, Path)) else "GroundedDinoAnnotations_instance",
                "strategy": conflict_strategy,
                "stats": {k: v for k, v in stats.items() if k != "conflicts_found"}
            }
            self.annotations["file_info"]["merge_history"].append(merge_record)
            
            self._unsaved_changes = True
        
        # Print summary
        if self.verbose:
            print(f"\n📊 Merge Summary:")
            print(f"   Images added: {stats['images_added']}")
            print(f"   Images merged: {stats['images_merged']}")
            print(f"   Annotations added: {stats['annotations_added']}")
            print(f"   Annotations skipped: {stats['annotations_skipped']}")
            print(f"   Annotations overwritten: {stats['annotations_overwritten']}")
            print(f"   High-quality experiments added: {stats['high_quality_experiments_added']}")
            print(f"   High-quality experiments skipped: {stats['high_quality_experiments_skipped']}")
            print(f"   Conflicts found: {len(stats['conflicts_found'])}")
            
            if dry_run:
                print(f"   🔍 DRY RUN - No changes were made")
            elif self._unsaved_changes:
                print(f"   💾 Remember to call .save() to persist changes")
        
        return stats

    @classmethod
    def merge_multiple_files(cls, 
                            file_paths: List[Union[str, Path]], 
                            output_path: Union[str, Path],
                            conflict_strategy: str = "skip",
                            merge_high_quality: bool = True,
                            verbose: bool = True) -> 'GroundedDinoAnnotations':
        """
        Merge multiple annotation files into a single file.
        
        Args:
            file_paths: List of paths to annotation JSON files
            output_path: Path for the merged output file
            conflict_strategy: How to handle conflicts ("skip", "overwrite", "merge")
            merge_high_quality: Whether to merge high-quality annotations
            verbose: Whether to print progress information
            
        Returns:
            New GroundedDinoAnnotations instance with merged data
        """
        if not file_paths:
            raise ValueError("At least one file path must be provided")
        
        if verbose:
            print(f"🔄 Merging {len(file_paths)} annotation files")
            print(f"   Strategy: {conflict_strategy}")
            print(f"   Output: {output_path}")
        
        # Start with the first file
        merged = cls(output_path, verbose=verbose)
        first_file_stats = merged.merge_annotations(
            file_paths[0], 
            conflict_strategy=conflict_strategy,
            merge_high_quality=merge_high_quality,
            dry_run=False
        )
        
        if verbose:
            print(f"   📁 Loaded base file: {file_paths[0]}")
        
        # Merge remaining files
        total_stats = first_file_stats.copy()
        
        for i, file_path in enumerate(file_paths[1:], 1):
            if verbose:
                print(f"   📁 Merging file {i+1}/{len(file_paths)}: {file_path}")
            
            file_stats = merged.merge_annotations(
                file_path,
                conflict_strategy=conflict_strategy, 
                merge_high_quality=merge_high_quality,
                dry_run=False
            )
            
            # Accumulate statistics
            for key in ["images_added", "images_merged", "annotations_added", 
                       "annotations_skipped", "annotations_overwritten",
                       "high_quality_experiments_added", "high_quality_experiments_skipped"]:
                total_stats[key] += file_stats[key]
            
            total_stats["conflicts_found"].extend(file_stats["conflicts_found"])
        
        # Save merged result
        merged.save()
        
        if verbose:
            print(f"\n🎯 Final Merge Summary:")
            print(f"   Total images: {len(merged.get_all_image_ids())}")
            print(f"   Total annotations added: {total_stats['annotations_added']}")
            print(f"   Total conflicts: {len(total_stats['conflicts_found'])}")
            print(f"   ✅ Merged file saved to: {output_path}")
        
        return merged

    @property
    def has_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        return self._unsaved_changes

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        status = "✅ saved" if not self._unsaved_changes else "⚠️ unsaved"
        metadata_status = "📂 metadata" if summary['metadata_loaded'] else "📂 no-metadata"
        hq_status = f", ⭐ {summary.get('high_quality_experiments', 0)} hq-exp" if "high_quality_experiments" in summary else ""
        return f"GroundedDinoAnnotations(images={summary['total_images']}, annotations={summary['total_annotations']}, {status}, {metadata_status}{hq_status})"

