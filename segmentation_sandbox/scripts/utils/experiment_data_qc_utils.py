#!/usr/bin/env python
"""
experiment_data_qc_utils.py

<<<<<<< HEAD
A comprehensive system for managing quality control flags in hierarchical experiment data 
for the MorphSeq pipeline. This system provides both procedural and object-oriented 
interfaces for flagging issues at experiment, video, image, and embryo levels.

🎯 CORE FUNCTIONALITY
=====================
This system enables you to:
- Flag quality issues at any level (experiment → video → image → embryo)
- Maintain consistent QC categories across all tools and users
- Perform efficient batch operations for large-scale QC
- Validate flag integrity automatically
- Export QC data for analysis and reporting

📊 EXPECTED DATA STRUCTURE
==========================
raw_data_organized/
├── experiment_metadata.json     # Contains experiment structure
└── quality_control/
    └── experiment_data_qc.json  # QC flags (created automatically)

🎨 DESIGN PHILOSOPHY
====================
- Valid QC flag categories are stored ONLY in the JSON file as single source of truth
- All operations work in memory until explicitly saved for performance
- Flag integrity is maintained automatically
- Supports both individual and batch operations

🚀 QUICK START EXAMPLES
========================

# 1. Initialize QC system from existing experiment data
initialize_qc_structure_from_metadata(
    quality_control_dir="/path/to/quality_control",
    experiment_metadata_path="/path/to/raw_data_organized/experiment_metadata.json"
)

# 2. Use the recommended class-based interface
qc = ExperimentDataQC("/path/to/quality_control", author_designation="automated_qc")
qc.flag_image("20241215_A01_t001", "BLUR", notes="Low variance: 45")
qc.flag_image("20241215_A01_t002", "DRY_WELL", notes="Visual inspection")
qc.save()

# 3. Batch operations for efficiency
qc = ExperimentDataQC("/path/to/quality_control", author_designation="batch_processor")
image_flags = {
    "20241215_A01_t001": "BLUR",
    "20241215_A01_t002": ["BLUR", "DRY_WELL"],
    "20241215_A01_t003": [
        {"qc_flag": "CORRUPT", "author": "auto", "notes": "Cannot read file"}
    ]
}
qc.flag_images_batch(image_flags)
qc.save()

# 4. Advanced batch with gen_flag_batch (MOST FLEXIBLE)
qc = ExperimentDataQC("/path/to/quality_control", author_designation="tech_reviewer")

# Option A: Return batches (DEFAULT - for accumulation)
batch = []
batch += qc.gen_flag_batch("experiment", "20241215", "PROTOCOL_DEVIATION", notes="Missing controls")
batch += qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Focus problems")
batch += qc.gen_flag_batch("image", "20241215_A01_t001", ["BLUR", "OVEREXPOSURE"], notes="Multiple issues")
qc.add_flag_batch(batch)  # Apply accumulated batch

# Option B: Apply directly in memory (apply_directly=True)
qc.gen_flag_batch("experiment", "20241215", "INCOMPLETE", notes="Missing data", apply_directly=True)
qc.gen_flag_batch("video", "20241215_A02", "TECHNICAL_ISSUE", notes="Camera issue", apply_directly=True)
qc.gen_flag_batch("image", "20241215_A02_t001", "DRY_WELL", notes="Detected drying", apply_directly=True)

# Option C: Mixed mode - some batched, some direct
batch = []
for image_id, analysis in automated_analysis.items():
    if analysis['needs_review']:
        batch += qc.gen_flag_batch("image", image_id, "BLUR", notes=f"Score: {analysis['blur_score']}")
    else:
        qc.gen_flag_batch("image", image_id, "DRY_WELL", notes="Clear case", apply_directly=True)

if batch:  # Apply accumulated flags that need review
    qc.add_flag_batch(batch)

qc.save()  # Save all changes

📋 QC FLAG LEVELS
==================
- experiment_level: Issues affecting entire experiments
- video_level: Issues affecting video sequences  
- image_level: Issues with individual images (blur, corruption, etc.)
- embryo_level: Issues with detected embryos (death, detection failures)

🔧 INSTALLATION REQUIREMENTS
=============================
No external dependencies beyond Python standard library.
Uses JSON for data storage and pathlib for file operations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Default QC flag categories - ONLY used when creating new JSON files
# The actual valid categories are stored in the JSON file as single source of truth
DEFAULT_QC_FLAG_CATEGORIES = {
    "experiment_level": {
        "PROTOCOL_DEVIATION": "Deviation from standard imaging protocol",
        "INCOMPLETE": "Experiment was not completed"
    },
    "video_level": {
        "TECHNICAL_ISSUE": "Technical problems during video acquisition"
=======
Utilities for managing hierarchical experiment data quality control flags for the MorphSeq pipeline.
Uses a JSON structure inspired by COCO and embryo_metadata.json for better organization
and maintenance of QC flags across experiment, video, image, and embryo levels.

🎯 CORE FUNCTIONALITY
=====================
This system enables you to:
- Flag quality issues at any level (experiment → video → image → embryo)
- Maintain consistent QC categories across all tools and users
- Perform efficient batch operations for large-scale QC
- Validate flag integrity automatically
- Export QC data for analysis and reporting

📊 EXPECTED DATA STRUCTURE
==========================
raw_data_organized/
├── experiment_metadata.json     # Contains experiment structure
└── quality_control/
    └── experiment_data_qc.json  # QC flags (created automatically)

🎨 DESIGN PHILOSOPHY
====================
- Valid QC flag categories are stored ONLY in the JSON file as single source of truth
- All operations work in memory until explicitly saved for performance
- Flag integrity is maintained automatically
- Supports both individual and batch operations

🚀 QUICK START EXAMPLES
========================

# 1. Initialize QC system from existing experiment data
initialize_qc_structure_from_metadata(
    quality_control_dir="/path/to/quality_control",
    experiment_metadata_path="/path/to/raw_data_organized/experiment_metadata.json"
)

# 2. Use the recommended class-based interface
qc = ExperimentDataQC("/path/to/quality_control", author_designation="automated_qc")
qc.flag_image("20241215_A01_t001", "BLUR", notes="Low variance: 45")
qc.flag_image("20241215_A01_t002", "DRY_WELL", notes="Visual inspection")
qc.save()

# 3. Batch operations for efficiency
qc = ExperimentDataQC("/path/to/quality_control", author_designation="batch_processor")
image_flags = {
    "20241215_A01_t001": "BLUR",
    "20241215_A01_t002": ["BLUR", "DRY_WELL"],
    "20241215_A01_t003": [
        {"qc_flag": "CORRUPT", "author": "auto", "notes": "Cannot read file"}
    ]
}
qc.flag_images_batch(image_flags)
qc.save()

# 4. Advanced batch with gen_flag_batch (MOST FLEXIBLE)
qc = ExperimentDataQC("/path/to/quality_control", author_designation="tech_reviewer")

# Option A: Return batches (DEFAULT - for accumulation)
batch = []
batch += qc.gen_flag_batch("experiment", "20241215", "PROTOCOL_DEVIATION", notes="Missing controls")
batch += qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Focus problems")
batch += qc.gen_flag_batch("image", "20241215_A01_t001", ["BLUR", "OVEREXPOSURE"], notes="Multiple issues")
qc.add_flag_batch(batch)  # Apply accumulated batch

# Option B: Apply directly in memory (apply_directly=True)
qc.gen_flag_batch("experiment", "20241215", "INCOMPLETE", notes="Missing data", apply_directly=True)
qc.gen_flag_batch("video", "20241215_A02", "TECHNICAL_ISSUE", notes="Camera issue", apply_directly=True)
qc.gen_flag_batch("image", "20241215_A02_t001", "DRY_WELL", notes="Detected drying", apply_directly=True)

# Option C: Mixed mode - some batched, some direct
batch = []
for image_id, analysis in automated_analysis.items():
    if analysis['needs_review']:
        batch += qc.gen_flag_batch("image", image_id, "BLUR", notes=f"Score: {analysis['blur_score']}")
    else:
        qc.gen_flag_batch("image", image_id, "DRY_WELL", notes="Clear case", apply_directly=True)

if batch:  # Apply accumulated flags that need review
    qc.add_flag_batch(batch)

qc.save()  # Save all changes

📋 QC FLAG LEVELS
==================
- experiment_level: Issues affecting entire experiments
- video_level: Issues affecting video sequences  
- image_level: Issues with individual images (blur, corruption, etc.)
- embryo_level: Issues with detected embryos (death, detection failures)

🔧 INSTALLATION REQUIREMENTS
=============================
No external dependencies beyond Python standard library.
Uses JSON for data storage and pathlib for file operations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Default QC flag categories - ONLY used when creating new JSON files
# The actual valid categories are stored in the JSON file as single source of truth
DEFAULT_QC_FLAG_CATEGORIES = {
    "experiment_level": {
        "PROTOCOL_DEVIATION": "Deviation from standard imaging protocol",
        "INCOMPLETE": "Experiment was not completed"
    },
    "video_level": {
    },
    "image_level": {
        "BLUR": "Image is blurry (low variance of Laplacian)",
        "DRY_WELL": "Well dried out during imaging",
        "CORRUPT": "Cannot read/process image"
    },
    "embryo_level": {
        "DEAD_EMBRYO": "Embryo appears dead",
        "EMBRYO_NOT_DETECTED": "No embryo detected in expected location",
        "ABNORMAL_DEVELOPMENT": "Embryo shows abnormal development patterns"
    }
}

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_qc_structure_from_metadata(
    quality_control_dir: Union[str, Path],
    experiment_metadata_path: Union[str, Path],
    overwrite: bool = False,
    validate_flags: bool = True
) -> Dict:
    """
    Initialize QC structure from experiment metadata - START HERE!
    
    This is typically the first function you'll call to set up the QC system.
    It creates the QC JSON file and populates it with entries for all experiments,
    videos, and images found in your experiment metadata.
    
    Args:
        quality_control_dir: Directory where QC JSON file will be created
        experiment_metadata_path: Path to your experiment_metadata.json
        overwrite: Whether to replace existing QC data
        validate_flags: Whether to validate existing flags against valid categories
    
    Returns:
        QC data dictionary
        
    Example:
        # Set up QC system for the first time
        qc_data = initialize_qc_structure_from_metadata(
            quality_control_dir="/data/morphseq/quality_control",
            experiment_metadata_path="/data/morphseq/raw_data_organized/experiment_metadata.json"
        )
        print(f"Initialized QC for {len(qc_data['experiments'])} experiments")
    """
    
    # Load existing QC data
    qc_data = load_qc_data(quality_control_dir) if not overwrite else {
        "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
        "experiments": {}
    }
    
    # Load experiment metadata
    if not Path(experiment_metadata_path).exists():
        print(f"Warning: Experiment metadata not found at {experiment_metadata_path}")
        return qc_data
    
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    experiments_data = metadata.get('experiments', {})
    added_count = {"experiments": 0, "videos": 0, "images": 0}
    
    for experiment_id, experiment_data in experiments_data.items():
        # Initialize experiment if not exists
        if experiment_id not in qc_data["experiments"]:
            qc_data["experiments"][experiment_id] = {
                "flags": [],
                "authors": [],
                "notes": [],
                "videos": {}
            }
            added_count["experiments"] += 1
        
        experiment_qc = qc_data["experiments"][experiment_id]
        
        # Process videos in this experiment
        videos_data = experiment_data.get('videos', {})
        for video_id, video_data in videos_data.items():
            # Initialize video if not exists
            if video_id not in experiment_qc["videos"]:
                experiment_qc["videos"][video_id] = {
                    "flags": [],
                    "authors": [],
                    "notes": [],
                    "images": {},
                    "embryos": {}
                }
                added_count["videos"] += 1
            
            video_qc = experiment_qc["videos"][video_id]
            
            # Process images in this video
            image_ids = video_data.get('image_ids', [])
            for image_id in image_ids:
                if image_id not in video_qc["images"]:
                    video_qc["images"][image_id] = {
                        "flags": [],
                        "authors": [],
                        "notes": []
                    }
                    added_count["images"] += 1
    
    # Check flag integrity if requested
    if validate_flags:
        print("🔍 Checking flag integrity...")
        try:
            check_flag_integrity(qc_data, fix_invalid=False, verbose=True)
        except ValueError as e:
            print(f"\n🚨 FLAG INTEGRITY ERROR: {e}")
            print("\n🛠️  To fix this, you can:")
            print("   1. Run clean_invalid_flags() to automatically remove invalid flags")
            print("   2. Manually edit the QC JSON file to remove/rename invalid flags")
            print("   3. Add missing categories to 'valid_qc_flag_categories' in the JSON")
            raise
    
    # Save updated QC data
    save_qc_data(qc_data, quality_control_dir)
    
    print(f"QC structure initialized: {added_count['experiments']} experiments, "
          f"{added_count['videos']} videos, {added_count['images']} images added")
    
    return qc_data

def show_valid_qc_categories(quality_control_dir: Union[str, Path]) -> None:
    """
    Display all valid QC flag categories from your JSON file.
    
    Use this to see what flags are available before adding new ones.
    
    Example:
        show_valid_qc_categories("/data/morphseq/quality_control")
        # Output:
        # 📋 Valid QC Flag Categories:
        # 🏷️ Image Level:
        #    • BLUR: Image is blurry (low variance of Laplacian)
        #    • DRY_WELL: Well dried out during imaging
        #    ...
    """
    valid_categories = get_valid_qc_flag_categories(quality_control_dir)
    
    print("📋 Valid QC Flag Categories (from JSON file):")
    print("=" * 60)
    
    for level, flags in valid_categories.items():
        print(f"\n🏷️  {level.replace('_', ' ').title()}:")
        if flags:
            for flag, description in flags.items():
                print(f"   • {flag}: {description}")
        else:
            print("   • (No flags defined at this level)")
    
    print("\n💡 To add new flag categories, edit the 'valid_qc_flag_categories'")
    print("   section in the experiment_data_qc.json file manually.")

# =============================================================================
# MAIN CLASS INTERFACE (RECOMMENDED)
# =============================================================================

class ExperimentDataQC:
    """
    Main interface for managing experiment QC data.
    
    This class provides the recommended way to work with QC flags. It handles
    loading, validation, batch operations, and saving automatically.
    
    Key Features:
    - Loads QC data once and works in memory for speed
    - Validates flags against JSON-defined categories
    - Supports individual and batch operations
    - Tracks unsaved changes
    - Automatic flag integrity checking
    - Default author designation for all flag operations
    
    Basic Usage:
        qc = ExperimentDataQC("/path/to/quality_control", author_designation="analyst_name")
        qc.flag_image("20241215_A01_t001", "BLUR", notes="Blur score: 45")
        qc.save()
    
    Batch Usage:
        qc = ExperimentDataQC("/path/to/quality_control", author_designation="automated_qc")
        qc.flag_images_batch({
            "img_001": "BLUR",
            "img_002": ["BLUR", "DRY_WELL"],
            "img_003": [{"qc_flag": "CORRUPT", "author": "override_author", "notes": "Read error"}]
        })
        qc.save()
    """
    
    def __init__(self, quality_control_dir: Union[str, Path], author_designation: str, auto_validate: bool = True):
        """
        Initialize the QC data manager.
        
        Args:
            quality_control_dir: Directory containing the QC JSON file
            author_designation: Default author for all flag operations (REQUIRED)
            auto_validate: Whether to validate flags on load (default: True)
            
        Example:
            qc = ExperimentDataQC("/path/to/quality_control", author_designation="john_analyst")
        """
        self.quality_control_dir = Path(quality_control_dir)
        self.qc_json_path = get_qc_json_path(quality_control_dir)
        self.auto_validate = auto_validate
        self.author_designation = author_designation  # Store default author
        
        # Load QC data into memory
        self._qc_data = load_qc_data(quality_control_dir)
        
        # Validate flag integrity if requested
        if auto_validate:
            try:
                check_flag_integrity(self._qc_data, fix_invalid=False, verbose=False)
            except ValueError as e:
                print(f"🚨 FLAG INTEGRITY WARNING: {e}")
                print("💡 Use qc.clean_invalid_flags() to fix automatically")
        
        self._unsaved_changes = False
    
    @property
    def data(self) -> Dict:
        """Access to the underlying QC data dictionary."""
        return self._qc_data
    
    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    # -------------------------------------------------------------------------
    # Individual Flag Operations
    # -------------------------------------------------------------------------
    
    def flag_experiment(self, experiment_id: str, qc_flag: Union[str, List[str]], 
                       author: str = None, notes: str = "") -> None:
        """
        Flag an experiment.
        
        Args:
            experiment_id: ID of the experiment to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_experiment("20241215", "PROTOCOL_DEVIATION", notes="Missing time points")
            qc.flag_experiment("20241215", "INCOMPLETE", author="override_author", notes="Special case")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("experiment", experiment_id, qc_flag, final_author, notes)

    def flag_video(self, video_id: str, qc_flag: Union[str, List[str]], 
                  author: str = None, notes: str = "") -> None:
        """
        Flag a video.
        
        Args:
            video_id: ID of the video to flag
            qc_flag: Single flag or list of flags  
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Examples:
            # Single flag
            qc.flag_video("20241215_A01", "TECHNICAL_ISSUE", notes="Camera malfunction")
            
            # Multiple flags
            qc.flag_video("20241215_A01", ["TECHNICAL_ISSUE", "FOCUS_DRIFT"], notes="Multiple problems")
            
        Note: For maximum flexibility across all entity types, use gen_flag_batch() instead:
            batch = qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Camera issue")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("video", video_id, qc_flag, final_author, notes)
    
    def flag_image(self, image_id: str, qc_flag: Union[str, List[str]], 
                  author: str = None, notes: str = "") -> None:
        """
        Flag an image.
        
        Args:
            image_id: ID of the image to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_image("20241215_A01_t001", "BLUR", notes="Variance of Laplacian: 45")
            qc.flag_image("20241215_A01_t002", ["BLUR", "DRY_WELL"], notes="Multiple issues")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("image", image_id, qc_flag, final_author, notes)

    def flag_embryo(self, embryo_id: str, qc_flag: Union[str, List[str]], 
                   author: str = None, notes: str = "") -> None:
        """
        Flag an embryo.
        
        Args:
            embryo_id: ID of the embryo to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_embryo("20241215_A01_t001_e01", "DEAD_EMBRYO", notes="No movement detected")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("embryo", embryo_id, qc_flag, final_author, notes)

    def flag_entity(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                   author: str = None, notes: str = "") -> None:
        """
        Flag an entity at any level (generic interface).
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
            
        Note: This method applies flags directly. For batch operations, use gen_flag_batch().
        """
        final_author = author if author is not None else self.author_designation
        batch = gen_flag_batch(level, entity_id, qc_flag, final_author, notes)
        self._qc_data = add_flags_to_qc_data(self._qc_data, batch, validate_flags=self.auto_validate)
        self._unsaved_changes = True
    
    def flag_directly(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                     author: str = None, notes: str = "") -> None:
        """
        Apply flags directly to QC data (alias for flag_entity with clearer intent).
        
        This method is equivalent to flag_entity() but makes the intent clearer
        when you want to apply flags immediately rather than building batches.
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Examples:
            # Apply flags immediately (using default author)
            qc.flag_directly("experiment", "20241215", "PROTOCOL_DEVIATION", notes="Missing controls")
            qc.flag_directly("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Focus problems")
            qc.flag_directly("image", "20241215_A01_t001", ["BLUR", "DRY_WELL"], notes="Multiple issues")
            qc.save()  # Save when ready
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity(level, entity_id, qc_flag, final_author, notes)
    
    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------
    
    def flag_images_batch(self, image_flags: Dict[str, Union[str, List[str], List[Dict]]], 
                         author: str = "batch_processor", notes: str = "") -> None:
        """
        Efficient batch flagging for multiple images.
        
        Args:
            image_flags: Dictionary mapping image_id to flags
            author: Default author for all flags
            notes: Default notes for all flags
        
        Format Options:
            1. Single flag per image: {"image_id": "BLUR"}
            2. Multiple flags per image: {"image_id": ["BLUR", "DRY_WELL"]}
            3. Detailed flags: {"image_id": [{"qc_flag": "BLUR", "author": "auto", "notes": "Score: 45"}]}
        
        Example:
            image_flags = {
                "20241215_A01_t001": "BLUR",
                "20241215_A01_t002": ["BLUR", "DRY_WELL"],
                "20241215_A01_t003": [
                    {"qc_flag": "CORRUPT", "author": "auto", "notes": "Cannot read file"}
                ]
            }
            qc.flag_images_batch(image_flags, author="automated_qc")
        """
        batch = []
        for image_id, flags in image_flags.items():
            if isinstance(flags, str):
                # Single flag as string
                batch += self.gen_flag_batch("image", image_id, flags, author, notes)
                
            elif isinstance(flags, list):
                if not flags:
                    continue
                    
                if isinstance(flags[0], str):
                    # Multiple flags as list of strings
                    batch += self.gen_flag_batch("image", image_id, flags, author, notes)
                    
                elif isinstance(flags[0], dict):
                    # Multiple flags with detailed info
                    for flag_info in flags:
                        flag_author = flag_info.get("author", author)
                        flag_notes = flag_info.get("notes", notes)
                        batch += self.gen_flag_batch("image", image_id, flag_info["qc_flag"], flag_author, flag_notes)
        
        self.add_flag_batch(batch)
    
    def gen_flag_batch(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                      author: str, notes: str = "", apply_directly: bool = False) -> Optional[List[Dict]]:
        """
        Generate flag batch entries OR apply directly to QC data (MOST FLEXIBLE METHOD).
        
        This is the most flexible way to create flags because it works uniformly
        across ALL entity types and offers two modes of operation:
        
        Mode 1 (DEFAULT): Return batch for accumulation (apply_directly=False)
        Mode 2: Apply directly to QC data in memory (apply_directly=True)
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single QC flag or list of QC flags
            author: Who added the flag(s)
            notes: Optional additional details
            apply_directly: If True, apply to QC data immediately. If False, return batch.
        
        Returns:
            List of batch entries (if apply_directly=False) or None (if apply_directly=True)
        
        Examples:
            # Mode 1: Build batches for later application (DEFAULT)
            batch = []
            batch += qc.gen_flag_batch("experiment", "20241215", "PROTOCOL_DEVIATION", "tech")
            batch += qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", "operator") 
            batch += qc.gen_flag_batch("image", "20241215_A01_t001", ["BLUR", "DRY_WELL"], "auto")
            qc.add_flag_batch(batch)  # Apply accumulated batch
            
            # Mode 2: Apply directly to memory (no batch needed)
            qc.gen_flag_batch("experiment", "20241215", "INCOMPLETE", "tech", apply_directly=True)
            qc.gen_flag_batch("video", "20241215_A02", "TECHNICAL_ISSUE", "op", apply_directly=True)
            qc.gen_flag_batch("image", "20241215_A02_t001", "DRY_WELL", "auto", apply_directly=True)
            # Flags applied immediately, just save when ready
            qc.save()
            
            # Mixed mode: Some batched, some direct
            batch = []
            if complex_condition:
                batch += qc.gen_flag_batch("image", "img_001", "BLUR", "auto")
            else:
                qc.gen_flag_batch("image", "img_001", "DRY_WELL", "auto", apply_directly=True)
        """
        if apply_directly:
            # Apply flags directly to QC data in memory
            batch = gen_flag_batch(level, entity_id, qc_flag, author, notes)
            self._qc_data = add_flags_to_qc_data(self._qc_data, batch, validate_flags=self.auto_validate)
            self._unsaved_changes = True
            return None
        else:
            # Return batch for accumulation (default behavior)
            return gen_flag_batch(level, entity_id, qc_flag, author, notes)
    
    def add_flag_batch(self, flag_batch: List[Dict]) -> None:
        """Apply a batch of flags to QC data in memory."""
        self._qc_data = add_flags_to_qc_data(self._qc_data, flag_batch, validate_flags=self.auto_validate)
        self._unsaved_changes = True
    
    # -------------------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------------------
    
    def save(self, backup: bool = False) -> None:
        """
        Save QC data to file.
        
        Example:
            qc.save(backup=True)  # Creates timestamped backup before saving
        """
        if backup and self.qc_json_path.exists():
            backup_path = self.qc_json_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(self.qc_json_path, backup_path)
            print(f"📋 Created backup: {backup_path}")
        
        save_qc_data(self._qc_data, self.quality_control_dir)
        self._unsaved_changes = False
        print(f"💾 Saved QC data to {self.qc_json_path}")
    
    def reload(self) -> None:
        """Reload QC data from file (discarding unsaved changes)."""
        if self._unsaved_changes:
            print("⚠️  Warning: Unsaved changes will be lost!")
        
        self._qc_data = load_qc_data(self.quality_control_dir)
        self._unsaved_changes = False
        print(f"🔄 Reloaded QC data from {self.qc_json_path}")
    
    # -------------------------------------------------------------------------
    # Validation and Utilities
    # -------------------------------------------------------------------------
    
    def show_valid_flags(self) -> None:
        """Display all valid QC flag categories."""
        valid_categories = self.get_valid_flag_categories()
        
        print("📋 Valid QC Flag Categories (from JSON file):")
        print("=" * 60)
        
        for level, flags in valid_categories.items():
            print(f"\n🏷️  {level.replace('_', ' ').title()}:")
            if flags:
                for flag, description in flags.items():
                    print(f"   • {flag}: {description}")
            else:
                print("   • (No flags defined at this level)")
    
    def get_valid_flag_categories(self) -> Dict:
        """Get valid QC flag categories from the JSON data."""
        return self._qc_data.get("valid_qc_flag_categories", {})
    
    def validate_flag(self, level: str, qc_flag: str) -> bool:
        """Validate that a QC flag is valid for the given level."""
        valid_flags = self._qc_data.get("valid_qc_flag_categories", {}).get(f"{level}_level", {})
        return qc_flag in valid_flags
    
    def check_flag_integrity(self, fix_invalid: bool = False) -> Dict:
        """
        Check integrity of QC flags against valid categories.
        
        Example:
            result = qc.check_flag_integrity(fix_invalid=True)
            print(f"Fixed {result['fixed_count']} invalid flags")
        """
        result = check_flag_integrity(self._qc_data, fix_invalid=fix_invalid, verbose=True)
        if fix_invalid and result["fixed_count"] > 0:
            self._unsaved_changes = True
        return result
    
    def clean_invalid_flags(self) -> Dict:
        """Clean invalid flags from QC data."""
        result = self.check_flag_integrity(fix_invalid=True)
        if result["fixed_count"] > 0:
            print(f"✅ Cleaned {result['fixed_count']} invalid flags")
            self._unsaved_changes = True
        return result
    
    def get_entity_flags(self, level: str, entity_id: str) -> Dict[str, List]:
        """
        Get QC flags for a specific entity.
        
        Example:
            flags = qc.get_entity_flags("image", "20241215_A01_t001")
            print(f"Flags: {flags['flags']}")
            print(f"Authors: {flags['authors']}")
        """
        parent_ids = None
        
        if level == "image":
            parent_ids = parse_image_id(entity_id)
        elif level == "embryo":
            parent_ids = parse_embryo_id(entity_id)
        elif level == "video":
            parts = entity_id.split('_')
            if len(parts) >= 2:
                parent_ids = {"experiment_id": parts[0]}
        
        return get_qc_flags(self.quality_control_dir, level, entity_id, parent_ids)
    
    def get_image_flags(self, image_id: str) -> Dict[str, List]:
        """Get QC flags for an image."""
        return self.get_entity_flags("image", image_id)
    
    def get_summary(self) -> Dict:
        """
        Generate a summary of QC flags across all levels.
        
        Example:
            summary = qc.get_summary()
            print(f"BLUR flags: {summary['image_level'].get('BLUR', 0)}")
        """
        return get_qc_summary(self.quality_control_dir)
    
    def initialize_from_metadata(self, experiment_metadata_path: Union[str, Path], 
                                overwrite: bool = False) -> None:
        """Initialize QC structure from experiment metadata."""
        self._qc_data = initialize_qc_structure_from_metadata(
            quality_control_dir=self.quality_control_dir,
            experiment_metadata_path=experiment_metadata_path,
            overwrite=overwrite,
            validate_flags=self.auto_validate
        )
        self._unsaved_changes = True
        print("✅ QC structure initialized from metadata")
    
    def __repr__(self) -> str:
        """String representation of the QC manager."""
        num_experiments = len(self._qc_data.get("experiments", {}))
        num_categories = len(self._qc_data.get("valid_qc_flag_categories", {}))
        status = "✅ saved" if not self._unsaved_changes else "⚠️  unsaved changes"
        
        return f"ExperimentDataQC(author='{self.author_designation}', experiments={num_experiments}, categories={num_categories}, status={status})"

# =============================================================================
# CORE DATA OPERATIONS
# =============================================================================

def get_qc_json_path(quality_control_dir: Union[str, Path]) -> Path:
    """Get the path to the experiment data QC JSON file."""
    return Path(quality_control_dir) / "experiment_data_qc.json"

def load_qc_data(quality_control_dir: Union[str, Path]) -> Dict:
    """
    Load QC data from JSON file.
    
    Creates a new file with default structure if none exists.
    """
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if not qc_json_path.exists():
        # Create empty structure with default categories
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }
    
    try:
        with open(qc_json_path, 'r') as f:
            qc_data = json.load(f)
        
        # Ensure structure is complete
        if "valid_qc_flag_categories" not in qc_data:
            qc_data["valid_qc_flag_categories"] = DEFAULT_QC_FLAG_CATEGORIES.copy()
        if "experiments" not in qc_data:
            qc_data["experiments"] = {}
            
        return qc_data
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load QC data from {qc_json_path}: {e}")
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }

def save_qc_data(qc_data: Dict, quality_control_dir: Union[str, Path]) -> None:
    """Save QC data to JSON file."""
    qc_json_path = get_qc_json_path(quality_control_dir)
    qc_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(qc_json_path, 'w') as f:
        json.dump(qc_data, f, indent=2)

def get_valid_qc_flag_categories(quality_control_dir: Union[str, Path]) -> Dict:
    """Get valid QC flag categories from the JSON file."""
    qc_data = load_qc_data(quality_control_dir)
    return qc_data.get("valid_qc_flag_categories", {})

# =============================================================================
# VALIDATION AND INTEGRITY CHECKING
# =============================================================================

def check_flag_integrity(qc_data: Dict, fix_invalid: bool = False, verbose: bool = True) -> Dict:
    """
    Check integrity of QC flags against valid categories stored in the JSON.
    
    Args:
        qc_data: QC data dictionary
        fix_invalid: If True, remove invalid flags. If False, raise error on invalid flags.
        verbose: Print detailed information about found issues
        
    Returns:
        Dict with integrity check results and optionally cleaned data
        
    Raises:
        ValueError: If invalid flags found and fix_invalid=False
    """
    valid_categories = qc_data.get("valid_qc_flag_categories", {})
    issues = []
    fixed_count = 0
    
    # Check each level
    for exp_id, exp_data in qc_data.get("experiments", {}).items():
        # Check experiment-level flags
        for flag in exp_data.get("flags", []):
            if flag not in valid_categories.get("experiment_level", {}):
                issue = f"Invalid experiment flag '{flag}' in experiment {exp_id}"
                issues.append(issue)
                if fix_invalid:
                    exp_data["flags"].remove(flag)
                    fixed_count += 1
        
        # Check video-level flags
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            for flag in vid_data.get("flags", []):
                if flag not in valid_categories.get("video_level", {}):
                    issue = f"Invalid video flag '{flag}' in video {vid_id}"
                    issues.append(issue)
                    if fix_invalid:
                        vid_data["flags"].remove(flag)
                        fixed_count += 1
            
            # Check image-level flags
            for img_id, img_data in vid_data.get("images", {}).items():
                for flag in img_data.get("flags", []):
                    if flag not in valid_categories.get("image_level", {}):
                        issue = f"Invalid image flag '{flag}' in image {img_id}"
                        issues.append(issue)
                        if fix_invalid:
                            img_data["flags"].remove(flag)
                            fixed_count += 1
            
            # Check embryo-level flags
            for emb_id, emb_data in vid_data.get("embryos", {}).items():
                for flag in emb_data.get("flags", []):
                    if flag not in valid_categories.get("embryo_level", {}):
                        issue = f"Invalid embryo flag '{flag}' in embryo {emb_id}"
                        issues.append(issue)
                        if fix_invalid:
                            emb_data["flags"].remove(flag)
                            fixed_count += 1
    
    # Report results
    if verbose:
        if issues:
            print(f"🚨 Flag integrity check found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more issues")
            
            if fix_invalid:
                print(f"✅ Fixed {fixed_count} invalid flags")
            else:
                print(f"💡 Run with fix_invalid=True to automatically remove invalid flags")
        else:
            print("✅ All flags are valid!")
    
    # Raise error if issues found and not fixing
    if issues and not fix_invalid:
        raise ValueError(
            f"Found {len(issues)} invalid QC flags. "
            f"Valid categories are defined in the JSON file under 'valid_qc_flag_categories'. "
            f"Either fix these manually or run check_flag_integrity() with fix_invalid=True"
        )
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "fixed_count": fixed_count if fix_invalid else 0,
        "qc_data": qc_data if fix_invalid else None
    }

def validate_qc_flag(qc_flag: str, level: str, qc_data: Dict) -> bool:
    """Validate that a QC flag is valid for the given level."""
    valid_flags = qc_data.get("valid_qc_flag_categories", {}).get(f"{level}_level", {})
    return qc_flag in valid_flags

def clean_invalid_flags(quality_control_dir: Union[str, Path], backup: bool = True) -> Dict:
    """
    Clean invalid flags from QC data file.
    
    Example:
        result = clean_invalid_flags("/path/to/quality_control", backup=True)
        print(f"Cleaned {result['fixed_count']} invalid flags")
    """
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if backup and qc_json_path.exists():
        backup_path = qc_json_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(qc_json_path, backup_path)
        print(f"📋 Created backup: {backup_path}")
    
    qc_data = load_qc_data(quality_control_dir)
    result = check_flag_integrity(qc_data, fix_invalid=True, verbose=True)
    
    if result["fixed_count"] > 0:
        # Mixed mode example (batch + direct)
        qc_data = load_qc_data(quality_control_dir)
        batch = []
        
        for image_id, analysis in image_results.items():
            if analysis['confidence'] < 0.8:  # Low confidence -> batch for review
                batch += gen_flag_batch("image", image_id, "BLUR", "auto", f"Low confidence: {analysis['confidence']}")
            else:  # High confidence -> apply directly
                gen_flag_batch("image", image_id, "DRY_WELL", "auto", "High confidence detection", 
                              qc_data=qc_data, apply_directly=True)
        
        # Apply batched flags that need review
        if batch:
            qc_data = add_flags_to_qc_data(qc_data, batch)
        
        save_qc_data(qc_data, quality_control_dir)
        print(f"💾 Saved cleaned QC data")
    
    return result

# =============================================================================
# INDIVIDUAL FLAG OPERATIONS
# =============================================================================

def flag_image(
    quality_control_dir: Union[str, Path],
    image_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """
    Convenience function to flag an image.
    
    Example:
        flag_image(
            quality_control_dir="/path/to/quality_control",
            image_id="20241215_A01_t001",
            qc_flag="BLUR",
            author="automatic",
            notes="Variance of Laplacian: 45"
        )
    """
    parent_ids = parse_image_id(image_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="image",
        entity_id=image_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_video(
    quality_control_dir: Union[str, Path],
    video_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag a video."""
    # Extract experiment_id from video_id (format: experiment_id_well_id)
    parts = video_id.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid video_id format: {video_id}")
    
    experiment_id = parts[0]
    parent_ids = {"experiment_id": experiment_id}
    
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="video",
        entity_id=video_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_experiment(
    quality_control_dir: Union[str, Path],
    experiment_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an experiment."""
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="experiment",
        entity_id=experiment_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_embryo(
    quality_control_dir: Union[str, Path],
    embryo_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an embryo."""
    parent_ids = parse_embryo_id(embryo_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="embryo",
        entity_id=embryo_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

# =============================================================================
# EFFICIENT BATCH OPERATIONS
# =============================================================================
# 
# gen_flag_batch() is the MOST FLEXIBLE method for creating QC flags:
# - Works across ALL entity types uniformly
# - Two modes: return batches for accumulation OR apply directly to QC data
# - Handles conditional flag building and complex workflows
# - Optimizes performance with deferred or immediate I/O
# - Enables mixed accumulation and direct application patterns
#

def gen_flag_batch(level: str, entity_id: str, qc_flag: Union[str, List[str]], author: str, 
                   notes: str = "", qc_data: Optional[Dict] = None, 
                   apply_directly: bool = False) -> Optional[List[Dict]]:
    """
    Generate flag batch entries OR apply directly to QC data (MOST FLEXIBLE METHOD).
    
    This is the most flexible and powerful function for creating QC flags because:
    1. Works uniformly across ALL entity types (experiment, video, image, embryo)
    2. Handles both single flags and lists of flags seamlessly
    3. Two modes: return batches for accumulation OR apply directly to QC data
    4. Enables conditional flag building and complex workflows
    5. Optimizes performance by enabling deferred or immediate I/O operations
    
    Args:
        quality_control_dir: Directory containing the QC JSON file
        
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This function displays them for user reference when adding flags.
    """
    valid_categories = get_valid_qc_flag_categories(quality_control_dir)
    
    print("📋 Valid QC Flag Categories (from JSON file):")
    print("=" * 60)
    
    for level, flags in valid_categories.items():
        print(f"\n🏷️  {level.replace('_', ' ').title()}:")
        if flags:
            for flag, description in flags.items():
                print(f"   • {flag}: {description}")
        else:
            print("   • (No flags defined at this level)")
    
    print("\n💡 To add new flag categories, edit the 'valid_qc_flag_categories'")
    print("   section in the experiment_data_qc.json file manually.")

def initialize_qc_structure_from_metadata(
    quality_control_dir: Union[str, Path],
    experiment_metadata_path: Union[str, Path],
    overwrite: bool = False,
    validate_flags: bool = True
) -> Dict:
    """
    Initialize QC structure from experiment metadata.
    Creates entries for all experiments, videos, and images found in metadata.
    
    Args:
        quality_control_dir: Directory where the QC JSON file will be stored
        experiment_metadata_path: Path to the experiment_metadata.json file
        overwrite: Whether to overwrite existing QC data
        validate_flags: Whether to validate existing flags against valid categories
    
    Design Philosophy:
        Valid QC flag categories are stored ONLY in the JSON file as the single source of truth.
        This ensures consistency and allows manual management of flag categories.
        By default, flag integrity is checked to prevent invalid flags from persisting.
    """
    
    # Load existing QC data
    qc_data = load_qc_data(quality_control_dir) if not overwrite else {
        "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
        "experiments": {}
    }
    
    # Load experiment metadata
    if not Path(experiment_metadata_path).exists():
        print(f"Warning: Experiment metadata not found at {experiment_metadata_path}")
        return qc_data
    
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    experiments_data = metadata.get('experiments', {})
    added_count = {"experiments": 0, "videos": 0, "images": 0}
    
    for experiment_id, experiment_data in experiments_data.items():
        # Initialize experiment if not exists
        if experiment_id not in qc_data["experiments"]:
            qc_data["experiments"][experiment_id] = {
                "flags": [],
                "authors": [],
                "notes": [],
                "videos": {}
            }
            added_count["experiments"] += 1
        
        experiment_qc = qc_data["experiments"][experiment_id]
        
        # Process videos in this experiment
        videos_data = experiment_data.get('videos', {})
        for video_id, video_data in videos_data.items():
            # Initialize video if not exists
            if video_id not in experiment_qc["videos"]:
                experiment_qc["videos"][video_id] = {
                    "flags": [],
                    "authors": [],
                    "notes": [],
                    "images": {},
                    "embryos": {}
                }
                added_count["videos"] += 1
            
            video_qc = experiment_qc["videos"][video_id]
            
            # Process images in this video
            image_ids = video_data.get('image_ids', [])
            for image_id in image_ids:
                if image_id not in video_qc["images"]:
                    video_qc["images"][image_id] = {
                        "flags": [],
                        "authors": [],
                        "notes": []
                    }
                    added_count["images"] += 1
    
    # Check flag integrity if requested
    if validate_flags:
        print("🔍 Checking flag integrity...")
        try:
            check_flag_integrity(qc_data, fix_invalid=False, verbose=True)
        except ValueError as e:
            print(f"\n🚨 FLAG INTEGRITY ERROR: {e}")
            print("\n🛠️  To fix this, you can:")
            print("   1. Run clean_invalid_flags() to automatically remove invalid flags")
            print("   2. Manually edit the QC JSON file to remove/rename invalid flags")
            print("   3. Add missing categories to 'valid_qc_flag_categories' in the JSON")
            raise
    
    # Save updated QC data
    save_qc_data(qc_data, quality_control_dir)
    
    print(f"QC structure initialized: {added_count['experiments']} experiments, "
          f"{added_count['videos']} videos, {added_count['images']} images added")
    
    return qc_data

<<<<<<< HEAD
def show_valid_qc_categories(quality_control_dir: Union[str, Path]) -> None:
    """
    Display all valid QC flag categories from your JSON file.
    
    Use this to see what flags are available before adding new ones.
    
    Example:
        show_valid_qc_categories("/data/morphseq/quality_control")
        # Output:
        # 📋 Valid QC Flag Categories:
        # 🏷️ Image Level:
        #    • BLUR: Image is blurry (low variance of Laplacian)
        #    • DRY_WELL: Well dried out during imaging
        #    ...
    """
    valid_categories = get_valid_qc_flag_categories(quality_control_dir)
    
    print("📋 Valid QC Flag Categories (from JSON file):")
    print("=" * 60)
    
    for level, flags in valid_categories.items():
        print(f"\n🏷️  {level.replace('_', ' ').title()}:")
        if flags:
            for flag, description in flags.items():
                print(f"   • {flag}: {description}")
        else:
            print("   • (No flags defined at this level)")
    
    print("\n💡 To add new flag categories, edit the 'valid_qc_flag_categories'")
    print("   section in the experiment_data_qc.json file manually.")

# =============================================================================
# MAIN CLASS INTERFACE (RECOMMENDED)
# =============================================================================

class ExperimentDataQC:
    """
    Main interface for managing experiment QC data.
    
    This class provides the recommended way to work with QC flags. It handles
    loading, validation, batch operations, and saving automatically.
    
    Key Features:
    - Loads QC data once and works in memory for speed
    - Validates flags against JSON-defined categories
    - Supports individual and batch operations
    - Tracks unsaved changes
    - Automatic flag integrity checking
    - Default author designation for all flag operations
    
    Basic Usage:
        qc = ExperimentDataQC("/path/to/quality_control", author_designation="analyst_name")
        qc.flag_image("20241215_A01_t001", "BLUR", notes="Blur score: 45")
        qc.save()
    
    Batch Usage:
        qc = ExperimentDataQC("/path/to/quality_control", author_designation="automated_qc")
        qc.flag_images_batch({
            "img_001": "BLUR",
            "img_002": ["BLUR", "DRY_WELL"],
            "img_003": [{"qc_flag": "CORRUPT", "author": "override_author", "notes": "Read error"}]
        })
        qc.save()
    """
    
    def __init__(self, quality_control_dir: Union[str, Path], author_designation: str, auto_validate: bool = True):
        """
        Initialize the QC data manager.
        
        Args:
            quality_control_dir: Directory containing the QC JSON file
            author_designation: Default author for all flag operations (REQUIRED)
            auto_validate: Whether to validate flags on load (default: True)
            
        Example:
            qc = ExperimentDataQC("/path/to/quality_control", author_designation="john_analyst")
        """
        self.quality_control_dir = Path(quality_control_dir)
        self.qc_json_path = get_qc_json_path(quality_control_dir)
        self.auto_validate = auto_validate
        self.author_designation = author_designation  # Store default author
        
        # Load QC data into memory
        self._qc_data = load_qc_data(quality_control_dir)
        
        # Validate flag integrity if requested
        if auto_validate:
            try:
                check_flag_integrity(self._qc_data, fix_invalid=False, verbose=False)
            except ValueError as e:
                print(f"🚨 FLAG INTEGRITY WARNING: {e}")
                print("💡 Use qc.clean_invalid_flags() to fix automatically")
        
        self._unsaved_changes = False
    
    @property
    def data(self) -> Dict:
        """Access to the underlying QC data dictionary."""
        return self._qc_data
    
    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    # -------------------------------------------------------------------------
    # Individual Flag Operations
    # -------------------------------------------------------------------------
    
    def flag_experiment(self, experiment_id: str, qc_flag: Union[str, List[str]], 
                       author: str = None, notes: str = "") -> None:
        """
        Flag an experiment.
        
        Args:
            experiment_id: ID of the experiment to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_experiment("20241215", "PROTOCOL_DEVIATION", notes="Missing time points")
            qc.flag_experiment("20241215", "INCOMPLETE", author="override_author", notes="Special case")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("experiment", experiment_id, qc_flag, final_author, notes)

    def flag_video(self, video_id: str, qc_flag: Union[str, List[str]], 
                  author: str = None, notes: str = "") -> None:
        """
        Flag a video.
        
        Args:
            video_id: ID of the video to flag
            qc_flag: Single flag or list of flags  
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Examples:
            # Single flag
            qc.flag_video("20241215_A01", "TECHNICAL_ISSUE", notes="Camera malfunction")
            
            # Multiple flags
            qc.flag_video("20241215_A01", ["TECHNICAL_ISSUE", "FOCUS_DRIFT"], notes="Multiple problems")
            
        Note: For maximum flexibility across all entity types, use gen_flag_batch() instead:
            batch = qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Camera issue")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("video", video_id, qc_flag, final_author, notes)
    
    def flag_image(self, image_id: str, qc_flag: Union[str, List[str]], 
                  author: str = None, notes: str = "") -> None:
        """
        Flag an image.
        
        Args:
            image_id: ID of the image to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_image("20241215_A01_t001", "BLUR", notes="Variance of Laplacian: 45")
            qc.flag_image("20241215_A01_t002", ["BLUR", "DRY_WELL"], notes="Multiple issues")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("image", image_id, qc_flag, final_author, notes)

    def flag_embryo(self, embryo_id: str, qc_flag: Union[str, List[str]], 
                   author: str = None, notes: str = "") -> None:
        """
        Flag an embryo.
        
        Args:
            embryo_id: ID of the embryo to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Example:
            qc.flag_embryo("20241215_A01_t001_e01", "DEAD_EMBRYO", notes="No movement detected")
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity("embryo", embryo_id, qc_flag, final_author, notes)

    def flag_entity(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                   author: str = None, notes: str = "") -> None:
        """
        Flag an entity at any level (generic interface).
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
            
        Note: This method applies flags directly. For batch operations, use gen_flag_batch().
        """
        final_author = author if author is not None else self.author_designation
        batch = gen_flag_batch(level, entity_id, qc_flag, final_author, notes)
        self._qc_data = add_flags_to_qc_data(self._qc_data, batch, validate_flags=self.auto_validate)
        self._unsaved_changes = True
    
    def flag_directly(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                     author: str = None, notes: str = "") -> None:
        """
        Apply flags directly to QC data (alias for flag_entity with clearer intent).
        
        This method is equivalent to flag_entity() but makes the intent clearer
        when you want to apply flags immediately rather than building batches.
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single flag or list of flags
            author: Author of the flag (defaults to instance's author_designation)
            notes: Optional notes
        
        Examples:
            # Apply flags immediately (using default author)
            qc.flag_directly("experiment", "20241215", "PROTOCOL_DEVIATION", notes="Missing controls")
            qc.flag_directly("video", "20241215_A01", "TECHNICAL_ISSUE", notes="Focus problems")
            qc.flag_directly("image", "20241215_A01_t001", ["BLUR", "DRY_WELL"], notes="Multiple issues")
            qc.save()  # Save when ready
        """
        final_author = author if author is not None else self.author_designation
        self.flag_entity(level, entity_id, qc_flag, final_author, notes)
    
    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------
    
    def flag_images_batch(self, image_flags: Dict[str, Union[str, List[str], List[Dict]]], 
                         author: str = "batch_processor", notes: str = "") -> None:
        """
        Efficient batch flagging for multiple images.
        
        Args:
            image_flags: Dictionary mapping image_id to flags
            author: Default author for all flags
            notes: Default notes for all flags
        
        Format Options:
            1. Single flag per image: {"image_id": "BLUR"}
            2. Multiple flags per image: {"image_id": ["BLUR", "DRY_WELL"]}
            3. Detailed flags: {"image_id": [{"qc_flag": "BLUR", "author": "auto", "notes": "Score: 45"}]}
        
        Example:
            image_flags = {
                "20241215_A01_t001": "BLUR",
                "20241215_A01_t002": ["BLUR", "DRY_WELL"],
                "20241215_A01_t003": [
                    {"qc_flag": "CORRUPT", "author": "auto", "notes": "Cannot read file"}
                ]
            }
            qc.flag_images_batch(image_flags, author="automated_qc")
        """
        batch = []
        for image_id, flags in image_flags.items():
            if isinstance(flags, str):
                # Single flag as string
                batch += self.gen_flag_batch("image", image_id, flags, author, notes)
                
            elif isinstance(flags, list):
                if not flags:
                    continue
                    
                if isinstance(flags[0], str):
                    # Multiple flags as list of strings
                    batch += self.gen_flag_batch("image", image_id, flags, author, notes)
                    
                elif isinstance(flags[0], dict):
                    # Multiple flags with detailed info
                    for flag_info in flags:
                        flag_author = flag_info.get("author", author)
                        flag_notes = flag_info.get("notes", notes)
                        batch += self.gen_flag_batch("image", image_id, flag_info["qc_flag"], flag_author, flag_notes)
        
        self.add_flag_batch(batch)
    
    def gen_flag_batch(self, level: str, entity_id: str, qc_flag: Union[str, List[str]], 
                      author: str, notes: str = "", apply_directly: bool = False) -> Optional[List[Dict]]:
        """
        Generate flag batch entries OR apply directly to QC data (MOST FLEXIBLE METHOD).
        
        This is the most flexible way to create flags because it works uniformly
        across ALL entity types and offers two modes of operation:
        
        Mode 1 (DEFAULT): Return batch for accumulation (apply_directly=False)
        Mode 2: Apply directly to QC data in memory (apply_directly=True)
        
        Args:
            level: "experiment", "video", "image", or "embryo"
            entity_id: ID of the entity to flag
            qc_flag: Single QC flag or list of QC flags
            author: Who added the flag(s)
            notes: Optional additional details
            apply_directly: If True, apply to QC data immediately. If False, return batch.
        
        Returns:
            List of batch entries (if apply_directly=False) or None (if apply_directly=True)
        
        Examples:
            # Mode 1: Build batches for later application (DEFAULT)
            batch = []
            batch += qc.gen_flag_batch("experiment", "20241215", "PROTOCOL_DEVIATION", "tech")
            batch += qc.gen_flag_batch("video", "20241215_A01", "TECHNICAL_ISSUE", "operator") 
            batch += qc.gen_flag_batch("image", "20241215_A01_t001", ["BLUR", "DRY_WELL"], "auto")
            qc.add_flag_batch(batch)  # Apply accumulated batch
            
            # Mode 2: Apply directly to memory (no batch needed)
            qc.gen_flag_batch("experiment", "20241215", "INCOMPLETE", "tech", apply_directly=True)
            qc.gen_flag_batch("video", "20241215_A02", "TECHNICAL_ISSUE", "op", apply_directly=True)
            qc.gen_flag_batch("image", "20241215_A02_t001", "DRY_WELL", "auto", apply_directly=True)
            # Flags applied immediately, just save when ready
            qc.save()
            
            # Mixed mode: Some batched, some direct
            batch = []
            if complex_condition:
                batch += qc.gen_flag_batch("image", "img_001", "BLUR", "auto")
            else:
                qc.gen_flag_batch("image", "img_001", "DRY_WELL", "auto", apply_directly=True)
        """
        if apply_directly:
            # Apply flags directly to QC data in memory
            batch = gen_flag_batch(level, entity_id, qc_flag, author, notes)
            self._qc_data = add_flags_to_qc_data(self._qc_data, batch, validate_flags=self.auto_validate)
            self._unsaved_changes = True
            return None
        else:
            # Return batch for accumulation (default behavior)
            return gen_flag_batch(level, entity_id, qc_flag, author, notes)
    
    def add_flag_batch(self, flag_batch: List[Dict]) -> None:
        """Apply a batch of flags to QC data in memory."""
        self._qc_data = add_flags_to_qc_data(self._qc_data, flag_batch, validate_flags=self.auto_validate)
        self._unsaved_changes = True
    
    # -------------------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------------------
    
    def save(self, backup: bool = False) -> None:
        """
        Save QC data to file.
        
        Example:
            qc.save(backup=True)  # Creates timestamped backup before saving
        """
        if backup and self.qc_json_path.exists():
            backup_path = self.qc_json_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(self.qc_json_path, backup_path)
            print(f"📋 Created backup: {backup_path}")
        
        save_qc_data(self._qc_data, self.quality_control_dir)
        self._unsaved_changes = False
        print(f"💾 Saved QC data to {self.qc_json_path}")
    
    def reload(self) -> None:
        """Reload QC data from file (discarding unsaved changes)."""
        if self._unsaved_changes:
            print("⚠️  Warning: Unsaved changes will be lost!")
        
        self._qc_data = load_qc_data(self.quality_control_dir)
        self._unsaved_changes = False
        print(f"🔄 Reloaded QC data from {self.qc_json_path}")
    
    # -------------------------------------------------------------------------
    # Validation and Utilities
    # -------------------------------------------------------------------------
    
    def show_valid_flags(self) -> None:
        """Display all valid QC flag categories."""
        valid_categories = self.get_valid_flag_categories()
        
        print("📋 Valid QC Flag Categories (from JSON file):")
        print("=" * 60)
        
        for level, flags in valid_categories.items():
            print(f"\n🏷️  {level.replace('_', ' ').title()}:")
            if flags:
                for flag, description in flags.items():
                    print(f"   • {flag}: {description}")
            else:
                print("   • (No flags defined at this level)")
    
    def get_valid_flag_categories(self) -> Dict:
        """Get valid QC flag categories from the JSON data."""
        return self._qc_data.get("valid_qc_flag_categories", {})
    
    def validate_flag(self, level: str, qc_flag: str) -> bool:
        """Validate that a QC flag is valid for the given level."""
        valid_flags = self._qc_data.get("valid_qc_flag_categories", {}).get(f"{level}_level", {})
        return qc_flag in valid_flags
    
    def check_flag_integrity(self, fix_invalid: bool = False) -> Dict:
        """
        Check integrity of QC flags against valid categories.
        
        Example:
            result = qc.check_flag_integrity(fix_invalid=True)
            print(f"Fixed {result['fixed_count']} invalid flags")
        """
        result = check_flag_integrity(self._qc_data, fix_invalid=fix_invalid, verbose=True)
        if fix_invalid and result["fixed_count"] > 0:
            self._unsaved_changes = True
        return result
    
    def clean_invalid_flags(self) -> Dict:
        """Clean invalid flags from QC data."""
        result = self.check_flag_integrity(fix_invalid=True)
        if result["fixed_count"] > 0:
            print(f"✅ Cleaned {result['fixed_count']} invalid flags")
            self._unsaved_changes = True
        return result
    
    def get_entity_flags(self, level: str, entity_id: str) -> Dict[str, List]:
        """
        Get QC flags for a specific entity.
        
        Example:
            flags = qc.get_entity_flags("image", "20241215_A01_t001")
            print(f"Flags: {flags['flags']}")
            print(f"Authors: {flags['authors']}")
        """
        parent_ids = None
        
        if level == "image":
            parent_ids = parse_image_id(entity_id)
        elif level == "embryo":
            parent_ids = parse_embryo_id(entity_id)
        elif level == "video":
            parts = entity_id.split('_')
            if len(parts) >= 2:
                parent_ids = {"experiment_id": parts[0]}
        
        return get_qc_flags(self.quality_control_dir, level, entity_id, parent_ids)
    
    def get_image_flags(self, image_id: str) -> Dict[str, List]:
        """Get QC flags for an image."""
        return self.get_entity_flags("image", image_id)
    
    def get_summary(self) -> Dict:
        """
        Generate a summary of QC flags across all levels.
        
        Example:
            summary = qc.get_summary()
            print(f"BLUR flags: {summary['image_level'].get('BLUR', 0)}")
        """
        return get_qc_summary(self.quality_control_dir)
    
    def initialize_from_metadata(self, experiment_metadata_path: Union[str, Path], 
                                overwrite: bool = False) -> None:
        """Initialize QC structure from experiment metadata."""
        self._qc_data = initialize_qc_structure_from_metadata(
            quality_control_dir=self.quality_control_dir,
            experiment_metadata_path=experiment_metadata_path,
            overwrite=overwrite,
            validate_flags=self.auto_validate
        )
        self._unsaved_changes = True
        print("✅ QC structure initialized from metadata")
    
    def __repr__(self) -> str:
        """String representation of the QC manager."""
        num_experiments = len(self._qc_data.get("experiments", {}))
        num_categories = len(self._qc_data.get("valid_qc_flag_categories", {}))
        status = "✅ saved" if not self._unsaved_changes else "⚠️  unsaved changes"
        
        return f"ExperimentDataQC(author='{self.author_designation}', experiments={num_experiments}, categories={num_categories}, status={status})"

# =============================================================================
# CORE DATA OPERATIONS
# =============================================================================

def get_qc_json_path(quality_control_dir: Union[str, Path]) -> Path:
    """Get the path to the experiment data QC JSON file."""
    return Path(quality_control_dir) / "experiment_data_qc.json"

def load_qc_data(quality_control_dir: Union[str, Path]) -> Dict:
    """
    Load QC data from JSON file.
    
    Creates a new file with default structure if none exists.
    """
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if not qc_json_path.exists():
        # Create empty structure with default categories
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }
    
    try:
        with open(qc_json_path, 'r') as f:
            qc_data = json.load(f)
        
        # Ensure structure is complete
        if "valid_qc_flag_categories" not in qc_data:
            qc_data["valid_qc_flag_categories"] = DEFAULT_QC_FLAG_CATEGORIES.copy()
        if "experiments" not in qc_data:
            qc_data["experiments"] = {}
            
        return qc_data
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load QC data from {qc_json_path}: {e}")
        return {
            "valid_qc_flag_categories": DEFAULT_QC_FLAG_CATEGORIES.copy(),
            "experiments": {}
        }

def save_qc_data(qc_data: Dict, quality_control_dir: Union[str, Path]) -> None:
    """Save QC data to JSON file."""
    qc_json_path = get_qc_json_path(quality_control_dir)
    qc_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(qc_json_path, 'w') as f:
        json.dump(qc_data, f, indent=2)

def get_valid_qc_flag_categories(quality_control_dir: Union[str, Path]) -> Dict:
    """Get valid QC flag categories from the JSON file."""
    qc_data = load_qc_data(quality_control_dir)
    return qc_data.get("valid_qc_flag_categories", {})

# =============================================================================
# VALIDATION AND INTEGRITY CHECKING
# =============================================================================

def check_flag_integrity(qc_data: Dict, fix_invalid: bool = False, verbose: bool = True) -> Dict:
    """
    Check integrity of QC flags against valid categories stored in the JSON.
    
    Args:
        qc_data: QC data dictionary
        fix_invalid: If True, remove invalid flags. If False, raise error on invalid flags.
        verbose: Print detailed information about found issues
        
    Returns:
        Dict with integrity check results and optionally cleaned data
        
    Raises:
        ValueError: If invalid flags found and fix_invalid=False
    """
    valid_categories = qc_data.get("valid_qc_flag_categories", {})
    issues = []
    fixed_count = 0
    
    # Check each level
    for exp_id, exp_data in qc_data.get("experiments", {}).items():
        # Check experiment-level flags
        for flag in exp_data.get("flags", []):
            if flag not in valid_categories.get("experiment_level", {}):
                issue = f"Invalid experiment flag '{flag}' in experiment {exp_id}"
                issues.append(issue)
                if fix_invalid:
                    exp_data["flags"].remove(flag)
                    fixed_count += 1
        
        # Check video-level flags
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            for flag in vid_data.get("flags", []):
                if flag not in valid_categories.get("video_level", {}):
                    issue = f"Invalid video flag '{flag}' in video {vid_id}"
                    issues.append(issue)
                    if fix_invalid:
                        vid_data["flags"].remove(flag)
                        fixed_count += 1
            
            # Check image-level flags
            for img_id, img_data in vid_data.get("images", {}).items():
                for flag in img_data.get("flags", []):
                    if flag not in valid_categories.get("image_level", {}):
                        issue = f"Invalid image flag '{flag}' in image {img_id}"
                        issues.append(issue)
                        if fix_invalid:
                            img_data["flags"].remove(flag)
                            fixed_count += 1
            
            # Check embryo-level flags
            for emb_id, emb_data in vid_data.get("embryos", {}).items():
                for flag in emb_data.get("flags", []):
                    if flag not in valid_categories.get("embryo_level", {}):
                        issue = f"Invalid embryo flag '{flag}' in embryo {emb_id}"
                        issues.append(issue)
                        if fix_invalid:
                            emb_data["flags"].remove(flag)
                            fixed_count += 1
    
    # Report results
    if verbose:
        if issues:
            print(f"🚨 Flag integrity check found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more issues")
            
            if fix_invalid:
                print(f"✅ Fixed {fixed_count} invalid flags")
            else:
                print(f"💡 Run with fix_invalid=True to automatically remove invalid flags")
        else:
            print("✅ All flags are valid!")
    
    # Raise error if issues found and not fixing
    if issues and not fix_invalid:
        raise ValueError(
            f"Found {len(issues)} invalid QC flags. "
            f"Valid categories are defined in the JSON file under 'valid_qc_flag_categories'. "
            f"Either fix these manually or run check_flag_integrity() with fix_invalid=True"
        )
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "fixed_count": fixed_count if fix_invalid else 0,
        "qc_data": qc_data if fix_invalid else None
    }

def validate_qc_flag(qc_flag: str, level: str, qc_data: Dict) -> bool:
    """Validate that a QC flag is valid for the given level."""
    valid_flags = qc_data.get("valid_qc_flag_categories", {}).get(f"{level}_level", {})
    return qc_flag in valid_flags

def clean_invalid_flags(quality_control_dir: Union[str, Path], backup: bool = True) -> Dict:
    """
    Clean invalid flags from QC data file.
    
    Example:
        result = clean_invalid_flags("/path/to/quality_control", backup=True)
        print(f"Cleaned {result['fixed_count']} invalid flags")
    """
    qc_json_path = get_qc_json_path(quality_control_dir)
    
    if backup and qc_json_path.exists():
        backup_path = qc_json_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(qc_json_path, backup_path)
        print(f"📋 Created backup: {backup_path}")
    
    qc_data = load_qc_data(quality_control_dir)
    result = check_flag_integrity(qc_data, fix_invalid=True, verbose=True)
    
    if result["fixed_count"] > 0:
        # Mixed mode example (batch + direct)
        qc_data = load_qc_data(quality_control_dir)
        batch = []
        
        for image_id, analysis in image_results.items():
            if analysis['confidence'] < 0.8:  # Low confidence -> batch for review
                batch += gen_flag_batch("image", image_id, "BLUR", "auto", f"Low confidence: {analysis['confidence']}")
            else:  # High confidence -> apply directly
                gen_flag_batch("image", image_id, "DRY_WELL", "auto", "High confidence detection", 
                              qc_data=qc_data, apply_directly=True)
        
        # Apply batched flags that need review
        if batch:
            qc_data = add_flags_to_qc_data(qc_data, batch)
        
        save_qc_data(qc_data, quality_control_dir)
        print(f"💾 Saved cleaned QC data")
    
    return result

# =============================================================================
# INDIVIDUAL FLAG OPERATIONS
# =============================================================================

=======
def add_qc_flag(
    quality_control_dir: Union[str, Path],
    level: str,  # "experiment", "video", "image", or "embryo"
    entity_id: str,  # experiment_id, video_id, image_id, or embryo_id
    qc_flag: str,
    author: str,
    notes: str = "",
    parent_ids: Optional[Dict[str, str]] = None,  # For nested entities
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Add a QC flag to an entity at the specified level."""
    qc_data = load_qc_data(quality_control_dir)
    
    # Validate flag
    if not validate_qc_flag(qc_flag, level, qc_data):
        valid_flags = list(qc_data["valid_qc_flag_categories"].get(f"{level}_level", {}).keys())
        raise ValueError(f"Invalid {level}_level flag '{qc_flag}'. Valid flags: {valid_flags}")
    
    # Navigate to the correct entity
    if level == "experiment":
        if entity_id not in qc_data["experiments"]:
            qc_data["experiments"][entity_id] = {
                "flags": [], "authors": [], "notes": [], "videos": {}
            }
        entity = qc_data["experiments"][entity_id]
        
    elif level == "video":
        if not parent_ids or "experiment_id" not in parent_ids:
            raise ValueError("video level flags require parent_ids with experiment_id")
        exp_id = parent_ids["experiment_id"]
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {
                "flags": [], "authors": [], "notes": [], "videos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][entity_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        entity = qc_data["experiments"][exp_id]["videos"][entity_id]
        
    elif level == "image":
        if not parent_ids or "experiment_id" not in parent_ids or "video_id" not in parent_ids:
            raise ValueError("image level flags require parent_ids with experiment_id and video_id")
        exp_id = parent_ids["experiment_id"]
        vid_id = parent_ids["video_id"]
        
        # Ensure structure exists
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {"flags": [], "authors": [], "notes": [], "videos": {}}
        if vid_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"][vid_id]["images"]:
            qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id] = {
                "flags": [], "authors": [], "notes": []
            }
        entity = qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id]
        
    elif level == "embryo":
        if not parent_ids or "experiment_id" not in parent_ids or "video_id" not in parent_ids:
            raise ValueError("embryo level flags require parent_ids with experiment_id and video_id")
        exp_id = parent_ids["experiment_id"]
        vid_id = parent_ids["video_id"]
        
        # Ensure structure exists
        if exp_id not in qc_data["experiments"]:
            qc_data["experiments"][exp_id] = {"flags": [], "authors": [], "notes": [], "videos": {}}
        if vid_id not in qc_data["experiments"][exp_id]["videos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id] = {
                "flags": [], "authors": [], "notes": [], "images": {}, "embryos": {}
            }
        if entity_id not in qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"]:
            qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id] = {
                "flags": [], "authors": [], "notes": []
            }
        entity = qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id]
    
    else:
        raise ValueError(f"Invalid level '{level}'. Must be experiment, video, image, or embryo")
    
    # Add or update flag
    if overwrite:
        entity["flags"] = [qc_flag]
        entity["authors"] = [author]
        entity["notes"] = [notes]
    else:
        if qc_flag not in entity["flags"]:
            entity["flags"].append(qc_flag)
            entity["authors"].append(author)
            entity["notes"].append(notes)
    
    # Save data only if write_directly is True
    if write_directly:
        save_qc_data(qc_data, quality_control_dir)
    return qc_data

def get_qc_flags(
    quality_control_dir: Union[str, Path],
    level: str,
    entity_id: str,
    parent_ids: Optional[Dict[str, str]] = None
) -> Dict[str, List]:
    """Get QC flags for a specific entity."""
    qc_data = load_qc_data(quality_control_dir)
    
    try:
        if level == "experiment":
            entity = qc_data["experiments"][entity_id]
        elif level == "video":
            exp_id = parent_ids["experiment_id"]
            entity = qc_data["experiments"][exp_id]["videos"][entity_id]
        elif level == "image":
            exp_id = parent_ids["experiment_id"]
            vid_id = parent_ids["video_id"]
            entity = qc_data["experiments"][exp_id]["videos"][vid_id]["images"][entity_id]
        elif level == "embryo":
            exp_id = parent_ids["experiment_id"]
            vid_id = parent_ids["video_id"]
            entity = qc_data["experiments"][exp_id]["videos"][vid_id]["embryos"][entity_id]
        else:
            raise ValueError(f"Invalid level: {level}")
            
        return {
            "flags": entity.get("flags", []),
            "authors": entity.get("authors", []),
            "notes": entity.get("notes", [])
        }
    except KeyError:
        return {"flags": [], "authors": [], "notes": []}

def get_qc_summary(quality_control_dir: Union[str, Path]) -> Dict:
    """Generate a summary of QC flags across all levels."""
    qc_data = load_qc_data(quality_control_dir)
    
    summary = {
        "experiment_level": {},
        "video_level": {},
        "image_level": {},
        "embryo_level": {}
    }
    
    for exp_id, exp_data in qc_data.get("experiments", {}).items():
        # Count experiment flags
        for flag in exp_data.get("flags", []):
            summary["experiment_level"][flag] = summary["experiment_level"].get(flag, 0) + 1
        
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            # Count video flags
            for flag in vid_data.get("flags", []):
                summary["video_level"][flag] = summary["video_level"].get(flag, 0) + 1
            
            # Count image flags
            for img_id, img_data in vid_data.get("images", {}).items():
                for flag in img_data.get("flags", []):
                    summary["image_level"][flag] = summary["image_level"].get(flag, 0) + 1
            
            # Count embryo flags
            for emb_id, emb_data in vid_data.get("embryos", {}).items():
                for flag in emb_data.get("flags", []):
                    summary["embryo_level"][flag] = summary["embryo_level"].get(flag, 0) + 1
    
    return summary

def parse_image_id(image_id: str) -> Dict[str, str]:
    """Parse image ID to extract experiment_id and video_id."""
    # Expected format: experiment_id_well_id_timepoint (e.g., "20241215_A01_t001")
    parts = image_id.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id format: {image_id}")
    
    experiment_id = parts[0]
    well_id = parts[1]
    video_id = f"{experiment_id}_{well_id}"
    
    return {
        "experiment_id": experiment_id,
        "video_id": video_id,
        "well_id": well_id
    }

def parse_embryo_id(embryo_id: str) -> Dict[str, str]:
    """Parse embryo ID to extract parent IDs."""
    # Expected format: experiment_id_well_id_timepoint_embryo_num (e.g., "20241215_A01_t001_e01")
    parts = embryo_id.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid embryo_id format: {embryo_id}")
    
    experiment_id = parts[0]
    well_id = parts[1]
    video_id = f"{experiment_id}_{well_id}"
    image_id = f"{experiment_id}_{well_id}_{parts[2]}"
    
    return {
        "experiment_id": experiment_id,
        "video_id": video_id,
        "image_id": image_id,
        "well_id": well_id
    }

# Convenience functions for common QC operations
def flag_image(
    quality_control_dir: Union[str, Path],
    image_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an image."""
    parent_ids = parse_image_id(image_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="image",
        entity_id=image_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_video(
    quality_control_dir: Union[str, Path],
    video_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag a video."""
    # Extract experiment_id from video_id (format: experiment_id_well_id)
    parts = video_id.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid video_id format: {video_id}")
    
    experiment_id = parts[0]
    parent_ids = {"experiment_id": experiment_id}
    
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="video",
        entity_id=video_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_experiment(
    quality_control_dir: Union[str, Path],
    experiment_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an experiment."""
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="experiment",
        entity_id=experiment_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        overwrite=overwrite,
        write_directly=write_directly
    )

def flag_embryo(
    quality_control_dir: Union[str, Path],
    embryo_id: str,
    qc_flag: str,
    author: str,
    notes: str = "",
    overwrite: bool = False,
    write_directly: bool = False
) -> Dict:
    """Convenience function to flag an embryo."""
    parent_ids = parse_embryo_id(embryo_id)
    return add_qc_flag(
        quality_control_dir=quality_control_dir,
        level="embryo",
        entity_id=embryo_id,
        qc_flag=qc_flag,
        author=author,
        notes=notes,
        parent_ids=parent_ids,
        overwrite=overwrite,
        write_directly=write_directly
    )

def save_qc_batch(qc_data: Dict, quality_control_dir: Union[str, Path]) -> None:
    """
    Save QC data after batch operations.
    
    Use this function when you've been adding flags with write_directly=False
    and want to save all changes at once for better performance.
    
    Args:
        qc_data: The QC data dictionary to save
        quality_control_dir: Directory where the QC JSON file will be stored
    """
    save_qc_data(qc_data, quality_control_dir)
    print(f"💾 Saved QC data to {get_qc_json_path(quality_control_dir)}")

# Legacy compatibility functions (for backward compatibility with CSV-based system)
def flag_qc(
    quality_control_dir: Union[str, Path],
    image_ids: List[str],
    qc_flag: str,
    annotator: str = 'manual',
    notes: str = '',
    overwrite: bool = False
) -> None:
    """Legacy compatibility function for flagging images."""
    for image_id in image_ids:
        flag_image(
            quality_control_dir=quality_control_dir,
            image_id=image_id,
            qc_flag=qc_flag,
            author=annotator,
            notes=notes,
            overwrite=overwrite
        )

def check_existing_qc(
    quality_control_dir: Union[str, Path],
    image_ids: List[str]
) -> Dict[str, List[str]]:
    """Legacy compatibility function to check existing QC flags for images."""
    result = {}
    for image_id in image_ids:
        try:
            parent_ids = parse_image_id(image_id)
            qc_info = get_qc_flags(quality_control_dir, "image", image_id, parent_ids)
            result[image_id] = qc_info["flags"]
        except (ValueError, KeyError):
            result[image_id] = []
<<<<<<< HEAD
    return result
