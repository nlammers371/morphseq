# AnnotationBatch Design: Inheritance-Based Implementation

## **Overview** 🎯

The AnnotationBatch provides a **temporary workspace** for safely manipulating embryo annotations before applying them to the persistent EmbryoMetadata store. It uses inheritance to reuse all validation logic while maintaining complete isolation from the main data.

**Core Philosophy:**
- ✅ **Isolation**: Batch never mutates metadata directly
- ✅ **Validation**: Same business rules as direct operations  
- ✅ **Simplicity**: Minimal code duplication through inheritance
- ✅ **Safety**: Explicit apply step with conflict policies

---

## **Architecture: Inheritance + Constructor Override** 🏗️

### **Class Hierarchy**
```python
class EmbryoMetadata:
    """Persistent annotation store with file I/O"""
    def __init__(self, sam2_path, annotations_path=None, validate=True):
        # File loading, data initialization, validation setup
    
    def add_phenotype(self, snip_id, phenotype, author, confidence=None, overwrite=False):
        # Core validation + storage logic
    
    def _canonicalize_phenotype(self, phenotype):
        # Synonym resolution
    
    def _validate_dead_exclusivity(self, snip_id, phenotype, overwrite):
        # DEAD coexistence rules
    
    # ... all validation methods

class AnnotationBatch(EmbryoMetadata):
    """Temporary workspace inheriting all validation methods"""
    def __init__(self, data_structure, author, validate=True):
        # Skip super().__init__() - manual setup instead
    
    def add_phenotype(self, embryo_id, phenotype, frames="all", **kwargs):
        # Frame resolution + inherited validation
```

### **Why Inheritance Over Composition**

**Benefits:**
- ✅ **Zero code duplication**: Inherits all validation, canonicalization, business logic
- ✅ **Consistent behavior**: Batch validates exactly like direct operations
- ✅ **Automatic updates**: Benefits from any improvements to parent validation
- ✅ **Simple API**: Same method signatures work for both batch and direct usage

**vs Composition Alternative:**
```python
# Composition would require forwarding every method
class AnnotationBatch:
    def add_phenotype(self, *args, **kwargs):
        return self.engine.add_phenotype(*args, **kwargs)
    def add_genotype(self, *args, **kwargs):
        return self.engine.add_genotype(*args, **kwargs)
    # ... repeat for 10+ methods
```

---

## **Constructor Implementation** ⚙️

### **Manual Attribute Setup**
```python
class AnnotationBatch(EmbryoMetadata):
    def __init__(self, data_structure, author, validate=True):
        """
        Initialize batch workspace without calling parent constructor
        
        Args:
            data_structure: Skeleton from metadata.initialize_batch()
            author: Default author for all operations
            validate: Enable/disable validation (same as parent)
        """
        # Essential parent attributes (what super().__init__ would set)
        self.data = data_structure              # Skeleton structure from metadata
        self.validate = validate                # Validation toggle
        self.validator = AnnotationValidator()  # Synonym resolution + validation
        self.frame_parser = FrameAwareRangeParser()  # Frame range parsing
        
        # Batch-specific attributes
        self.author = author                    # Default author for operations
        
        # Robust parsing utilities (inherited from parent but ensure availability)
        self.parsing_utils = {
            'extract_frame_number': extract_frame_number,
            'validate_id_format': validate_id_format,
            'get_entity_type': get_entity_type,
            'extract_embryo_id': extract_embryo_id,
            'build_snip_id': build_snip_id,
            'parse_entity_id': parse_entity_id
        }
        
        # Safety check (optional insurance)
        self._verify_contract()
    
    def _verify_contract(self):
        """Ensure required attributes exist for parent methods"""
        required = ["data", "validate", "validator", "frame_parser"]
        missing = [attr for attr in required if not hasattr(self, attr)]
        if missing:
            raise AttributeError(f"AnnotationBatch missing required attributes: {missing}")
```

### **Why Skip super().__init__()**

**Parent constructor does heavy I/O:**
- Loads SAM2 files
- Reads existing annotation files  
- Builds indexes and caches
- Sets up file paths

**Batch needs clean workspace:**
- No file dependencies
- Empty annotation structure
- Minimal memory footprint
- Fast initialization

---

## **Unified Flexible API Strategy** 📊

### **Inherits Improved Parent API with Robust Parsing**
```python
# Import robust parsing utilities at module level
from scripts.utils.parsing_utils import (
    extract_frame_number, validate_id_format, get_entity_type,
    extract_embryo_id, build_snip_id, parse_entity_id
)

class AnnotationBatch(EmbryoMetadata):
    def add_phenotype(self, embryo_id, phenotype, target="all", author=None, overwrite_dead=False, **kwargs):
        """Add phenotype using inherited flexible API with robust parsing and DEAD safety"""
        
        # Use parent's unified API with batch author and DEAD safety
        return super().add_phenotype(
            embryo_id, phenotype, 
            author=author or self.author,
            target=target,
            overwrite_dead=overwrite_dead,
            **kwargs
        )
        # Inherits all target types: "all", "30:50", [30, 31], ["snip_id1"]
        # Inherits smart filtering, strict modes, return values
        # Inherits robust parsing utilities for ID validation and frame extraction
```

### **Flexible Target Examples**
```python
# All equivalent ways to specify frames - inherits from parent
batch.add_phenotype("embryo_e01", "NORMAL", target="all")
batch.add_phenotype("embryo_e01", "EDEMA", target="30:50")        # Flexible zero-padding
batch.add_phenotype("embryo_e01", "EDEMA", target="030:050")      # Also works
batch.add_phenotype("embryo_e01", "DEAD", target="200:")          # Open-ended range
batch.add_phenotype("embryo_e01", "BLUR", target=[45, 67, 89])    # Frame numbers
batch.add_phenotype("embryo_e01", "CORRUPT", target=["embryo_e01_s0123"]) # Snip IDs

# DEAD safety examples - new overwrite_dead parameter
batch.add_phenotype("embryo_e01", "EDEMA", target="all")                    # Safe: skips DEAD frames silently
batch.add_phenotype("embryo_e01", "DEAD", target="180:", overwrite_dead=True) # Correcting death frame

# Smart handling of missing frames (normal in SAM2 data)
result = batch.add_phenotype("embryo_e01", "EDEMA", target="30:70")
print(f"Applied to {result['count']} frames, skipped missing")
print(f"Applied snips: {result['applied_to'][:3]}...")

# Strict validation when needed
try:
    batch.add_phenotype("embryo_e01", "TEST", target=["nonexistent_snip"], strict="all")
except ValueError as e:
    print(e)  # Clear error with context
```

### **Benefits of Inherited API with Robust Parsing**
- ✅ **Zero code duplication**: Inherits all parent functionality automatically
- ✅ **Consistent behavior**: Same validation rules as direct metadata operations
- ✅ **Smart filtering**: Missing frames handled gracefully for ranges
- ✅ **Flexible input**: All target types supported (strings, slices, lists)
- ✅ **Clear feedback**: Returns list of actually applied snip_ids
- ✅ **Future-proof**: Benefits from parent API improvements automatically
- ✅ **Robust parsing**: Uses proven parsing_utils for ID validation and frame extraction
- ✅ **Consistent ID handling**: Leverages established patterns from parsing_utils
- ✅ **Error prevention**: Reduces risk of ID parsing failures and false positives

---

## **Validation Inheritance** ✅

### **Methods Inherited from Parent**
```python
# Automatic inheritance - no code duplication needed
batch._canonicalize_phenotype("edema")           # → "EDEMA" (synonym resolution)
batch._validate_dead_exclusivity(snip_id, "DEAD", False)  # → DEAD coexistence rules
batch._validate_dead_permanence(embryo_id, 200, "NORMAL") # → temporal DEAD logic  
batch._append_phenotype_record(snip_id, "EDEMA", "user1", 0.9)  # → data storage

# Data access helpers - safe navigation with centralized error handling
batch._get_embryo_data(embryo_id)                # → Safe embryo data access with consistent errors
batch._get_snip_data(snip_id)                    # → Safe snip data access with ID extraction
```

### **Business Rules Enforced**
- **DEAD Exclusivity**: DEAD cannot coexist with other phenotypes at same frame
- **DEAD Permanence**: Once dead at frame N, all frames ≥ N must be DEAD
- **DEAD Safety**: Non-DEAD phenotypes silently skip already-DEAD frames unless `overwrite_dead=True`
- **Single Genotype**: One genotype per embryo unless overwrite=True
- **Controlled Vocabularies**: Phenotypes/genes validated against registry
- **Synonym Resolution**: "edema" → "EDEMA", "ce_defect" → "CONVERGENCE_EXTENSION"

### **Validation Toggle**
```python
# Strict validation (production)
batch = metadata.initialize_batch(validate=True)
batch.add_phenotype("embryo_e01", "invalid_phenotype")  # → ValueError

# Permissive mode (development)  
batch = metadata.initialize_batch(validate=False)
batch.add_phenotype("embryo_e01", "experimental_phenotype")  # → Allowed
```

---

## **DEAD Phenotype Safety Workflow** 🛡️

### **Enhanced Safety with `overwrite_dead` Parameter**

The DEAD phenotype workflow implements a safer default behavior when annotating ranges that might overlap with an embryo's DEAD status. The principle of permanence for the DEAD phenotype protects against accidental overwrites by non-DEAD annotations, making the system more robust and intuitive for biologists.

The solution makes the system silently and safely ignore attempts to annotate already DEAD frames, while providing an explicit flag for rare cases where an override is intended.

### **Implementation Details**

**New Parameter**: `overwrite_dead: bool = False` added to the `add_phenotype` method.

- **`overwrite_dead=False` (Default - Safe Mode)**: When a user tries to apply a phenotype like "EDEMA" to a range of frames (e.g., "all"), the method will process each frame. If it encounters a frame that is already marked DEAD, it will silently skip that frame and continue to the next. No error will be raised, and the operation will be considered a partial success. The method's return value will accurately reflect that only the pre-death frames were modified.

- **`overwrite_dead=True` (Explicit Override)**: Setting this flag to True signals the user's intent to modify DEAD frames. However, it does not disable other business rules. Attempting to add "EDEMA" to a DEAD frame would still fail validation because of the "DEAD Exclusivity" rule (DEAD cannot coexist with other phenotypes). This flag's primary purpose is to be used in conjunction with re-applying the DEAD phenotype itself to correct the frame of death.

### **Core Logic Implementation**

```python
# Inside the add_phenotype method loop that iterates over resolved snip_ids
for snip_id in snip_ids:
    # Use helper for clean data access
    snip_data = self._get_snip_data(snip_id)
    existing_phenotypes = {p['value'] for p in snip_data.get("phenotypes", [])}

    # ⏬ NEW SAFETY CHECK ⏬
    # If the snip is already DEAD and we are not trying to overwrite it...
    if "DEAD" in existing_phenotypes and not overwrite_dead:
        # Silently skip applying any non-DEAD phenotypes to this frame.
        if "DEAD" not in phenotypes: # Ensure we're not just re-applying DEAD
            continue # Move to the next snip_id

    # ... proceed with existing validation and phenotype application ...
    # self._validate_dead_exclusivity(...)
    # self._validate_dead_permanence(...)
    # self._append_phenotype_record(...)
```

### **Usage Examples**

**Scenario 1: Safe Default Behavior (Silent Skip)**
Let's assume embryo_e01 has been marked DEAD starting from frame 200.

```python
# The default, overwrite_dead=False, is used
result = metadata.add_phenotype(
    "embryo_e01",
    "EDEMA",
    target="all",
    author="researcher1"
)

# Expected Outcome:
# - The operation succeeds without any errors.
# - Frames before 200 are successfully annotated with "EDEMA".
# - Frames 200 and later are silently skipped, preserving their "DEAD" status.
# - The return dictionary reflects this partial application:
print(f"Applied EDEMA to {result['count']} frames.")
# Output: Applied EDEMA to 199 frames. (Assuming frames 1-199 exist)
print(f"Frame range affected: {result['frame_range']}")
# Output: Frame range affected: 0001:0199
```

**Scenario 2: Correcting the Frame of Death**
The researcher realizes the embryo actually died at frame 180.

```python
# Using the explicit overwrite flag to change the history
result = metadata.add_phenotype(
    "embryo_e01",
    "DEAD",
    target="180:", # Apply DEAD from the new, earlier frame
    author="researcher1",
    overwrite_dead=True # Explicitly allow overwriting existing DEAD markers
)

# Expected Outcome:
# - The operation succeeds.
# - The DEAD annotation is applied from frame 180 onwards, including backfilling any frames
#   that might not have been marked DEAD between 180 and 200.
```

**Benefits:**
- ✅ **Safety First**: Prevents accidental overwriting of critical DEAD annotations
- ✅ **User-Friendly**: Silent skipping allows bulk operations without manual frame exclusion
- ✅ **Explicit Override**: Clear intent when corrections are needed
- ✅ **Consistent Returns**: Accurate reporting of what was actually applied
- ✅ **Business Rule Preservation**: Maintains DEAD exclusivity even with override flag

---

## **Internal Data Access Helpers** 🏗️

### **Foundational Design Pattern for Maintainability**

Adopting private helper methods like `_get_embryo_data` and `_get_snip_data` is a foundational software design practice that significantly improves the codebase's long-term health and quality. It moves beyond just making the code work; it makes it clean, maintainable, and robust.

### **Core Helper Methods**

```python
class EmbryoMetadata:
    def _get_embryo_data(self, embryo_id: str) -> Dict:
        """Single point of embryo data access with consistent error handling"""
        if embryo_id not in self.data["embryos"]:
            raise KeyError(f"Embryo not found: {embryo_id}")
        return self.data["embryos"][embryo_id]
    
    def _get_snip_data(self, snip_id: str) -> Dict:
        """Single point of snip data access with consistent error handling"""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        embryo_data = self._get_embryo_data(embryo_id)
        if snip_id not in embryo_data["snips"]:
            raise KeyError(f"Snip not found: {snip_id}")
        return embryo_data["snips"][snip_id]
```

### **Primary Benefits**

**1. Improved Maintainability** (Most Critical)
The internal data structure is an implementation detail that other parts of the class shouldn't need to know about. By hiding the `self.data["embryos"][embryo_id]["snips"][snip_id]` path inside helpers, you create a single point of control. If you ever decide to optimize the data structure (e.g., change to `self.data["embryo_annotations"]`), you only need to update the helper methods, and the rest of the code remains unchanged. This saves significant refactoring effort and reduces the risk of introducing bugs.

**2. Increased Readability**
Code becomes easier to understand because it focuses on what it's doing, not how it's accessing data.

**Before:**
```python
def some_method(self, snip_id):
    # Logic is cluttered with data structure navigation
    embryo_id = extract_embryo_id(snip_id)
    if embryo_id in self.data["embryos"] and snip_id in self.data["embryos"][embryo_id]["snips"]:
        phenotypes = self.data["embryos"][embryo_id]["snips"][snip_id].get("phenotypes", [])
        # ... do something with phenotypes
```

**After:**
```python
def some_method(self, snip_id):
    # Logic is clean and expresses intent
    snip_data = self._get_snip_data(snip_id) # Handles lookup and errors
    phenotypes = snip_data.get("phenotypes", [])
    # ... do something with phenotypes
```

**3. Centralized Error Handling**
The helpers provide a single, consistent place to handle errors like a missing embryo_id or snip_id. This prevents logic from being duplicated across multiple methods and ensures that users always get the same, high-quality KeyError message, making the API more predictable and easier to debug.

**4. Abstraction of Complexity**
The `_get_snip_data` helper is particularly valuable because it abstracts away the two-step process of first extracting the embryo_id from the snip_id and then drilling down into the data structure. This simplifies the logic in every method that operates at the snip level.

### **Perfect Integration with Inheritance Design**

These helper methods work seamlessly with the inheritance-based AnnotationBatch approach:
- ✅ **Automatic availability**: Helper methods in base class are immediately available to AnnotationBatch
- ✅ **No code duplication**: Both batch and direct operations use the same data access patterns  
- ✅ **Consistent error handling**: Same error messages across all operations
- ✅ **Future-proof inheritance**: Changes to data access automatically benefit both classes

---

## **Apply Mechanism: Simple Data Merge** 🔄

### **No Operation Logging Required**
Based on our analysis, we're implementing a **simplified approach without _ops logging**:

**Why no _ops:**
- ✅ **Data is source of truth**: Final state in batch.data is what matters
- ✅ **Validation already happened**: Batch validated changes when added
- ✅ **Simpler apply**: Direct data merge instead of operation replay
- ✅ **Less complexity**: Fewer moving parts, easier to understand

### **Apply Implementation**
```python
class EmbryoMetadata:
    def apply_batch(self, batch, on_conflict="error", dry_run=False):
        """
        Apply batch changes to metadata with conflict resolution
        
        Args:
            batch: AnnotationBatch instance
            on_conflict: "error" | "skip" | "overwrite" | "merge"
            dry_run: If True, validate without applying changes
        
        Returns:
            Report with applied count, conflicts, errors
        """
        # 1. Validate batch data consistency
        validation_report = self._validate_batch_data(batch.data, on_conflict)
        
        if validation_report["errors"] and on_conflict == "error":
            raise ValueError(f"Batch validation failed: {validation_report['errors']}")
        
        # 2. Apply changes if not dry run
        if not dry_run:
            applied_count = self._merge_batch_data(batch.data, on_conflict)
            validation_report["applied_count"] = applied_count
        
        return validation_report
    
    def _merge_batch_data(self, batch_data, on_conflict):
        """Merge batch annotations into metadata"""
        applied_count = 0
        
        for embryo_id, embryo_data in batch_data["embryos"].items():
            # Merge genotype
            if embryo_data.get("genotype"):
                if self._should_apply_genotype(embryo_id, embryo_data["genotype"], on_conflict):
                    self.data["embryos"][embryo_id]["genotype"] = embryo_data["genotype"]
                    applied_count += 1
            
            # Merge phenotypes (append rather than replace)
            for snip_id, snip_data in embryo_data.get("snips", {}).items():
                batch_phenotypes = snip_data.get("phenotypes", [])
                if batch_phenotypes:
                    if self._should_apply_phenotypes(snip_id, batch_phenotypes, on_conflict):
                        existing_phenotypes = self.data["embryos"][embryo_id]["snips"][snip_id].get("phenotypes", [])
                        
                        if on_conflict == "overwrite":
                            # Replace all phenotypes
                            self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = batch_phenotypes
                        elif on_conflict == "merge":
                            # Intelligent merge - update existing, add new
                            merged_phenotypes = existing_phenotypes.copy()
                            for new_pheno in batch_phenotypes:
                                # Find existing phenotype with same value
                                existing_idx = None
                                for i, existing_pheno in enumerate(merged_phenotypes):
                                    if existing_pheno["value"] == new_pheno["value"]:
                                        existing_idx = i
                                        break
                                
                                if existing_idx is not None:
                                    # Update existing phenotype with new metadata
                                    merged_phenotypes[existing_idx] = new_pheno
                                else:
                                    # Add new phenotype
                                    merged_phenotypes.append(new_pheno)
                            
                            self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = merged_phenotypes
                        else:  # append mode (default for skip/error that passed validation)
                            # Simple append, avoiding exact duplicates
                            merged_phenotypes = existing_phenotypes.copy()
                            for new_pheno in batch_phenotypes:
                                if not any(p["value"] == new_pheno["value"] for p in merged_phenotypes):
                                    merged_phenotypes.append(new_pheno)
                            
                            self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = merged_phenotypes
                        
                        applied_count += 1
            
            # Merge treatments, flags, etc.
            # ...
        
        return applied_count
```

### **Conflict Resolution**
```python
def _should_apply_genotype(self, embryo_id, new_genotype, on_conflict):
    """Determine if genotype should be applied based on conflict policy"""
    existing = self.data["embryos"][embryo_id].get("genotype")
    
    if not existing:
        return True  # No conflict
    
    if on_conflict == "error":
        raise ValueError(f"Genotype conflict for {embryo_id}: {existing['value']} vs {new_genotype['value']}")
    elif on_conflict == "skip":
        return False  # Keep existing
    elif on_conflict == "overwrite":
        return True   # Replace with new

def _should_apply_phenotypes(self, snip_id, new_phenotypes, on_conflict):
    """Determine if phenotypes should be applied based on conflict policy"""
    embryo_id = self.get_embryo_id_from_snip(snip_id)
    existing = self.data["embryos"][embryo_id]["snips"][snip_id].get("phenotypes", [])
    
    if not existing:
        return True  # No conflict
    
    # Check for actual conflicts (same phenotype value)
    existing_values = {p["value"] for p in existing}
    new_values = {p["value"] for p in new_phenotypes}
    conflicts = existing_values & new_values
    
    if not conflicts:
        return True  # No actual conflicts, can merge
    
    if on_conflict == "error":
        raise ValueError(f"Phenotype conflicts for {snip_id}: {list(conflicts)}")
    elif on_conflict == "skip":
        return False  # Skip all new phenotypes if any conflict
    elif on_conflict in ["overwrite", "merge"]:
        return True   # Apply new phenotypes (merge strategy handled by caller)
```

---

## **Usage Patterns** 🚀

### **Basic Workflow**
```python
# 1. Initialize batch from metadata skeleton
metadata = EmbryoMetadata("sam2_annotations.json")
batch = metadata.initialize_batch(mode="skeleton", author="researcher1")

# 2. Add annotations with flexible target specifications
batch.add_genotype("embryo_e01", "tmem67", zygosity="homozygous")
batch.add_phenotype("embryo_e01", "EDEMA", target="200:400")     # Natural frame format
batch.add_phenotype("embryo_e01", "DEAD", target="350:") # Open-ended range

# 3. Preview changes
print(batch.preview())
# Output: Shows embryos, genotypes, phenotype ranges, affected snip counts

# 4. Validate without applying
report = metadata.apply_batch(batch, dry_run=True)
print(f"Validation: {report['errors']} errors, {report['conflicts']} conflicts")

# 5. Apply if validation passes
if not report["errors"]:
    final_report = metadata.apply_batch(batch, on_conflict="error")
    print(f"Applied {final_report['applied_count']} changes")
```

### **Team Collaboration Workflow**
```python
# Researcher 1: Genotype annotation
batch1 = metadata.initialize_batch(mode="skeleton", author="researcher1") 
for embryo_id in selected_embryos:
    batch1.add_genotype(embryo_id, determine_genotype(embryo_id))

# Researcher 2: Phenotype annotation with DEAD safety
batch2 = metadata.initialize_batch(mode="skeleton", author="researcher2")
for embryo_id in selected_embryos:
    # Safe default - automatically skips any DEAD frames
    batch2.add_phenotype(embryo_id, score_phenotype(embryo_id), target="all")

# Apply in sequence with conflict handling
metadata.apply_batch(batch1, on_conflict="error")     # Genotypes first
metadata.apply_batch(batch2, on_conflict="skip")      # Skip genotype conflicts
```

### **Iterative Refinement Workflow**
```python
# Start with skeleton for clean annotation pass
batch = metadata.initialize_batch(mode="skeleton", author="curator")

# Add initial annotations with flexible targeting and DEAD safety
batch.add_phenotype("embryo_e01", "NORMAL", target="100:200")     # Natural format
batch.add_phenotype("embryo_e01", "EDEMA", target="200:300")      # Inclusive range

# Preview and refine
print(batch.preview())

# Add death annotation (triggers backfilling validation) 
result = batch.add_phenotype("embryo_e01", "DEAD", target="250:", overwrite_dead=True)
print(f"Death applied to {result['count']} frames: {result['frame_range']}")
print(f"Sample applied snips: {result['applied_to'][:3]}...{result['applied_to'][-3:]}")

# Later correction - move death earlier (safe with overwrite_dead=True)
correction = batch.add_phenotype("embryo_e01", "DEAD", target="240:", overwrite_dead=True)
print(f"Death corrected to frame 240, backfilled {correction['count']} frames")

# Apply with validation
metadata.apply_batch(batch, on_conflict="error")
```

---

## **Preview and Debugging** 🔍

### **Preview Implementation**
```python
def preview(self, limit=10):
    """Generate human-readable summary with statistics"""
    lines = [f"AnnotationBatch (Author: {self.author})", ""]
    
    embryo_count = 0
    for embryo_id, embryo_data in self.data["embryos"].items():
        if embryo_count >= limit:
            lines.append(f"... and {len(self.data['embryos']) - limit} more embryos")
            break
        
        # Collect statistics for this embryo
        phenotype_counts = self._count_phenotypes_by_type(embryo_data)
        total_phenotype_snips = sum(phenotype_counts.values())
        
        # Compact summary line with key statistics
        stats_parts = []
        if embryo_data.get("genotype"):
            g = embryo_data["genotype"]
            stats_parts.append(f"🧬 {g['value']}")
        
        if total_phenotype_snips > 0:
            # Show top 3 phenotypes with counts
            top_phenotypes = sorted(phenotype_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            pheno_summary = ", ".join([f"{pheno}:{count}" for pheno, count in top_phenotypes])
            stats_parts.append(f"🔬 {total_phenotype_snips} phenotypes ({pheno_summary})")
        
        treatment_count = len(embryo_data.get("treatments", []))
        if treatment_count > 0:
            stats_parts.append(f"💊 {treatment_count} treatments")
        
        # Single compact line per embryo
        if stats_parts:
            lines.append(f"📋 {embryo_id}: {' | '.join(stats_parts)}")
        else:
            lines.append(f"📋 {embryo_id}: (no annotations)")
        
        embryo_count += 1
    
    return "\n".join(lines)

def _count_phenotypes_by_type(self, embryo_data):
    """Count phenotypes by type for summary statistics"""
    from collections import Counter
    phenotype_counts = Counter()
    
    for snip_data in embryo_data.get("snips", {}).values():
        for phenotype in snip_data.get("phenotypes", []):
            phenotype_counts[phenotype["value"]] += 1
    
    return phenotype_counts

def _compress_snip_ranges(self, snip_ids):
    """Convert snip list back to readable ranges using robust frame extraction"""
    frame_nums = []
    for snip_id in snip_ids:
        try:
            # Use robust frame extraction from parsing_utils
            frame_num = extract_frame_number(snip_id)
            if frame_num is not None:
                frame_nums.append(frame_num)
        except:
            continue  # Skip invalid snip IDs
    
    frame_nums.sort()
    
    if len(frame_nums) <= 3:
        return f"{len(frame_nums)} frames"
    else:
        return f"{frame_nums[0]:04d}:{frame_nums[-1]+1:04d} ({len(frame_nums)} frames)"
```

### **Example Preview Output**
```
AnnotationBatch (Author: researcher1)

📋 embryo_e01: 🧬 tmem67 | 🔬 456 phenotypes (NORMAL:200, EDEMA:150, DEAD:106) | 💊 2 treatments
📋 embryo_e02: 🧬 WT | 🔬 400 phenotypes (NORMAL:400) | 💊 1 treatments  
📋 embryo_e03: 🧬 tmem67 | 🔬 89 phenotypes (EDEMA:45, BLUR:44) 
📋 embryo_e04: (no annotations)
```

**Benefits of compact format:**
- ✅ **Quick scanning**: Key info visible at-a-glance
- ✅ **Statistics focus**: Counts more useful than frame ranges for overview
- ✅ **Less clutter**: No need to scroll through detailed phenotype breakdowns
- ✅ **Pattern recognition**: Easy to spot annotation patterns across embryos

---

## **Error Handling and Robustness** 🛡️

### **Contract Verification**
```python
def _verify_contract(self):
    """Ensure batch has required attributes for parent methods"""
    required_attrs = {
        "data": "Annotation data structure",
        "validate": "Validation toggle",  
        "validator": "Vocabulary validation",
        "frame_parser": "Frame range parsing"
    }
    
    missing = []
    for attr, description in required_attrs.items():
        if not hasattr(self, attr):
            missing.append(f"{attr} ({description})")
    
    if missing:
        raise AttributeError(
            f"AnnotationBatch missing required attributes:\n" + 
            "\n".join(f"  - {item}" for item in missing)
        )
```

### **Frame Resolution with Robust Parsing**
```python
def _resolve_frames(self, embryo_id, frames):
    """Robust frame resolution using data helpers and parsing_utils with clear error messages"""
    # Use helper method for consistent error handling
    embryo_data = self._get_embryo_data(embryo_id)
    
    available_snips = list(embryo_data["snips"].keys())
    if not available_snips:
        raise ValueError(f"No snips available for embryo: {embryo_id}")
    
    try:
        if frames == "all":
            return available_snips
        elif isinstance(frames, list):
            # Validate each snip ID using robust parsing
            valid_snips = []
            for snip_candidate in frames:
                if isinstance(snip_candidate, str):
                    # Validate snip ID format using parsing_utils
                    if validate_id_format(snip_candidate, "snip") and snip_candidate in available_snips:
                        valid_snips.append(snip_candidate)
                elif isinstance(snip_candidate, int):
                    # Convert frame number to snip ID using robust building
                    snip_id = build_snip_id(embryo_id, snip_candidate)
                    if snip_id in available_snips:
                        valid_snips.append(snip_id)
            
            if not valid_snips:
                raise ValueError(f"None of the specified snips exist for {embryo_id}")
            return valid_snips
        else:
            return self.frame_parser.parse_snip_range(frames, available_snips)
    
    except Exception as e:
        raise ValueError(
            f"Frame resolution failed for {embryo_id} with frames='{frames}': {e}\n"
            f"Available snips: {len(available_snips)} ({available_snips[0]} to {available_snips[-1]})"
        )
```

### **Validation Error Context**
```python
def add_phenotype(self, embryo_id, phenotype, frames="all", overwrite_dead=False, **kwargs):
    """Enhanced error messages with context and DEAD safety"""
    try:
        snip_ids = self._resolve_frames(embryo_id, frames)
        phenotype = self._canonicalize_phenotype(phenotype)
        
        for snip_id in snip_ids:
            # Use helper method for clean data access
            snip_data = self._get_snip_data(snip_id)
            existing_phenotypes = {p['value'] for p in snip_data.get("phenotypes", [])}
            
            # DEAD safety check using helper method
            if "DEAD" in existing_phenotypes and not overwrite_dead:
                if "DEAD" not in [phenotype]:  # Skip if not re-applying DEAD
                    continue
            
            super().add_phenotype(snip_id, phenotype, overwrite_dead=overwrite_dead, **kwargs)
    
    except ValueError as e:
        # Add batch context to validation errors
        raise ValueError(
            f"Batch annotation failed for {embryo_id}:\n"
            f"  Phenotype: {phenotype}\n" 
            f"  Frames: {frames}\n"
            f"  Error: {e}"
        )
```

---

## **Design Decisions Summary** 📋

### **Key Choices Made**

**1. Inheritance over Composition**
- **Rationale**: Maximum code reuse with minimal complexity
- **Trade-off**: Small risk of brittleness vs large reduction in duplicate code
- **Mitigation**: Contract verification catches missing attributes early

**2. Skip super().__init__()**  
- **Rationale**: Avoid file I/O dependencies for temporary workspace
- **Trade-off**: Manual attribute setup vs automatic parent initialization
- **Mitigation**: Explicit required attribute list with verification

**3. Immediate Frame Resolution**
- **Rationale**: Clear preview, early error detection, explicit operations
- **Trade-off**: Slightly more memory vs deferred parsing complexity
- **Benefit**: Skeleton structure makes resolution fast and reliable

**4. No Operation Logging**
- **Rationale**: Data merge is simpler than operation replay
- **Trade-off**: Less audit detail vs reduced complexity
- **Benefit**: Fewer moving parts, easier to understand and debug

**5. Granular Conflict Policies**
- **Rationale**: Cover common real-world collaboration scenarios
- **Options**: 
  - `error` (strict): Fail on any conflict
  - `skip` (conservative): Keep existing data, skip conflicts  
  - `overwrite` (permissive): Replace existing data completely
  - `merge` (collaborative): Intelligently combine phenotypes
- **Benefit**: Clear semantics, supports team workflows

**6. Robust Parsing Integration**
- **Rationale**: Leverage proven parsing_utils instead of custom ID parsing
- **Trade-off**: Slight dependency vs robust, tested parsing logic
- **Benefit**: Consistent ID handling, reduced parsing errors, future-proof design

### **Robustness Measures**
- ✅ **Contract verification** ensures required attributes exist
- ✅ **Frame resolution validation** with clear error messages
- ✅ **Inherited business rule enforcement** (DEAD logic, vocabularies)
- ✅ **Conflict detection and resolution** during apply
- ✅ **Dry-run validation** before committing changes
- ✅ **Robust ID parsing** using proven parsing_utils functions
- ✅ **Consistent frame extraction** across all operations
- ✅ **Inheritance contract testing** catches parent class evolution issues

### **Inheritance Contract Smoke Test**

**Simple insurance policy to catch parent class evolution:**

```python
def test_annotation_batch_inheritance_contract():
    """Smoke test to ensure AnnotationBatch inheritance works correctly"""
    
    # Create minimal test data structure
    test_data = {
        "embryos": {
            "test_embryo_e01": {
                "genotype": None,
                "treatments": [],
                "snips": {
                    "test_embryo_e01_s0100": {"phenotypes": []}
                }
            }
        }
    }
    
    # Test 1: Basic instantiation
    try:
        batch = AnnotationBatch(test_data, author="test_user")
        assert hasattr(batch, 'data'), "Missing data attribute"
        assert hasattr(batch, 'author'), "Missing author attribute"
        assert hasattr(batch, 'validate'), "Missing validate attribute"
        print("✅ Basic instantiation works")
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False
    
    # Test 2: Inherited method access
    try:
        # Should inherit validation methods from parent
        result = batch._canonicalize_phenotype("NORMAL")
        assert result == "NORMAL", "Canonicalization inheritance broken"
        print("✅ Method inheritance works")
    except Exception as e:
        print(f"❌ Method inheritance failed: {e}")
        return False
    
    # Test 3: Core operation
    try:
        # Should be able to perform basic phenotype operation
        result = batch.add_phenotype("test_embryo_e01", "NORMAL", target="100")
        assert isinstance(result, dict), "Return format incorrect"
        assert "operation" in result, "Missing operation key in result"
        print("✅ Core operation works")
    except Exception as e:
        print(f"❌ Core operation failed: {e}")
        return False
    
    print("🎉 All inheritance contract tests passed")
    return True

# Run during development/CI to catch parent evolution issues
if __name__ == "__main__":
    test_annotation_batch_inheritance_contract()
```

**Benefits:**
- ✅ **Early detection**: Catches parent class changes before they break batch operations
- ✅ **Minimal overhead**: Simple test runs quickly in CI
- ✅ **Clear failure modes**: Specific error messages for different inheritance failures
- ✅ **Development safety**: Runs during development to catch issues immediately

---

## **Future Extensions** 🔮

### **Potential Enhancements**
1. **Operation Logging**: Add _ops tracking if detailed audit trails become needed
2. **Batch Merging**: Combine multiple batches before applying
3. **Partial Apply**: Apply only specific annotation types from batch
4. **Undo/Redo**: Store reverse operations for rollback capability
5. **Batch Templates**: Save/load common annotation patterns
6. **Parallel Apply**: Concurrent application of non-conflicting batches

### **Integration Points**
1. **Pipeline Scripts**: Batch creation from experimental protocols
2. **Interactive Tools**: GUI editors that build batches incrementally  
3. **Quality Control**: Automated batch validation and reporting
4. **Team Workflows**: Batch review and approval processes
5. **Data Export**: Batch-based annotation export for analysis

---

This design provides a robust, simple foundation for batch annotation workflows while maintaining clean separation from the persistent metadata store. The inheritance approach maximizes code reuse while the simplified apply mechanism keeps complexity manageable.