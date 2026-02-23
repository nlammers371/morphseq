# Module 3 Simplified: Data Structure & SAM2 Import Process

## **1. What We're Building** 🎯

**Goal**: Add biological annotations (phenotypes, genotypes, treatments) to SAM2 segmentation results

**Problem**: SAM2 gives us "where the embryos are" but biologists need to record "what's wrong with them"

**Solution**: Layer biological metadata on top of existing SAM2 structure

---

## **2. SAM2 Input Structure** 📥

**What SAM2 Gives Us:**
```json
{
  "experiments": {
    "20240418": {
      "videos": {
        "20240418_A01": {
          "image_ids": {
            "20240418_A01_t0100": {
              "embryos": {
                "20240418_A01_e01": {
                  "segmentation": {"counts": "...", "size": [1440, 3420]},
                  "mask_confidence": 0.85
                },
                "20240418_A01_e02": {
                  "segmentation": {"counts": "...", "size": [1440, 3420]},
                  "mask_confidence": 0.92
                }
              }
            },
            "20240418_A01_t0101": { /* more frames... */ }
          }
        }
      }
    }
  }
}
```

**Key Information We Extract:**
- **Embryo IDs**: `"20240418_A01_e01"`, `"20240418_A01_e02"`
- **Snip IDs**: `"20240418_A01_e01_s0100"` (embryo + frame)
- **Hierarchical Structure**: experiment → video → image → embryo

---

## **3. Target Annotation Structure** 📤

**What We Want to Create:**
```json
{
  "metadata": {
    "source_sam2": "/path/to/sam2_annotations.json",
    "created": "2025-01-15T10:30:00",
    "version": "simplified_v1"
  },
  "embryos": {
    "20240418_A01_e01": {
      "embryo_id": "20240418_A01_e01",
      "experiment_id": "20240418",
      "video_id": "20240418_A01",
      
      "genotype": {
        "gene": "tmem67",
        "allele": "sa1423", 
        "zygosity": "homozygous",
        "author": "researcher1",
        "timestamp": "2025-01-15T10:35:00"
      },
      
      "treatments": [
        {
          "value": "PTU",
          "temperature_celsius": 28.5,
          "concentration": "200μM",
          "notes": "24-48hpf pigment inhibition",
          "author": "researcher1",
          "timestamp": "2025-01-15T10:36:00"
        },
        {
          "value": "heat_shock", 
          "temperature_celsius": 37.0,
          "concentration": null,
          "notes": "1hr at 30hpf",
          "author": "researcher1", 
          "timestamp": "2025-01-15T10:45:00"
        }
      ],
      
      "snips": {
        "20240418_A01_e01_s0100": {
          "snip_id": "20240418_A01_e01_s0100",
          "frame_number": 100,
          "phenotypes": [
            {
              "value": "NORMAL",
              "confidence": 0.9,
              "author": "researcher1",
              "timestamp": "2025-01-15T10:37:00"
            }
          ],
          "flags": []
        },
        "20240418_A01_e01_s0150": {
          "snip_id": "20240418_A01_e01_s0150", 
          "frame_number": 150,
          "phenotypes": [
            {
              "value": "EDEMA",
              "confidence": 0.95,
              "author": "researcher1",
              "timestamp": "2025-01-15T10:38:00"
            },
            {
              "value": "CONVERGENCE_EXTENSION",
              "confidence": 0.8,
              "author": "researcher1",
              "timestamp": "2025-01-15T10:38:00"
            }
          ],
          "flags": [
            {
              "type": "MOTION_BLUR",
              "description": "Tail movement during capture",
              "author": "researcher1"
            }
          ]
        },
        "20240418_A01_e01_s0200": {
          "snip_id": "20240418_A01_e01_s0200", 
          "frame_number": 200,
          "phenotypes": [
            {
              "value": "DEAD",
              "confidence": 1.0,
              "author": "researcher1",
              "timestamp": "2025-01-15T10:39:00"
            }
          ],
          "flags": []
        }
      }
    }
  }
}
```

---

## **4. SAM2 Import Process** 🔄

**Step-by-Step Initialization:**

**Step 1: Load SAM2 Data**
```python
def __init__(self, sam2_path, annotations_path=None):
    # Load the SAM2 JSON file
    with open(sam2_path) as f:
        self.sam2_data = json.load(f)
    
    # Determine where to save annotations
    if annotations_path is None:
        # Default: sam2_annotations.json → sam2_annotations_biology.json
        annotations_path = sam2_path.replace('.json', '_biology.json')
```

**Step 2: Extract Embryo Structure**
```python
def _extract_embryos_from_sam2(self):
    """
    PURPOSE: Find all embryos in SAM2 data and create annotation structure
    
    PROCESS:
    - Scan through experiments → videos → image_ids → embryos
    - For each embryo found, create annotation placeholder
    - Generate all possible snip IDs (embryo + frame combinations)
    """
    embryos = {}
    
    for exp_id, exp_data in self.sam2_data["experiments"].items():
        for video_id, video_data in exp_data["videos"].items():
            for image_id, image_data in video_data["image_ids"].items():
                for embryo_id in image_data["embryos"].keys():
                    
                    if embryo_id not in embryos:
                        embryos[embryo_id] = self._create_embryo_structure(embryo_id)
                    
                    # Add snip for this frame
                    frame_num = self._extract_frame_number(image_id)  # "t0100" → 100
                    snip_id = f"{embryo_id}_s{frame_num:04d}"
                    embryos[embryo_id]["snips"][snip_id] = self._create_snip_structure(snip_id, frame_num)
    
    return embryos
```

**Step 3: Create Empty Structures**
```python
def _create_embryo_structure(self, embryo_id):
    """Create empty annotation structure for one embryo"""
    return {
        "embryo_id": embryo_id,
        "experiment_id": self._extract_experiment_id(embryo_id),  # "20240418_A01_e01" → "20240418"
        "video_id": self._extract_video_id(embryo_id),           # "20240418_A01_e01" → "20240418_A01"
        "genotype": None,        # Will be filled by biologist
        "treatments": [],        # List of treatment objects (preserves order)
        "snips": {}             # Will contain frame-by-frame data
    }

def _create_snip_structure(self, snip_id, frame_number):
    """Create empty annotation structure for one snip (embryo at specific frame)"""
    return {
        "snip_id": snip_id,
        "frame_number": frame_number,
        "phenotypes": [],        # Array for multiple phenotypes
        "flags": []             # QC flags for this specific frame
    }
```

---

## **5. Data Type Architecture** 🏗️

**Three Levels of Annotations:**

**Embryo-Level** (applies to whole embryo across all frames):
- **Genotype**: `tmem67`, `lmx1b`, `WT` - genetic background
- **Treatments**: `PTU`, `DMSO`, `heat_shock` - experimental conditions

**Snip-Level** (specific to one frame):
- **Phenotypes**: `NORMAL`, `EDEMA`, `DEAD` - what we see at this timepoint
- **QC Flags**: `MOTION_BLUR`, `OUT_OF_FOCUS` - technical issues

**Why This Split?**
- **Genotype doesn't change** → embryo level
- **Phenotypes evolve over time** → snip level  
- **Treatments affect whole embryo** → embryo level
- **Technical issues are frame-specific** → snip level

---

## **6. Business Rules** ⚖️

**Critical Validation Rules:**

### **6.1 DEAD Phenotype Logic** 💀
**Rule**: DEAD phenotype has two strict requirements:
1. **Exclusivity**: DEAD cannot coexist with any other phenotype at the same frame
2. **Permanence**: Once an embryo is marked DEAD at frame N, ALL subsequent frames (N+1, N+2, ...) MUST also be marked DEAD

**Valid Example**:
```json
"snips": {
  "embryo_e01_s0100": {"phenotype": {"value": "NORMAL"}},
  "embryo_e01_s0150": {"phenotype": {"value": "EDEMA"}},
  "embryo_e01_s0200": {"phenotype": {"value": "DEAD"}},    // Death occurs at frame 200
  "embryo_e01_s0250": {"phenotype": {"value": "DEAD"}},    // Must remain DEAD
  "embryo_e01_s0300": {"phenotype": {"value": "DEAD"}}     // Must remain DEAD
}
```

**Invalid Examples**:
```json
// ❌ INVALID: DEAD coexisting with other phenotype
"embryo_e01_s0200": {"phenotype": {"value": "DEAD"}}
"embryo_e01_s0200": {"phenotype": {"value": "EDEMA"}}  // Cannot have both!

// ❌ INVALID: Resurrection after death  
"embryo_e01_s0200": {"phenotype": {"value": "DEAD"}},
"embryo_e01_s0250": {"phenotype": {"value": "NORMAL"}}  // Cannot come back to life!
```

### **6.2 Single Genotype Rule** 🧬
**Rule**: Each embryo gets exactly one genotype
- **Rationale**: Genetic background is fixed for entire embryo
- **Enforcement**: Attempting to add second genotype raises error (unless `overwrite=True`)

### **6.3 Level Enforcement** 📊
**Rule**: Different annotation types go to appropriate levels
- **Genotypes** → embryo level (don't change over time)
- **Phenotypes** → snip level (temporal observations)
- **Treatments** → embryo level (affect whole organism)
- **QC Flags** → snip level (frame-specific technical issues)

### **6.4 Valid Values** ✅
**Rule**: Only pre-approved values allowed
```python
VALID_PHENOTYPES = ["NORMAL", "EDEMA", "CONVERGENCE_EXTENSION", "DEAD"]
VALID_GENES = ["WT", "tmem67", "lmx1b", "sox9a", "cep290", "b9d2", "rpgrip1l"]
VALID_ZYGOSITY = ["homozygous", "heterozygous", "compound_heterozygous", "crispant", "morpholino"]
VALID_TREATMENTS = ["control", "DMSO", "PTU", "BIO", "SB431542", "DAPT", "heat_shock", "cold_shock"]
VALID_FLAGS = ["MOTION_BLUR", "OUT_OF_FOCUS", "DARK", "CORRUPT"]
```

---

## **7. Multiple Phenotype API Design** 🛠️

**Two Methods for Flexibility:**

```python
# Single phenotype (primary implementation)
def add_phenotype(self, snip_id, phenotype, author, confidence=None):
    """Add single phenotype to snip with DEAD validation"""
    # Validate DEAD exclusivity with existing phenotypes
    existing = self._get_snip_phenotypes(snip_id)
    
    if phenotype == "DEAD" and existing:
        raise ValueError("DEAD cannot coexist with other phenotypes")
    
    if "DEAD" in existing and phenotype != "DEAD":
        raise ValueError("Cannot add phenotypes to DEAD snip")
    
    # Validate DEAD permanence (temporal logic)
    self._validate_dead_permanence(snip_id, phenotype)
    
    # Add the phenotype directly to the array
    embryo_id = self._get_embryo_id_from_snip(snip_id)
    if "phenotypes" not in self.data["embryos"][embryo_id]["snips"][snip_id]:
        self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"] = []
    
    self.data["embryos"][embryo_id]["snips"][snip_id]["phenotypes"].append({
        "value": phenotype,
        "confidence": confidence,
        "author": author,
        "timestamp": datetime.now().isoformat()
    })

# Single unified method handles both single and multiple phenotypes
# No separate add_phenotypes method needed
```

**Example Usage:**
```python
# Single phenotype
annotations.add_phenotype("embryo_e01", "NORMAL", "user1", target="100")

# Multiple phenotypes at once (unified API)
annotations.add_phenotype("embryo_e01", ["EDEMA", "CONVERGENCE_EXTENSION"], "user1", target="150")

# DEAD phenotype (must be alone) 
annotations.add_phenotype("embryo_e01", "DEAD", "user1", target="200:")
```

---

## **8. Range Semantics and Target Specifications** 🎯

**Simplified, Inclusive String Range Format**

### **8.1 Range Philosophy**

**All frame ranges are inclusive by default** - no confusing exclusive endpoints:
- `"30:50"` means frames 30, 31, 32, ..., 49, 50 (includes both endpoints)
- `"30:"` means frame 30 to the last available frame 
- `":50"` means first available frame to frame 50
- No need for Python slice objects or complex endpoint calculations

### **8.2 Supported Target Formats**

**String Ranges (Recommended):**
```python
# Inclusive frame ranges - both endpoints included
"30:50"    # Frames 30, 31, 32, ..., 49, 50
"030:050"  # Same as above - zero-padding optional
"200:"     # Frame 200 to end of embryo timeline
":100"     # Start of timeline to frame 100 
"all"      # All available snips for the embryo
```

**List Formats:**
```python
# Specific frame numbers
[30, 31, 32, 45, 67]          # Only these exact frames

# Specific snip IDs  
["embryo_e01_s0030", "embryo_e01_s0031"]  # Only these exact snips
```

### **8.3 Range Behavior Examples**

**Natural inclusive semantics:**
```python
# Standard inclusive range
result = metadata.add_phenotype("embryo_e01", "EDEMA", "user1", target="100:105")
# Applied to: embryo_e01_s0100 through embryo_e01_s0105 (inclusive)
print(f"Applied {result['phenotype']} to {result['count']} frames ({result['frame_range']})")
print(f"Applied to snips: {result['applied_to'][:3]}...{result['applied_to'][-1]}")

# Open-ended ranges 
death_result = metadata.add_phenotype("embryo_e01", "DEAD", "user1", target="200:")
# Applied to all frames from 200 onward: [s0200, s0201, s0202, ..., s0500]

early_result = metadata.add_phenotype("embryo_e02", "NORMAL", "user1", target=":50") 
# Applied to all frames up to 50: [s0000, s0001, s0002, ..., s0050]
```

**Gap handling in ranges:**
```python
# Missing frames skipped gracefully (normal in SAM2 data)
result = metadata.add_phenotype("embryo_e01", "TEST", "user1", target="100:110")
# If frames 103 and 107 are missing from SAM2 data:
# Applied to: [s0100, s0101, s0102, s0104, s0105, s0106, s0108, s0109, s0110]
# (missing s0103, s0107 skipped automatically in smart mode)
```

### **8.4 Range Parsing Implementation**

**Robust and simple parsing logic:**
```python
def _normalize_range_string(self, range_str: str) -> str:
    """Normalize range format for consistent parsing
    
    Examples:
        "030:059" → "30:59" 
        "200:" → "200:"
        ":100" → ":100"
        "0042" → "42" (single frame)
    """
    if ":" not in range_str:
        # Single frame number - remove leading zeros
        return str(int(range_str))
    
    parts = range_str.split(":")
    normalized_parts = []
    
    for part in parts:
        if part:  # Non-empty part
            normalized_parts.append(str(int(part)))
        else:  # Empty part (e.g., ":100" or "200:")
            normalized_parts.append("")
    
    return ":".join(normalized_parts)

def _parse_frame_range(self, embryo_id: str, range_str: str, available_snips: list, strict: str) -> list[str]:
    """Parse inclusive frame ranges with automatic normalization"""
    # Normalize first to reduce parsing complexity
    normalized_range = self._normalize_range_string(range_str)
    
    parts = normalized_range.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid range format: {range_str}. Use 'start:end', 'start:', or ':end'")
    
    # Parse start/end (now guaranteed to be clean integers)
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    
    # Filter available snips to range (inclusive on both ends)
    result = []
    for snip_id in available_snips:
        frame_num = extract_frame_number(snip_id)
        
        if frame_num is None:
            continue
        if start is not None and frame_num < start:  # Before range start
            continue
        if end is not None and frame_num > end:      # After range end (inclusive!)
            continue
        
        result.append(snip_id)
    
    return sorted(result, key=lambda s: extract_frame_number(s))
```

### **8.5 Benefits of Inclusive Ranges**

- ✅ **Intuitive**: `"30:50"` includes both 30 and 50 (natural expectation)
- ✅ **Consistent**: No confusion about exclusive vs inclusive endpoints
- ✅ **Flexible padding**: `"30:50"` and `"030:050"` work identically
- ✅ **Open-ended support**: `"30:"` and `":50"` handle partial ranges naturally
- ✅ **Gap tolerant**: Missing frames in ranges skipped gracefully (smart mode)
- ✅ **Simple parsing**: No slice objects or complex boundary calculations needed

**Range validation edge case handling:**
```python
# Edge case: single frame as range
"100:100"  # Valid - includes only frame 100 (start == end)

# Edge case: empty range detection
"200:100"  # Invalid - start > end, raises ValueError with helpful message

# Edge case: partial availability
"100:200" with only frames [100, 101, 150, 151, 200] available
# Result: [s0100, s0101, s0150, s0151, s0200] in smart mode
# Error in strict="all" mode due to gaps
```

---

## **9. Dictionary-Style Data Access API** 📋

**Intuitive data exploration using entity detection and summaries**

### **9.1 Auto-Detection Access Pattern**

**Core Implementation:**
```python
# Import robust parsing utilities for entity detection
from scripts.utils.parsing_utils import get_entity_type

def __getitem__(self, entity_id: str):
    """Auto-detect entity type and return appropriate summary"""
    entity_type = get_entity_type(entity_id)
    
    if entity_type == "embryo":
        return self.get_embryo_summary(entity_id)
    elif entity_type == "snip":
        return self.get_snip_summary(entity_id)
    else:
        raise KeyError(f"Unsupported entity type: {entity_type} for {entity_id}")
```

**Universal Access Pattern:**
```python
# Works identically on both metadata and batch objects
metadata = EmbryoMetadata("sam2_annotations.json")
batch = metadata.initialize_batch()

# Auto-detects embryo type, returns embryo summary
embryo_info = metadata["embryo_e01"]
batch_embryo = batch["embryo_e01"]

# Auto-detects snip type, returns snip summary  
snip_info = metadata["embryo_e01_s0150"]
batch_snip = batch["embryo_e01_s0150"]
```

### **9.2 Embryo Summary Method**

```python
def get_embryo_summary(self, embryo_id: str) -> dict:
    """Get comprehensive embryo state overview"""
    embryo_data = self.data["embryos"][embryo_id]
    
    # Count phenotypes by type using stored frame numbers
    phenotype_counts = {}
    for snip_data in embryo_data["snips"].values():
        for pheno in snip_data.get("phenotypes", []):
            value = pheno["value"]
            phenotype_counts[value] = phenotype_counts.get(value, 0) + 1
    
    # Extract frame range using stored data instead of parsing
    snip_frames = [snip_data.get("frame_number") for snip_data in embryo_data["snips"].values() 
                   if snip_data.get("frame_number") is not None]
    frame_range = f"{min(snip_frames):04d}-{max(snip_frames):04d}" if snip_frames else "none"
    
    return {
        "embryo_id": embryo_id,
        "genotype": self._format_genotype(embryo_data.get("genotype")),
        "phenotypes": phenotype_counts,
        "treatments": [t["value"] for t in embryo_data.get("treatments", [])],
        "flags": len(embryo_data.get("flags", {})),
        "total_snips": len(embryo_data["snips"]),
        "frame_range": frame_range,
        "last_modified": embryo_data.get("metadata", {}).get("last_updated", "unknown")
    }

def _format_genotype(self, genotype_data):
    """Format genotype for display"""
    if not genotype_data:
        return None
    gene = genotype_data.get("value", "unknown")
    zygosity = genotype_data.get("zygosity", "unknown")
    allele = genotype_data.get("allele", "")
    return f"{gene}:{allele} ({zygosity})" if allele else f"{gene} ({zygosity})"
```

### **9.3 Snip Summary Method**

```python
def get_snip_summary(self, snip_id: str) -> dict:
    """Get detailed snip state and context"""
    # Use robust parsing to get parent embryo
    embryo_id = extract_embryo_id(snip_id)
    if not embryo_id or embryo_id not in self.data["embryos"]:
        raise ValueError(f"Cannot find embryo for snip: {snip_id}")
    
    snip_data = self.data["embryos"][embryo_id]["snips"].get(snip_id, {})
    
    # Extract frame number from stored data instead of parsing
    frame_number = snip_data.get("frame_number") or extract_frame_number(snip_id)
    
    return {
        "snip_id": snip_id,
        "embryo_id": embryo_id,
        "frame_number": frame_number,
        "phenotypes": [p["value"] for p in snip_data.get("phenotypes", [])],
        "flags": [f["value"] for f in snip_data.get("flags", [])],
        "has_mask": "segmentation" in snip_data,
        "confidence_scores": [p.get("confidence") for p in snip_data.get("phenotypes", []) 
                             if p.get("confidence") is not None],
        "last_modified": max([p.get("timestamp", "") for p in snip_data.get("phenotypes", [])] or ["unknown"])
    }
```

### **9.4 Usage Examples**

**Interactive Data Exploration:**
```python
# Quick embryo overview
summary = metadata["embryo_e01"]
print(f"Embryo: {summary['genotype']}")
print(f"Phenotypes: {summary['phenotypes']}")
print(f"Frame range: {summary['frame_range']}")

# Detailed snip inspection
snip = metadata["embryo_e01_s0150"]
print(f"Frame {snip['frame_number']}: {snip['phenotypes']}")
print(f"Flags: {snip['flags']}")

# Works identically on batch objects
batch_summary = batch["embryo_e01"]
print(f"Batch embryo state: {batch_summary['phenotypes']}")
```

**Programmatic Analysis:**
```python
# Check phenotype distributions across embryos
embryo_ids = ["embryo_e01", "embryo_e02", "embryo_e03"]
for embryo_id in embryo_ids:
    summary = metadata[embryo_id]
    edema_count = summary["phenotypes"].get("EDEMA", 0)
    total_frames = summary["total_snips"]
    print(f"{embryo_id}: {edema_count}/{total_frames} frames with EDEMA")

# Quality control checks
for embryo_id in metadata.data["embryos"]:
    summary = metadata[embryo_id]
    if summary["genotype"] is None:
        print(f"⚠️ Missing genotype: {embryo_id}")
    if "DEAD" in summary["phenotypes"] and len(summary["phenotypes"]) > 1:
        print(f"⚠️ DEAD coexistence issue: {embryo_id}")
```

### **9.5 Benefits**

- ✅ **Intuitive access**: Natural dictionary-like pattern
- ✅ **Auto-routing**: No need to remember method names
- ✅ **Consistent interface**: Same pattern for metadata and batch
- ✅ **Rich summaries**: Comprehensive state information
- ✅ **Performance optimized**: Uses stored data instead of parsing
- ✅ **Future-proof**: Easily extensible to other entity types

---

## **10. Enhanced Return Values** 📊

**Standardized, informative feedback from all annotation operations**

### **10.1 Operation Summary Structure**

**All annotation methods return comprehensive operation summaries:**
```python
# Phenotype operation returns applied details
def add_phenotype(self, embryo_id: str, phenotype: str, author: str | None = None, *,
                  target: str | list[str] | list[int] = "all",
                  strict: str = "smart",
                  overwrite: bool = False,
                  confidence: float | None = None,
                  **kwargs) -> dict:
    """Add phenotype with detailed operation feedback"""
    # ... existing logic ...
    
    return {
        "operation": "add_phenotype",
        "embryo_id": embryo_id,
        "phenotype": phenotype,
        "applied_to": applied_snips,
        "count": len(applied_snips),
        "frame_range": f"{min_frame:04d}-{max_frame:04d}" if applied_snips else None,
        "author": author,
        "timestamp": datetime.now().isoformat(),
        "strict_mode": strict
    }

# Genotype operation returns confirmation
def add_genotype(self, embryo_id: str, gene_name: str, allele: str = "",
                zygosity: str = "heterozygous", author: str | None = None,
                overwrite: bool = False) -> dict:
    """Add genotype with operation confirmation"""
    # ... existing logic ...
    
    return {
        "operation": "add_genotype", 
        "embryo_id": embryo_id,
        "genotype": f"{gene_name}:{allele} ({zygosity})" if allele else f"{gene_name} ({zygosity})",
        "gene": gene_name,
        "allele": allele,
        "zygosity": zygosity,
        "success": True,
        "overwrite": overwrite,
        "author": author,
        "timestamp": datetime.now().isoformat()
    }

# Treatment operation returns summary
def add_treatment(self, embryo_id: str, treatment_name: str,
                 temperature_celsius: float = 28.5,
                 concentration: str | None = None,
                 notes: str = "", author: str | None = None) -> dict:
    """Add treatment with biological metadata"""
    # ... existing logic ...
    
    return {
        "operation": "add_treatment",
        "embryo_id": embryo_id,
        "treatment": treatment_name,
        "temperature_celsius": temperature_celsius,
        "concentration": concentration,
        "notes": notes,
        "success": True,
        "author": author,
        "timestamp": datetime.now().isoformat()
    }
```

### **10.2 Usage Patterns**

**Immediate Operation Feedback:**
```python
# Get detailed feedback on phenotype operations
result = metadata.add_phenotype("embryo_e01", "EDEMA", "user1", target="100:150")
print(f"Applied {result['phenotype']} to {result['count']} frames")
print(f"Frames: {result['frame_range']}")
print(f"Strict mode: {result['strict_mode']}")

# Confirm genotype operations
genotype_result = metadata.add_genotype("embryo_e01", "tmem67", "sa1423", "homozygous", "user1")
print(f"Added genotype: {genotype_result['genotype']}")
print(f"Success: {genotype_result['success']}")

# Track treatment details
treatment_result = metadata.add_treatment("embryo_e01", "PTU", temperature_celsius=28.5, 
                                         concentration="200μM", notes="24-48hpf")
print(f"Treatment: {treatment_result['treatment']} at {treatment_result['temperature_celsius']}°C")
```

**Operation Chaining and Validation:**
```python
# Chain operations with validation
phenotype_ops = []
for embryo_id in ["embryo_e01", "embryo_e02", "embryo_e03"]:
    result = metadata.add_phenotype(embryo_id, "NORMAL", "user1", target="all")
    phenotype_ops.append(result)
    
    if result["count"] == 0:
        print(f"⚠️ No frames annotated for {embryo_id}")

# Aggregate results
total_annotated = sum(op["count"] for op in phenotype_ops)
print(f"Total frames annotated: {total_annotated}")
```

**Programmatic Error Handling:**
```python
try:
    result = metadata.add_genotype("embryo_e01", "invalid_gene", author="user1")
    print(f"Genotype added: {result['genotype']}")
except ValueError as e:
    print(f"Genotype failed: {e}")
    # Result dict not available on exception - use return values for success cases
```

### **10.3 Consistent Structure Benefits**

- ✅ **Rich feedback**: Know exactly what changed and how
- ✅ **Operation tracking**: Timestamp and author for audit trails  
- ✅ **Count information**: Immediate feedback on scope of changes
- ✅ **Chaining support**: Use results from one operation in the next
- ✅ **Debugging aid**: Clear context for troubleshooting issues
- ✅ **Logging ready**: Structured data perfect for automated logging

### **10.4 Deterministic Sorting and Deduplication**

**Guaranteed Consistent Ordering:**

All annotation operations ensure deterministic, reproducible results through systematic sorting and deduplication:

```python
def add_phenotype(self, embryo_id: str, phenotype: str, ...) -> dict:
    # ... operation logic ...
    
    # STEP 1: Remove duplicates while preserving functionality  
    applied_snips_deduped = sorted(set(applied_snips), key=lambda s: extract_frame_number(s))
    
    # STEP 2: Sort by frame number for temporal consistency
    # Uses robust frame extraction from parsing_utils for reliable ordering
    # Result: ['embryo_e01_s0030', 'embryo_e01_s0031', 'embryo_e01_s0032', ...]
    
    return {
        "applied_to": applied_snips_deduped,  # Deterministic, deduplicated list
        "count": len(applied_snips_deduped),  # Accurate count after deduplication
        # ... other fields
    }
```

**Deduplication Handling:**

```python
# Example: Multiple operations might target overlapping snips
operations = [
    add_phenotype("embryo_e01", "NORMAL", target="100:110"),
    add_phenotype("embryo_e01", "BLUR", target=[105, 106, 107, 108]),  # Overlaps with range
]

# Each operation automatically deduplicates:
result1 = operations[0]  # applied_to: ['s0100', 's0101', ..., 's0110']
result2 = operations[1]  # applied_to: ['s0105', 's0106', 's0107', 's0108'] (no duplicates)
```

**Frame Number Extraction for Sorting:**

Uses robust `extract_frame_number()` from parsing_utils to ensure reliable sorting:

```python
# Handles various ID formats consistently:
extract_frame_number("embryo_e01_s0042")    # → 42
extract_frame_number("embryo_e01_s042")     # → 42  (flexible padding)
extract_frame_number("experiment_H01_e01_s0150")  # → 150 (complex IDs)

# Sorting result is always temporally consistent:
sorted_snips = sorted(snip_ids, key=lambda s: extract_frame_number(s))
# → ['embryo_e01_s0042', 'embryo_e01_s0150', ...] (chronological order)
```

**Benefits of Deterministic Processing:**

- ✅ **Reproducible results**: Same inputs always produce identical outputs
- ✅ **Temporal consistency**: Snips always ordered chronologically by frame
- ✅ **No duplicate processing**: Each snip annotated exactly once per operation
- ✅ **Reliable counting**: Accurate counts reflect actual snips affected
- ✅ **Testing friendly**: Predictable outputs enable robust unit tests
- ✅ **Debugging support**: Consistent ordering aids in troubleshooting
- ✅ **Audit trail reliability**: Operation logs contain precise, ordered information

---

## **11. Frame Number Optimization** ⚡

**Use stored data instead of parsing for better performance and reliability**

### **11.1 Optimization Strategy**

**Problem**: Heavy reliance on regex parsing for frame extraction from IDs:
```python
# INEFFICIENT: Parse frame numbers from IDs repeatedly
def sort_snips(snip_ids):
    return sorted(snip_ids, key=lambda s: extract_frame_number(s))  # Regex parsing each time

def get_frame_range(snip_ids):
    frames = [extract_frame_number(s) for s in snip_ids]  # More regex parsing
    return min(frames), max(frames)
```

**Solution**: Use stored `frame_number` from data structure:
```python
# OPTIMIZED: Use stored frame numbers from data structure
def sort_snips_optimized(self, embryo_id: str, snip_ids: list[str]) -> list[str]:
    """Sort snips using stored frame numbers instead of parsing"""
    embryo_data = self.data["embryos"][embryo_id]
    
    def get_stored_frame(snip_id):
        snip_data = embryo_data["snips"].get(snip_id, {})
        return snip_data.get("frame_number", 0)  # Use stored value
    
    return sorted(snip_ids, key=get_stored_frame)

def get_frame_range_optimized(self, embryo_id: str) -> tuple[int, int]:
    """Get frame range using stored data instead of parsing"""
    embryo_data = self.data["embryos"][embryo_id]
    frames = [snip_data.get("frame_number") for snip_data in embryo_data["snips"].values()
              if snip_data.get("frame_number") is not None]
    return (min(frames), max(frames)) if frames else (0, 0)
```

### **11.2 Data Structure Advantage**

**During SAM2 import, we already store frame numbers:**
```json
{
  "embryos": {
    "embryo_e01": {
      "snips": {
        "embryo_e01_s0100": {
          "frame_number": 100,  // Already available!
          "phenotypes": [],
          "flags": []
        },
        "embryo_e01_s0101": {
          "frame_number": 101,  // No parsing needed
          "phenotypes": [],
          "flags": []
        }
      }
    }
  }
}
```

### **11.3 Optimized Implementation Examples**

**Frame-based operations using stored data:**
```python
def _get_snips_in_range_optimized(self, embryo_id: str, start_frame: int, end_frame: int) -> list[str]:
    """Get snips in frame range using stored frame numbers"""
    embryo_data = self.data["embryos"][embryo_id]
    result = []
    
    for snip_id, snip_data in embryo_data["snips"].items():
        frame_num = snip_data.get("frame_number")
        if frame_num is not None and start_frame <= frame_num <= end_frame:
            result.append(snip_id)
    
    # Sort using stored frame numbers
    result.sort(key=lambda s: embryo_data["snips"][s]["frame_number"])
    return result

def _validate_temporal_order_optimized(self, embryo_id: str, phenotype_changes: list) -> bool:
    """Validate temporal phenotype order using stored frame numbers"""
    embryo_data = self.data["embryos"][embryo_id]
    
    # Group changes by frame number using stored data
    frame_changes = []
    for snip_id, phenotype in phenotype_changes:
        frame_num = embryo_data["snips"][snip_id].get("frame_number")
        if frame_num is not None:
            frame_changes.append((frame_num, snip_id, phenotype))
    
    # Sort by stored frame numbers (no parsing)
    frame_changes.sort(key=lambda x: x[0])
    
    # Validate DEAD permanence using sorted data
    found_dead = False
    for frame_num, snip_id, phenotype in frame_changes:
        if phenotype == "DEAD":
            found_dead = True
        elif found_dead and phenotype != "DEAD":
            return False  # Temporal violation
    return True
```

### **11.4 Fallback Strategy**

**When stored frame number is unavailable, use parsing as fallback:**
```python
def get_frame_number_hybrid(self, embryo_id: str, snip_id: str) -> int:
    """Get frame number with stored data preferred, parsing fallback"""
    # Try stored data first
    if embryo_id in self.data["embryos"]:
        snip_data = self.data["embryos"][embryo_id]["snips"].get(snip_id, {})
        stored_frame = snip_data.get("frame_number")
        if stored_frame is not None:
            return stored_frame
    
    # Fallback to parsing if stored data unavailable
    return extract_frame_number(snip_id) or 0
```

### **11.5 Performance Benefits**

**Stored data lookup vs regex parsing:**
- ✅ **~10x faster**: Direct dictionary access vs regex compilation/matching
- ✅ **More reliable**: No regex pattern dependencies or edge cases
- ✅ **Already available**: Frame numbers stored during SAM2 import
- ✅ **Consistent format**: Integer values vs string parsing variations

**Use cases where this matters:**
- Large datasets (1000s of embryos × 100s of frames each)
- Frequent sorting operations during annotation
- Real-time frame range operations
- Batch processing and temporal validation

---

## **12. Implementation Philosophy** 💡

**Simplicity Principles:**
- ✅ **Single class** (no inheritance chains)
- ✅ **Direct SAM2 integration** (no complex adapters)
- ✅ **Hard-coded validation** (no external schema dependencies)
- ✅ **Essential business rules only** (no over-engineering)
- ✅ **Clear error messages** (biologist-friendly)

---

## **13. Treatment System Design** 🧪

**Simple Treatment Structure:**
```python
# Minimal but practical treatment validation
VALID_TREATMENTS = [
    # Controls
    "control", 
    "DMSO",           # Vehicle control
    
    # Chemical treatments  
    "PTU",            # Pigmentation inhibitor
    "BIO",            # GSK-3 inhibitor
    "SB431542",       # TGF-β inhibitor
    "DAPT",           # Notch inhibitor
    
    # Physical treatments
    "heat_shock",
    "cold_shock"
    
    # Add more as actually needed - no over-engineering
]
```

**Treatment API with Biological Metadata:**
```python
def add_treatment(self, embryo_id: str, treatment_name: str, 
                 temperature_celsius: float = 28.5,  # Standard zebrafish temperature
                 concentration: str | None = None,
                 notes: str = "", author: str | None = None) -> dict:
    """Add treatment with essential biological metadata"""
    if treatment_name not in VALID_TREATMENTS:
        raise ValueError(f"Invalid treatment '{treatment_name}'. Valid: {VALID_TREATMENTS}")
    
    # Temperature is crucial - affects development rates and phenotypes
    if not (20.0 <= temperature_celsius <= 35.0):
        raise ValueError(f"Temperature {temperature_celsius}°C outside viable range (20-35°C)")
    
    treatment_data = {
        "value": treatment_name,
        "temperature_celsius": temperature_celsius,
        "concentration": concentration,
        "notes": notes,
        "author": author or self.config.get("default_author", "unknown"),
        "timestamp": datetime.now().isoformat()
    }
    
    # Store treatment (append to list)
    self.data["embryos"][embryo_id]["treatments"].append(treatment_data)
    
    return {
        "operation": "add_treatment",
        "embryo_id": embryo_id,
        "treatment": treatment_name,
        "temperature_celsius": temperature_celsius,
        "concentration": concentration,
        "notes": notes,
        "success": True,
        "author": author,
        "timestamp": treatment_data["timestamp"]
    }
```

**Enhanced Treatment Usage Examples:**
```python
# Standard treatment with temperature tracking
result = metadata.add_treatment("embryo_e01", "PTU", 
                               temperature_celsius=28.5,
                               concentration="200μM", 
                               notes="24-48hpf pigment inhibition")
print(f"Added {result['treatment']} at {result['temperature_celsius']}°C")

# Temperature variation study
result = metadata.add_treatment("embryo_e02", "control", 
                               temperature_celsius=32.0,
                               notes="elevated temperature study")

# Multiple treatments per embryo allowed
metadata.add_treatment("embryo_e01", "DMSO", temperature_celsius=28.5)
metadata.add_treatment("embryo_e01", "BIO", temperature_celsius=28.5, concentration="5μM")
```

**Temperature Importance in Zebrafish Development:**
- **28.5°C**: Standard laboratory temperature for zebrafish
- **32°C**: Accelerated development (shorter timeframes)
- **25°C**: Slower development (extended timeframes)
- Temperature affects gene expression, developmental timing, and phenotype severity
- Crucial metadata for reproducible experiments and data interpretation

---

**Updated Design Philosophy:**
- ✅ **Validate treatment names** (prevents typos: "ptu" vs "PTU")
- ✅ **Temperature tracking** (essential biological metadata)
- ✅ **Optional structured fields** (concentration, notes) with flexibility
- ✅ **Multiple treatments allowed** (unlike single genotype rule)
- ✅ **Biological validation** (temperature range checking)
- ✅ **Extensible and practical** (add treatments as actually needed)

---

## **14. Genotype System Design** 🧬

**Single Genotype Rule:**
```python
def add_genotype(self, embryo_id, gene, allele="", zygosity="heterozygous", author=""):
    """Add single genotype to embryo - only one allowed per embryo"""
    # Validate gene and zygosity
    if gene not in VALID_GENES:
        raise ValueError(f"Invalid gene '{gene}'. Valid: {VALID_GENES}")
    if zygosity not in VALID_ZYGOSITY:
        raise ValueError(f"Invalid zygosity '{zygosity}'. Valid: {VALID_ZYGOSITY}")
    
    # Check for existing genotype
    if self.data["embryos"][embryo_id]["genotype"]:
        raise ValueError(f"Embryo already has genotype. Use overwrite=True to change.")
    
    # Add genotype
    self.data["embryos"][embryo_id]["genotype"] = {
        "gene": gene,
        "allele": allele,  # e.g., "sa1423", "e1127" - can be empty for WT
        "zygosity": zygosity,
        "author": author,
        "timestamp": datetime.now().isoformat()
    }
```

**Valid Zygosity Types:**
```python
VALID_ZYGOSITY = [
    "homozygous",           # +/+ or -/-
    "heterozygous",         # +/-  
    "compound_heterozygous", # Two different alleles
    "crispant",            # CRISPR-induced mutation
    "morpholino"           # Morpholino knockdown
]
```

**Example Usage:**
```python
# Wild type
annotations.add_genotype("embryo_e01", "WT", author="user1")

# Mutant with allele
annotations.add_genotype("embryo_e02", "tmem67", "sa1423", "homozygous", "user1")

# CRISPR experiment
annotations.add_genotype("embryo_e03", "cep290", "cr001", "crispant", "user1")
```

---

## **15. Validation Toggle System** ⚙️

**Flexible Validation Control:**
```python
class EmbryoAnnotations:
    def __init__(self, sam2_path, annotations_path=None, validate=True):
        """
        Initialize annotation system with optional validation
        
        Args:
            validate: Enable/disable validation (default: True)
                     - True: Strict validation (production use)
                     - False: Accept any values (development/testing)
        """
        self.validate = validate
        self.sam2_data = self._load_sam2_data(sam2_path)
        self.data = self._initialize_annotations()
```

**Validation-Aware Methods:**
```python
def add_phenotype(self, snip_id, phenotype, author, confidence=None):
    """Add phenotype with optional validation"""
    if self.validate:
        # Strict validation - production mode
        if phenotype not in VALID_PHENOTYPES:
            raise ValueError(f"Invalid phenotype '{phenotype}'. Valid: {VALID_PHENOTYPES}")
        
        # DEAD exclusivity validation
        existing = self._get_snip_phenotypes(snip_id)
        if phenotype == "DEAD" and existing:
            raise ValueError("DEAD cannot coexist with other phenotypes")
    
    # Always store the annotation (validation or not)
    self._add_phenotype_to_data(snip_id, phenotype, author, confidence)

def add_genotype(self, embryo_id, gene, allele="", zygosity="heterozygous", author=""):
    """Add genotype with optional validation"""
    if self.validate:
        # Strict validation - production mode
        if gene not in VALID_GENES:
            raise ValueError(f"Invalid gene '{gene}'. Valid: {VALID_GENES}")
        if zygosity not in VALID_ZYGOSITY:
            raise ValueError(f"Invalid zygosity '{zygosity}'. Valid: {VALID_ZYGOSITY}")
        
        # Single genotype rule
        if self.data["embryos"][embryo_id]["genotype"]:
            raise ValueError("Embryo already has genotype. Use overwrite=True to change.")
    
    # Always store the annotation (validation or not)
    self._add_genotype_to_data(embryo_id, gene, allele, zygosity, author)
```

**Use Cases:**

**Production Mode** (validate=True):
```python
# Strict validation - prevents bad data
annotations = EmbryoAnnotations("sam2.json", validate=True)
annotations.add_phenotype("snip_s0100", "INVALID_PHENOTYPE", "user1")  # ❌ Raises error
```

**Development/Testing Mode** (validate=False):
```python
# Flexible mode - accepts anything
annotations = EmbryoAnnotations("sam2.json", validate=False)
annotations.add_phenotype("snip_s0100", "test_phenotype", "user1")     # ✅ Works
annotations.add_phenotype("snip_s0100", "another_test", "user1")       # ✅ Works
annotations.add_genotype("embryo_e01", "experimental_gene", author="user1")  # ✅ Works

# Later, add to valid lists and switch to validation mode
```

**Pipeline Integration:**
```python
# Pipeline script with validation toggle
python scripts/pipelines/07_biological_annotations.py \
  --sam2-annotations "$SAM2_ANNOTATIONS" \
  --output-annotations "$EMBRYO_ANNOTATIONS" \
  --no-validation \     # Disable validation for testing
  --dry-run
```

**Benefits:**
- ✅ **Production safety** - strict validation prevents bad data
- ✅ **Development flexibility** - test with experimental values
- ✅ **Gradual adoption** - start loose, tighten validation later
- ✅ **Pipeline testing** - validate functionality without data constraints
- ✅ **Research exploration** - try new phenotypes/genes before standardizing

**Validation Summary Method:**
```python
def get_validation_report(self):
    """Get validation report showing what would fail in strict mode"""
    if self.validate:
        return {"status": "strict_mode", "violations": []}
    
    violations = []
    for embryo_id, embryo_data in self.data["embryos"].items():
        # Check phenotypes
        for snip_id, snip_data in embryo_data["snips"].items():
            for phenotype_data in snip_data.get("phenotypes", []):
                if phenotype_data["value"] not in VALID_PHENOTYPES:
                    violations.append({
                        "type": "invalid_phenotype",
                        "snip_id": snip_id,
                        "value": phenotype_data["value"]
                    })
        
        # Check genotype
        if embryo_data["genotype"]:
            gene = embryo_data["genotype"]["gene"]
            if gene not in VALID_GENES:
                violations.append({
                    "type": "invalid_gene", 
                    "embryo_id": embryo_id,
                    "value": gene
                })
    
    return {"status": "permissive_mode", "violations": violations}
```

---

## **16. Temporal DEAD Validation Implementation** ⏰

**Death Frame Tracking System:**
```python
def _validate_dead_permanence(self, embryo_id, frame_num, phenotype):
    """
    Validate DEAD phenotype temporal consistency using embryo-level death tracking.
    
    APPROACH: Track death_frame at embryo level for efficient temporal validation
    """
    death_frame = self.data["embryos"][embryo_id].get("death_frame")
    
    if phenotype == "DEAD":
        # Adding DEAD phenotype
        if death_frame is None:
            # First time marking DEAD - record the death frame
            self.data["embryos"][embryo_id]["death_frame"] = frame_num
        elif frame_num < death_frame:
            # Trying to mark DEAD earlier than existing death - requires overwrite
            raise ValueError(f"Cannot mark DEAD at frame {frame_num} - embryo already died at frame {death_frame}. Use overwrite=True to change death frame.")
    else:
        # Adding non-DEAD phenotype
        if death_frame is not None and frame_num >= death_frame:
            # Trying to add non-DEAD after death - temporal violation
            raise ValueError(f"Cannot annotate '{phenotype}' at frame {frame_num} - embryo died at frame {death_frame}.")
#note @claude, there should be a mark_dead function for the api 
```

**Unified Flexible API Implementation:**
```python
# Import robust parsing utilities
from scripts.utils.parsing_utils import (
    extract_frame_number, validate_id_format, get_entity_type,
    extract_embryo_id, build_snip_id, parse_entity_id
)

def add_phenotype(self, embryo_id: str, phenotype: str | list[str], author: str | None = None, *,
                  target: str | list[str] | list[int] = "all",
                  strict: str = "smart",
                  overwrite: bool = False,
                  confidence: float | None = None,
                  **kwargs) -> dict:
    """
    Add phenotype(s) to embryo snips with flexible targeting
    
    IMPORTANT - DEAD Backfilling Behavior:
    When adding "DEAD" phenotype, the system automatically applies DEAD to ALL 
    subsequent frames (biological reality: death is permanent). This ensures 
    temporal consistency but means adding DEAD at frame 200 will also mark 
    frames 201, 202, 203... as DEAD.
    
    Args:
        embryo_id: Target embryo ID
        phenotype: Single phenotype ("EDEMA") or list of phenotypes (["EDEMA", "BLUR"])
                  Note: DEAD cannot be combined with other phenotypes in the same operation
        author: Author name
        target: Snip specification:
               - "all": all available snips
               - "30:50": inclusive frame range (frames 30, 31, 32, ..., 50)
               - "30:": open-ended range (frame 30 to end)
               - ":50": open-ended range (start to frame 50) 
               - [30, 31, 32]: specific frame numbers
               - ["snip_id1", "snip_id2"]: specific snip IDs
        strict: Validation level:
               - "smart": intelligent handling - skip missing frames in ranges, validate explicit IDs (default)
               - "all": everything specified must exist (strict validation)
               - "none": skip any missing, never error (permissive mode)
        overwrite: Allow overwriting existing phenotypes
        confidence: Confidence score for annotation
    
    Returns:
        Dictionary with operation details:
        {
            "operation": "add_phenotype",
            "embryo_id": str,
            "phenotype": str,  # Display name for operation
            "phenotypes": list[str],  # Full list of phenotypes applied
            "applied_to": list[str],  # Directly targeted snips
            "backfilled": list[str],  # Automatically added DEAD frames (if applicable)
            "count": int,  # Total snips affected (applied_to + backfilled)
            "frame_range": str,  # "start:end" of all affected frames
            "author": str,
            "timestamp": str,
            "strict_mode": str
        }
        
    Raises:
        ValueError: Invalid target spec, missing embryo, validation failures
    """
    # 1. Normalize phenotype input to list for consistent processing
    if isinstance(phenotype, str):
        phenotypes = [phenotype]
    else:
        phenotypes = phenotype.copy()  # Don't mutate caller's list
    
    # 2. Canonicalize and validate phenotypes
    if self.validate:
        for i, pheno in enumerate(phenotypes):
            phenotypes[i] = self._canonicalize_phenotype(pheno)
        
        # Validate DEAD exclusivity within the list
        if "DEAD" in phenotypes and len(phenotypes) > 1:
            raise ValueError("DEAD cannot coexist with other phenotypes in the same operation")
    
    # Store original for return info
    phenotype_display = phenotypes[0] if len(phenotypes) == 1 else f"{len(phenotypes)} phenotypes"
    
    # 2. Resolve target specification to snip_ids
    try:
        snip_ids = self._resolve_target_specification(embryo_id, target, strict)
    except Exception as e:
        available_frames = self._get_available_frame_range(embryo_id)
        raise ValueError(
            f"Target resolution failed for {embryo_id} with target='{target}':\n"
            f"  Available frames: {available_frames[0]}-{available_frames[1]}\n"
            f"  Available snips: {len(self.data['embryos'][embryo_id]['snips'])}\n"
            f"  Error: {e}\n"
            f"  Valid examples: 'all', '30:50', '30:', [30, 31, 32]"
        )
    
    if not snip_ids:
        if strict == "all":
            raise ValueError(f"No valid snips found for target='{target}' in {embryo_id}")
        return {
            "operation": "add_phenotype",
            "embryo_id": embryo_id,
            "phenotype": phenotype_display,
            "applied_to": [],
            "count": 0,
            "frame_range": None,
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "strict_mode": strict
        }  # Permissive modes return empty operation result
    
    # 3. Apply to each resolved snip
    applied_snips = []
    backfilled_snips = []
    
    for snip_id in snip_ids:
        try:
            # Extract context for validation using robust parsing
            frame_num = extract_frame_number(snip_id)  # Robust extraction from snip_id
            
            # Apply each phenotype to this snip
            for pheno in phenotypes:
                # Validation (if enabled)
                if self.validate:
                    self._validate_dead_exclusivity(snip_id, pheno, overwrite)
                    self._validate_dead_permanence(embryo_id, frame_num, pheno)
                
                # Store the phenotype
                self._append_phenotype_record(snip_id, pheno, author, confidence)
            
            applied_snips.append(snip_id)
            
            # Handle DEAD backfilling (if any phenotype in list is DEAD)
            if "DEAD" in phenotypes:
                backfilled = self._backfill_dead_forward(embryo_id, from_frame=frame_num)
                backfilled_snips.extend(backfilled)
                
        except Exception as e:
            if strict == "all":
                raise ValueError(f"Failed to apply phenotype to {snip_id}: {e}")
            # Permissive modes skip problematic snips
            continue
    
    # Return standardized operation summary with backfilling transparency
    applied_snips_deduped = sorted(set(applied_snips), key=lambda s: extract_frame_number(s))
    backfilled_snips_deduped = sorted(set(backfilled_snips), key=lambda s: extract_frame_number(s))
    
    # Calculate frame range from all affected snips (applied + backfilled)
    all_affected_snips = applied_snips_deduped + backfilled_snips_deduped
    if all_affected_snips:
        frame_nums = [extract_frame_number(s) for s in all_affected_snips if extract_frame_number(s) is not None]
        frame_range = f"{min(frame_nums):04d}:{max(frame_nums):04d}" if frame_nums else None
    else:
        frame_range = None
    
    return {
        "operation": "add_phenotype",
        "embryo_id": embryo_id,
        "phenotype": phenotype_display,
        "phenotypes": phenotypes,  # Full list for detailed inspection
        "applied_to": applied_snips_deduped,  # Directly targeted snips
        "backfilled": backfilled_snips_deduped,  # Automatically added DEAD frames
        "count": len(applied_snips_deduped) + len(backfilled_snips_deduped),  # Total affected
        "frame_range": frame_range,  # Range of all affected frames
        "author": author or "unknown",
        "timestamp": datetime.now().isoformat(),
        "strict_mode": strict
    }

def _resolve_target_specification(self, embryo_id: str, target, strict: str) -> list[str]:
    """
    Convert flexible target specification to snip_ids using robust parsing utilities
    
    Handles missing frames gracefully for ranges, strict validation for explicit specs
    """
    if embryo_id not in self.data["embryos"]:
        raise ValueError(f"Embryo not found: {embryo_id}")
    
    available_snips = list(self.data["embryos"][embryo_id]["snips"].keys())
    available_set = set(available_snips)
    
    if target == "all":
        return available_snips
    
    
    elif isinstance(target, str):
        entity_type = get_entity_type(target)  # Use robust entity type detection
        
        if entity_type == "snip":
            # Single snip_id: "embryo_e01_s0200" - validate it belongs to this embryo
            snip_embryo = extract_embryo_id(target)  # Robust embryo extraction
            if snip_embryo == embryo_id and target in available_set:
                return [target]
            elif strict == "all":
                raise ValueError(f"Snip {target} not found for embryo {embryo_id}")
            else:
                return []  # Skip missing in permissive modes
        else:
            # Frame range string: "30:50", "150:", ":100"
            return self._parse_frame_range(embryo_id, target, available_snips, strict)
    
    elif isinstance(target, list):
        if not target:
            return []
        
        # Determine list type from first element
        first_item = target[0]
        
        if isinstance(first_item, int):
            # List of frame numbers: [30, 31, 32] → build snip_ids consistently
            return self._frame_numbers_to_snips(embryo_id, target, available_snips, strict)
        
        elif isinstance(first_item, str):
            # List of snip_ids: ["embryo_e01_s0200", "embryo_e01_s0201"]
            return self._validate_snip_list(embryo_id, target, available_set, strict)
        
        else:
            raise ValueError(f"Invalid list item type: {type(first_item)}")
    
    else:
        raise ValueError(f"Invalid target type: {type(target)}")

def _validate_snip_list(self, embryo_id: str, snip_list: list, available_set: set, strict: str) -> list[str]:
    """Validate list of snip_ids using robust parsing utilities"""
    valid_snips = []
    
    for snip_id in snip_list:
        # Use robust validation instead of regex
        if (validate_id_format(snip_id, "snip") and  # Robust format validation
            extract_embryo_id(snip_id) == embryo_id and  # Correct parent embryo
            snip_id in available_set):  # Actually exists
            valid_snips.append(snip_id)
        elif strict == "all":
            raise ValueError(f"Invalid snip_id for {embryo_id}: {snip_id}")
    
    return valid_snips

def _parse_frame_range(self, embryo_id: str, range_str: str, available_snips: list, strict: str) -> list[str]:
    """Parse flexible frame ranges: '30:50', '030:050', '150:', ':100'"""
    parts = range_str.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid range format: {range_str}. Use 'start:end', 'start:', or ':end'")
    
    # Parse start/end, handling variable zero-padding
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    
    # Convert to snip_ids, filtering for existing frames using robust parsing
    result = []
    for snip_id in available_snips:
        frame_num = extract_frame_number(snip_id)  # Use robust extraction
        
        if frame_num is None:  # Skip if frame extraction fails
            continue
        if start is not None and frame_num < start:
            continue
        if end is not None and frame_num > end:
            continue
        
        result.append(snip_id)
    
    return sorted(result, key=lambda s: extract_frame_number(s))  # Use robust sorting


def _get_available_frame_range(self, embryo_id: str) -> tuple[int, int]:
    """Get min/max frame numbers for embryo using robust parsing"""
    snips = list(self.data["embryos"][embryo_id]["snips"].keys())
    if not snips:
        return (0, 0)
    
    frame_numbers = [extract_frame_number(s) for s in snips 
                    if extract_frame_number(s) is not None]  # Use robust extraction
    return (min(frame_numbers), max(frame_numbers)) if frame_numbers else (0, 0)

def _frame_numbers_to_snips(self, embryo_id: str, frame_numbers: list, available_snips: list, strict: str) -> list[str]:
    """Convert frame numbers to snip_ids using consistent formatting"""
    available_set = set(available_snips)
    result = []
    
    for frame_num in frame_numbers:
        # Use consistent snip_id builder from parsing_utils
        expected_snip = build_snip_id(embryo_id, frame_num)
        
        if expected_snip in available_set:
            result.append(expected_snip)
        elif strict == "all":
            raise ValueError(f"Frame {frame_num} not found for embryo {embryo_id}")
    
    return sorted(result, key=lambda s: extract_frame_number(s))  # Use robust sorting

def _validate_dead_exclusivity(self, snip_id, phenotype, overwrite):
    """Validate DEAD exclusivity rules at snip level"""
    existing = self._get_snip_phenotypes(snip_id)
    
    if phenotype == "DEAD" and existing and not (overwrite and set(existing) == {"DEAD"}):
        raise ValueError("DEAD cannot coexist with other phenotypes.")
    
    if "DEAD" in existing and phenotype != "DEAD":
        raise ValueError("Cannot add non-DEAD to a DEAD snip.")

def _backfill_dead_forward(self, embryo_id, from_frame):
    """Backfill DEAD phenotypes for all frames after death using robust parsing"""
    backfilled_snips = []
    
    for snip_id in self.data["embryos"][embryo_id]["snips"].keys():
        snip_frame = extract_frame_number(snip_id)  # Use robust extraction
        
        if snip_frame is not None and snip_frame > from_frame:
            # Only backfill if not already DEAD
            existing_phenotypes = self._get_snip_phenotypes(snip_id)
            if "DEAD" not in existing_phenotypes:
                self._append_phenotype_record(
                    snip_id, "DEAD", 
                    author="system_backfill", 
                    confidence=1.0
                )
                backfilled_snips.append(snip_id)
    
    return backfilled_snips
```

**Flexible API Usage Examples:**
```python
# Range operations with flexible formatting
result = annotations.add_phenotype("embryo_e01", "NORMAL", "user1", target="100:150")
print(f"Applied {result['phenotype']} to {result['count']} frames")
print(f"Frame range: {result['frame_range']}")  # Shows 0100:0149 inclusive
print(f"Sample snips: {result['applied_to'][:3]}...")  # Shows first 3 applied snips

# Natural frame ranges (no zero-padding required)
annotations.add_phenotype("embryo_e01", "EDEMA", "user1", target="30:50")     # Works!
annotations.add_phenotype("embryo_e01", "EDEMA", "user1", target="030:050")   # Also works!

# Open-ended string ranges (natural and explicit)
annotations.add_phenotype("embryo_e01", "DEAD", "user1", target="200:")    # From 200 onward
annotations.add_phenotype("embryo_e01", "NORMAL", "user1", target=":100")  # Up to frame 99

# Specific frame numbers or snip_ids
annotations.add_phenotype("embryo_e01", "BLUR", "user1", target=[45, 67, 89])  # Specific frames
annotations.add_phenotype("embryo_e01", "CORRUPT", "user1", target=["embryo_e01_s0123"])  # Specific snip

# Smart handling of missing frames (normal in SAM2 data)
# If frames 34, 67 are missing due to segmentation failures:
result = annotations.add_phenotype("embryo_e01", "EDEMA", "user1", target="30:70")
print(f"Applied to {result['count']} frames (skipped missing frames)")
print(f"Frames: {result['frame_range']} - includes gaps for missing snips")
# result['applied_to'] contains: ['embryo_e01_s0030', 'embryo_e01_s0031', ..., 'embryo_e01_s0033', 
#                                'embryo_e01_s0035', ..., 'embryo_e01_s0066', 'embryo_e01_s0068', ...]
# (automatically skips missing s0034, s0067 in smart mode)

# Strict validation for explicit specifications
try:
    annotations.add_phenotype("embryo_e01", "TEST", "user1", 
                             target=["embryo_e01_s9999"], strict="all")
except ValueError as e:
    print(e)  # "Snip not found: embryo_e01_s9999"

# Death sequence with automatic backfilling (now with transparency)
death_result = annotations.add_phenotype("embryo_e01", "DEAD", "user1", target="200")
print(f"Directly applied: {len(death_result['applied_to'])} frames")  # 1 frame (200)
print(f"Auto-backfilled: {len(death_result['backfilled'])} frames")   # 300 frames (201-500)
print(f"Total affected: {death_result['count']} frames")              # 301 frames total
print(f"Frame range: {death_result['frame_range']}")                  # "0200:0500"
# death_result['applied_to']: ['embryo_e01_s0200'] 
# death_result['backfilled']: ['embryo_e01_s0201', 'embryo_e01_s0202', ..., 'embryo_e01_s0500']

# Temporal validation still enforced
try:
    annotations.add_phenotype("embryo_e01", "NORMAL", "user1", target="250:")  # After death
except ValueError as e:
    print(e)  # "Cannot annotate 'NORMAL' at frame 250 - embryo died at frame 200"
```

### **16.1 Strict Mode Options** ⚙️

**Clear, intuitive naming for validation behavior:**

- **`strict="smart"`** (default for interactive tools): Intelligent handling based on target type
  - Range targets (`"30:50"`, `"30:"`): Skip missing frames gracefully
  - Explicit targets (`["snip_id1", "snip_id2"]`): Validate all IDs exist
  - Best for interactive use where ranges often have gaps
  
- **`strict="all"`** (recommended for pipeline imports): Everything specified must exist (strict validation)
  - All targets must exist or operation fails
  - Useful for automated scripts that need guaranteed results
  - Clear error messages for debugging
  - Prevents silent partial imports in production pipelines
  
- **`strict="none"`**: Skip any missing, never error (permissive mode)
  - Always succeeds, applies only to valid targets
  - Useful for bulk operations where partial success is acceptable

### **16.2 Recommended Defaults by Usage Context** 🎯

**Interactive Annotation Tools:**
```python
# Default to "smart" for human-friendly behavior
def create_annotation_interface():
    metadata = EmbryoMetadata("sam2_data.json")
    # Users expect ranges to handle gaps gracefully
    metadata.add_phenotype("embryo_e01", "EDEMA", "user1", 
                          target="100:200")  # strict="smart" by default
```

**Pipeline Import Scripts:**
```python  
# Default to "all" for reliable automated processing
def import_sam2_batch(sam2_files):
    for sam2_file in sam2_files:
        metadata = EmbryoMetadata(sam2_file)
        # Pipelines need guaranteed complete imports
        metadata.add_phenotype("embryo_e01", "DETECTED", "pipeline", 
                              target="all", strict="all")  # Explicit strict mode
```

**Data Analysis/Export:**
```python
# Use "none" for exploratory analysis
def export_phenotype_data():
    metadata = EmbryoMetadata("annotations.json") 
    # Analysis can work with partial data
    metadata.add_phenotype("embryo_e01", "ANALYSIS_FLAG", "script",
                          target=suspicious_frames, strict="none")  # Permissive
```
  - Returns list of actually applied targets for verification

**Usage Examples:**
```python
# Smart mode (default) - natural behavior
result = annotations.add_phenotype("embryo_e01", "EDEMA", "user1", 
                                  target="100:200")  # Skips missing frames in range
print(f"Applied to {result['count']} available frames in range {result['frame_range']}")

# Strict mode - everything must exist
annotations.add_phenotype("embryo_e01", "TEST", "user1", 
                         target=["embryo_e01_s0100", "embryo_e01_s0101"], 
                         strict="all")  # Fails if either snip missing

# Permissive mode - apply what you can
annotations.add_phenotype("embryo_e01", "FLAG", "user1", 
                         target=["existing_snip", "missing_snip", "another_snip"], 
                         strict="none")  # Applies only to existing snips
```

**API Benefits:**
- ✅ **Flexible input**: String ranges, Python slices, frame numbers, snip_ids all work
- ✅ **Smart filtering**: Missing frames skipped gracefully in ranges
- ✅ **Clear feedback**: Returns list of actually applied snip_ids
- ✅ **Error context**: Helpful error messages with available frame ranges
- ✅ **Natural syntax**: "30:50" and "30:" work intuitively
- ✅ **Strict when needed**: Explicit snips validated strictly by default

**Benefits of Death Frame Tracking:**
- ✅ **O(1) temporal validation** (no sorting required)
- ✅ **Simple embryo-level state** (single death_frame field)
- ✅ **Clear error messages** with specific frame violations
- ✅ **Efficient cross-frame consistency** without complex temporal queries
- ✅ **Optional overwrite support** for correcting death annotations

**Validation Toggle Integration:**
```python
def add_phenotype(self, snip_id, phenotype, author, confidence=None):
    if self.validate:
        # Full validation including temporal logic
        self._validate_dead_permanence(embryo_id, frame_num, phenotype)
    else:
        # Store without validation - for development/testing
        pass
    
    # Always store the data
    self._add_phenotype_to_data(snip_id, phenotype, author, confidence)
```

**Frame Number Extraction:**
```python
def _extract_frame_number(self, snip_id):
    """Extract frame number from snip ID: 'embryo_e01_s0150' → 150"""
    match = re.search(r'_s(\d{4})$', snip_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract frame number from snip_id: {snip_id}")
```

---

## **17. Validation Configuration System** 📁

**File Structure:**
```
config/
├── valid_annotation_types.json      # Main vocabulary registry
└── annotation_schema.json           # Optional: extended schema definitions

utils/
└── annotation_validation.py         # Simple validation utilities
```

**Main Vocabulary File: `config/valid_annotation_types.json`**
```json
{
  "phenotypes": {
    "NORMAL": {
      "description": "Normal embryonic development",
      "synonyms": ["normal", "wild-type", "wt", "wildtype"]
    },
    "EDEMA": {
      "description": "Fluid accumulation in tissues", 
      "synonyms": ["edema", "swelling", "fluid_accumulation"]
    },
    "CONVERGENCE_EXTENSION": {
      "description": "Convergence and extension defects",
      "synonyms": ["ce_defect", "convergence_extension", "ce"]
    },
    "DEAD": {
      "description": "Embryonic death",
      "synonyms": ["dead", "died", "death", "deceased"]
    }
  },
  
  "genes": {
    "WT": {
      "description": "Wild type control",
      "synonyms": ["wildtype", "wild-type", "control"]
    },
    "tmem67": {
      "description": "TMEM67 gene (MKS3)",
      "synonyms": ["mks3", "TMEM67"]
    },
    "lmx1b": {
      "description": "LMX1B transcription factor",
      "synonyms": ["LMX1B"]
    },
    "cep290": {
      "description": "CEP290 centrosomal protein",
      "synonyms": ["CEP290", "rd16"]
    }
  },
  
  "treatments": {
    "control": {
      "description": "No treatment control",
      "synonyms": ["ctrl", "untreated", "vehicle"]
    },
    "DMSO": {
      "description": "Dimethyl sulfoxide vehicle control",
      "synonyms": ["dmso", "vehicle"]
    },
    "PTU": {
      "description": "1-phenyl-2-thiourea pigment inhibitor",
      "synonyms": ["ptu", "phenylthiourea"]
    },
    "heat_shock": {
      "description": "Heat shock treatment",
      "synonyms": ["heat", "hs", "thermal_stress"]
    }
  },
  
  "zygosity": {
    "homozygous": {
      "description": "Homozygous genotype",
      "synonyms": ["homo", "hom", "homozygous"]
    },
    "heterozygous": {
      "description": "Heterozygous genotype", 
      "synonyms": ["hetero", "het", "heterozygous"]
    },
    "crispant": {
      "description": "CRISPR-induced mutation",
      "synonyms": ["crispr", "crispant", "cas9"]
    },
    "morpholino": {
      "description": "Morpholino knockdown",
      "synonyms": ["mo", "morpholino", "knockdown"]
    }
  },
  
  "flags": {
    "snip_level": {
      "MOTION_BLUR": {"description": "Motion blur in image"},
      "OUT_OF_FOCUS": {"description": "Image out of focus"}, 
      "DARK": {"description": "Image too dark"},
      "CORRUPT": {"description": "Corrupted image data"}
    },
    "embryo_level": {
      "POOR_QUALITY": {"description": "Overall poor embryo quality"},
      "DEVELOPMENTAL_DELAY": {"description": "Developmental timing issues"}
    }
  }
}
```

**Simple Validation Utils: `utils/annotation_validation.py`**
```python
"""
Simple annotation validation utilities
No complex logic - just load config and validate against lists
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set


class AnnotationValidator:
    """Simple validator that loads config and checks values"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Load validation config from JSON file"""
        if config_path is None:
            # Default to config/valid_annotation_types.json
            config_path = Path(__file__).parent.parent / "config" / "valid_annotation_types.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Build lookup tables for fast validation
        self._phenotype_lookup = self._build_lookup("phenotypes")
        self._gene_lookup = self._build_lookup("genes") 
        self._treatment_lookup = self._build_lookup("treatments")
        self._zygosity_lookup = self._build_lookup("zygosity")
    
    def _load_config(self) -> Dict:
        """Load config with fallback to hardcoded defaults"""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Config load failed: {e}. Using hardcoded defaults.")
            return self._get_hardcoded_defaults()
    
    def _build_lookup(self, category: str) -> Dict[str, str]:
        """Build synonym → canonical name lookup table"""
        lookup = {}
        
        for canonical, data in self.config.get(category, {}).items():
            # Add canonical name
            lookup[canonical.lower()] = canonical
            
            # Add synonyms
            for synonym in data.get("synonyms", []):
                lookup[synonym.lower()] = canonical
        
        return lookup
    
    def _get_hardcoded_defaults(self) -> Dict:
        """Fallback validation config if JSON fails"""
        return {
            "phenotypes": {
                "NORMAL": {"synonyms": ["normal", "wt"]},
                "EDEMA": {"synonyms": ["edema"]}, 
                "DEAD": {"synonyms": ["dead", "died"]}
            },
            "genes": {
                "WT": {"synonyms": ["wildtype"]},
                "tmem67": {"synonyms": ["mks3"]}
            },
            "treatments": {
                "control": {"synonyms": ["ctrl"]},
                "DMSO": {"synonyms": ["dmso"]}
            },
            "zygosity": {
                "homozygous": {"synonyms": ["homo"]},
                "heterozygous": {"synonyms": ["hetero"]}
            }
        }
    
    # Simple validation methods
    def validate_phenotype(self, value: str) -> str:
        """Return canonical phenotype or raise ValueError"""
        canonical = self._phenotype_lookup.get(value.lower())
        if canonical:
            return canonical
        
        valid_options = list(self.config["phenotypes"].keys())
        raise ValueError(f"Invalid phenotype '{value}'. Valid: {valid_options}")
    
    def validate_gene(self, value: str) -> str:
        """Return canonical gene or raise ValueError"""
        canonical = self._gene_lookup.get(value.lower())
        if canonical:
            return canonical
            
        valid_options = list(self.config["genes"].keys())
        raise ValueError(f"Invalid gene '{value}'. Valid: {valid_options}")
    
    def validate_treatment(self, value: str) -> str:
        """Return canonical treatment or raise ValueError"""
        canonical = self._treatment_lookup.get(value.lower())
        if canonical:
            return canonical
            
        valid_options = list(self.config["treatments"].keys())
        raise ValueError(f"Invalid treatment '{value}'. Valid: {valid_options}")
    
    def validate_zygosity(self, value: str) -> str:
        """Return canonical zygosity or raise ValueError"""
        canonical = self._zygosity_lookup.get(value.lower())
        if canonical:
            return canonical
            
        valid_options = list(self.config["zygosity"].keys())
        raise ValueError(f"Invalid zygosity '{value}'. Valid: {valid_options}")
    
    def get_valid_phenotypes(self) -> List[str]:
        """Get list of valid phenotypes"""
        return list(self.config["phenotypes"].keys())
    
    def get_valid_genes(self) -> List[str]:
        """Get list of valid genes"""
        return list(self.config["genes"].keys())
    
    def get_valid_treatments(self) -> List[str]:
        """Get list of valid treatments"""
        return list(self.config["treatments"].keys())
    
    def get_valid_zygosity(self) -> List[str]:
        """Get list of valid zygosity types"""
        return list(self.config["zygosity"].keys())


# Simple convenience functions for direct use
def validate_phenotype(value: str, validator: Optional[AnnotationValidator] = None) -> str:
    """Validate single phenotype value"""
    if validator is None:
        validator = AnnotationValidator()
    return validator.validate_phenotype(value)


def validate_gene(value: str, validator: Optional[AnnotationValidator] = None) -> str:
    """Validate single gene value"""
    if validator is None:
        validator = AnnotationValidator()
    return validator.validate_gene(value)


def get_phenotype_synonyms(canonical: str, validator: Optional[AnnotationValidator] = None) -> List[str]:
    """Get synonyms for a canonical phenotype"""
    if validator is None:
        validator = AnnotationValidator()
    return validator.config["phenotypes"].get(canonical, {}).get("synonyms", [])
```

**Integration with EmbryoAnnotations:**
```python
class EmbryoAnnotations:
    def __init__(self, sam2_path, annotations_path=None, validate=True, validator=None):
        self.validate = validate
        self.validator = validator or AnnotationValidator()
        # ... rest of init
    
    def add_phenotype(self, snip_id, phenotype, author, confidence=None):
        if self.validate:
            # Use validator to get canonical form and validate
            phenotype = self.validator.validate_phenotype(phenotype)
        
        # ... rest of method
```

**Benefits:**
- ✅ **Separation of concerns**: Config in config/, logic in utils/
- ✅ **Synonym support**: "edema" automatically becomes "EDEMA"  
- ✅ **Extensible**: Add new terms without touching code
- ✅ **Robust fallback**: Hardcoded defaults if JSON fails
- ✅ **Simple API**: Just validate_*() functions
- ✅ **Future-ready**: Can extend to ZFIN/HPO later

---

## **18. Atomic File Operations and Edge Case Handling** 🛡️

**Robust data persistence with atomic operations and comprehensive error handling**

### **18.1 Atomic Save Operations**

**Write-then-rename pattern for data safety:**

```python
def save(self, backup: bool = True):
    """Save metadata with atomic operations and validation"""
    
    # STEP 1: Pre-save validation
    current_entities = EntityIDTracker.extract_entities(self.data)
    EntityIDTracker.validate_hierarchy(current_entities, raise_on_violations=True)
    
    # STEP 2: Update metadata before save
    self.data["entity_tracking"]["metadata"] = {
        entity_type: list(ids) for entity_type, ids in current_entities.items()
    }
    self.data["file_info"]["last_updated"] = self.get_timestamp()
    
    # STEP 3: Atomic file operations
    if backup and self.filepath.exists():
        backup_path = self.filepath.with_suffix(f'.backup.{int(time.time())}.json')
        shutil.copy2(self.filepath, backup_path)
        
        # Cleanup old backups (keep last 5)
        self._cleanup_old_backups(max_keep=5)
    
    # STEP 4: Write to temporary file first
    temp_path = self.filepath.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # STEP 5: Atomic rename (OS guarantees atomicity)
        temp_path.rename(self.filepath)
        
        if self.verbose:
            embryo_count = len(self.data["embryos"])
            snip_count = len(self._snip_to_embryo)
            print(f"💾 Saved: {embryo_count} embryos, {snip_count} snips (validated)")
            
    except Exception as e:
        # Cleanup temporary file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Save failed: {e}")

def _cleanup_old_backups(self, max_keep: int = 5):
    """Remove old backup files, keeping only the most recent"""
    backup_pattern = f"{self.filepath.stem}.backup.*.json"
    backup_files = sorted(
        self.filepath.parent.glob(backup_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for old_backup in backup_files[max_keep:]:
        try:
            old_backup.unlink()
        except OSError:
            pass  # Ignore cleanup failures
```

### **18.2 Edge Case Validation and Recovery**

**Comprehensive input validation with helpful error messages:**

```python
def add_phenotype(self, embryo_id: str, phenotype: str, ...) -> dict:
    """Add phenotype with comprehensive edge case handling"""
    
    # EDGE CASE 1: Empty or invalid inputs
    if not embryo_id or not embryo_id.strip():
        raise ValueError("Embryo ID cannot be empty or whitespace-only")
    
    if not phenotype or not phenotype.strip():
        raise ValueError("Phenotype cannot be empty or whitespace-only")
    
    # EDGE CASE 2: Embryo existence validation
    if embryo_id not in self.data["embryos"]:
        available_embryos = list(self.data["embryos"].keys())[:5]  # Show sample
        suggestion = f"Available embryos (sample): {available_embryos}"
        
        # Check for common typos
        similar = [e for e in self.data["embryos"].keys() 
                  if embryo_id.lower() in e.lower() or e.lower() in embryo_id.lower()]
        if similar:
            suggestion += f". Did you mean: {similar[:3]}"
        
        raise ValueError(f"Embryo not found: {embryo_id}. {suggestion}")
    
    # EDGE CASE 3: Target specification validation with context
    if target == "":
        raise ValueError("Target cannot be empty string. Use 'all' for all snips")
    
    if isinstance(target, str) and ":" in target:
        # Validate range format
        parts = target.split(":")
        if len(parts) > 2:
            raise ValueError(f"Invalid range format: '{target}'. Use 'start:end', 'start:', or ':end'")
        
        # Check for invalid range values
        try:
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            
            if start is not None and end is not None and start > end:
                raise ValueError(f"Invalid range: start ({start}) > end ({end})")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Non-numeric value in range: '{target}'")
            raise
    
    # EDGE CASE 4: No available snips
    available_snips = list(self.data["embryos"][embryo_id]["snips"].keys())
    if not available_snips:
        raise ValueError(f"No snips available for embryo {embryo_id}. Check SAM2 data import.")
    
    # Continue with normal processing...
```

### **18.3 Data Corruption Recovery**

**Graceful handling of corrupted or incomplete data:**

```python
def load_with_recovery(self, filepath: Path) -> dict:
    """Load annotations with automatic recovery from corruption"""
    
    try:
        # ATTEMPT 1: Normal load
        with open(filepath) as f:
            data = json.load(f)
        
        # Basic structure validation
        required_keys = ["file_info", "embryos"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        
        return data
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Primary file corrupted: {e}")
        
        # ATTEMPT 2: Try backup files
        backup_files = sorted(
            filepath.parent.glob(f"{filepath.stem}.backup.*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )
        
        for backup_file in backup_files:
            try:
                print(f"🔄 Attempting recovery from {backup_file.name}")
                with open(backup_file) as f:
                    data = json.load(f)
                
                print(f"✅ Successfully recovered from backup")
                
                # Save recovered data to primary location
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return data
                
            except Exception as backup_error:
                print(f"❌ Backup recovery failed: {backup_error}")
                continue
        
        # ATTEMPT 3: Create minimal structure from SAM2 if available
        if hasattr(self, 'sam_annotation_path') and self.sam_annotation_path.exists():
            print(f"🏗️ Rebuilding from SAM2 source data")
            return self._create_from_sam2()
        
        # FAILURE: No recovery possible
        raise IOError(
            f"Cannot recover annotation data from {filepath}. "
            f"Tried primary file, {len(backup_files)} backups, and SAM2 rebuild. "
            f"Original error: {e}"
        )

def validate_and_repair_data_integrity(self) -> dict:
    """Comprehensive data integrity check with automatic repairs"""
    
    report = {
        "errors_found": [],
        "repairs_made": [],
        "warnings": []
    }
    
    # CHECK 1: Orphaned snips (snips without parent embryo data)
    for embryo_id, embryo_data in self.data["embryos"].items():
        for snip_id in list(embryo_data["snips"].keys()):
            try:
                expected_embryo = extract_embryo_id(snip_id)
                if expected_embryo != embryo_id:
                    # Repair: Move to correct embryo or remove
                    if expected_embryo in self.data["embryos"]:
                        self.data["embryos"][expected_embryo]["snips"][snip_id] = embryo_data["snips"][snip_id]
                        del embryo_data["snips"][snip_id]
                        report["repairs_made"].append(f"Moved orphaned snip {snip_id} to correct embryo {expected_embryo}")
                    else:
                        del embryo_data["snips"][snip_id]
                        report["repairs_made"].append(f"Removed orphaned snip {snip_id} (no matching embryo)")
                        
            except Exception as e:
                report["errors_found"].append(f"Invalid snip ID format: {snip_id} ({e})")
    
    # CHECK 2: Missing frame numbers in snip data
    for embryo_id, embryo_data in self.data["embryos"].items():
        for snip_id, snip_data in embryo_data["snips"].items():
            if "frame_number" not in snip_data:
                try:
                    frame_num = extract_frame_number(snip_id)
                    snip_data["frame_number"] = frame_num
                    report["repairs_made"].append(f"Added missing frame_number to {snip_id}")
                except Exception as e:
                    report["errors_found"].append(f"Cannot extract frame number from {snip_id}: {e}")
    
    # CHECK 3: DEAD phenotype consistency
    dead_violations = self._check_dead_conflicts()
    if dead_violations["conflicts"]:
        report["errors_found"].extend([
            f"DEAD coexistence violation in {conflict['embryo_id']}"
            for conflict in dead_violations["conflicts"]
        ])
    
    return report
```

### **18.4 Benefits of Robust Operations**

- ✅ **Data safety**: Atomic operations prevent partial writes and corruption
- ✅ **Automatic backup**: Configurable backup retention with automatic cleanup
- ✅ **Recovery capability**: Multiple fallback strategies for data recovery
- ✅ **Validation integration**: Pre-save validation prevents invalid data persistence
- ✅ **Helpful error messages**: Context-rich errors with suggestions and alternatives
- ✅ **Self-repair**: Automatic detection and fixing of common data inconsistencies
- ✅ **Graceful degradation**: System continues operating even with minor data issues

---

## **19. AnnotationBatch Integration** 📝

**For detailed batch implementation design, see:** [`annotation_batch_design.md`](./annotation_batch_design.md)

**Key Integration Points:**

**Batch Initialization from Metadata:**
```python
class EmbryoAnnotations:
    def initialize_batch(self, mode="skeleton", author="unknown", validate=True):
        """Create temporary annotation workspace from this metadata"""
        if mode == "skeleton":
            # Create empty annotation structure with embryo/snip hierarchy
            batch_data = self._create_skeleton_structure()
        elif mode == "copy":
            # Deep copy existing annotations for revision workflows  
            batch_data = copy.deepcopy(self.data)
        
        # Return isolated batch workspace
        return AnnotationBatch(batch_data, author, validate)
    
    def apply_batch(self, batch, on_conflict="error", dry_run=False):
        """Apply batch annotations to metadata with conflict resolution"""
        # Validate batch data + merge with specified conflict policy
        # See annotation_batch_design.md for detailed implementation
```

**Batch Class Overview:**
```python
class AnnotationBatch(EmbryoAnnotations):
    """Temporary workspace inheriting all validation methods"""
    def __init__(self, data_structure, author, validate=True):
        # Skip super().__init__() - manual setup for lightweight workspace
        self.data = data_structure              # From skeleton/copy
        self.validator = AnnotationValidator()   # Inherited validation needs
        # ... other required attributes
    
    def add_phenotype(self, embryo_id, phenotype, frames="all", **kwargs):
        # Immediate frame resolution + inherited validation
        snip_ids = self._resolve_frames(embryo_id, frames)  # "0200:0400" → snip list
        for snip_id in snip_ids:
            super().add_phenotype(snip_id, phenotype, self.author, **kwargs)
```

**Benefits:**
- ✅ **Safe isolation**: Batch never mutates metadata directly
- ✅ **Inherited validation**: Same business rules as direct operations
- ✅ **Frame range support**: "0200:0400" syntax for bulk operations  
- ✅ **Conflict resolution**: Explicit policies for team workflows

---

**Next Steps**: After understanding this structure, we'll implement the actual EmbryoAnnotations class and AnnotationBatch system based on the documented designs, incorporating death frame tracking, validation configuration, and batch workflow support.