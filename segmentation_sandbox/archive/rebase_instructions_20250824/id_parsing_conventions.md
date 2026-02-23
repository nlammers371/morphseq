# Entity ID Parsing Conventions & Guardrails

## Overview
Entity IDs in the pipeline follow a hierarchical structure with strict formatting rules. The key principle is to **parse backwards** from the end of the ID string because experiment IDs can contain arbitrary underscores and be complex.

## GUARDRAILS & REQUIREMENTS

### ✅ MUST Requirements
1. **All experiment IDs MUST start with YYYYMMDD date format** (20XXXXXX)
2. **Date MUST be in 21st century** (2000-2099) to avoid conflicts
3. **Well IDs MUST follow [A-H][01-99] format** (A01, B12, H07, etc.)
4. **Frame numbers MUST be 3-4 digits** (000, 0042, 1234)
5. **No file paths, URLs, or special characters** in entity IDs
6. **Minimum length 8 characters** (simple date: 20240506)

### ❌ FORBIDDEN Patterns
- **File paths**: `/path/to/file`, `C:\Windows\path`, `./relative/path`
- **URLs**: `http://example.com`, `ftp://server.com`
- **Hidden files**: `.hidden_file`, `..parent_dir`
- **Non-date prefixes**: `experiment_20240506`, `ctrl_group_1`
- **Wrong century**: `19990101` (too old), `21000101` (too far future)
- **Invalid wells**: `I01` (no I row), `A00` (no 00 column), `Z99` (no Z row)
- **Too short**: `2024`, `ctrl`, `exp1`
- **Special chars**: `20240506@test`, `exp#1`, `data&analysis`

## Valid ID Examples by Complexity

### Simple IDs
- `20240506` - Date only experiment
- `20240506_A01` - Minimal video  
- `20240506_A01_t0001` - Minimal image
- `20240506_A01_e01` - Minimal embryo

### Named IDs  
- `20240506_ctrl` - Named experiment
- `20240506_ctrl_A01` - Named video
- `20240506_ctrl_A01_t0042` - Named image
- `20240506_ctrl_A01_e01_s0042` - Named snip

### Complex IDs (Current Standard)
- `20250703_chem3_28C_T00_1325` - Full experiment
- `20250703_chem3_28C_T00_1325_G04` - Full video
- `20250703_chem3_28C_T00_1325_G04_t0000` - Full image
- `20250703_chem3_28C_T00_1325_G04_e01` - Full embryo

## ID Structure Hierarchy

```
experiment_id (MUST start with YYYYMMDD)
    └── video_id = {experiment_id}_{WELL}
            ├── image_id = {video_id}_t{FRAME}
            └── embryo_id = {video_id}_e{NN}
                    └── snip_id = {embryo_id}_{FRAME} or {embryo_id}_s{FRAME}
```

## Parsing Rules (MUST Parse Backwards!)

### 1. video_id
- **Format**: `{experiment_id}_{WELL}`
- **WELL Pattern**: `[A-H][01-99]` (A01-H99, always at the END)
- **Guardrails**: Well MUST be valid plate position (A-H rows, 01-99 columns)
- **Examples**:
  - `20240411_A01` → experiment_id: `20240411`, well: `A01`
  - `20250529_36hpf_ctrl_atf6_A01` → experiment_id: `20250529_36hpf_ctrl_atf6`, well: `A01`
  - `20250624_chem02_28C_T00_1356_H01` → experiment_id: `20250624_chem02_28C_T00_1356`, well: `H01`

### 2. image_id
- **Format**: `{video_id}_t{FRAME}` (NEW: 't' prefix for future-proofing)
- **FRAME Pattern**: `_t[0-9]{3,4}` (t + 3-4 digits at the END)
- **Guardrails**: Frame number MUST be 000-9999, 't' prefix REQUIRED
- **Examples**:
  - `20240411_A01_t0042` → video_id: `20240411_A01`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_t0042` → video_id: `20250529_36hpf_ctrl_atf6_A01`, frame: `0042`
  
**IMPORTANT TRANSITION NOTE**: We are transitioning from `{video_id}_{FRAME}` to `{video_id}_t{FRAME}` format. The 't' prefix makes parsing significantly easier by disambiguating image IDs from other entity types that might end with numbers. This prevents conflicts with complex experiment IDs that may contain trailing numbers.

### 3. embryo_id
- **Format**: `{video_id}_e{NN}`
- **Embryo Pattern**: `_e[0-9]+` (at the END)
- **Guardrails**: Embryo number MUST be 1-999, 'e' prefix REQUIRED
- **Examples**:
  - `20240411_A01_e01` → video_id: `20240411_A01`, embryo_number: `01`
  - `20250529_36hpf_ctrl_atf6_A01_e03` → video_id: `20250529_36hpf_ctrl_atf6_A01`, embryo_number: `03`

### 4. snip_id
- **Format**: `{embryo_id}_{FRAME}` OR `{embryo_id}_s{FRAME}`
- **FRAME Pattern**: `_s?[0-9]{3,4}` (optional 's' prefix, at the END)
- **Guardrails**: Frame MUST be 000-9999, 's' prefix OPTIONAL
- **Examples**:
  - `20240411_A01_e01_s0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20240411_A01_e01_0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_e01_s0042` → embryo_id: `20250529_36hpf_ctrl_atf6_A01_e01`, frame: `0042`

## Validation & Error Prevention

### Common Mistakes to Avoid
- ❌ `experiment_20240506` - Don't prefix with words
- ❌ `20240506_I01` - No 'I' row in well plates  
- ❌ `20240506_A00` - Wells start at 01, not 00
- ❌ `20240506_A01_42` - Missing 't' prefix for images
- ❌ `19990101_test` - Wrong century
- ❌ `/data/20240506` - File paths not allowed

### Regex Patterns for Validation
```regex
experiment_id: ^20\d{6}(_\w+)*$
video_id:      ^20\d{6}(_\w+)*_[A-H](0[1-9]|[1-9]\d)$
image_id:      ^20\d{6}(_\w+)*_[A-H](0[1-9]|[1-9]\d)_t\d{3,4}$
embryo_id:     ^20\d{6}(_\w+)*_[A-H](0[1-9]|[1-9]\d)_e\d+$
snip_id:       ^20\d{6}(_\w+)*_[A-H](0[1-9]|[1-9]\d)_e\d+_s?\d{3,4}$
```

## Parsing Strategy

Always parse from the END of the string:

1. **For video_id**: Look for `_[A-H](0[1-9]|[1-9]\d)$` at the end
2. **For image_id**: Look for `_t\d{3,4}$` at the end
3. **For embryo_id**: Look for `_e\d+$` at the end
4. **For snip_id**: Look for `_s?\d{3,4}$` at the end

## EntityIDTracker Role & Data Integrity

The `EntityIDTracker` class enforces these conventions and provides:

### Data Validation
- **Automatic filtering** of file paths, URLs, and invalid formats
- **Regex-based validation** using the patterns above
- **Hierarchy checking** to ensure parent-child relationships are valid
- **Orphan detection** to find entities without valid parents

### Quality Assurance
- **Real-time validation** during metadata loading
- **Clear error messages** when violations are found
- **Automatic cleanup** of invalid or orphaned entities
- **Change tracking** between pipeline runs

### Usage in Pipeline
```python
# Extract entities from metadata
entities = EntityIDTracker.extract_entities(metadata)

# Validate hierarchy
result = EntityIDTracker.validate_hierarchy(entities)
if not result["valid"]:
    print("Violations found:", result["violations"])

# Clean up orphaned entities
cleaned = EntityIDTracker.remove_orphaned_entities(entities)
```

## Common Pitfalls to Avoid

❌ **DON'T** try to parse experiment_id first by looking for patterns
❌ **DON'T** assume experiment_id follows any specific format beyond YYYYMMDD prefix
❌ **DON'T** split on underscores and assume positions
❌ **DON'T** allow non-YYYYMMDD prefixes (e.g., "experiment_20240506")
❌ **DON'T** use invalid well positions (I01, A00, Z99)

✅ **DO** always work backwards from the end
✅ **DO** use the specific end patterns for each entity type  
✅ **DO** handle complex experiment IDs like `20250624_chem02_28C_T00_1356`
✅ **DO** enforce YYYYMMDD prefix for all experiment IDs
✅ **DO** validate well positions are A01-H99

## Benefits of These Guardrails

1. **Prevents parsing ambiguity** - YYYYMMDD prefix makes experiment boundaries clear
2. **Catches data entry errors** - Invalid wells/frames rejected early
3. **Avoids file path confusion** - Strict patterns prevent metadata corruption
4. **Enables reliable automation** - Consistent format allows robust pipeline processing
5. **Future-proofs the system** - 21st century dates give us 80+ years of runway

## Examples of Complex Parsing

```python
# Complex experiment ID with multiple underscores
snip_id = "20250624_chem02_28C_T00_1356_H01_e01_s034"

# Parse backwards:
# 1. Find frame: _s034 → frame = "034", remainder = "20250624_chem02_28C_T00_1356_H01_e01"
# 2. Find embryo: _e01 → embryo_number = "01", remainder = "20250624_chem02_28C_T00_1356_H01"
# 3. Find well: _H01 → well = "H01", remainder = "20250624_chem02_28C_T00_1356"
# 4. What's left is experiment_id: "20250624_chem02_28C_T00_1356"

# Result:
{
    'experiment_id': '20250624_chem02_28C_T00_1356',
    'well_id': 'H01',
    'video_id': '20250624_chem02_28C_T00_1356_H01',
    'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
    'embryo_number': '01',
    'frame_number': '034',
    'snip_id': '20250624_chem02_28C_T00_1356_H01_e01_s034'
}
```