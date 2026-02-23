# Entity ID Parsing Conventions & Guardrails

## Overview
Entity IDs in the pipeline follow a hierarchical structure with strict formatting rules. The key principle is to **parse backwards** from the end of the ID string because experiment IDs can contain arbitrary underscores and be complex.

**NEW**: All parsing patterns are now centralized in `parsing_utils.py` as a single source of truth.

## GUARDRAILS & REQUIREMENTS

### ✅ MUST Requirements
1. **All experiment IDs MUST start with YYYYMMDD date format** (20XXXXXX)
2. **Date MUST be in 21st century** (2000-2099) to avoid conflicts
3. **Well IDs MUST follow [A-H]\d{2} format** (A00-H99, including A00)
4. **Frame numbers MUST be 3-4 digits** (000, 0042, 1234)
5. **Channels MUST be zero-padded** (00, 01, 02, etc.)
6. **No file paths, URLs, or special characters** in entity IDs
7. **Minimum length 8 characters** (simple date: 20240506)

### ❌ FORBIDDEN Patterns
- **File paths**: `/path/to/file`, `C:\Windows\path`, `./relative/path`
- **URLs**: `http://example.com`, `ftp://server.com`
- **Hidden files**: `.hidden_file`, `..parent_dir`
- **Non-date prefixes**: `experiment_20240506`, `ctrl_group_1`
- **Wrong century**: `19990101` (too old), `21000101` (too far future)
- **Invalid wells**: `I01` (no I row), `Z99` (no Z row)
- **Too short**: `2024`, `ctrl`, `exp1`
- **Special chars**: `20240506@test`, `exp#1`, `data&analysis`

## Valid ID Examples by Complexity

### Simple IDs
- `20240506` - Date only experiment
- `20240506_A00` - Video with A00 well (A00 is now valid)
- `20240506_A01_ch00_t0001` - Image with channel and 't' prefix
- `20240506_A01_e01` - Minimal embryo

### Named IDs  
- `20240506_ctrl` - Named experiment
- `20240506_ctrl_A01` - Named video
- `20240506_ctrl_A01_ch01_t0042` - Named image with channel
- `20240506_ctrl_A01_t0042` - Named image (legacy format)
- `20240506_ctrl_A01_e01_s0042` - Named snip

### Complex IDs (Current Standard)
- `20250703_chem3_28C_T00_1325` - Full experiment
- `20250703_chem3_28C_T00_1325_G04` - Full video
- `20250703_chem3_28C_T00_1325_G04_ch00_t0000` - Full image with channel
- `20250703_chem3_28C_T00_1325_G04_e01` - Full embryo

## ID Structure Hierarchy

```
experiment_id (MUST start with YYYYMMDD)
    └── video_id = {experiment_id}_{WELL}
            ├── image_id = {video_id}_ch{CHANNEL}_t{FRAME} or {video_id}_t{FRAME}
            └── embryo_id = {video_id}_e{NN}
                    └── snip_id = {embryo_id}_{FRAME} or {embryo_id}_s{FRAME}
```

## Centralized Parsing Patterns (Single Source of Truth)

The `parsing_utils.py` module now contains all regex patterns in one place:

```python
# Core component patterns
WELL_PATTERN = r'[A-H]\d{2}'                    # A00-H99 consistently everywhere
FRAME_PATTERN = r'\d{3,4}'                      # 3-4 digit frames  
EMBRYO_NUM_PATTERN = r'\d+'                     # Embryo numbers
CHANNEL_PATTERN = r'\d+'                        # Channel numbers

# End patterns for backward parsing
WELL_END_PATTERN = rf'_({WELL_PATTERN})$'       # _A01$ at end
EMBRYO_END_PATTERN = rf'_e({EMBRYO_NUM_PATTERN})$'  # _e01$ at end
SNIP_END_PATTERN = rf'_s?({FRAME_PATTERN})$'    # _s0042$ or _0042$ at end  
IMAGE_CH_END_PATTERN = rf'_ch({CHANNEL_PATTERN})_t({FRAME_PATTERN})$'  # _ch00_t0042$ at end
IMAGE_LEGACY_END_PATTERN = rf'_t({FRAME_PATTERN})$'  # _t0042$ at end

# Detection patterns for get_entity_type  
HAS_WELL_PATTERN = rf'_{WELL_PATTERN}$'         # Ends with well
HAS_EMBRYO_PATTERN = rf'_e{EMBRYO_NUM_PATTERN}$'  # Ends with embryo
HAS_SNIP_PATTERN = rf'_s?{FRAME_PATTERN}$'      # Ends with snip frame
HAS_IMAGE_CH_PATTERN = rf'_ch{CHANNEL_PATTERN}_t{FRAME_PATTERN}$'  # Ends with ch+frame
HAS_IMAGE_LEGACY_PATTERN = rf'_t{FRAME_PATTERN}$'  # Ends with frame only
```

**Benefits**: 
- Change well range once → applies everywhere
- No more contradictory patterns between functions
- Easy maintenance and updates

## Parsing Rules (MUST Parse Backwards!)

### 1. video_id
- **Format**: `{experiment_id}_{WELL}`
- **WELL Pattern**: `[A-H]\d{2}` (A00-H99, always at the END)
- **Guardrails**: Well MUST be valid plate position (A-H rows, 00-99 columns)
- **Examples**:
  - `20240411_A00` → experiment_id: `20240411`, well: `A00`
  - `20250529_36hpf_ctrl_atf6_A01` → experiment_id: `20250529_36hpf_ctrl_atf6`, well: `A01`
  - `20250624_chem02_28C_T00_1356_H01` → experiment_id: `20250624_chem02_28C_T00_1356`, well: `H01`

### 2. image_id
- **Format**: `{video_id}_ch{CHANNEL}_t{FRAME}` (preferred) or `{video_id}_t{FRAME}` (legacy)
- **CHANNEL Pattern**: `_ch\d+` (channel with zero-padding, e.g., ch00, ch01)
- **FRAME Pattern**: `_t\d{3,4}` (t + 3-4 digits at the END)
- **Guardrails**: Frame number MUST be 0000-9999, channel MUST be 00-99, 't' prefix REQUIRED
- **Examples**:
  - `20240411_A01_ch00_t0042` → video_id: `20240411_A01`, channel: `00`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_ch01_t0042` → video_id: `20250529_36hpf_ctrl_atf6_A01`, channel: `01`, frame: `0042`
  - `20240411_A01_t0042` → video_id: `20240411_A01`, frame: `0042` (legacy format, channel defaults to `00`)

**CURRENT STATE**: The implementation supports both channel-inclusive format (`_ch00_t0042`) and legacy format without channel (`_t0042`). New image IDs should use the channel format for clarity. **Legacy format now consistently returns zero-padded channel `"00"` instead of `"0"`.**

### 3. embryo_id
- **Format**: `{video_id}_e{NN}`
- **Embryo Pattern**: `_e\d+` (at the END)
- **Guardrails**: Embryo number MUST be 1-999, 'e' prefix REQUIRED
- **Examples**:
  - `20240411_A01_e01` → video_id: `20240411_A01`, embryo_number: `01`
  - `20250529_36hpf_ctrl_atf6_A01_e03` → video_id: `20250529_36hpf_ctrl_atf6_A01`, embryo_number: `03`

### 4. snip_id
- **Format**: `{embryo_id}_{FRAME}` OR `{embryo_id}_s{FRAME}`
- **FRAME Pattern**: `_s?\d{3,4}` (optional 's' prefix, at the END)
- **Guardrails**: Frame MUST be 000-9999, 's' prefix OPTIONAL
- **Examples**:
  - `20240411_A01_e01_s0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20240411_A01_e01_0042` → embryo_id: `20240411_A01_e01`, frame: `0042`
  - `20250529_36hpf_ctrl_atf6_A01_e01_s0042` → embryo_id: `20250529_36hpf_ctrl_atf6_A01_e01`, frame: `0042`

## Validation & Error Prevention

### Common Mistakes to Avoid
- ❌ `experiment_20240506` - Don't prefix with words
- ❌ `20240506_I01` - No 'I' row in well plates  
- ❌ `20240506_A01_42` - Missing 't' prefix for images
- ❌ `20240506_A01_t42` - Frame should be 4 digits (t0042)
- ❌ `20240506_A01_ch1_t42` - Channel should be 2 digits (ch01)
- ❌ `19990101_test` - Wrong century
- ❌ `/data/20240506` - File paths not allowed

### Current Regex Patterns (Centralized)
```regex
experiment_id: ^20\d{6}(_\w+)*$
video_id:      ^20\d{6}(_\w+)*_[A-H]\d{2}$
image_id:      ^20\d{6}(_\w+)*_[A-H]\d{2}_(ch\d+_)?t\d{3,4}$
embryo_id:     ^20\d{6}(_\w+)*_[A-H]\d{2}_e\d+$
snip_id:       ^20\d{6}(_\w+)*_[A-H]\d{2}_e\d+_s?\d{3,4}$
```

## Parsing Strategy

Always parse from the END of the string:

1. **For video_id**: Look for `_[A-H]\d{2}$` at the end
2. **For image_id**: Look for `_(ch\d+_)?t\d{3,4}$` at the end (channel + 't' or just 't')
3. **For embryo_id**: Look for `_e\d+$` at the end
4. **For snip_id**: Look for `_s?\d{3,4}$` at the end (but ensure it has `_e` earlier)

## parsing_utils.py Functions

The `parsing_utils.py` module provides core ID parsing functionality:

### Main Functions
- **`parse_entity_id()`** - Auto-detect and parse any ID type, returns dict with all components
- **`get_entity_type()`** - Determine ID type ('experiment', 'video', 'image', 'embryo', 'snip')
- **`build_image_id()`** - Create image ID with channel + 't' prefix
- **`build_video_id()`, `build_embryo_id()`, `build_snip_id()`** - Builder functions

### Extraction Functions
- **`extract_frame_number()`** - Get frame from image/snip ID
- **`extract_experiment_id()`** - Get experiment ID from any child ID
- **`extract_video_id()`** - Get video ID from image/embryo/snip ID
- **`extract_embryo_id()`** - Get embryo ID from snip ID

### Path Utilities
- **`get_image_filename_from_id()`** - Convert image_id to filename.jpg
- **`get_relative_image_path()`** - Get relative path: images/video_id/filename.jpg
- **`build_image_path_from_base()`** - Build full path from base directory

### Validation Functions  
- **`is_valid_well_id()`** - Check if well is in A00-H99 range
- **`is_valid_experiment_id()`** - Check if ID could be valid experiment
- **`validate_id_format()`** - Check ID matches expected type

### Usage Examples
```python
# Parse any entity ID
result = parse_entity_id("20250624_chem02_28C_T00_1356_H01_ch00_t0042")
# Returns: {'experiment_id': '20250624_chem02_28C_T00_1356', 'video_id': '...', 
#           'channel': '00', 'frame_number': '0042', 'entity_type': 'image', ...}

# Build image ID with channel
image_id = build_image_id("20240411_A01", 42, channel=1)
# Returns: "20240411_A01_ch01_t0042"

# Get image path
path = get_relative_image_path("20240411_A01_ch00_t0042")
# Returns: "images/20240411_A01/20240411_A01_ch00_t0042.jpg"

# Check if A00 is valid (now True!)
is_valid_well_id("A00")  # Returns: True
```

## Common Pitfalls to Avoid

❌ **DON'T** try to parse experiment_id first by looking for patterns
❌ **DON'T** assume experiment_id follows any specific format beyond YYYYMMDD prefix
❌ **DON'T** split on underscores and assume positions
❌ **DON'T** allow non-YYYYMMDD prefixes (e.g., "experiment_20240506")
❌ **DON'T** use invalid well positions (I01, Z99) - but A00 is now valid!

✅ **DO** always work backwards from the end
✅ **DO** use the centralized patterns from `parsing_utils.py`
✅ **DO** handle complex experiment IDs like `20250624_chem02_28C_T00_1356`
✅ **DO** enforce YYYYMMDD prefix for all experiment IDs
✅ **DO** validate well positions are A00-H99 (including A00)
✅ **DO** use zero-padded channels ("00", "01", not "0", "1")

## Benefits of These Guardrails

1. **Prevents parsing ambiguity** - YYYYMMDD prefix makes experiment boundaries clear
2. **Catches data entry errors** - Invalid wells/frames rejected early
3. **Avoids file path confusion** - Strict patterns prevent metadata corruption
4. **Enables reliable automation** - Consistent format allows robust pipeline processing
5. **Future-proofs the system** - 21st century dates give us 80+ years of runway
6. **Single source of truth** - Centralized patterns eliminate inconsistencies
7. **Easy maintenance** - Change pattern once, applies everywhere

## Examples of Complex Parsing

```python
# Complex experiment ID with multiple underscores
snip_id = "20250624_chem02_28C_T00_1356_H01_e01_s034"

# Parse backwards using centralized patterns:
# 1. Find frame: _s034 → frame = "034", remainder = "20250624_chem02_28C_T00_1356_H01_e01"
# 2. Find embryo: _e01 → embryo_number = "01", remainder = "20250624_chem02_28C_T00_1356_H01"
# 3. Find well: _H01 → well = "H01", remainder = "20250624_chem02_28C_T00_1356"
# 4. What's left is experiment_id: "20250624_chem02_28C_T00_1356"

# Result using parse_entity_id():
{
    'experiment_id': '20250624_chem02_28C_T00_1356',
    'well_id': 'H01',
    'video_id': '20250624_chem02_28C_T00_1356_H01',
    'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
    'embryo_number': '01',
    'frame_number': '034',
    'snip_id': '20250624_chem02_28C_T00_1356_H01_e01_s034',
    'entity_type': 'snip'
}
```

## Migration Notes

### Changes from Previous Version:
1. **A00 wells are now valid** - Previously forbidden, now supported consistently
2. **Channel handling is consistent** - Legacy format returns "00" not "0" 
3. **Centralized patterns** - All regex moved to single source of truth
4. **Well range updated** - A00-H99 instead of A01-H99
5. **EntityIDTracker replaced** - Use `parsing_utils.py` functions instead