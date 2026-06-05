# analyze.utils.coord

`analyze.utils.coord` is the coordinate-frame layer for morphseq. It defines how
masks, images, and downstream analyses move between raw image space, canonical
grid space, and explicit registration space.

## What this package is for

This package exists to keep coordinate handling explicit and auditable:

- define stable coordinate-frame contracts
- capture transformation provenance
- separate canonicalization from registration
- provide reusable affine / grid transform primitives

It is used by downstream pipelines that need a consistent geometry frame, such as
canonical OT morphometrics and other grid-based analyses.

## Package map

- [types.py](types.py) — dataclasses and result containers for coordinate outputs
- [transforms.py](transforms.py) — affine/grid transform primitives and chains
- [register.py](register.py) — explicit Stage 2 registration into a fixed frame
- [grids/](grids/) — canonical grid mapping utilities and grid-specific logic
- [grids/back_direction.py](grids/back_direction.py) — yolk-centered back-point helper used by canonical alignment

## What the API looks like

There are two main entry points:

1. canonicalization: map a mask, image, or frame into the canonical grid
2. registration: explicitly align one already-canonical mask to another fixed mask

The package root exports the common data types and the batch mask helper:

- BoxYX
- CanonicalGrid
- CanonicalMaskResult
- CanonicalImageResult
- CanonicalFrameResult
- GridTransform
- TransformChain
- to_canonical_grid_mask_batch(...)

The single-item canonicalization helpers live in [grids/canonical.py](grids/canonical.py):

- to_canonical_grid_mask(...)
- to_canonical_grid_image(...)
- to_canonical_grid_frame(...)

Explicit registration lives in [register.py](register.py):

- register_to_fixed(...)

## How someone would actually use this

### Step 1 — decide which operation you need

Use canonicalization when you want to standardize an embryo into the package’s
canonical orientation and scale.

Use explicit registration when you already have two masks in the same frame and
want to align the moving one to the fixed one.

### Step 2 — canonicalize a mask

The most common call is to canonicalize a binary embryo mask, optionally with a
yolk mask:

- pass the embryo mask
- pass um_per_px for the original image
- optionally pass yolk_mask
- optionally customize the grid through CanonicalGridConfig

The result is a CanonicalMaskResult containing:

- mask: the canonicalized mask array
- grid: the canonical grid descriptor
- transform_chain: the transform provenance
- meta: alignment decisions and QC-style metadata
- content_bbox_yx: a tight bounding box in canonical coordinates

Example flow:

1. load a raw mask and optional yolk mask
2. call to_canonical_grid_mask(...)
3. use result.mask for downstream analysis
4. inspect result.meta if you want to know why it oriented the way it did

### Step 3 — canonicalize a full frame

If you have an image and a mask together, use to_canonical_grid_frame(...).

That function chooses the right path automatically:

- if the frame has a mask, it uses segmentation-driven canonicalization
- if it only has an image, it does an image-only scale-and-center mapping

It returns a CanonicalFrameResult with a rewritten Frame object and the same
transform provenance fields.

### Step 4 — work in batches when needed

If you already have a list of masks, use to_canonical_grid_mask_batch(...).

This is the easiest way to canonicalize a cohort with the same settings.

It returns a list of CanonicalMaskResult objects, one per input mask.

### Step 5 — inspect the metadata, not just the pixels

Each canonical result carries a meta dictionary that records:

- whether canonicalization was applied
- the chosen rotation and flip
- the anchor shift
- whether yolk-aware orientation was used
- clipping or fallback conditions

That metadata is the audit trail. If something looks off, start there.

### Step 6 — use the transform chain downstream

The transform_chain field is the programmatic record of the mapping.
Downstream code can use it to:

- reproduce the same warp
- compose additional transforms
- keep coordinate provenance explicit

### Step 7 — use registration only when you mean registration

register_to_fixed(...) is for the second stage of alignment.
It compares a moving mask to a fixed mask and returns a RegisterResult that
contains:

- transform: the transform chain
- applied: whether a transform was accepted
- meta: registration diagnostics
- moving_in_fixed: the registered mask, if apply=True

## Minimal API Examples

Here is exactly what the code looks like for the main operations.

### 1. Canonicalizing a raw embryo mask

When you have a raw mask (e.g. from the segmenter) and its physical pixel size:

```python
import numpy as np
from analyze.utils.coord.grids.canonical import (
    to_canonical_grid_mask,
    CanonicalGridConfig,
)

# 1. Prepare inputs (e.g. loaded via OpenCV or skimage)
raw_mask = np.zeros((1024, 1024), dtype=np.uint8)  # Sample embryo mask
raw_yolk = np.zeros((1024, 1024), dtype=np.uint8)  # Optional yolk mask
um_per_px = 1.3  # known resolution of the raw image

# 2. (Optional) configure grid behavior
#    Defaults to 256x576 at 10 um/px, orienting by the yolk
cfg = CanonicalGridConfig(
    grid_shape_hw=(256, 576),
    reference_um_per_pixel=10.0,
    align_mode="yolk",
)

# 3. Call the mapping function
result = to_canonical_grid_mask(
    raw_mask,
    um_per_px=um_per_px,
    yolk_mask=raw_yolk,
    cfg=cfg
)

# 4. Use the result
canonical_mask_array = result.mask            # The standardized 256x576 mask
print(result.meta["rotation_deg"])            # See how it was rotated
print(result.transform_chain)                 # Programmatic sequence of transforms
```

### 2. Canonicalizing a full frame

If you need to move the brightfield image along with its mask, wrap them in a `Frame`.

```python
from analyze.utils.coord import Frame
from analyze.utils.coord.grids.canonical import to_canonical_grid_frame

# Group inputs into a single Frame
my_frame = Frame(
    image=raw_brightfield,
    mask=raw_mask,
    yolk_mask=raw_yolk,
    um_per_px=1.3,
)

# Map the whole frame
# (The image will be transformed using the exact parameters determined from the mask)
frame_result = to_canonical_grid_frame(my_frame)

canonical_image = frame_result.frame.image
canonical_mask = frame_result.frame.mask
```

### 3. Registering explicitly aligned frames ("Stage 2")

When you want to fine-tune alignment between two *already canonical* masks (for instance, registering an embryo to a developmental template):

```python
from analyze.utils.coord.register import register_to_fixed, RegisterConfig

# Configure explicit registration behavior
reg_cfg = RegisterConfig(
    mode="rotate_then_pivot_translate",
    angle_step_deg=1.0,
)

# Register moving into fixed
reg_result = register_to_fixed(
    moving=current_canonical_mask,
    fixed=template_mask,
    cfg=reg_cfg,
    apply=True  # so it actually warps the moving mask and returns it
)

if reg_result.applied:
    warped_mask = reg_result.moving_in_fixed
    dyx = reg_result.meta["register_to_fixed"]["translate_dyx"]
    angle = reg_result.meta["register_to_fixed"]["angle_deg"]
```

## Canonicalization vs registration

The package uses a two-stage mental model:

1. **Stage 1: canonicalization**
   - handled by [grids/canonical.py](grids/canonical.py)
   - maps raw embryo masks/images into a canonical grid
   - for embryo-specific cases, this includes yolk-aware orientation logic

2. **Stage 2: explicit registration**
   - handled by [register.py](register.py)
   - registers one already-prepared mask into another fixed mask/frame
   - never runs implicitly inside the OT solver

The embryo canonical aligner also uses a small helper in
[grids/back_direction.py](grids/back_direction.py) to derive the back point
from the yolk-centered sampling disk.

## Public surface

The package entrypoint [__init__.py](__init__.py) exports the core coordinate
contracts and helper types used by downstream code.

The most important public concepts are:

- `CanonicalGrid`
- `CanonicalFrameResult`
- `CanonicalImageResult`
- `CanonicalMaskResult`
- `GridTransform`
- `TransformChain`
- `register_to_fixed(...)`

## Coordinate conventions

- Coordinates are generally stored in **yx** order in metadata and results.
- OpenCV affine transforms use **xy** matrix conventions internally.
- Canonical outputs are written onto a fixed canonical grid shape.
- Transform provenance is preserved in the result metadata and transform chain.

## Reading order

1. [__init__.py](__init__.py)
2. [types.py](types.py)
3. [transforms.py](transforms.py)
4. [grids/canonical.py](grids/canonical.py)
5. [register.py](register.py)

## What this package is not

- It is not the OT solver layer.
- It is not the embryo-specific OT wrapper layer.
- It is not the place to hide implicit alignment behavior.

If you are looking for canonical embryo alignment, start with
[grids/canonical.py](grids/canonical.py). If you are looking for explicit
frame-to-frame registration, start with [register.py](register.py).
