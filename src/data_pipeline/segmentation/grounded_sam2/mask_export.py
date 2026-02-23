"""
SAM2 Mask Export to PNG

Export SAM2 segmentation masks as integer-labeled PNG images.
Each pixel value corresponds to an embryo instance (0=background, 1=embryo_1, etc.).

Key Functions:
    - export_frame_masks: Export masks for a single frame
    - export_all_masks: Batch export masks for all frames
    - create_labeled_mask_image: Combine multiple masks into labeled PNG

Example Usage:
    ```python
    from pathlib import Path
    import numpy as np

    # Single frame with multiple embryo masks
    frame_masks = {
        "embryo_0": np.array([[0, 0, 1, 1], [0, 0, 1, 1]]),
        "embryo_1": np.array([[0, 1, 0, 0], [0, 1, 0, 0]]),
    }

    # Export as labeled PNG
    export_frame_masks(
        frame_masks=frame_masks,
        output_path=Path("masks/frame_0000_masks.png"),
        image_id="exp_A01_t0000"
    )

    # Batch export all frames
    all_results = {
        0: {"embryo_0": mask_array_0, "embryo_1": mask_array_1},
        1: {"embryo_0": mask_array_2, "embryo_1": mask_array_3},
    }

    export_all_masks(
        results=all_results,
        output_dir=Path("masks"),
        image_id_template="exp_A01_t{frame:04d}"
    )
    ```

Output Format:
    PNG files with integer labels:
    - 0: Background
    - 1: Embryo 1
    - 2: Embryo 2
    - ... etc.

    Filename convention: {image_id}_masks.png
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def create_labeled_mask_image(
    embryo_masks: Dict[str, np.ndarray],
    height: Optional[int] = None,
    width: Optional[int] = None
) -> np.ndarray:
    """
    Combine multiple binary masks into single integer-labeled image.

    Args:
        embryo_masks: Dict mapping embryo_id to binary mask array
        height: Output height (inferred from masks if not provided)
        width: Output width (inferred from masks if not provided)

    Returns:
        Integer-labeled mask array (H, W) with values 0, 1, 2, ...

    Example:
        >>> masks = {
        ...     "embryo_0": np.array([[0, 0, 1], [0, 0, 1]]),
        ...     "embryo_1": np.array([[1, 1, 0], [1, 1, 0]]),
        ... }
        >>> labeled = create_labeled_mask_image(masks)
        >>> labeled
        array([[1, 1, 2],
               [1, 1, 2]], dtype=uint8)
    """
    if not embryo_masks:
        # Empty masks
        if height is None or width is None:
            raise ValueError("Must provide height/width for empty masks")
        return np.zeros((height, width), dtype=np.uint8)

    # Infer dimensions from first mask if not provided
    first_mask = next(iter(embryo_masks.values()))
    if height is None:
        height = first_mask.shape[0]
    if width is None:
        width = first_mask.shape[1]

    # Create output array (0 = background)
    labeled_mask = np.zeros((height, width), dtype=np.uint8)

    # Sort embryo IDs for consistent labeling
    sorted_embryos = sorted(embryo_masks.keys())

    # Assign each embryo a unique label (1, 2, 3, ...)
    for label_idx, embryo_id in enumerate(sorted_embryos, start=1):
        mask = embryo_masks[embryo_id]

        # Resize if needed
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        # Assign label where mask is positive
        labeled_mask[mask > 0] = label_idx

    return labeled_mask


def export_frame_masks(
    frame_masks: Dict[str, np.ndarray],
    output_path: Path,
    image_id: str
) -> Path:
    """
    Export masks for a single frame as labeled PNG.

    Args:
        frame_masks: Dict mapping embryo_id to mask array
        output_path: Path for output PNG file
        image_id: Image ID for metadata

    Returns:
        Path to created PNG file

    Raises:
        ValueError: If frame_masks is empty or invalid

    Example:
        >>> masks = {"embryo_0": np.ones((100, 100), dtype=np.uint8)}
        >>> path = export_frame_masks(
        ...     masks,
        ...     Path("output/frame_masks.png"),
        ...     "exp_A01_t0000"
        ... )
        >>> path.exists()
        True
    """
    if not frame_masks:
        raise ValueError(f"No masks to export for {image_id}")

    # Create labeled mask image
    labeled_mask = create_labeled_mask_image(frame_masks)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PNG
    cv2.imwrite(str(output_path), labeled_mask)

    return output_path


def export_all_masks(
    results: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    image_id_template: str = "{frame:04d}",
    filename_template: str = "{image_id}_masks.png"
) -> List[Path]:
    """
    Batch export masks for all frames.

    Args:
        results: Dict mapping frame index to embryo masks
        output_dir: Output directory for PNG files
        image_id_template: Template for generating image IDs (format with frame=idx)
        filename_template: Template for filenames (format with image_id=id)

    Returns:
        List of paths to created PNG files

    Example:
        >>> results = {
        ...     0: {"embryo_0": np.ones((100, 100), dtype=np.uint8)},
        ...     1: {"embryo_0": np.ones((100, 100), dtype=np.uint8)},
        ... }
        >>> paths = export_all_masks(
        ...     results,
        ...     Path("output"),
        ...     image_id_template="exp_A01_t{frame:04d}"
        ... )
        >>> len(paths)
        2
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = []

    for frame_idx, frame_masks in results.items():
        # Generate image ID
        image_id = image_id_template.format(frame=frame_idx)

        # Generate filename
        filename = filename_template.format(image_id=image_id)
        output_path = output_dir / filename

        # Export frame
        try:
            path = export_frame_masks(frame_masks, output_path, image_id)
            exported_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to export frame {frame_idx}: {e}")
            continue

    return exported_paths


def load_labeled_mask(mask_path: Path) -> np.ndarray:
    """
    Load integer-labeled mask from PNG file.

    Args:
        mask_path: Path to labeled mask PNG

    Returns:
        Integer-labeled mask array

    Example:
        >>> mask = load_labeled_mask(Path("masks/frame_0000_masks.png"))
        >>> mask.dtype
        dtype('uint8')
        >>> np.unique(mask)
        array([0, 1, 2])  # background + 2 embryos
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def extract_individual_masks(labeled_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract individual binary masks from labeled mask image.

    Args:
        labeled_mask: Integer-labeled mask array

    Returns:
        Dict mapping embryo_id to binary mask

    Example:
        >>> labeled = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
        >>> masks = extract_individual_masks(labeled)
        >>> masks.keys()
        dict_keys(['embryo_0', 'embryo_1'])
        >>> masks['embryo_0']
        array([[0, 1, 1],
               [0, 0, 0]], dtype=uint8)
    """
    unique_labels = np.unique(labeled_mask)
    # Remove background (0)
    embryo_labels = unique_labels[unique_labels > 0]

    individual_masks = {}
    for label in embryo_labels:
        embryo_id = f"embryo_{label - 1}"  # 1-indexed to 0-indexed
        binary_mask = (labeled_mask == label).astype(np.uint8)
        individual_masks[embryo_id] = binary_mask

    return individual_masks


def visualize_masks(
    labeled_mask: np.ndarray,
    output_path: Optional[Path] = None,
    colormap: str = "tab20"
) -> np.ndarray:
    """
    Create RGB visualization of labeled masks.

    Args:
        labeled_mask: Integer-labeled mask array
        output_path: Optional path to save visualization
        colormap: Matplotlib colormap name

    Returns:
        RGB visualization array (H, W, 3)

    Example:
        >>> labeled = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
        >>> viz = visualize_masks(labeled)
        >>> viz.shape
        (2, 3, 3)  # RGB
    """
    import matplotlib.pyplot as plt

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Normalize labels to [0, 1] for colormap
    max_label = labeled_mask.max()
    if max_label == 0:
        normalized = labeled_mask
    else:
        normalized = labeled_mask / max_label

    # Apply colormap
    colored = cmap(normalized)

    # Convert to 8-bit RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    # Make background black
    rgb[labeled_mask == 0] = [0, 0, 0]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return rgb


def get_mask_statistics(
    labeled_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each embryo mask.

    Args:
        labeled_mask: Integer-labeled mask array

    Returns:
        Dict mapping embryo_id to statistics dict with:
        - area: pixel count
        - bbox: [x_min, y_min, x_max, y_max]
        - centroid: [x, y]

    Example:
        >>> labeled = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
        >>> stats = get_mask_statistics(labeled)
        >>> stats['embryo_0']['area']
        2.0
    """
    unique_labels = np.unique(labeled_mask)
    embryo_labels = unique_labels[unique_labels > 0]

    statistics = {}

    for label in embryo_labels:
        embryo_id = f"embryo_{label - 1}"
        mask = (labeled_mask == label).astype(np.uint8)

        # Area
        area = float(np.sum(mask))

        # Bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        else:
            bbox = [0, 0, 0, 0]

        # Centroid
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            centroid = [float(np.mean(x_coords)), float(np.mean(y_coords))]
        else:
            centroid = [0.0, 0.0]

        statistics[embryo_id] = {
            "area": area,
            "bbox": bbox,
            "centroid": centroid
        }

    return statistics
