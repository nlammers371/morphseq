"""
Frame Organization for SAM2 Video Processing

Organizes image frames into the strict temporal order required by SAM2's video predictor.
SAM2 expects frames to be sequentially numbered (00000.jpg, 00001.jpg, etc.) in a directory.

This module provides utilities to create temporary frame directories from arbitrary image paths
while maintaining the original frame index mapping needed for result reconstruction.

Key Functions:
    - organize_frames_for_sam2: Create symlinked frame directory for SAM2
    - prepare_bidirectional_propagation: Organize forward and backward frame sequences
    - cleanup_frame_directory: Remove temporary symlink directories

Example Usage:
    ```python
    from pathlib import Path

    # List of frame paths in chronological order
    frame_paths = [
        Path("/data/exp_A01_t0000.jpg"),
        Path("/data/exp_A01_t0001.jpg"),
        Path("/data/exp_A01_t0002.jpg"),
    ]

    # Organize for SAM2
    temp_dir = organize_frames_for_sam2(frame_paths)
    # temp_dir now contains: 00000.jpg -> /data/exp_A01_t0000.jpg
    #                        00001.jpg -> /data/exp_A01_t0001.jpg
    #                        00002.jpg -> /data/exp_A01_t0002.jpg

    # Use with SAM2 predictor
    # ... SAM2 processing ...

    # Cleanup
    cleanup_frame_directory(temp_dir)
    ```

Notes:
    - Uses symlinks for efficiency (no file copying)
    - Frame paths must be in chronological order
    - Temporary directories should be cleaned up after processing
    - For bidirectional propagation, use prepare_bidirectional_propagation
"""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Dict, Any


def organize_frames_for_sam2(
    frame_paths: List[Path],
    start_index: int = 0,
    reverse: bool = False
) -> Path:
    """
    Organize frame paths into a SAM2-compatible directory structure.

    Creates a temporary directory with symlinked frames numbered sequentially
    starting from 00000.jpg. Maintains mapping to original frame indices.

    Args:
        frame_paths: List of image paths in chronological order
        start_index: Starting frame index for mapping (default: 0)
        reverse: If True, reverse the frame order (for backward propagation)

    Returns:
        Path to temporary directory containing organized frames

    Example:
        >>> frames = [Path("img_0005.jpg"), Path("img_0006.jpg")]
        >>> temp_dir = organize_frames_for_sam2(frames, start_index=5)
        >>> list(temp_dir.glob("*.jpg"))
        [PosixPath('.../00000.jpg'), PosixPath('.../00001.jpg')]
    """
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))

    # Optionally reverse for backward propagation
    paths_to_process = list(reversed(frame_paths)) if reverse else frame_paths

    # Create symlinks with sequential numbering
    for idx, src_path in enumerate(paths_to_process):
        if not src_path.exists():
            cleanup_frame_directory(temp_dir)
            raise FileNotFoundError(f"Source frame not found: {src_path}")

        # SAM2 expects format: 00000.jpg, 00001.jpg, etc.
        dst_name = f"{idx:05d}.jpg"
        dst_path = temp_dir / dst_name

        # Create symlink
        dst_path.symlink_to(src_path.absolute())

    return temp_dir


def cleanup_frame_directory(frame_dir: Path) -> None:
    """
    Remove temporary frame directory and all contents.

    Args:
        frame_dir: Path to temporary directory to remove

    Example:
        >>> temp_dir = Path("/tmp/sam2_frames_xyz")
        >>> cleanup_frame_directory(temp_dir)
    """
    if frame_dir.exists():
        shutil.rmtree(frame_dir)


@contextmanager
def sam2_frame_context(frame_paths: List[Path], start_index: int = 0, reverse: bool = False):
    """
    Context manager for temporary SAM2 frame organization.

    Automatically creates and cleans up temporary frame directory.

    Args:
        frame_paths: List of image paths in chronological order
        start_index: Starting frame index for mapping
        reverse: If True, reverse frame order

    Yields:
        Path to temporary directory with organized frames

    Example:
        >>> with sam2_frame_context(frame_paths) as temp_dir:
        ...     # Use temp_dir with SAM2
        ...     predictor.init_state(video_path=str(temp_dir))
        ... # temp_dir automatically cleaned up
    """
    temp_dir = organize_frames_for_sam2(frame_paths, start_index, reverse)
    try:
        yield temp_dir
    finally:
        cleanup_frame_directory(temp_dir)


def prepare_bidirectional_propagation(
    frame_paths: List[Path],
    seed_frame_idx: int
) -> Tuple[Path, Path]:
    """
    Prepare frame directories for bidirectional SAM2 propagation.

    Creates two temporary directories:
    - Forward: frames from seed_frame_idx to end
    - Backward: frames from 0 to seed_frame_idx (reversed)

    Args:
        frame_paths: List of all frame paths in chronological order
        seed_frame_idx: Index of seed frame in frame_paths

    Returns:
        Tuple of (forward_dir, backward_dir) temporary directories

    Example:
        >>> frames = [Path(f"frame_{i}.jpg") for i in range(10)]
        >>> fwd_dir, bwd_dir = prepare_bidirectional_propagation(frames, seed_frame_idx=5)
        >>> # fwd_dir contains frames 5-9 as 00000.jpg-00004.jpg
        >>> # bwd_dir contains frames 0-5 (reversed) as 00000.jpg-00005.jpg
        >>> cleanup_frame_directory(fwd_dir)
        >>> cleanup_frame_directory(bwd_dir)
    """
    if not (0 <= seed_frame_idx < len(frame_paths)):
        raise ValueError(
            f"seed_frame_idx {seed_frame_idx} out of range for {len(frame_paths)} frames"
        )

    # Forward: seed to end
    forward_frames = frame_paths[seed_frame_idx:]
    forward_dir = organize_frames_for_sam2(forward_frames, start_index=seed_frame_idx)

    # Backward: 0 to seed (reversed for SAM2)
    backward_dir = None
    if seed_frame_idx > 0:
        backward_frames = frame_paths[:seed_frame_idx + 1]
        backward_dir = organize_frames_for_sam2(
            backward_frames,
            start_index=0,
            reverse=True
        )

    return forward_dir, backward_dir


def remap_frame_indices(
    results: Dict[int, Any],
    start_index: int,
    reverse: bool = False
) -> Dict[int, Any]:
    """
    Remap SAM2 frame indices back to original chronological indices.

    SAM2 always outputs indices starting from 0. This function converts
    those indices back to the original frame indices.

    Args:
        results: Dict mapping SAM2 frame offsets to results
        start_index: Original starting frame index
        reverse: If True, indices were from reversed propagation

    Returns:
        Dict mapping original frame indices to results

    Example:
        >>> # Forward propagation starting at frame 5
        >>> sam2_results = {0: "data0", 1: "data1", 2: "data2"}
        >>> remapped = remap_frame_indices(sam2_results, start_index=5)
        >>> remapped
        {5: "data0", 6: "data1", 7: "data2"}

        >>> # Backward propagation from frame 5 to 0
        >>> backward_results = {0: "data0", 1: "data1", 2: "data2"}
        >>> remapped = remap_frame_indices(backward_results, start_index=5, reverse=True)
        >>> remapped
        {5: "data0", 4: "data1", 3: "data2"}
    """
    remapped = {}

    for sam2_offset, result in results.items():
        if reverse:
            # Backward: seed_idx - offset
            original_idx = start_index - sam2_offset
        else:
            # Forward: seed_idx + offset
            original_idx = start_index + sam2_offset

        remapped[original_idx] = result

    return remapped


def merge_bidirectional_results(
    forward_results: Dict[int, Any],
    backward_results: Dict[int, Any],
    prefer_forward: bool = True
) -> Dict[int, Any]:
    """
    Merge results from forward and backward SAM2 propagation.

    When frame indices overlap, prefer forward results by default
    (as forward propagation is typically more accurate).

    Args:
        forward_results: Results from forward propagation (already remapped)
        backward_results: Results from backward propagation (already remapped)
        prefer_forward: If True, use forward results for overlaps (default: True)

    Returns:
        Merged results dict

    Example:
        >>> fwd = {5: "fwd5", 6: "fwd6", 7: "fwd7"}
        >>> bwd = {3: "bwd3", 4: "bwd4", 5: "bwd5"}
        >>> merged = merge_bidirectional_results(fwd, bwd)
        >>> merged
        {3: "bwd3", 4: "bwd4", 5: "fwd5", 6: "fwd6", 7: "fwd7"}
    """
    if prefer_forward:
        # Start with backward, overlay forward (forward wins on conflicts)
        merged = {**backward_results, **forward_results}
    else:
        # Start with forward, overlay backward (backward wins on conflicts)
        merged = {**forward_results, **backward_results}

    return merged
