from __future__ import annotations

from typing import Callable, Optional, Tuple

from torchvision import transforms


def basic_transform(target_size: Optional[Tuple[int, int]] = None) -> Callable:
    """Minimal transform used for legacy embedding generation.

    The legacy VAE expects grayscale tensors. Many call sites pass
    `target_size=(288, 128)`.
    """
    ops = [transforms.Grayscale(num_output_channels=1)]
    if target_size is not None:
        ops.append(transforms.Resize(target_size))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)

