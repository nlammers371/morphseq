from __future__ import annotations

import numpy as np
import torch

from data_pipeline.image_building.shared.log_focus import LoG_focus_stacker, im_rescale
from src.build.export_utils import LoG_focus_stacker as legacy_log_focus_stacker
from src.build.export_utils import im_rescale as legacy_im_rescale


def test_log_focus_matches_legacy_cpu() -> None:
    rng = np.random.default_rng(42)
    stack = (rng.random((7, 32, 32)) * 65535).astype(np.uint16)

    ff_new, score_new = LoG_focus_stacker(stack, filter_size=3, device="cpu")
    ff_old, score_old = legacy_log_focus_stacker(stack, filter_size=3, device="cpu")

    assert torch.allclose(ff_new, ff_old)
    assert torch.allclose(score_new, score_old)


def test_im_rescale_matches_legacy() -> None:
    rng = np.random.default_rng(123)
    stack = (rng.random((5, 24, 24)) * 65535).astype(np.uint16)

    out_new, lo_new, hi_new = im_rescale(stack)
    out_old, lo_old, hi_old = legacy_im_rescale(stack)

    assert np.allclose(out_new, out_old)
    assert lo_new == lo_old
    assert hi_new == hi_old
