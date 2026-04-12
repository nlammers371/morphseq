"""Compatibility data package.

Some legacy analysis/embedding code imports modules as `data.*` after adding
`<repo>/src` to `sys.path`. In this repository, the canonical training and
embedding utilities live under `src/core`, but a small subset of code still
expects `data.dataset_configs` and `data.data_transforms`.

This package provides lightweight shims used by Build06 embedding generation.
"""

from __future__ import annotations

