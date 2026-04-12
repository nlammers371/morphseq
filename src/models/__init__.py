"""Compatibility models package.

Some legacy analysis utilities import `models.*` after adding `<repo>/src` to
`sys.path`. The canonical code lives under `src/core/models`, so this package
re-exports small pieces needed by those utilities.
"""

from __future__ import annotations

