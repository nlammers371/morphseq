# Stream B: Backend Plumbing — Decisions

## Default backend
`backend: Optional[UOTBackend] = None` defaults to `POTBackend()` if None — matches existing behavior.

## data_root plumbing
Added `data_root` parameter to `load_mask_series_from_csv` so yolk masks can be loaded during timeseries runs, improving canonical alignment quality.

## No breaking changes
All new parameters are optional with backward-compatible defaults.
