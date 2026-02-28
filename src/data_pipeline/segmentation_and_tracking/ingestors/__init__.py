from __future__ import annotations

import importlib
from types import ModuleType


REQUIRED_DETECTOR_FUNCTIONS = {"ingest_detections", "ingest_seed_selection"}
REQUIRED_TRACKER_FUNCTIONS = {"ingest_propagation", "tracks_to_raw_masks"}


def _import_module(path: str) -> ModuleType:
    return importlib.import_module(path)


def get_detector_ingestor(backend: str) -> ModuleType:
    backend = str(backend).strip().lower()
    mod_path = {
        "groundingdino": "data_pipeline.segmentation_and_tracking.ingestors.gdino_ingestor",
        "detectron2": "data_pipeline.segmentation_and_tracking.ingestors.detectron2_ingestor",
    }.get(backend)
    if not mod_path:
        raise ValueError(f"Unsupported detector backend: {backend}")
    mod = _import_module(mod_path)
    missing = [fn for fn in sorted(REQUIRED_DETECTOR_FUNCTIONS) if not hasattr(mod, fn)]
    if missing:
        raise AttributeError(f"{backend} ingestor missing required functions: {missing}")
    return mod


def get_tracker_ingestor(backend: str) -> ModuleType:
    backend = str(backend).strip().lower()
    mod_path = {
        "sam2": "data_pipeline.segmentation_and_tracking.ingestors.sam2_ingestor",
        "sam3": "data_pipeline.segmentation_and_tracking.ingestors.sam3_ingestor",
    }.get(backend)
    if not mod_path:
        raise ValueError(f"Unsupported tracker backend: {backend}")
    mod = _import_module(mod_path)
    missing = [fn for fn in sorted(REQUIRED_TRACKER_FUNCTIONS) if not hasattr(mod, fn)]
    if missing:
        raise AttributeError(f"{backend} ingestor missing required functions: {missing}")
    return mod

