"""Label alignment helpers for bootstrap resamples."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _contingency_counts(
    reference_labels: Sequence[int],
    bootstrap_labels: Sequence[int],
    sampled_indices: Sequence[int],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Build a contingency table between reference and bootstrap IDs."""
    ref_sample = np.asarray(reference_labels)[sampled_indices]
    boot_sample = np.asarray(bootstrap_labels)[sampled_indices]

    ref_ids = sorted(np.unique(ref_sample))
    boot_ids = sorted(np.unique(boot_sample[boot_sample >= 0]))

    contingency = np.zeros((len(ref_ids), len(boot_ids)), dtype=int)
    ref_to_idx = {lab: i for i, lab in enumerate(ref_ids)}
    boot_to_idx = {lab: j for j, lab in enumerate(boot_ids)}

    for r, b in zip(ref_sample, boot_sample):
        if b < 0:
            continue
        contingency[ref_to_idx[r], boot_to_idx[b]] += 1

    return contingency, ref_ids, boot_ids


def _hungarian_mapping(contingency: np.ndarray) -> Dict[int, int]:
    cost = contingency.max() - contingency
    row_ind, col_ind = linear_sum_assignment(cost)
    return dict(zip(col_ind, row_ind))


def _greedy_mapping(contingency: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    remaining_rows = set(range(contingency.shape[0]))

    for j in range(contingency.shape[1]):
        column = contingency[:, j]
        candidates = [(count, i) for i, count in enumerate(column) if i in remaining_rows]
        if not candidates:
            continue
        _, best_row = max(candidates)
        mapping[j] = best_row
        remaining_rows.remove(best_row)

    return mapping


def infer_label_mapping(
    reference_labels: Sequence[int],
    bootstrap_labels: Sequence[int],
    sampled_indices: Sequence[int],
    prefer_hungarian: bool = True,
) -> Dict[int, int]:
    contingency, ref_ids, boot_ids = _contingency_counts(
        reference_labels, bootstrap_labels, sampled_indices
    )

    if contingency.size == 0:
        return {}

    if prefer_hungarian and linear_sum_assignment is not None:
        mapping_idx = _hungarian_mapping(contingency)
    else:
        mapping_idx = _greedy_mapping(contingency)

    return {boot_ids[j]: ref_ids[i] for j, i in mapping_idx.items()}


def align_bootstrap_labels(
    reference_labels: Sequence[int],
    bootstrap_labels: Sequence[int],
    sampled_indices: Sequence[int],
    prefer_hungarian: bool = True,
) -> np.ndarray:
    aligned = np.full_like(reference_labels, fill_value=-1, dtype=int)
    bootstrap_labels = np.asarray(bootstrap_labels)

    label_map = infer_label_mapping(
        reference_labels, bootstrap_labels, sampled_indices, prefer_hungarian
    )

    for idx in sampled_indices:
        boot_label = int(bootstrap_labels[idx])
        if boot_label < 0:
            continue
        aligned[idx] = label_map.get(boot_label, -1)

    return aligned


def collect_aligned_histories(
    bootstrap_results: Iterable[Dict[str, np.ndarray]],
    reference_labels: Sequence[int],
    prefer_hungarian: bool = True,
) -> List[np.ndarray]:
    histories: List[np.ndarray] = []

    for res in bootstrap_results:
        aligned = align_bootstrap_labels(
            reference_labels=reference_labels,
            bootstrap_labels=res["labels"],
            sampled_indices=res["indices"],
            prefer_hungarian=prefer_hungarian,
        )
        histories.append(aligned)

    return histories


def stack_histories_to_matrix(histories: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    if not histories:
        raise ValueError("histories must contain at least one aligned label vector.")

    shapes = {np.asarray(h).shape for h in histories}
    if len(shapes) != 1:
        raise ValueError("All histories must share the same shape to stack into a matrix.")

    matrix = np.stack([np.asarray(h, dtype=int) for h in histories], axis=0)
    sample_mask = matrix >= 0

    return {"labels": matrix, "sample_mask": sample_mask}
