"""
Minimal CEP290 within-bin confound check (no pandas/sklearn required).

This script answers the concrete question:
"If Penetrant vs Control looks identical by eye in Panel C, why is AUROC significant?"

It tests a common confound in coarse time bins:
  Within a time bin (e.g. 12–16 hpf for bin_width=4), do Penetrant embryos have
  systematically different *mean predicted_stage_hpf* than Control embryos?

If yes, and if embeddings/curvature correlate with time, a classifier can
"predict group" by predicting time-within-bin, even when trajectories overlap
visually at the bin scale.

What it computes (per embryo, within the target bin)
---------------------------------------------------
- mean_hpf, median_hpf, n_rows
- mean curvature metric (baseline_deviation_normalized)
- AUC for mean_hpf predicting label
- AUC for mean metric predicting label
- AUC for n_rows predicting label (proxy for missingness / sampling bias)

Outputs
-------
Writes a small TSV summary + per-embryo table under:
`results/mcolon/20260105_refined_embedding_and_metric_classification/output/cep290/validation_minimal/`
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"),
        help="Path to embryo_data_with_labels.csv",
    )
    p.add_argument("--time-col", type=str, default="predicted_stage_hpf")
    p.add_argument("--metric-col", type=str, default="baseline_deviation_normalized")
    p.add_argument("--cluster-col", type=str, default="cluster_categories")
    p.add_argument("--embryo-id-col", type=str, default="embryo_id")
    p.add_argument("--bin-width", type=float, default=4.0)
    p.add_argument("--time-bin", type=float, default=12.0, help="Left edge of the bin to audit (e.g. 12 for 12–16 when bin_width=4).")
    p.add_argument("--time-max", type=float, default=20.0, help="Ignore rows with time >= time-max (speeds up parsing).")
    p.add_argument("--max-rows", type=int, default=0, help="Optional cap on parsed rows (0 = no cap).")
    return p.parse_args()


PENETRANT_CATEGORIES = {"Low_to_High", "High_to_Low", "Intermediate"}
CONTROL_CATEGORY = "Not Penetrant"


@dataclass
class Agg:
    n: int = 0
    sum_time: float = 0.0
    sum_metric: float = 0.0
    times: List[float] = None

    def __post_init__(self):
        if self.times is None:
            self.times = []

    def add(self, t: float, metric: float) -> None:
        self.n += 1
        self.sum_time += t
        self.sum_metric += metric
        self.times.append(t)

    def mean_time(self) -> float:
        return self.sum_time / self.n if self.n else float("nan")

    def median_time(self) -> float:
        if not self.times:
            return float("nan")
        s = sorted(self.times)
        m = len(s) // 2
        if len(s) % 2 == 1:
            return float(s[m])
        return float((s[m - 1] + s[m]) / 2.0)

    def mean_metric(self) -> float:
        return self.sum_metric / self.n if self.n else float("nan")


def _auc_from_scores(y: List[int], scores: List[float]) -> float:
    """
    Compute ROC AUC via the Mann–Whitney U statistic with average ranks for ties.

    Returns NaN if only one class is present.
    """
    n = len(y)
    if n == 0:
        return float("nan")
    n_pos = sum(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs = sorted(zip(scores, y), key=lambda x: x[0])

    # Assign average ranks for ties (1-based ranks)
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        rank += (j - i)
        i = j

    rank_sum_pos = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def _write_tsv(path: Path, header: List[str], rows: Iterable[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = _parse_args()

    out_dir = Path("results/mcolon/20260105_refined_embedding_and_metric_classification/output/cep290/validation_minimal")
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_start = float(args.time_bin)
    bin_end = bin_start + float(args.bin_width)

    agg_by_embryo: dict[Tuple[str, str], Agg] = defaultdict(Agg)

    parsed = 0
    kept = 0

    with args.csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {args.embryo_id_col, args.time_col, args.cluster_col, args.metric_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            parsed += 1
            if args.max_rows and parsed > args.max_rows:
                break

            embryo_id = row[args.embryo_id_col]
            cluster = row[args.cluster_col]

            # Assign label
            if cluster in PENETRANT_CATEGORIES:
                label = "Penetrant"
            elif cluster == CONTROL_CATEGORY:
                label = "Control"
            else:
                continue

            try:
                t = float(row[args.time_col])
            except Exception:
                continue

            if t >= float(args.time_max):
                continue

            # Filter to the requested bin (consistent with floor binning)
            if not (bin_start <= t < bin_end):
                continue

            try:
                metric = float(row[args.metric_col])
            except Exception:
                continue
            if math.isnan(metric) or math.isnan(t):
                continue

            kept += 1
            agg_by_embryo[(embryo_id, label)].add(t, metric)

    per_embryo_rows = []
    y = []
    mean_hpf = []
    mean_metric = []
    n_rows = []

    for (embryo_id, label), agg in agg_by_embryo.items():
        y_val = 1 if label == "Penetrant" else 0
        y.append(y_val)
        mean_hpf.append(agg.mean_time())
        mean_metric.append(agg.mean_metric())
        n_rows.append(agg.n)
        per_embryo_rows.append(
            [
                embryo_id,
                label,
                f"{agg.n:d}",
                f"{agg.mean_time():.4f}",
                f"{agg.median_time():.4f}",
                f"{agg.mean_metric():.6g}",
            ]
        )

    auc_time = _auc_from_scores(y, mean_hpf)
    auc_metric = _auc_from_scores(y, mean_metric)
    auc_nrows = _auc_from_scores(y, [float(v) for v in n_rows])

    per_embryo_path = out_dir / f"per_embryo__bin{args.bin_width:g}_t{args.time_bin:g}.tsv"
    _write_tsv(
        per_embryo_path,
        header=["embryo_id", "label", "n_rows", "mean_hpf", "median_hpf", f"mean_{args.metric_col}"],
        rows=per_embryo_rows,
    )

    summary_row = [
        f"{args.bin_width:g}",
        f"{bin_start:g}",
        f"{bin_end:g}",
        str(parsed),
        str(kept),
        str(sum(y)),
        str(len(y) - sum(y)),
        f"{auc_time:.6f}",
        f"{auc_metric:.6f}",
        f"{auc_nrows:.6f}",
    ]

    summary_header = [
        "bin_width",
        "time_bin_start",
        "time_bin_end",
        "rows_parsed",
        "rows_used_in_bin",
        "n_embryos_pos",
        "n_embryos_neg",
        "auc_mean_hpf_predicts_label",
        f"auc_mean_{args.metric_col}_predicts_label",
        "auc_n_rows_predicts_label",
    ]

    summary_path = out_dir / f"summary__bin{args.bin_width:g}_t{args.time_bin:g}.tsv"
    _write_tsv(
        summary_path,
        header=[
            *summary_header,
        ],
        rows=[
            summary_row
        ],
    )

    # Append to a combined summary table for convenience
    summary_all_path = out_dir / "summary_all.tsv"
    if not summary_all_path.exists():
        _write_tsv(summary_all_path, summary_header, [summary_row])
    else:
        with summary_all_path.open("a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(summary_row)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {summary_all_path}")
    print(f"Wrote: {per_embryo_path}")
    print("")
    print("Interpretation:")
    print(f"- If auc_mean_hpf_predicts_label is far from 0.5, you have within-bin time imbalance.")
    print(f"- If auc_n_rows_predicts_label is far from 0.5, sampling/missingness differs by group within bin.")


if __name__ == "__main__":
    main()
