from pathlib import Path

import pytest

from analyze.classification.misclassification.pipeline import run_misclassification_pipeline


def test_pipeline_on_local_stage1_artifacts_if_available(tmp_path: Path):
    """Dataset-backed smoke test; skips when local artifacts are absent."""
    candidates = [
        Path("results/mcolon/20260213_subtle_phenotype_methods/output"),
        Path("results/mcolon/20260222_dev_consistently_misclassified_embryos"),
    ]

    input_dir = None
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("embryo_predictions_augmented.parquet"):
            if (p.parent / "null" / "null_metadata.json").exists():
                input_dir = p.parent
                break
        if input_dir is not None:
            break

    if input_dir is None:
        pytest.skip("No local Stage 1 artifact directory with embryo_predictions_augmented.parquet + null_metadata.json")

    out_dir = tmp_path / "misclassification_out"
    result = run_misclassification_pipeline(
        input_dir=input_dir,
        output_dir=out_dir,
        config={
            "n_permutations": 50,
            "n_sim": 200,
            "random_state": 42,
        },
    )

    assert (out_dir / "tables" / "per_embryo_metrics.csv").exists()
    assert (out_dir / "metadata.json").exists()
    assert "per_embryo_metrics" in result
