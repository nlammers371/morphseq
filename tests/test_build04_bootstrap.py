import os
from pathlib import Path
import shutil
import pandas as pd

from src.run_morphseq_pipeline.steps.run_build04 import run_build04


def make_min_df01(tmp_root: Path) -> None:
    meta_dir = tmp_root / "metadata" / "combined_metadata_files"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # Minimal df01 with required columns
    df01 = pd.DataFrame([
        {
            "snip_id": "20250101_expA_A01_e00_t0000",
            "embryo_id": "A01_e00",
            "experiment_date": "20250101",
            "predicted_stage_hpf": 10.0,
            "surface_area_um": 5e5,
            "use_embryo_flag": 1,
            "chem_perturbation": "None",
            "genotype": "wik",
            "time_int": 0,
        },
        {
            "snip_id": "20250101_expA_A01_e01_t0000",
            "embryo_id": "A01_e01",
            "experiment_date": "20250101",
            "predicted_stage_hpf": 12.0,
            "surface_area_um": 6e5,
            "use_embryo_flag": 1,
            "chem_perturbation": "None",
            "genotype": "ab",
            "time_int": 0,
        },
    ])
    df01.to_csv(meta_dir / "embryo_metadata_df01.csv", index=False)

    # Minimal stage ref (sa_um vs stage_hpf) for interpolation
    stage_ref = pd.DataFrame({
        "sa_um": [0.0, 4e5, 8e5, 1.2e6],
        "stage_hpf": [0, 8, 16, 24],
    })
    stage_ref.to_csv(tmp_root / "metadata" / "stage_ref_df.csv", index=False)


def test_build04_bootstrap_key(tmp_path: Path):
    root = tmp_path / "build04_smoke"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    make_min_df01(root)

    # Ensure no key exists
    key_path = root / "metadata" / "perturbation_name_key.csv"
    assert not key_path.exists()

    # Run Build04; should bootstrap key and produce df02
    run_build04(root=root)

    # Assert key written and contains traceability column
    assert key_path.exists()
    key_df = pd.read_csv(key_path)
    assert "time_auto_constructed" in key_df.columns
    assert len(key_df) >= 1

    # Assert df02 written
    df02_path = root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
    assert df02_path.exists()
    df02 = pd.read_csv(df02_path)
    # Basic schema presence
    for c in ["snip_id", "master_perturbation", "predicted_stage_hpf"]:
        assert c in df02.columns
    # Stage inference should have run or at least produced a column
    assert "inferred_stage_hpf" in df02.columns

