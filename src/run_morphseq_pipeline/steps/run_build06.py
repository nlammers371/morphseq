from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union
import logging
import os
import pandas as pd

from ..services.gen_embeddings import ensure_latents_for_experiments


def _coerce_bool_flag(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    try:
        if str(s.dtype).startswith('int') or str(s.dtype).startswith('float'):
            return s.fillna(0).astype(int).astype(bool)
        vals = s.astype(str).str.strip().str.lower()
        mapping = {
            'true': True, 't': True, '1': True, 'yes': True, 'y': True,
            'false': False, 'f': False, '0': False, 'no': False, 'n': False,
            '': False, 'nan': False
        }
        return vals.map(mapping).fillna(False).astype(bool)
    except Exception:
        return s.astype(bool)


def _load_build04_per_exp(root: Path, exp: str, logger: logging.Logger) -> pd.DataFrame:
    path = root / "metadata" / "build04_output" / f"qc_staged_{exp}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Build04 per-experiment CSV not found: {path}")
    logger.info(f"📄 Loading Build04 per-experiment CSV: {path}")
    df = pd.read_csv(path)
    if 'use_embryo_flag' in df.columns:
        flag = _coerce_bool_flag(df['use_embryo_flag'])
        df = df[flag].copy()
        logger.info(f"✅ Quality filtering: {len(flag)} → {len(df)} embryos (use_embryo_flag=True)")
    else:
        logger.warning("use_embryo_flag not found in Build04 CSV; using all rows")
    return df


def _latents_path(data_root: Path, model_name: str, exp: str) -> Path:
    return data_root / "analysis" / "latent_embeddings" / "legacy" / model_name / f"morph_latents_{exp}.csv"


def run_build06(
    root: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    model_name: str = "20241107_ds_sweep01_optimum",
    experiments: Optional[List[str]] = None,
    generate_missing: bool = False,
    overwrite_latents: bool = False,
    export_analysis: bool = False,
    train_name: Optional[str] = None,
    write_train_output: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    """Per-experiment Build06: merge per-exp Build04 with latents.

    Behavior:
      - Requires `experiments` list. For each exp:
        • Load metadata/build04_output/qc_staged_<exp>.csv
        • Filter to use_embryo_flag=True (robust coercion)
        • Ensure/generate latents under analysis/latent_embeddings/legacy/<model_name>/
        • Merge on snip_id and write per-exp df03 to metadata/build06_output/df03_final_output_with_latents_<exp>.csv

      - Global df03 is deprecated; this function no longer writes or returns a global df03.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    root = Path(root)
    data_root = Path(data_root) if data_root else root

    if not experiments:
        raise ValueError("Build06 now requires per-experiment usage: provide experiments=[...] list")

    last_out: Optional[Path] = None
    for exp in experiments:
        # 1) Load per-experiment Build04 and filter
        df02 = _load_build04_per_exp(root, exp, logger)
        if len(df02) == 0:
            logger.warning(f"No quality embryos to merge for {exp}; skipping")
            continue

        # 2) Ensure latents exist (overwrite if requested)
        logger.info(f"🧬 Ensuring latents for {exp} (model={model_name})")
        ensure_latents_for_experiments(
            data_root=data_root,
            model_name=model_name,
            experiments=[exp],
            generate_missing=generate_missing or overwrite_latents,
            overwrite=overwrite_latents,
            repo_root=root,
            logger=logger,
        )
        lat_path = _latents_path(data_root, model_name, exp)
        if not lat_path.exists():
            raise FileNotFoundError(f"Latents not found for {exp}: {lat_path}")

        # 3) Load latents and merge
        lat_df = pd.read_csv(lat_path)
        z_cols = [c for c in lat_df.columns if c.startswith('z_')]
        if 'snip_id' not in lat_df.columns or not z_cols:
            raise ValueError(f"Invalid latents file (missing snip_id or z_*): {lat_path}")

        merge_df = df02.merge(lat_df[['snip_id'] + z_cols], on='snip_id', how='left')
        match = merge_df[z_cols[0]].notna().sum()
        logger.info(f"🔗 Joined {match}/{len(merge_df)} embryos with latents for {exp}")

        # 4) Write per-experiment df03
        out_dir = root / "metadata" / "build06_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"df03_final_output_with_latents_{exp}.csv"
        if out_path.exists() and not overwrite:
            logger.info(f"📄 Output exists (use --force/overwrite to replace): {out_path}")
        merge_df.to_csv(out_path, index=False)
        logger.info(f"✅ Wrote per-experiment df03: {out_path}")
        last_out = out_path

    return last_out
