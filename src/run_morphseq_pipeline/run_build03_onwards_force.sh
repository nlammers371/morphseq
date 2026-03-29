#!/usr/bin/env bash
#$ -N morphseq_sam2_onwards             # job name
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=8G
#$ -l h_rt=120:00:00                    # walltime (adjust as needed)
#$ -j y                                 # merge stdout/stderr
#$ -pe serial 8
#$ -cwd
#$ -V
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/logs
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/logs

# SAM2-onwards pipeline with optional metadata rebuild.
# Usage with array jobs:
#   qsub -t 1-14 -tc 3 -v EXP_FILE=/path/to/experiment_list.txt run_build03_onwards_force.sh
# Override behaviour with env vars:
#   RUN_METADATA_REBUILD=0      # skip Build01 metadata-only refresh
#   RUN_BUILD02=1               # run Build02 UNet masks (legacy QC segmentation)
#   RUN_SAM2=0                  # skip re-running SAM2
#   RUN_BUILD03=0               # skip Build03 action
#   BUILD03_SKIP_GEOMETRY_QC=1  # skip geometry QC computation (fast mode, all embryos marked usable)
#   RUN_SNIP_EXPORT=1           # export snips only (use existing Build03 metadata)
#   RUN_BUILD04=0               # skip Build04 action
#   MSEQ_SKIP_CURVATURE=1       # skip curvature metrics in Build04 (fast; keeps curvature cols NaN)
#   RUN_BUILD06=0               # skip Build06 action
#   SAM2_WORKERS=8              # SAM2 worker count (default 8)
#   SAM2_CONFIDENCE=0.45        # SAM2 confidence threshold
#   SAM2_IOU=0.5                # SAM2 IoU threshold

set -euo pipefail

# --- CONFIGURATION -----------------------------------------------------------
REPO_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
ALT_DATA_ROOT="${REPO_ROOT%/morphseq}/morphseq-docs/morphseq_playground"
# Allow override.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/morphseq_playground}"
MODEL_NAME="20241107_ds_sweep01_optimum"
ENV_NAME="segmentation_grounded_sam"
PYTHON_EXEC="${PYTHON_EXEC:-/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/${ENV_NAME}/bin/python}"

# Ensure the per-experiment well-metadata Excel is discoverable by Build01.
# Build01 validation prefers:
#   <data-root>/metadata/well_metadata/<exp>_well_metadata.xlsx
#   <data-root>/metadata/plate_metadata/<exp>_well_metadata.xlsx
# Some runs also keep a copy under:
#   <data-root>/metadata/build03_output/<exp>_well_metadata.xlsx
_ensure_well_metadata_excel() {
  local exp_name="$1"

  local dest_dir="${DATA_ROOT}/metadata/well_metadata"
  local dest_path="${dest_dir}/${exp_name}_well_metadata.xlsx"

  # If already in the standard place(s), nothing to do.
  if [[ -f "${dest_path}" || -f "${DATA_ROOT}/metadata/plate_metadata/${exp_name}_well_metadata.xlsx" ]]; then
    return 0
  fi

  # Try known alternate locations.
  local -a candidates=(
    "${DATA_ROOT}/metadata/build03_output/${exp_name}_well_metadata.xlsx"
    "${DATA_ROOT}/metadata/build03_output/${exp_name}_well_metadata.xlsm"
  )
  for src in "${candidates[@]}"; do
    if [[ -f "${src}" ]]; then
      mkdir -p "${dest_dir}"
      # Copy/update into the standard location so Build01 can validate and ingest it.
      cp -u "${src}" "${dest_path}"
      echo "[sam2-onwards] ✓ Well metadata Excel staged for Build01: ${dest_path} (from ${src})"
      return 0
    fi
  done

  # No-op if missing; Build01 will emit a detailed validation error.
  return 0
}

# Default experiment list (used if not running as array job)
DEFAULT_EXPERIMENTS="20251207_pbx,20260213,20260223,20260224,20260304,20260306,20260319,20260320"

# Tunable defaults — override by exporting the variable before invoking this script.
# Example: RUN_SAM2=0 SAM2_WORKERS=2 EXP_LIST=20250305 bash run_build03_onwards_force.sh
: "${METADATA_MICROSCOPE:=YX1}"

# SAM2 inference knobs
: "${SAM2_WORKERS:=8}"
: "${SAM2_CONFIDENCE:=0.45}"
: "${SAM2_IOU:=0.5}"

# Pipeline stage toggles (1=run, 0=skip)
: "${RUN_METADATA_REBUILD:=0}"
: "${RUN_BUILD02:=0}"
: "${RUN_SAM2:=0}"
: "${RUN_BUILD03:=0}"
: "${BUILD03_SKIP_GEOMETRY_QC:=0}"  # 0=compute full geometry QC (default), 1=fast mode (skip QC, mark all embryos usable)
: "${RUN_BUILD04:=0}"
: "${RUN_BUILD06:=1}"
: "${RUN_SNIP_EXPORT:=0}"

# Build04 curvature knobs
# Default to skipping curvature for quick reruns. Set MSEQ_SKIP_CURVATURE=0 to compute curvature.
: "${MSEQ_SKIP_CURVATURE:=1}"
: "${MSEQ_CURVATURE_WORKERS:=}"      # optional override (e.g. 8)
: "${MSEQ_CURVATURE_FALLBACK:=skip}" # skip | sequential

# Build02 knobs

: "${BUILD02_MODE:=legacy}"         # legacy | skip
: "${BUILD02_NUM_WORKERS:=1}"       # DataLoader workers for UNet inference
: "${BUILD02_OVERWRITE:=1}"         # 1=overwrite existing masks, 0=skip existing

# Snip export knobs (outscale fixed at 7.8 to match embedding expectations)
: "${SNIP_WORKERS:=1}"
: "${SNIP_DL_RAD_UM:=50}"
: "${SNIP_OVERWRITE:=0}"
# ----------------------------------------------------------------------------

if [[ "${SNIP_OVERWRITE}" == "1" ]]; then
  SNIP_OVERWRITE_PY="True"
else
  SNIP_OVERWRITE_PY="False"
fi

echo "[sam2-onwards] JOB_ID=${JOB_ID:-unknown} TASK=${SGE_TASK_ID:-0}"
echo "[sam2-onwards] Repo root : ${REPO_ROOT}"
echo "[sam2-onwards] Data root : ${DATA_ROOT}"

if [[ ! -d "${DATA_ROOT}/raw_image_data" && -d "${ALT_DATA_ROOT}/raw_image_data" ]]; then
  echo "[sam2-onwards] NOTE: Falling back to alternate playground: ${ALT_DATA_ROOT}"
  DATA_ROOT="${ALT_DATA_ROOT}"
fi

echo "[sam2-onwards] Run flags - metadata:${RUN_METADATA_REBUILD} b02:${RUN_BUILD02} sam2:${RUN_SAM2} b03:${RUN_BUILD03} snip:${RUN_SNIP_EXPORT} b04:${RUN_BUILD04} b06:${RUN_BUILD06}"
echo "[sam2-onwards] Build02 params - mode:${BUILD02_MODE} workers:${BUILD02_NUM_WORKERS} overwrite:${BUILD02_OVERWRITE}"
echo "[sam2-onwards] Build03 flags - skip_geometry_qc:${BUILD03_SKIP_GEOMETRY_QC}"
echo "[sam2-onwards] SAM2 params - workers:${SAM2_WORKERS} conf:${SAM2_CONFIDENCE} iou:${SAM2_IOU}"
echo "[sam2-onwards] Snip params - workers:${SNIP_WORKERS} dl_rad:${SNIP_DL_RAD_UM} overwrite:${SNIP_OVERWRITE}"
if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "[sam2-onwards] ERROR: PYTHON_EXEC not found/executable: ${PYTHON_EXEC}" >&2
  exit 2
fi

# Support SGE array jobs: select one experiment per task using SGE_TASK_ID
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  if [[ "${SGE_TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "[sam2-onwards] Array task ID: ${SGE_TASK_ID}"
    if [[ -n "${EXP_FILE:-}" && -f "${EXP_FILE}" ]]; then
      mapfile -t _EXPS < "${EXP_FILE}"
    elif [[ -n "${EXP_LIST:-}" ]]; then
      IFS=',' read -r -a _EXPS <<< "${EXP_LIST}"
    else
      IFS=',' read -r -a _EXPS <<< "${DEFAULT_EXPERIMENTS}"
    fi
    _IDX=$(( SGE_TASK_ID - 1 ))
    if (( _IDX < 0 || _IDX >= ${#_EXPS[@]} )); then
      echo "[sam2-onwards] ERROR: Task index $_IDX out of range (len=${#_EXPS[@]})" >&2
      exit 1
    fi
    EXPERIMENT="${_EXPS[_IDX]}"
  else
    echo "[sam2-onwards] SGE_TASK_ID='${SGE_TASK_ID}' is non-numeric; ignoring array selection"
    EXPERIMENT="${DEFAULT_EXPERIMENTS}"
  fi
else
  # Running interactively or non-array job
  if [[ -n "${EXP_FILE:-}" && -f "${EXP_FILE}" ]]; then
    mapfile -t _EXPS < "${EXP_FILE}"
    EXPERIMENT="$(IFS=','; echo "${_EXPS[*]}")"
  else
    EXPERIMENT="${EXP_LIST:-${DEFAULT_EXPERIMENTS}}"
  fi
fi

echo "[sam2-onwards] Processing experiment(s): ${EXPERIMENT}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[sam2-onwards] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a SELECTED_EXPERIMENTS <<< "${EXPERIMENT}"

# Create logs dir if running interactively
mkdir -p logs

# This script expects `PYTHON_EXEC` to point at the correct interpreter; avoid `conda activate`.

# Prefer CUDA 11.8 toolkit to match PyTorch cu118 builds
CUDA_MODULE="cuda/11.8.0"
CUDA_BASE="/net/gs/vol3/software/modules-sw/cuda/11.8.0/Linux/Ubuntu22.04/x86_64"
if type module >/dev/null 2>&1; then
  module unload cuda >/dev/null 2>&1 || true
  module load "${CUDA_MODULE}" >/dev/null 2>&1 || true
fi
if [[ -d "${CUDA_BASE}" ]]; then
  export CUDA_HOME="${CUDA_BASE}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
  export CPATH="${CUDA_HOME}/include:${CPATH:-}"
fi
# Ensure Torch shared libs are on the runtime path (libc10, libtorch_*)
TORCH_LIB="$("${PYTHON_EXEC}" - <<'PY' 2>/dev/null || true
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
if [[ -n "${TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

# Environment diagnostics (helpful for GroundingDINO/CUDA issues)
echo "[sam2-onwards] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none}"
echo "[sam2-onwards] python exec: ${PYTHON_EXEC}"
if command -v conda >/dev/null 2>&1; then
  echo "[sam2-onwards] conda: $(conda --version 2>/dev/null || echo 'unknown')"
fi
"${PYTHON_EXEC}" - <<'PY'
import sys
try:
    import torch
    print("[sam2-onwards] python:", sys.executable)
    print("[sam2-onwards] torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
except Exception as e:
    print("[sam2-onwards] torch import failed:", e)
try:
    import groundingdino
    print("[sam2-onwards] groundingdino:", groundingdino.__file__)
    try:
        from groundingdino import _C  # noqa: F401
        print("[sam2-onwards] groundingdino._C: OK")
    except Exception as e:
        print("[sam2-onwards] groundingdino._C import failed:", e)
except Exception as e:
    print("[sam2-onwards] groundingdino import failed:", e)
PY
if compgen -G "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/GroundingDINO/groundingdino/_C*.so" > /dev/null; then
  ls -l /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/GroundingDINO/groundingdino/_C*.so
else
  echo "[sam2-onwards] groundingdino _C*.so not found in repo checkout"
fi

# Ensure Python can import the repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "🚀 Starting SAM2 onwards pipeline for ${EXPERIMENT}..."

if [[ "${RUN_METADATA_REBUILD}" == "1" ]]; then
  for exp_name in "${SELECTED_EXPERIMENTS[@]}"; do
    _ensure_well_metadata_excel "${exp_name}"
    echo "🔄 Pre-step: Build01 metadata-only for ${exp_name}"
    "${PYTHON_EXEC}" -m src.run_morphseq_pipeline.cli build01 \
      --data-root "${DATA_ROOT}" \
      --exp "${exp_name}" \
      --microscope "${METADATA_MICROSCOPE}" \
      --metadata-only \
      --overwrite \
      || echo "[sam2-onwards] WARNING: Build01 metadata-only failed for ${exp_name}"
  done
fi

if [[ "${RUN_BUILD02}" == "1" ]]; then
  echo "🔄 Step 0.5: Running Build02 UNet masks for ${EXPERIMENT}..."
  BUILD02_ARGS=(
    -m src.run_morphseq_pipeline.cli build02
    --data-root "${DATA_ROOT}"
    --mode "${BUILD02_MODE}"
    --num-workers "${BUILD02_NUM_WORKERS}"
    --experiments "${EXPERIMENT}"
  )
  if [[ "${BUILD02_OVERWRITE}" == "1" ]]; then
    BUILD02_ARGS+=(--overwrite)
  fi
  "${PYTHON_EXEC}" "${BUILD02_ARGS[@]}"
fi

if [[ "${RUN_SAM2}" == "1" ]]; then
  echo "🔄 Step 1: Running SAM2 for ${EXPERIMENT}..."
  "${PYTHON_EXEC}" -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action sam2 \
    --sam2-workers "${SAM2_WORKERS}" \
    --sam2-confidence "${SAM2_CONFIDENCE}" \
    --sam2-iou "${SAM2_IOU}" \
    --force
fi

if [[ "${RUN_BUILD03}" == "1" ]]; then
  echo "🔄 Step 2: Running Build03 for ${EXPERIMENT}..."
  # Export flag for Python to read
  export BUILD03_SKIP_GEOMETRY_QC
  "${PYTHON_EXEC}" -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build03 \
    --model-name "${MODEL_NAME}" \
    --force
fi

if [[ "${RUN_SNIP_EXPORT}" == "1" ]]; then
  echo "🖼️  Step 2b: Exporting BF snips for ${EXPERIMENT}..."
  for exp_name in "${SELECTED_EXPERIMENTS[@]}"; do
    "${PYTHON_EXEC}" - <<PY
from pathlib import Path
import pandas as pd
from src.build.build03A_process_images import extract_embryo_snips

data_root = Path("${DATA_ROOT}")
exp = "${exp_name}"
candidates = [
    data_root / "metadata" / "build03_output" / f"expr_embryo_metadata_{exp}.csv",
    data_root / "metadata" / "build03" / "per_experiment" / f"expr_embryo_metadata_{exp}.csv",
]
for csv_path in candidates:
    if csv_path.exists():
        stats_df = pd.read_csv(csv_path)
        break
else:
    checked = "\n   - ".join(str(p) for p in candidates)
    raise SystemExit(f"❌ Build03 metadata not found for {exp}. Checked paths:\\n   - {checked}")

extract_embryo_snips(
    root=data_root,
    stats_df=stats_df,
    outscale=7.8,
    dl_rad_um=float("${SNIP_DL_RAD_UM}"),
    overwrite_flag=${SNIP_OVERWRITE_PY},
    n_workers=int("${SNIP_WORKERS}"),
)
print(f"✅ Snip export complete for {exp}")
PY
  done
fi

if [[ "${RUN_BUILD04}" == "1" ]]; then
  echo "🔄 Step 3: Running Build04 for ${EXPERIMENT}..."
  # Build04 can be very expensive if curvature is enabled; default to skipping it.
  export MSEQ_SKIP_CURVATURE
  export MSEQ_CURVATURE_FALLBACK
  [[ -n "${MSEQ_CURVATURE_WORKERS}" ]] && export MSEQ_CURVATURE_WORKERS
  "${PYTHON_EXEC}" -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build04 \
    --model-name "${MODEL_NAME}" \
    --force
fi

if [[ "${RUN_BUILD06}" == "1" ]]; then
  echo "🔄 Step 4: Running Build06 for ${EXPERIMENT}..."
  "${PYTHON_EXEC}" -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build06 \
    --model-name "${MODEL_NAME}" \
    --force
fi

echo "🎉 SAM2 onwards pipeline completed for ${EXPERIMENT}!"

# Example usage with array jobs:
#
# Run for all experiments in list:
# qsub -t 1-27 -tc 2 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh

# Run for single experiment (test):
# qsub -t 1 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   run_build03_onwards_force.sh
# qsub -t 1-2 -tc 1 \
#   /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh
# qsub -t 1 -tc 1 /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh

# qsub /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh

# qsub -t 1-4 -tc 3 /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh


  # # SAM2 regeneration (11 experiments)
  # python -m src.run_morphseq_pipeline.cli pipeline \
  #   --data-root morphseq_playground \
  #   --experiments 20250305,20250501,20251017_part1,20251017_part2,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  #   --action sam2 --force

  # # Build03 regeneration (9 experiments)  
  # python -m src.run_morphseq_pipeline.cli pipeline \
  #   --data-root morphseq_playground \
  #   --experiments 20250501,20250912,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  #   --action build03 --force
