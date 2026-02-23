#!/usr/bin/env bash
# Render SAM2 evaluation videos using VideoGenerator.
#
# Usage:
#   bash make_eval_video.sh                              # Use default videos from script
#   bash make_eval_video.sh 20250711_A02                 # Single video
#   bash make_eval_video.sh 20250711_A02,20250711_B03    # Multiple videos
#
# Auto-detects per-experiment SAM2 JSONs under the MorphSeq playground:
#   morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_<EXP>.json
# Falls back to monolithic grounded_sam_segmentations.json if per-experiment is not found.

set -euo pipefail

# --- User-configurable inputs (optional) -----------------------------------
# EXP_ID is optional; if empty, it will be derived from VIDEO_ID/VIDEOS.
EXP_ID=""

# Default videos (used if no command line arguments provided)
DEFAULT_VIDEOS="20250711_F06,20250711_H07"  # e.g., "20250529_36hpf_ctrl_atf6_A04,20250529_36hpf_ctrl_atf6_G06"

# Check for command line arguments
if [[ $# -gt 0 ]]; then
    # Use command line arguments
    VIDEOS="$1"
    VIDEO_ID=""
    echo "🎬 Using videos from command line: ${VIDEOS}"
else
    # Use defaults from script
    VIDEO_ID=""      # e.g., "20250529_36hpf_ctrl_atf6_A04"
    VIDEOS="${DEFAULT_VIDEOS}"            # e.g., "20250529_36hpf_ctrl_atf6_A04,20250529_36hpf_ctrl_atf6_G06"
    echo "🎬 Using default videos from script: ${VIDEOS}"
fi

# Explicit JSON override (leave empty to auto-detect).
RESULTS_JSON=""

# Output policy: directory + suffix → video filenames.
OUT_DIR=""
OUT_SUFFIX="_eval"

# Overlay options.
SHOW_BBOX=true
SHOW_MASK=true
SHOW_METRICS=true
SHOW_QC=false
# --------------------------------------------------------------------------

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
# segmentation_sandbox (3 up), repo root (4 up)
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

# Default output directory if not provided.
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${repo_root}/segmentation_sandbox/results/sam2_eval_videos/$(date +%Y%m%d)"
fi
mkdir -p "${OUT_DIR}"

# Ensure Python can import the video_generation package.
export PYTHONPATH="${script_dir}/..:${PYTHONPATH:-}"

# Derive EXP_ID from VIDEO_ID/VIDEOS if not set.
derive_exp_from_video() {
  local vid="$1"
  if [[ "${vid}" =~ ^(.+)_([A-H][0-9]{2})$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "${vid%_*}"
  fi
}

if [[ -z "${EXP_ID}" ]]; then
  if [[ -n "${VIDEO_ID}" ]]; then
    EXP_ID="$(derive_exp_from_video "${VIDEO_ID}")"
  elif [[ -n "${VIDEOS}" ]]; then
    IFS="," read -r first_video _ <<< "${VIDEOS}"
    EXP_ID="$(derive_exp_from_video "${first_video}")"
  fi
fi

# SAM2 root inside MorphSeq playground.
sam2_root="${repo_root}/morphseq_playground/sam2_pipeline_files"
seg_dir="${sam2_root}/segmentation"

# Auto-detect JSON when not explicitly provided.
if [[ -z "${RESULTS_JSON}" ]]; then
  if [[ -n "${EXP_ID}" && -f "${seg_dir}/grounded_sam_segmentations_${EXP_ID}.json" ]]; then
    RESULTS_JSON="${seg_dir}/grounded_sam_segmentations_${EXP_ID}.json"
  elif [[ -f "${seg_dir}/grounded_sam_segmentations.json" ]]; then
    RESULTS_JSON="${seg_dir}/grounded_sam_segmentations.json"
  else
    echo "❌ Could not find SAM2 JSON. Looked for:" >&2
    echo "   • ${seg_dir}/grounded_sam_segmentations_<EXP>.json (EXP=${EXP_ID:-unset})" >&2
    echo "   • ${seg_dir}/grounded_sam_segmentations.json" >&2
    echo "Available files in ${seg_dir}:" >&2
    ls -1 "${seg_dir}" 2>/dev/null || true
    exit 1
  fi
fi

echo "📁 Repo root: ${repo_root}"
echo "💾 SAM2 root: ${sam2_root}"
echo "🧪 Experiment: ${EXP_ID:-<derived per video>}"
echo "📄 Using JSON: ${RESULTS_JSON}"
echo "📂 Output dir: ${OUT_DIR}"

# Build flags based on booleans.
flags=( )
[[ "${SHOW_BBOX}" == "true" ]] && flags+=("--show-bbox")
[[ "${SHOW_MASK}" != "true" ]] && flags+=("--no-mask")
[[ "${SHOW_METRICS}" != "true" ]] && flags+=("--no-metrics")
[[ "${SHOW_QC}" == "true" ]] && flags+=("--show-qc")

# Base args.
args=( --json "${RESULTS_JSON}" --out-dir "${OUT_DIR}" --suffix "${OUT_SUFFIX}" )

# Pass video ids: either comma list or single.
if [[ -n "${VIDEOS}" ]]; then
  args+=( --videos "${VIDEOS}" )
elif [[ -n "${VIDEO_ID}" ]]; then
  args+=( --video "${VIDEO_ID}" )
else
  echo "❌ Provide VIDEO_ID or VIDEOS (comma-separated)." >&2
  exit 1
fi

# Only pass --exp if set; otherwise the CLI derives it from video_id.
if [[ -n "${EXP_ID}" ]]; then
  args+=( --exp "${EXP_ID}" )
fi

python3 "${script_dir}/render_eval_video.py" "${args[@]}" "${flags[@]}"
echo "✅ Done. Outputs written under: ${OUT_DIR}"

