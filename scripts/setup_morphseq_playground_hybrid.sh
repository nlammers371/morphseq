#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLAYGROUND="${REPO_ROOT}/morphseq_playground"

# Defaults: override with env vars or flags below.
CENTRAL_BASE_DEFAULT="/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
OLD_PLAYGROUND_DEFAULT="/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/morphseq_playground"

CENTRAL_BASE="${CENTRAL_BASE:-${CENTRAL_BASE_DEFAULT}}"
OLD_PLAYGROUND="${OLD_PLAYGROUND:-${OLD_PLAYGROUND_DEFAULT}}"

DRY_RUN=0

usage() {
  cat <<EOF
Create a "hybrid" morphseq_playground layout:
  - Keep metadata/ local and writable
  - Symlink large/shared trees to canonical locations

Usage:
  $(basename "$0") [--central-base PATH] [--old-playground PATH] [--dry-run]

Defaults:
  --central-base   ${CENTRAL_BASE_DEFAULT}
  --old-playground ${OLD_PLAYGROUND_DEFAULT}

Notes:
  - This does NOT copy data.
  - Existing paths are moved aside with a timestamped .bak_* suffix.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --central-base)
      CENTRAL_BASE="$2"
      shift 2
      ;;
    --old-playground)
      OLD_PLAYGROUND="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ts="$(date +%Y%m%d_%H%M%S)"

doit() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

backup_if_exists() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    doit mv "${path}" "${path}.bak_${ts}"
  fi
}

link_path() {
  local dest="$1"
  local target="$2"

  if [[ -L "${dest}" ]]; then
    local cur
    cur="$(readlink "${dest}" || true)"
    if [[ "${cur}" == "${target}" ]]; then
      echo "[ok] ${dest} already links to ${target}"
      return 0
    fi
  fi

  backup_if_exists "${dest}"
  doit ln -s "${target}" "${dest}"
  echo "[link] ${dest} -> ${target}"
}

doit mkdir -p "${PLAYGROUND}"

# Keep metadata local/writable (pipeline writes state + outputs here)
doit mkdir -p "${PLAYGROUND}/metadata/experiments"

# Well metadata: pipeline expects metadata/well_metadata/<EXP>_well_metadata.xlsx
# We keep the curated XLSX inputs in the repo at metadata/plate_metadata.
if [[ ! -e "${PLAYGROUND}/metadata/well_metadata" && ! -L "${PLAYGROUND}/metadata/well_metadata" ]]; then
  link_path "${PLAYGROUND}/metadata/well_metadata" "${REPO_ROOT}/metadata/plate_metadata"
fi

# Canonical shared locations (symlinks)
link_path "${PLAYGROUND}/raw_image_data" "${CENTRAL_BASE}/raw_image_data"
link_path "${PLAYGROUND}/models" "${CENTRAL_BASE}/models"
link_path "${PLAYGROUND}/outside_models" "${CENTRAL_BASE}/outside_models"
link_path "${PLAYGROUND}/segmentation" "${CENTRAL_BASE}/segmentation"
link_path "${PLAYGROUND}/built_image_data" "${CENTRAL_BASE}/built_image_data"

# Optional legacy/abandoned artifacts (symlink if present)
for d in analysis sam2_pipeline_files training_data videos mask_data safe_test_outputs; do
  if [[ -e "${OLD_PLAYGROUND}/${d}" || -L "${OLD_PLAYGROUND}/${d}" ]]; then
    link_path "${PLAYGROUND}/${d}" "${OLD_PLAYGROUND}/${d}"
  fi
done

echo "Done. Data root: ${PLAYGROUND}"

