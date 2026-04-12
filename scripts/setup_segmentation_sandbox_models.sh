#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_DIR="${REPO_ROOT}/segmentation_sandbox/models"

# Override these if your model repos live elsewhere.
GROUNDINGDINO_TARGET_DEFAULT="/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/GroundingDINO"
SAM2_TARGET_DEFAULT="/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/sam2"

GROUNDINGDINO_TARGET="${GROUNDINGDINO_TARGET:-${GROUNDINGDINO_TARGET_DEFAULT}}"
SAM2_TARGET="${SAM2_TARGET:-${SAM2_TARGET_DEFAULT}}"

DRY_RUN=0

usage() {
  cat <<EOF
Create local symlinks under segmentation_sandbox/models/ for external model repos.

This repo does not vendor model weights; it expects machine-specific symlinks.

Usage:
  $(basename "$0") [--dry-run]

Env overrides:
  GROUNDINGDINO_TARGET  (default: ${GROUNDINGDINO_TARGET_DEFAULT})
  SAM2_TARGET           (default: ${SAM2_TARGET_DEFAULT})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

doit() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
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
    echo "[warn] ${dest} links to ${cur}; expected ${target}"
    echo "       Remove it first if you want to replace it."
    return 1
  fi

  if [[ -e "${dest}" ]]; then
    echo "[warn] ${dest} exists and is not a symlink; refusing to overwrite."
    return 1
  fi

  if [[ ! -e "${target}" ]]; then
    echo "[warn] target missing: ${target}"
    return 1
  fi

  doit ln -s "${target}" "${dest}"
  echo "[link] ${dest} -> ${target}"
}

doit mkdir -p "${MODELS_DIR}"

link_path "${MODELS_DIR}/GroundingDINO" "${GROUNDINGDINO_TARGET}"
link_path "${MODELS_DIR}/sam2" "${SAM2_TARGET}"

echo "Done. Models dir: ${MODELS_DIR}"

