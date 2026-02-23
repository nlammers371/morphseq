#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DEFAULT="${REPO_ROOT}/../morphseq_CORRUPT_OLD/morphseq_playground"
TARGET_DEFAULT="${REPO_ROOT}/morphseq_playground"

SOURCE="${SOURCE_DEFAULT}"
TARGET="${TARGET_DEFAULT}"
MODE="symlink"
FORCE=0

usage() {
  cat <<EOF
Restore morphseq_playground into this repository.

Usage:
  $(basename "$0") [--source PATH] [--target PATH] [--symlink|--copy] [--force]

Options:
  --source PATH   Source playground directory
                  (default: ${SOURCE_DEFAULT})
  --target PATH   Destination path
                  (default: ${TARGET_DEFAULT})
  --symlink       Create a single symlink at target -> source (default)
  --copy          Copy directory contents while preserving internal symlinks
  --force         Remove an existing target first
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --symlink)
      MODE="symlink"
      shift
      ;;
    --copy)
      MODE="copy"
      shift
      ;;
    --force)
      FORCE=1
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

if [[ ! -d "${SOURCE}" ]]; then
  echo "Source does not exist or is not a directory: ${SOURCE}" >&2
  exit 1
fi

if [[ -e "${TARGET}" || -L "${TARGET}" ]]; then
  if [[ "${FORCE}" -eq 1 ]]; then
    rm -rf "${TARGET}"
  else
    echo "Target already exists: ${TARGET}" >&2
    echo "Re-run with --force to replace it." >&2
    exit 1
  fi
fi

if [[ "${MODE}" == "symlink" ]]; then
  ln -s "${SOURCE}" "${TARGET}"
  echo "Created symlink:"
  echo "  ${TARGET} -> ${SOURCE}"
else
  mkdir -p "${TARGET}"
  rsync -a --links "${SOURCE}/" "${TARGET}/"
  echo "Copied playground with symlinks preserved:"
  echo "  ${SOURCE} -> ${TARGET}"
fi

echo "Done."
