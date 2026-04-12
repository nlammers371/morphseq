import sys
from pathlib import Path

# Ensure repo root is on sys.path for src/ imports when running `pytest`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add src/ so imports like `import data_pipeline...` work without needing
# to set PYTHONPATH externally.
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
