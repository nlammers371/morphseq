"""Architectural guard: morphology_geometry must not import classification internals.

Rules enforced:
1. Only io.py and validation.py may import from analyze.classification.
2. The ONLY permitted import from analyze.classification in those files is:
       from analyze.classification.directions.artifact import ClassifierDirections
   (validation.py needs it as the input type annotation; io.py needs it to load artifacts.)
3. No other file under morphology_geometry may import from analyze.classification.
4. Direction-math symbols (fit_classifier_direction,
   build_classifier_directions_payload) must not be defined in any
   morphology_geometry file (they belong in classification/directions/).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MG_ROOT = Path(__file__).resolve().parents[1]  # src/analyze/morphology_geometry/
_PERMITTED_IMPORT = "from analyze.classification.directions.artifact import ClassifierDirections"
# Only these two files may import from analyze.classification (and only the artifact type).
_PERMITTED_CLASSIFICATION_FILES = {"io.py", "validation.py"}
_BANNED_DIRECTION_SYMBOLS = {
    "fit_classifier_direction",
    "build_classifier_directions_payload",
}


def _iter_py_files(root: Path, exclude_tests: bool = True):
    """Yield all .py files under root, skipping tests/ if requested."""
    for p in sorted(root.rglob("*.py")):
        if exclude_tests and "tests" in p.parts:
            continue
        yield p


def _collect_classification_imports(source: str) -> list[str]:
    """Return import lines that reference analyze.classification."""
    lines = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "analyze.classification" in module:
                # Reconstruct a canonical import string for inspection
                names = ", ".join(alias.name for alias in node.names)
                lines.append(f"from {module} import {names}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "analyze.classification" in alias.name:
                    lines.append(f"import {alias.name}")
    return lines


def _collect_function_defs(source: str) -> list[str]:
    """Return names of all top-level function definitions."""
    names = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


class TestNoClassificationInternalImports:
    def test_only_permitted_files_import_classification(self):
        """Only io.py and validation.py may import from analyze.classification."""
        violations = []
        for py_file in _iter_py_files(_MG_ROOT, exclude_tests=True):
            if py_file.name in _PERMITTED_CLASSIFICATION_FILES:
                continue
            source = py_file.read_text(encoding="utf-8")
            bad = _collect_classification_imports(source)
            if bad:
                violations.append(f"{py_file.relative_to(_MG_ROOT)}: {bad}")
        assert not violations, (
            "Files outside {io.py, validation.py} import from analyze.classification:\n"
            + "\n".join(violations)
        )

    def test_permitted_files_only_import_artifact(self):
        """io.py and validation.py may only import ClassifierDirections from directions.artifact."""
        for fname in _PERMITTED_CLASSIFICATION_FILES:
            py_path = _MG_ROOT / fname
            if not py_path.exists():
                pytest.skip(f"{fname} not yet created")
            source = py_path.read_text(encoding="utf-8")
            classification_imports = _collect_classification_imports(source)
            if not classification_imports:
                continue
            for imp in classification_imports:
                assert imp.strip() == _PERMITTED_IMPORT, (
                    f"{fname} has a disallowed classification import: {imp!r}\n"
                    f"Only permitted: {_PERMITTED_IMPORT!r}"
                )

    def test_no_direction_math_symbols_defined(self):
        """fit_classifier_direction and build_classifier_directions_payload must
        not be defined anywhere in morphology_geometry — those live in
        classification/directions/.
        """
        violations = []
        for py_file in _iter_py_files(_MG_ROOT, exclude_tests=False):
            source = py_file.read_text(encoding="utf-8")
            defs = set(_collect_function_defs(source))
            bad = defs & _BANNED_DIRECTION_SYMBOLS
            if bad:
                violations.append(f"{py_file.relative_to(_MG_ROOT)}: defines {bad}")
        assert not violations, (
            "Direction-math symbols defined inside morphology_geometry (must stay in "
            "classification/directions/):\n" + "\n".join(violations)
        )
