from pathlib import Path


def test_backends_do_not_import_coord_or_working_grid():
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "src" / "analyze" / "utils" / "optimal_transport" / "backends"

    forbidden = ("analyze.utils.coord", "working_grid")
    for p in sorted(backend_dir.glob("*.py")):
        text = p.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"Backend module {p} contains forbidden import/reference: {needle}"

