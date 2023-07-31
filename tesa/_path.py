from pathlib import Path

# Path of the root of the Python package, where the `pyproject.toml` is
_ROOT: Path = Path(__file__).parents[1]

_DATA: Path = _ROOT / "data"
DATASET: Path = _ROOT / "dataset"
