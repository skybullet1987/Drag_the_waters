"""
test_file_sizes.py — verify all Vox Python source files are under QuantConnect's
63KB per-file limit and within the ~60KB safety threshold.

Run with: python -m pytest Vox/tests/test_file_sizes.py -v
"""
import os
import pathlib
import pytest

# ── Constants ──────────────────────────────────────────────────────────────────

# Hard limit imposed by QuantConnect/LEAN
QC_HARD_LIMIT_BYTES = 63_000

# Soft safety threshold; files over this get a warning-level check
QC_SOFT_LIMIT_BYTES = 60_000

# Source files to check (relative to the Vox/ package directory)
_VOX_DIR = pathlib.Path(__file__).parent.parent  # .../Vox/


def _vox_source_files():
    """Yield (name, path) for every .py file in Vox/ excluding tests and caches."""
    skip_dirs = {"tests", "__pycache__"}
    for p in sorted(_VOX_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue  # skip __init__ etc.
        yield p.name, p


@pytest.mark.parametrize("name,path", list(_vox_source_files()))
def test_file_under_hard_limit(name, path):
    """Every Vox source file must be strictly below the QuantConnect 63KB limit."""
    size = path.stat().st_size
    assert size < QC_HARD_LIMIT_BYTES, (
        f"{name} is {size:,} bytes — exceeds QuantConnect's {QC_HARD_LIMIT_BYTES:,}-byte limit. "
        f"Split large code blocks into smaller modules."
    )


@pytest.mark.parametrize("name,path", list(_vox_source_files()))
def test_file_under_soft_limit(name, path):
    """Soft check: flag files approaching the 63KB limit (over 60KB).

    This is a warning-level check.  Failures here are non-blocking but should
    be addressed to leave room for future edits.
    """
    size = path.stat().st_size
    if size >= QC_SOFT_LIMIT_BYTES:
        pytest.warns(
            UserWarning,
            match=".*",
        )
        import warnings
        warnings.warn(
            f"{name} is {size:,} bytes — within 3KB of the QuantConnect limit. "
            f"Consider splitting before it grows further.",
            UserWarning,
            stacklevel=1,
        )
