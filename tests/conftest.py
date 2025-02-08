import tempfile
from pathlib import Path

import pytest
from rootutils import find_root


@pytest.fixture
def data_path() -> Path:
    """A pytest fixture for creating a temporary directory for data.

    Returns:
        Path: A Path object pointing to a temporary directory.
    """
    return Path(tempfile.gettempdir()) / "data"


@pytest.fixture
def configs_dir() -> Path:
    """A pytest fixture for path to the configs directory.

    Returns:
        Path: A Path object pointing to the configs directory.
    """
    return find_root(__file__, indicator="pyproject.toml") / "configs"
