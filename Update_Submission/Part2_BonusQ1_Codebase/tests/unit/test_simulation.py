"""Package-level tests for simulation/data path wiring."""

import pytest

from dyninv.config import PathsPart2

pytestmark = pytest.mark.unit


def test_simulation_paths_default_csv():
    """Verify the default simulation output path is a CSV file."""
    paths = PathsPart2()
    assert paths.data_csv.endswith(".csv")
