"""Package-level tests for policy training configuration wiring."""

import pytest

from dyninv.config import BasicTrainingParams

pytestmark = pytest.mark.unit


def test_training_defaults_are_positive():
    """Ensure key training defaults are strictly positive."""
    cfg = BasicTrainingParams()
    assert cfg.batch_size > 0
    assert cfg.train_steps > 0
