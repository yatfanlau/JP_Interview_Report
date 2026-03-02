"""Package-level tests for SMM estimator wiring."""

import pytest

from dyninv.estimation.smm import SMMEstimator

pytestmark = pytest.mark.unit


def test_smm_estimator_constructs():
    """Confirm SMM estimator initialization exposes default settings."""
    est = SMMEstimator()
    assert est.cfg.n_reps_eval > 0
