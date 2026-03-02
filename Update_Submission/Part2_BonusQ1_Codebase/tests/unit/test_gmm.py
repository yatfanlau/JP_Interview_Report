"""Package-level tests for GMM estimator wiring."""

import pytest

from dyninv.estimation.gmm import GMMEstimator

pytestmark = pytest.mark.unit


def test_gmm_estimator_constructs():
    """Confirm GMM estimator initialization exposes default settings."""
    est = GMMEstimator()
    assert est.cfg.n_reps_eval > 0
