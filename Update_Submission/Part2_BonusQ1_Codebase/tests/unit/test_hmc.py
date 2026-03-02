"""Package-level tests for HMC estimator wiring."""

import pytest

from dyninv.estimation.bayesian_hmc import HMCEstimator

pytestmark = pytest.mark.unit


def test_hmc_estimator_constructs():
    """Confirm HMC estimator initialization exposes default settings."""
    est = HMCEstimator()
    assert est.cfg.n_reps_eval > 0
