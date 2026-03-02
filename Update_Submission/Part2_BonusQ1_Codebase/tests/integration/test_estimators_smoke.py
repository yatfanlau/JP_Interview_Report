"""Integration smoke tests for end-to-end estimator execution paths."""

from __future__ import annotations

import pytest

from dyninv.config import (
    BasicModelParams,
    GMMConfigPart2,
    HMCColumnsPart2,
    HMCConfigPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    PathsPart2,
    SMMConfigPart2,
)
from dyninv.estimation.bayesian_hmc import HMCEstimator
from dyninv.estimation.gmm import GMMEstimator
from dyninv.estimation.smm import SMMEstimator

pytestmark = pytest.mark.integration


def test_gmm_estimator_smoke(synthetic_panel_csv):
    """Run a short GMM estimation and verify expected output columns."""
    est = GMMEstimator(
        paths=PathsPart2(data_csv=synthetic_panel_csv),
        cols=PanelColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=GMMConfigPart2(n_reps_eval=1, steps_1=2, steps_2=2, lr_1=0.01, lr_2=0.01),
    )
    out = est.run()
    assert len(out.res_df) == 1
    assert "theta_hat" in out.res_df.columns
    assert "phi_hat" in out.res_df.columns


def test_smm_estimator_smoke(synthetic_panel_csv, dummy_policy_path):
    """Run a short SMM estimation and verify expected output columns."""
    est = SMMEstimator(
        paths=PathsPart2(data_csv=synthetic_panel_csv, policy_path=dummy_policy_path),
        cols=PanelColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=SMMConfigPart2(
            n_reps_eval=1,
            n_firms_sim=20,
            t_data=6,
            t_burnin=5,
            steps_1=2,
            steps_2=2,
            lr_1=0.01,
            lr_2=0.01,
            sims_per_obj_1=1,
            sims_per_obj_2=1,
            w_n_boot=3,
            w_n_sims=3,
            metric2_n_boot=3,
            metric2_n_sims=3,
            jtest_n_boot=3,
            jtest_n_sims=3,
        ),
        mp=BasicModelParams(),
    )
    out = est.run()
    assert len(out.res_df) == 1
    assert "theta_hat" in out.res_df.columns
    assert "phi_hat" in out.res_df.columns


def test_hmc_estimator_smoke(synthetic_panel_csv):
    """Run a short HMC estimation and verify interval output columns."""
    est = HMCEstimator(
        paths=PathsPart2(data_csv=synthetic_panel_csv),
        cols=PanelColumnsPart2(),
        hmc_cols=HMCColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=HMCConfigPart2(
            n_reps_eval=1,
            num_results=10,
            num_burnin=10,
            num_adaptation_steps=5,
            num_leapfrog_steps=2,
            step_size=0.01,
            seed=123,
        ),
        mp=BasicModelParams(),
    )
    out = est.run()
    assert len(out.res_df) == 1
    assert "theta_lo" in out.res_df.columns
    assert "phi_hi" in out.res_df.columns
