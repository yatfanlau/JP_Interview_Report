# tests/test_risky_debt_diagnostic_integration.py
import numpy as np

from config_part1 import RiskyDebtFinalTestParams
from model_core import tf, DTYPE
from risky_debt_diagnostic import (
    compute_residuals,
    run_final_diagnostics_and_plots,
    simulate_ergodic_panel,
)


def test_compute_residuals_returns_finite_scalars(risky_small):
    model, _, _ = risky_small
    states = model.coverage_sampler(16)
    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]

    out = compute_residuals(model, k, b, z, n_mc_rhs=2)

    expected_keys = {
        "bell_rel_mean",
        "bell_rel_max",
        "foc_mean_sq",
        "foc_max_abs",
        "ll_rel_mean",
        "ll_rel_max",
    }
    assert expected_keys.issubset(out.keys())
    for v in out.values():
        assert np.isfinite(v)


def test_simulate_ergodic_panel_shapes_and_bounds(risky_small):
    model, params, train_params = risky_small

    k_panel, b_panel, z_panel = simulate_ergodic_panel(
        model,
        params=params,
        train_params=train_params,
        n_firms=8,
        T=10,
        burn_in=2,
        k0_min=0.6,
        k0_max=1.0,
    )

    # Observations recorded for t >= burn_in: (T-burn_in)*n_firms
    expected_n = (10 - 2) * 8
    assert k_panel.shape == (expected_n, 1)
    assert b_panel.shape == (expected_n, 1)
    assert z_panel.shape == (expected_n, 1)

    assert np.all(k_panel >= params.k_cov_min - 1e-6)
    assert np.all(k_panel <= params.k_cov_max + 1e-6)
    assert np.all(b_panel >= params.b_cov_min - 1e-6)
    assert np.all(b_panel <= params.b_cov_max + 1e-6)
    assert np.all(z_panel >= train_params.z_clip_min - 1e-6)
    assert np.all(z_panel <= train_params.z_clip_max + 1e-6)


def test_run_final_diagnostics_and_plots_smoke(risky_small):
    model, params, train_params = risky_small

    # tiny "training" so train_step + Polyak path is exercised
    states = model.coverage_sampler(train_params.batch_size)
    model.lambda_foc = 0.1
    _ = model.train_step(states)

    final_params = RiskyDebtFinalTestParams(
        n_coverage=32,
        n_onpolicy=32,
        n_mc_rhs=2,
        ergodic_n_firms=8,
        ergodic_T=12,
        ergodic_burn_in=2,
        stats_batch_size=64,
        default_prob_draws=10,
        hist_bins=10,
        grid_z_fixed=0.3,
        grid_k_points=15,
        grid_b_points=15,
        grid_b_indices=(0, 7, 14),
        default_region_n_levels=8,
        default_region_contour_levels=(0.01, 0.05),
    )

    # Should run without exceptions (plots are suppressed by conftest.py)
    run_final_diagnostics_and_plots(
        model=model,
        params=params,
        train_params=train_params,
        final_test_params=final_params,
    )

    # Quick sanity: default prob on a small batch is in [0,1]
    k = tf.ones((5, 1), dtype=DTYPE)
    b = tf.zeros((5, 1), dtype=DTYPE)
    z = tf.ones((5, 1), dtype=DTYPE)
    pd = model.default_probability(k, b, z, n_draws=5, use_target=True).numpy()
    assert np.all(pd >= -1e-8)
    assert np.all(pd <= 1.0 + 1e-8)