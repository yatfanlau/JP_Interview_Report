"""Integration test for a compact end-to-end training/evaluation pipeline.

The test intentionally uses tiny grids and short training horizons so it
remains fast while still touching the major project components.
"""

from __future__ import annotations

import math

from investment_dl.config import BasicModelParams, BasicTrainingParams
from investment_dl.core.math_utils import tf_quantile_1d
from investment_dl.core.stochastic import tauchen_ln_z_grid
from investment_dl.core.tf_env import tf
from investment_dl.evaluation.euler_residuals import EulerResidualEvaluator
from investment_dl.evaluation.panels import compute_panel_moments
from investment_dl.evaluation.regimes import compute_regime_stats
from investment_dl.evaluation.values import PolicyValueEvaluator
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.simulation.simulators import PolicySimulator
from investment_dl.solvers.euler_dl import EulerEquationTrainer
from investment_dl.solvers.vfi import VFIInterpolatedPolicy, VFISolver


def test_small_end_to_end_pipeline() -> None:
    """Run a minimal full pipeline and validate key outputs are well-formed.

    This includes DL training, VFI solving, residual diagnostics, simulation,
    panel moments, regime stats, and policy interpolation checks.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=23))
    # Keep the run lightweight while exercising all major code paths.
    training_params = BasicTrainingParams(
        hidden_sizes=(16, 16),
        activation="tanh",
        buffer_size=256,
        n_paths=32,
        roll_steps=1,
        batch_size=64,
        pretrain_steps=1,
        train_steps=2,
        coverage_final_share=0.2,
        lr=1e-3,
        log_every=1000,
        eval_every=1000,
        test_size=128,
        test_mc=64,
        k_cov_m_minus=0.5,
        k_cov_m_plus=1.5,
    )

    # Short DL training pass should still produce non-empty history.
    trainer = EulerEquationTrainer(model, training_params)
    history = trainer.train(return_history=True)
    assert history is not None
    assert len(history["train_steps"]) >= 1

    policy_fn = trainer.make_policy_function()

    # Build a small VFI benchmark grid for downstream comparisons.
    vfi_solver = VFISolver(model)
    k_grid, z_grid, V_vfi, iota_vfi = vfi_solver.solve(
        n_k=31,
        n_z=7,
        k_min_mul=0.5,
        k_max_mul=2.0,
        max_iter=80,
        tol=1e-3,
        verbose=False,
        return_history=False,
    )

    # Learned policy evaluated on the VFI grid must be finite everywhere.
    iota_dl = vfi_solver.evaluate_policy_on_grid(k_grid, z_grid, policy_fn)
    assert tuple(iota_dl.shape) == (31, 7)
    assert bool(tf.reduce_all(tf.math.is_finite(iota_dl)).numpy())

    # Euler diagnostics should return finite scalar summary metrics.
    euler_eval = EulerResidualEvaluator(model)
    K, Z = tf.meshgrid(k_grid, z_grid, indexing="ij")
    k_flat = tf.reshape(K, [-1])
    z_flat = tf.reshape(Z, [-1])
    euler_stats = euler_eval.stats(policy_fn, k_flat, z_flat, gh_nodes=5)
    for value in euler_stats.values():
        assert math.isfinite(value)

    # Value and Bellman residual diagnostics with Tauchen transitions.
    value_eval = PolicyValueEvaluator(model)
    _, p_z = tauchen_ln_z_grid(model.params, n_z=int(z_grid.shape[0]), m_std=3.0)
    V_dl = value_eval.evaluate_policy_value_on_grid(
        k_grid,
        z_grid,
        iota_dl,
        p_z,
        max_iter=80,
        tol=1e-3,
        init_V=V_vfi,
    )
    bellman = value_eval.compute_bellman_residual_on_grid(
        k_grid,
        z_grid,
        iota_dl,
        V_dl,
        p_z,
    )
    bellman_stats = value_eval.bellman_residual_stats(bellman)
    for value in bellman_stats.values():
        assert math.isfinite(value)

    # Simulate both flattened sample and panel outputs for downstream stats.
    simulator = PolicySimulator(model, default_n_paths=32)
    k_sim, z_sim, iota_sim = simulator.simulate_sample(
        policy_fn,
        burn_in_steps=5,
        T=64,
        n_paths=8,
        seed=23,
    )
    assert tuple(k_sim.shape) == (64,)
    assert tuple(z_sim.shape) == (64,)
    assert tuple(iota_sim.shape) == (64,)

    k_panel, z_panel, iota_panel = simulator.simulate_panel(
        policy_fn,
        burn_in_steps=5,
        T=12,
        n_paths=8,
        seed=23,
    )
    assert tuple(k_panel.shape) == (12, 8)
    assert tuple(z_panel.shape) == (12, 8)
    assert tuple(iota_panel.shape) == (12, 8)

    # Panel moments should include the expected variable set.
    moments = compute_panel_moments(k_panel, z_panel, iota_panel, model, lags=[1, 2])
    assert set(moments.keys()) == {"k", "iota", "I", "y"}

    # Quantile-based 2x2 regime binning for compact regime-stat checks.
    q = tf.constant([0.0, 0.5, 1.0], dtype=tf.float64)
    k_edges = tf_quantile_1d(k_sim, q)
    z_edges = tf_quantile_1d(z_sim, q)
    mean, std, counts, share = compute_regime_stats(k_sim, z_sim, iota_sim, k_edges, z_edges)

    assert tuple(mean.shape) == (2, 2)
    assert tuple(std.shape) == (2, 2)
    assert tuple(counts.shape) == (2, 2)
    assert tuple(share.shape) == (2, 2)
    assert int(tf.reduce_sum(counts).numpy()) == 64
    assert math.isclose(float(tf.reduce_sum(share).numpy()), 1.0, rel_tol=1e-6, abs_tol=1e-6)

    # Interpolated VFI policy should evaluate cleanly on the base grid.
    vfi_interp = VFIInterpolatedPolicy(k_grid, z_grid, iota_vfi)
    iota_vfi_on_grid = vfi_solver.evaluate_policy_on_grid(k_grid, z_grid, vfi_interp)
    assert tuple(iota_vfi_on_grid.shape) == (31, 7)
