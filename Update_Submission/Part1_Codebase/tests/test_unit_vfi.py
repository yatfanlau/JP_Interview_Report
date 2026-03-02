"""Unit tests for VFI solver outputs and interpolation behavior.

These checks focus on shape sanity, feasibility bounds, and finite-value
behavior for off-grid policy queries.
"""

from __future__ import annotations

from investment_dl.config import BasicModelParams
from investment_dl.core.tf_env import DTYPE, tf
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.solvers.vfi import VFIInterpolatedPolicy, VFISolver


def test_vfi_solver_small_grid_outputs() -> None:
    """VFI should return consistent arrays and history on a small grid.

    A small configuration keeps the test fast while still exercising the full
    Bellman iteration path and policy extraction logic.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=13))
    solver = VFISolver(model)

    k_grid, z_grid, V, iota, history = solver.solve(
        n_k=25,
        n_z=7,
        k_min_mul=0.5,
        k_max_mul=2.0,
        max_iter=80,
        tol=1e-4,
        verbose=False,
        return_history=True,
    )

    # Returned tensors should match requested grid dimensions.
    assert tuple(k_grid.shape) == (25,)
    assert tuple(z_grid.shape) == (7,)
    assert tuple(V.shape) == (25, 7)
    assert tuple(iota.shape) == (25, 7)

    # History keys are required for convergence plotting and diagnostics.
    assert "iter" in history
    assert "sup_norm" in history
    assert "converged" in history
    assert "converged_iter" in history
    assert len(history["iter"]) > 0

    # Extracted policy must honor model-imposed iota bounds.
    iota_min = float(tf.reduce_min(iota).numpy())
    iota_max = float(tf.reduce_max(iota).numpy())
    assert iota_min >= model.iota_min - 1e-6
    assert iota_max <= model.iota_max + 1e-6


def test_vfi_evaluate_policy_on_grid_and_interpolator() -> None:
    """Grid evaluation and interpolation utilities should return finite outputs.

    The test covers both an externally provided policy function and the VFI
    interpolator used during simulation/evaluation.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=17))
    solver = VFISolver(model)

    k_grid, z_grid, _, iota_vfi = solver.solve(
        n_k=21,
        n_z=5,
        max_iter=50,
        tol=1e-3,
        verbose=False,
        return_history=False,
    )

    def zero_policy(k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        del z
        return tf.zeros_like(k, dtype=DTYPE)

    # Evaluation helper should broadcast a simple policy over the (k,z) grid.
    iota_zero = solver.evaluate_policy_on_grid(k_grid, z_grid, zero_policy)
    assert tuple(iota_zero.shape) == (21, 5)
    assert float(tf.reduce_max(tf.abs(iota_zero)).numpy()) == 0.0

    # Interpolated policy queries at mixed boundary/interior points.
    interpolator = VFIInterpolatedPolicy(k_grid, z_grid, iota_vfi)
    k_query = tf.stack([k_grid[0], k_grid[10], k_grid[-1]])
    z_query = tf.stack([z_grid[0], z_grid[2], z_grid[-1]])
    iota_query = interpolator(k_query, z_query)

    assert tuple(iota_query.shape) == (3,)
    assert bool(tf.reduce_all(tf.math.is_finite(iota_query)).numpy())
