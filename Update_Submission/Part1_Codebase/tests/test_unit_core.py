"""Unit tests for core numerical and stochastic utilities.

The goal of this module is to verify small reusable building blocks with
fast, deterministic checks before they are used by higher-level pipelines.
"""

from __future__ import annotations

import math

from investment_dl.config import BasicModelParams
from investment_dl.core.math_utils import steady_state_k, tf_quantile_1d, tf_var
from investment_dl.core.replay_buffer import ReplayBuffer
from investment_dl.core.stochastic import get_gh_nodes, tauchen_ln_z_grid
from investment_dl.core.tf_env import DTYPE, tf


def test_steady_state_k_matches_closed_form() -> None:
    """`steady_state_k` should reproduce the analytical closed-form value.

    This validates the algebraic formula used throughout model initialization.
    """
    theta, delta, r = 0.7, 0.10, 0.04
    expected = (theta / (r + delta)) ** (1.0 / (1.0 - theta))
    got = steady_state_k(theta, delta, r)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_tf_quantile_and_variance() -> None:
    """Quantile/variance helpers should match simple hand-checkable cases.

    The chosen sample has exact expected values under linear interpolation and
    population-variance conventions.
    """
    x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float64)
    q = tf.constant([0.25, 0.50, 0.75], dtype=tf.float64)

    # Verify quantiles at common percentiles against known reference values.
    got_q = tf_quantile_1d(x, q).numpy().tolist()
    exp_q = [1.75, 2.5, 3.25]
    for got, exp in zip(got_q, exp_q):
        assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9)

    # Population variance of [1,2,3,4] equals 1.25.
    got_var = float(tf_var(tf.cast(x, DTYPE)).numpy())
    assert math.isclose(got_var, 1.25, rel_tol=1e-6, abs_tol=1e-6)


def test_replay_buffer_ring_behavior() -> None:
    """Replay buffer should preserve capacity and overwrite oldest entries.

    The test checks size bookkeeping and sampling interface after wrap-around.
    """
    rb = ReplayBuffer(max_size=5, state_dim=2, seed=123)

    # Initial fill below capacity.
    rb.push_batch(tf.constant([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=DTYPE))
    assert len(rb) == 3

    # Push beyond capacity to trigger circular overwrite behavior.
    rb.push_batch(
        tf.constant(
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
            dtype=DTYPE,
        ),
    )
    assert len(rb) == 5

    sample = rb.sample(batch_size=3)
    assert tuple(sample.shape) == (3, 2)
    assert sample.dtype == DTYPE


def test_gh_nodes_properties() -> None:
    """Gauss-Hermite nodes and weights should satisfy core identities.

    These checks guard against regressions in node construction and scaling.
    """
    n = 7
    x, w, factor, sqrt2 = get_gh_nodes(n)

    assert tuple(x.shape) == (n,)
    assert tuple(w.shape) == (n,)
    assert float(tf.reduce_min(w).numpy()) > 0.0
    assert math.isclose(float(tf.reduce_sum(w).numpy()), math.sqrt(math.pi), rel_tol=1e-4)
    assert math.isclose(float(factor.numpy()), 1.0 / math.sqrt(math.pi), rel_tol=1e-6)
    assert math.isclose(float(sqrt2.numpy()), math.sqrt(2.0), rel_tol=1e-6)


def test_tauchen_transition_rows_sum_to_one() -> None:
    """Tauchen transition matrix should define a valid Markov kernel.

    All probabilities must be nonnegative and each row must sum to one.
    """
    mp = BasicModelParams()
    z_grid, p_z = tauchen_ln_z_grid(mp, n_z=9, m_std=3.0)

    assert tuple(z_grid.shape) == (9,)
    assert tuple(p_z.shape) == (9, 9)
    assert float(tf.reduce_min(p_z).numpy()) >= 0.0

    # Check row stochasticity tolerance after numerical normalization.
    row_sums = tf.reduce_sum(p_z, axis=1)
    max_err = float(tf.reduce_max(tf.abs(row_sums - 1.0)).numpy())
    assert max_err < 1e-5
