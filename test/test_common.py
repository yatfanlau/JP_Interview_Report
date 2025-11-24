import math
import random

import numpy as np
import pytest

from common import (
    tf,
    DTYPE,
    set_global_seed,
    steady_state_k,
    ReplayBuffer,
    ar1_step_ln_z,
    get_gh_nodes,
    risky_coverage_sampler,
)
from config import RiskyDebtParams


def test_set_global_seed_reproducible():
    set_global_seed(123)
    a1 = random.random()
    b1 = np.random.rand(3)
    c1 = tf.random.uniform((3,), dtype=DTYPE)

    set_global_seed(123)
    a2 = random.random()
    b2 = np.random.rand(3)
    c2 = tf.random.uniform((3,), dtype=DTYPE)

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert np.allclose(c1.numpy(), c2.numpy())


def test_steady_state_k_first_order_condition():
    theta = 0.7
    delta = 0.1
    r = 0.04
    k_star = steady_state_k(theta, delta, r)

    # Check that θ k^(θ-1) ≈ r + δ
    lhs = theta * (k_star ** (theta - 1.0))
    rhs = r + delta
    assert math.isclose(lhs, rhs, rel_tol=1e-7, abs_tol=1e-9)


def test_replay_buffer_push_and_sample():
    max_size = 10
    state_dim = 3
    buf = ReplayBuffer(max_size=max_size, state_dim=state_dim, seed=0)

    # Push less than capacity
    data1 = np.random.rand(5, state_dim).astype(np.float32)
    buf.push_batch(data1)
    assert len(buf) == 5

    # Push more to trigger wrap-around
    data2 = np.random.rand(10, state_dim).astype(np.float32)
    buf.push_batch(data2)
    assert len(buf) == max_size

    # Sampling returns correct shape and uses only known rows
    sample = buf.sample(batch_size=4)
    assert sample.shape == (4, state_dim)
    # Every sampled row must be in buffer.buffer (not strict equality set, just shape/assert not NaN)
    assert np.isfinite(sample).all()


def test_ar1_step_ln_z_matches_formula():
    B = 5
    z = tf.ones((B,), dtype=DTYPE) * 2.0
    rho = tf.constant(0.8, dtype=DTYPE)
    eps = tf.random.normal((B,), mean=0.1, stddev=0.2, dtype=DTYPE)
    mu_ln_z = tf.constant(-0.05, dtype=DTYPE)

    z_next = ar1_step_ln_z(z, rho, eps, mu_ln_z)
    z_next_manual = tf.exp(mu_ln_z + rho * tf.math.log(z) + eps)

    assert np.allclose(z_next.numpy(), z_next_manual.numpy(), atol=1e-6)
    # AR(1) in logs should yield strictly positive z
    assert (z_next.numpy() > 0).all()


def test_get_gh_nodes_basic_properties():
    n = 8
    x, w, factor, sqrt2 = get_gh_nodes(n)

    assert x.shape == (n,)
    assert w.shape == (n,)
    assert factor.dtype == DTYPE
    assert sqrt2.dtype == DTYPE

    # factor = 1 / sqrt(pi)
    assert math.isclose(float(factor.numpy()), 1.0 / math.sqrt(math.pi), rel_tol=1e-6)
    assert math.isclose(float(sqrt2.numpy()), math.sqrt(2.0), rel_tol=1e-6)

    # Calling again should hit cache (same object identity is not required,
    # but same numerical values)
    x2, w2, factor2, sqrt22 = get_gh_nodes(n)
    assert np.allclose(x.numpy(), x2.numpy())
    assert np.allclose(w.numpy(), w2.numpy())
    assert float(factor.numpy()) == float(factor2.numpy())
    assert float(sqrt2.numpy()) == float(sqrt22.numpy())


def test_risky_coverage_sampler_shapes_and_ranges():
    rp = RiskyDebtParams()
    batch_size = 256
    states = risky_coverage_sampler(batch_size, rp)  # [B, 3]

    assert states.shape == (batch_size, 3)

    k = states[:, 0].numpy()
    b = states[:, 1].numpy()
    z = states[:, 2].numpy()

    assert (k >= rp.k_cov_min - 1e-6).all()
    assert (k <= rp.k_cov_max + 1e-6).all()

    assert (b >= rp.b_cov_min - 1e-6).all()
    assert (b <= rp.b_cov_max + 1e-6).all()

    # lognormal with mean ~ lnz_cov_mean and std ~ lnz_cov_std (just basic checks)
    assert (z > 0).all()
    lnz = np.log(z)
    assert abs(lnz.mean() - rp.lnz_cov_mean) < 0.5  # loose sanity band
    assert lnz.std() > 0.0