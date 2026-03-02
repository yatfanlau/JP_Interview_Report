"""Unit tests for utility-layer helpers and numerical wrappers."""

from __future__ import annotations

import math
import pytest

from dyninv.utils import ReplayBuffer, chi2_sf, log_sigmoid_prime_tf, set_global_seed, tf

pytestmark = pytest.mark.unit


def test_seed_reproducible_tf():
    """Verify repeated global seeding reproduces TensorFlow draws."""
    set_global_seed(123)
    a = tf.random.uniform([5], dtype=tf.float32)
    set_global_seed(123)
    b = tf.random.uniform([5], dtype=tf.float32)
    tf.debugging.assert_near(a, b)


def test_replay_buffer_push_sample_shapes():
    """Ensure replay-buffer sampling returns expected shapes and size."""
    buf = ReplayBuffer(max_size=5, state_dim=2, seed=0)
    buf.push_batch(tf.zeros([3, 2], dtype=tf.float32))
    samp = buf.sample(2)
    assert samp.shape == (2, 2)
    assert len(buf) == 3


def test_log_sigmoid_prime_tf_zero():
    """Check log-sigmoid-prime identity at zero against log(1/4)."""
    x = tf.constant(0.0, dtype=tf.float32)
    val = float(log_sigmoid_prime_tf(x).numpy())
    assert abs(val - math.log(0.25)) < 1e-6


def test_chi2_sf_zero_is_one():
    """Check the chi-square survival function at zero equals one."""
    assert abs(chi2_sf(0.0, 4) - 1.0) < 1e-12
