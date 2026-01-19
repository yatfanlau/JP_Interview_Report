# tests/test_utils_part2.py
import numpy as np
import pytest
import utils_part2 as u

tf = u.tf


def test_set_global_seed_reproducible_numpy_and_tf():
    u.set_global_seed(123)
    a1 = np.random.rand(5)
    t1 = tf.random.uniform([5], dtype=tf.float32).numpy()

    u.set_global_seed(123)
    a2 = np.random.rand(5)
    t2 = tf.random.uniform([5], dtype=tf.float32).numpy()

    assert np.allclose(a1, a2)
    assert np.allclose(t1, t2)


def test_logit_np_clip_finite():
    assert np.isfinite(u.logit_np(0.0))
    assert np.isfinite(u.logit_np(1.0))


def test_log_sigmoid_prime_tf_zero():
    x = tf.constant(0.0, dtype=u.DTYPE)
    val = float(u.log_sigmoid_prime_tf(x).numpy())
    assert val == pytest.approx(np.log(0.25), abs=1e-6)


def test_replay_buffer_push_sample_shapes():
    buf = u.ReplayBuffer(max_size=5, state_dim=2, seed=0)
    buf.push_batch(np.zeros((3, 2), dtype=np.float32))
    samp = buf.sample(2)
    assert samp.shape == (2, 2)
    assert len(buf) == 3


def test_replay_buffer_overwrite_oldest():
    buf = u.ReplayBuffer(max_size=3, state_dim=1, seed=0)
    buf.push_batch(np.array([[0], [1], [2]], dtype=np.float32))
    buf.push_batch(np.array([[3]], dtype=np.float32))
    assert len(buf) == 3
    vals = set(buf.buffer[: len(buf)].reshape(-1).astype(int))
    assert vals == {1, 2, 3}


def test_pinv_psd_np_identity():
    I = np.eye(3, dtype=np.float64)
    Ipinv = u.pinv_psd_np(I)
    assert np.allclose(Ipinv, I)


def test_safe_corr_np_degenerate_zero():
    x = np.ones(10)
    y = np.arange(10)
    assert u.safe_corr_np(x, y) == 0.0


def test_ols_slopes_2reg_with_intercept_np_recovers_slopes():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 1.0 + 2.0 * x1 - 3.0 * x2  # no noise
    b1, b2 = u.ols_slopes_2reg_with_intercept_np(y, x1, x2)
    assert b1 == pytest.approx(2.0, abs=1e-6)
    assert b2 == pytest.approx(-3.0, abs=1e-6)


def test_chi2_sf_at_zero_is_one():
    assert u.chi2_sf(0.0, df=5) == pytest.approx(1.0, abs=1e-12)