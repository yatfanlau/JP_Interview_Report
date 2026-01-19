# tests/test_model_core.py
import random
import numpy as np

from model_core import (
    DTYPE,
    ReplayBuffer,
    ar1_step_ln_z,
    get_gh_nodes,
    set_global_seed,
    steady_state_k,
    tf,
)


def test_steady_state_k_matches_formula():
    theta, delta, r = 0.7, 0.10, 0.04
    k = steady_state_k(theta, delta, r)
    k_expected = (theta / (r + delta)) ** (1.0 / (1.0 - theta))
    assert np.isclose(k, k_expected)


def test_set_global_seed_reproducible_across_py_np_tf():
    set_global_seed(123)
    a1 = random.random()
    b1 = np.random.rand(5)
    c1 = tf.random.uniform((5,), dtype=DTYPE).numpy()

    set_global_seed(123)
    a2 = random.random()
    b2 = np.random.rand(5)
    c2 = tf.random.uniform((5,), dtype=DTYPE).numpy()

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert np.allclose(c1, c2)


def test_replay_buffer_push_and_sample_shapes():
    buf = ReplayBuffer(max_size=10, state_dim=3, seed=7)
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    buf.push_batch(x)

    assert len(buf) == 4
    sample = buf.sample(batch_size=2)
    assert sample.shape == (2, 3)
    assert sample.dtype == np.float32


def test_replay_buffer_overwrites_oldest_when_full():
    buf = ReplayBuffer(max_size=10, state_dim=1, seed=1)
    states = np.arange(12, dtype=np.float32).reshape(-1, 1)
    buf.push_batch(states)

    assert len(buf) == 10
    kept = np.sort(buf.buffer[: buf.size, 0])
    expected = np.arange(2, 12, dtype=np.float32)  # last 10 entries: 2..11
    assert np.allclose(kept, expected)


def test_replay_buffer_sampling_deterministic_given_seed_and_data():
    states = np.arange(20, dtype=np.float32).reshape(10, 2)

    b1 = ReplayBuffer(max_size=10, state_dim=2, seed=999)
    b1.push_batch(states)
    s1 = b1.sample(batch_size=4)

    b2 = ReplayBuffer(max_size=10, state_dim=2, seed=999)
    b2.push_batch(states)
    s2 = b2.sample(batch_size=4)

    assert np.allclose(s1, s2)


def test_ar1_step_ln_z_basic_cases_and_positivity():
    z = tf.constant([[1.0], [2.0], [0.0]], dtype=DTYPE)
    rho = tf.constant(0.0, dtype=DTYPE)
    eps = tf.zeros((3, 1), dtype=DTYPE)
    mu = tf.constant(0.0, dtype=DTYPE)

    z_next = ar1_step_ln_z(z=z, rho=rho, eps=eps, mu_ln_z=mu).numpy()
    # With rho=0, mu=0, eps=0: ln z_{t+1}=0 => z_{t+1}=1 for all entries
    assert np.allclose(z_next, np.ones((3, 1), dtype=np.float32))
    assert np.all(np.isfinite(z_next))
    assert np.all(z_next > 0.0)


def test_get_gh_nodes_cache_and_normalization():
    t1 = get_gh_nodes(10, dtype=DTYPE)
    t2 = get_gh_nodes(10, dtype=DTYPE)

    # cached tuple object reused
    assert t1 is t2

    x, w, factor, sqrt2 = t1
    assert tuple(x.shape) == (10,)
    assert tuple(w.shape) == (10,)
    assert factor.shape == ()
    assert sqrt2.shape == ()

    # For Gauss–Hermite: sum(w) == sqrt(pi); factor == 1/sqrt(pi) => factor*sum(w) == 1
    val = float((factor * tf.reduce_sum(w)).numpy())
    assert abs(val - 1.0) < 1e-6
    assert np.all(w.numpy() > 0.0)