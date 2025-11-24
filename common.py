# Shared TensorFlow setup, random seeds, replay buffer, AR(1) process,
# Gauss–Hermite nodes, and coverage samplers used by both models.

import os
import math
import random
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------
# TensorFlow setup: CPU only, float32 everywhere
# ---------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32

# ---------------------------------------------------------------------
# Global seeding helper
# ---------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------
# Simple analytical steady state for capital
# ---------------------------------------------------------------------


def steady_state_k(theta: float, delta: float, r: float) -> float:
    """
    Steady-state capital for a frictionless model with E[z]=1 and no adj. costs:
        θ * k^(θ-1) = r + δ  =>  k = (θ/(r+δ))^(1/(1-θ)).
    This is used as a scale reference for coverage sampling.
    """
    return (theta / (r + delta)) ** (1.0 / (1.0 - theta))


# ---------------------------------------------------------------------
# Generic replay buffer for vector-valued states
# ---------------------------------------------------------------------


class ReplayBuffer:
    """
    Simple ring-buffer replay buffer for vector-valued states.
    Stores states as a 2D NumPy array of shape [max_size, state_dim].
    """

    def __init__(self, max_size: int, state_dim: int, seed: int = 42):
        self.max_size = int(max_size)
        self.state_dim = int(state_dim)
        self.buffer = np.empty((self.max_size, self.state_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.RandomState(seed)

    def push_batch(self, states: np.ndarray) -> None:
        """
        Insert a batch of states with shape [batch_size, state_dim].
        Oldest entries are overwritten when the buffer is full.
        """
        states = np.asarray(states, dtype=np.float32)
        assert states.ndim == 2 and states.shape[1] == self.state_dim, \
            f"Expected states of shape [B,{self.state_dim}], got {states.shape}"
        n = states.shape[0]
        for i in range(n):
            self.buffer[self.ptr] = states[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> np.ndarray:
        """Uniformly sample a batch of states, returned as a NumPy array."""
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty.")
        idx = self.rng.randint(0, self.size, size=batch_size)
        return self.buffer[idx]

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------
# AR(1) process in ln z
# ---------------------------------------------------------------------


def ar1_step_ln_z(
    z: tf.Tensor,
    rho: tf.Tensor,
    eps: tf.Tensor,
    mu_ln_z: tf.Tensor,
) -> tf.Tensor:
    """
    AR(1) process in ln z with intercept mu_ln_z:
        ln z_{t+1} = mu_ln_z + rho * ln z_t + eps,  eps ~ N(0, sigma^2).

    Args:
        z: current productivity level, shape (B,) or (B,1)
        rho: scalar tf.Tensor
        eps: innovations, same batch shape as z
        mu_ln_z: scalar tf.Tensor intercept in ln z

    Returns:
        z_next: next-period productivity level, same shape as z
    """
    z_clipped = tf.maximum(z, tf.constant(1e-12, dtype=z.dtype))
    lnz = tf.math.log(z_clipped)
    lnz_next = mu_ln_z + rho * lnz + eps
    return tf.exp(lnz_next)


# ---------------------------------------------------------------------
# Gauss–Hermite nodes and weights (cached)
# ---------------------------------------------------------------------

_GH_CACHE = {}


def get_gh_nodes(n: int, dtype=DTYPE) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Return (x, w, factor, sqrt2) for n-node Gauss–Hermite quadrature.

    For ε ~ N(0, σ^2) and function f:
        E[f(ε)] ≈ factor * Σ w_i f(sqrt2 * σ * x_i)

    Returns:
        x: nodes, shape (n,)
        w: weights, shape (n,)
        factor: scalar 1/sqrt(pi)
        sqrt2: scalar sqrt(2)
    """
    if n in _GH_CACHE:
        return _GH_CACHE[n]

    x_np, w_np = np.polynomial.hermite.hermgauss(n)
    x = tf.constant(x_np, dtype=dtype)
    w = tf.constant(w_np, dtype=dtype)
    factor = tf.constant(1.0 / math.sqrt(math.pi), dtype=dtype)
    sqrt2 = tf.constant(math.sqrt(2.0), dtype=dtype)

    _GH_CACHE[n] = (x, w, factor, sqrt2)
    return _GH_CACHE[n]


# ---------------------------------------------------------------------
# Coverage sampler for risky-debt model states (k, b, z)
# ---------------------------------------------------------------------

from config import RiskyDebtParams  # late import to avoid circular issues


def risky_coverage_sampler(batch_size: int, rp: RiskyDebtParams) -> tf.Tensor:
    """
    Coverage sampler for risky-debt model states.

    Draws:
        k ~ Uniform[k_cov_min, k_cov_max]
        b ~ Uniform[b_cov_min, b_cov_max]
        ln z ~ Normal(lnz_cov_mean, lnz_cov_std^2)
        z = exp(ln z)

    Returns:
        states: Tensor of shape (batch_size, 3) with columns [k, b, z].
    """
    k = tf.random.uniform(
        (batch_size, 1),
        minval=rp.k_cov_min,
        maxval=rp.k_cov_max,
        dtype=DTYPE,
    )
    b = tf.random.uniform(
        (batch_size, 1),
        minval=rp.b_cov_min,
        maxval=rp.b_cov_max,
        dtype=DTYPE,
    )
    lnz = tf.random.normal(
        (batch_size, 1),
        mean=rp.lnz_cov_mean,
        stddev=rp.lnz_cov_std,
        dtype=DTYPE,
    )
    z = tf.exp(lnz)
    states = tf.concat([k, b, z], axis=1)
    return states