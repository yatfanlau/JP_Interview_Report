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


def symmetrize_np(A):
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)


def pinv_psd_np(A, rcond=1e-10, ridge=0.0):
    A = symmetrize_np(A)
    if ridge and ridge > 0:
        A = A + float(ridge) * np.eye(A.shape[0], dtype=np.float64)

    evals, evecs = np.linalg.eigh(A)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    thresh = float(rcond) * max(max_eval, 1.0)
    inv_evals = np.where(evals > thresh, 1.0 / evals, 0.0)
    A_pinv = (evecs * inv_evals) @ evecs.T
    return symmetrize_np(A_pinv)


def safe_corr_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ols_slopes_2reg_with_intercept_np(y, x1, x2, eps=1e-12):
    y = np.asarray(y, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    if y.size < 5:
        return 0.0, 0.0

    my, mx1, mx2 = y.mean(), x1.mean(), x2.mean()
    dy, dx1, dx2 = y - my, x1 - mx1, x2 - mx2

    S11 = float(np.mean(dx1 * dx1))
    S22 = float(np.mean(dx2 * dx2))
    S12 = float(np.mean(dx1 * dx2))
    C1y = float(np.mean(dx1 * dy))
    C2y = float(np.mean(dx2 * dy))

    den = S11 * S22 - S12 * S12
    if not np.isfinite(den) or abs(den) <= eps:
        return 0.0, 0.0

    b1 = (S22 * C1y - S12 * C2y) / den
    b2 = (-S12 * C1y + S11 * C2y) / den
    if not np.isfinite(b1) or not np.isfinite(b2):
        return 0.0, 0.0
    return float(b1), float(b2)


def zcrit(alpha=0.05):
    try:
        import tensorflow_probability as tfp
        return float(
            tfp.distributions.Normal(0.0, 1.0).quantile(1.0 - alpha / 2.0).numpy()
        )
    except Exception:
        return 1.959963984540054


def chi2_sf(x, df):
    x = tf.constant(float(x), dtype=tf.float64)
    a = tf.constant(0.5 * float(df), dtype=tf.float64)
    return float(tf.math.igammac(a, 0.5 * x).numpy())


# ---------------------------------------------------------------------
# Extra helpers (used by HMC/UKF)
# ---------------------------------------------------------------------


def logit_np(p, eps=1e-6):
    """Numerically safe logit for NumPy arrays/scalars."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def log_sigmoid_tf(x: tf.Tensor) -> tf.Tensor:
    """log(sigmoid(x)) stable."""
    return -tf.nn.softplus(-x)


def log1m_sigmoid_tf(x: tf.Tensor) -> tf.Tensor:
    """log(1 - sigmoid(x)) stable."""
    return -tf.nn.softplus(x)


def log_sigmoid_prime_tf(x: tf.Tensor) -> tf.Tensor:
    """log(sigmoid'(x)) = log(sigmoid(x)) + log(1-sigmoid(x)) stable."""
    return log_sigmoid_tf(x) + log1m_sigmoid_tf(x)