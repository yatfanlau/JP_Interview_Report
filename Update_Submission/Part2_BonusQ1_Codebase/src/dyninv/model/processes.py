"""Stochastic-process and steady-state helpers.

These utilities are intentionally small and pure so they can be reused by both
simulation code and objective evaluations inside estimators.
"""

from __future__ import annotations

import math

from dyninv.utils import DTYPE, tf


@tf.function
def ar1_step_ln_z(z: tf.Tensor, rho: tf.Tensor, eps: tf.Tensor, mu_ln_z: tf.Tensor) -> tf.Tensor:
    """Advance productivity by one AR(1)-in-logs step.

    The function accepts productivity levels ``z`` as input, maps to logs,
    applies the AR(1) update with intercept and shock, and exponentiates back
    to levels.
    """
    z_clipped = tf.maximum(z, tf.constant(1e-12, dtype=z.dtype))
    lnz = tf.math.log(z_clipped)
    return tf.exp(mu_ln_z + rho * lnz + eps)


@tf.function
def steady_state_ln_k(theta: tf.Tensor, delta: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
    """Return ``log(k*)`` implied by the frictionless first-order condition."""
    one = tf.constant(1.0, dtype=theta.dtype)
    return tf.math.log(theta / (r + delta)) / (one - theta)


@tf.function
def steady_state_k(theta: tf.Tensor, delta: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
    """Return level steady-state capital implied by ``steady_state_ln_k``."""
    return tf.exp(steady_state_ln_k(theta, delta, r))


def stationary_lnz_moments(rho: float, sigma_eps: float) -> tuple[float, float]:
    """Return stationary mean/std of log productivity under AR(1).

    The mean uses the normalization implied by ``E[z] = 1``, and the standard
    deviation is the closed-form AR(1) stationary value.
    """
    mean = -0.5 * (sigma_eps**2) / (1.0 - rho * rho)
    std = sigma_eps / math.sqrt(1.0 - rho * rho)
    return mean, std
