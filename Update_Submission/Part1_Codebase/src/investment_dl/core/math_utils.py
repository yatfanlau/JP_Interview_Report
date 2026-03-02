"""Small numerical helpers reused across solvers, simulation, and evaluation.

These routines are intentionally minimal and framework-compatible so they can
be called from both eager code and TensorFlow graph-decorated functions.
"""

from __future__ import annotations

from investment_dl.core.tf_env import tf


def steady_state_k(theta: float, delta: float, r: float) -> float:
    """Return steady-state capital from the frictionless first-order condition.

    The implied condition is ``theta * k^(theta-1) = r + delta``.
    """
    return (theta / (r + delta)) ** (1.0 / (1.0 - theta))


def tf_quantile_1d(x: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    """Return 1D linear quantiles using NumPy-style interpolation weights.

    Both inputs are cast to float64 to reduce rounding differences when
    quantiles are used in diagnostics and report comparisons.
    """
    # Flatten to a vector so quantiles are computed over all elements.
    x = tf.reshape(tf.cast(tf.convert_to_tensor(x), tf.float64), [-1])
    q = tf.cast(tf.convert_to_tensor(q), tf.float64)
    x_sorted = tf.sort(x)
    n = tf.shape(x_sorted)[0]
    n_f = tf.cast(n, tf.float64)
    idx = q * (n_f - 1.0)
    idx_lower = tf.cast(tf.floor(idx), tf.int32)
    idx_upper = tf.cast(tf.math.ceil(idx), tf.int32)
    weight = idx - tf.cast(idx_lower, tf.float64)
    x_lower = tf.gather(x_sorted, idx_lower)
    x_upper = tf.gather(x_sorted, idx_upper)
    return (1.0 - weight) * x_lower + weight * x_upper


def tf_var(x: tf.Tensor) -> tf.Tensor:
    """Return population variance for ``x`` without Bessel correction.

    This matches the moment convention used throughout the project diagnostics.
    """
    x = tf.convert_to_tensor(x)
    mean = tf.reduce_mean(x)
    return tf.reduce_mean(tf.square(x - mean))
