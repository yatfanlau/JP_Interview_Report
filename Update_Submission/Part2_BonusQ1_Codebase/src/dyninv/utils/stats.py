"""Statistical helper functions built on TensorFlow operations.

The utilities here provide numerically stable low-dimensional statistics that
are reused across estimators and diagnostics.
"""

from __future__ import annotations

import tensorflow as tf


def safe_corr_tf(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    """Return a numerically stable Pearson-correlation estimate.

    Inputs are flattened, cast to float64, and guarded against zero-variance
    edge cases that would otherwise produce NaN or Inf.
    """
    # Work in float64 to reduce cancellation error in centered products.
    x = tf.cast(tf.reshape(x, [-1]), tf.float64)
    y = tf.cast(tf.reshape(y, [-1]), tf.float64)
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    dx = x - mx
    dy = y - my
    sx = tf.sqrt(tf.reduce_mean(dx * dx) + eps)
    sy = tf.sqrt(tf.reduce_mean(dy * dy) + eps)
    # If either series is effectively constant, return 0 instead of NaN/Inf.
    return tf.where((sx > 0.0) & (sy > 0.0), tf.reduce_mean(dx * dy) / (sx * sy), 0.0)


def ols_slopes_2reg_with_intercept_tf(
    y: tf.Tensor, x1: tf.Tensor, x2: tf.Tensor, eps: float = 1e-12
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute closed-form OLS slopes for ``y ~ 1 + x1 + x2``.

    The intercept is absorbed by demeaning, so the solve reduces to a stable
    2x2 moment-system inversion with safe fallback on near-singular designs.
    """
    y = tf.cast(tf.reshape(y, [-1]), tf.float64)
    x1 = tf.cast(tf.reshape(x1, [-1]), tf.float64)
    x2 = tf.cast(tf.reshape(x2, [-1]), tf.float64)
    my = tf.reduce_mean(y)
    mx1 = tf.reduce_mean(x1)
    mx2 = tf.reduce_mean(x2)
    dy = y - my
    dx1 = x1 - mx1
    dx2 = x2 - mx2
    # These are centered second moments; intercept is handled by demeaning.
    s11 = tf.reduce_mean(dx1 * dx1)
    s22 = tf.reduce_mean(dx2 * dx2)
    s12 = tf.reduce_mean(dx1 * dx2)
    c1y = tf.reduce_mean(dx1 * dy)
    c2y = tf.reduce_mean(dx2 * dy)
    den = s11 * s22 - s12 * s12
    ok = tf.abs(den) > eps
    # Explicit 2x2 inverse formula with safe fallback when near singular.
    b1 = tf.where(ok, (s22 * c1y - s12 * c2y) / den, 0.0)
    b2 = tf.where(ok, (-s12 * c1y + s11 * c2y) / den, 0.0)
    return b1, b2


def logit_tf(p: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Apply a clipped logit transform to avoid boundary overflow."""
    # Clipping avoids `log(0)` and keeps gradients finite near boundaries.
    p = tf.clip_by_value(tf.cast(p, tf.float64), eps, 1.0 - eps)
    return tf.math.log(p) - tf.math.log1p(-p)


def log_sigmoid_tf(x: tf.Tensor) -> tf.Tensor:
    """Return numerically stable ``log(sigmoid(x))``."""
    return -tf.nn.softplus(-x)


def log1m_sigmoid_tf(x: tf.Tensor) -> tf.Tensor:
    """Return numerically stable ``log(1 - sigmoid(x))``."""
    return -tf.nn.softplus(x)


def log_sigmoid_prime_tf(x: tf.Tensor) -> tf.Tensor:
    """Return numerically stable ``log(sigmoid'(x))``."""
    return log_sigmoid_tf(x) + log1m_sigmoid_tf(x)


def safe_corr_np(x, y):
    """Compatibility wrapper returning a Python float correlation."""
    return float(safe_corr_tf(x, y).numpy())


def ols_slopes_2reg_with_intercept_np(y, x1, x2, eps: float = 1e-12):
    """Compatibility wrapper returning NumPy-compatible slope scalars."""
    b1, b2 = ols_slopes_2reg_with_intercept_tf(y, x1, x2, eps=eps)
    return float(b1.numpy()), float(b2.numpy())


def logit_np(p, eps: float = 1e-6):
    """Compatibility wrapper returning NumPy-style logit output."""
    return logit_tf(p, eps=eps).numpy()
