"""Numerical linear-algebra and distribution helpers.

These helpers focus on robust covariance-matrix manipulation and compact
probability utilities used repeatedly by estimators.
"""

from __future__ import annotations

import tensorflow as tf


def symmetrize_tf(a: tf.Tensor) -> tf.Tensor:
    """Return the symmetric part ``0.5 * (A + A^T)`` in float64."""
    a = tf.cast(a, tf.float64)
    return 0.5 * (a + tf.transpose(a))


def pinv_psd_tf(a: tf.Tensor, rcond: float = 1e-10, ridge: float = 0.0) -> tf.Tensor:
    """Compute a stable pseudo-inverse for near-PSD matrices.

    The routine applies non-finite cleanup, explicit symmetrization, optional
    ridge stabilization, and eigenvalue thresholding before inversion.
    """
    # Replace non-finite entries so covariance inversions fail gracefully.
    a = tf.where(tf.math.is_finite(a), tf.cast(a, tf.float64), tf.zeros_like(tf.cast(a, tf.float64)))
    # Enforce symmetry because covariance estimates may have tiny asymmetric noise.
    a = symmetrize_tf(a)
    if ridge > 0.0:
        # Optional diagonal ridge to push near-singular matrices away from zero.
        a = a + tf.cast(ridge, tf.float64) * tf.eye(tf.shape(a)[0], dtype=tf.float64)
    try:
        evals, evecs = tf.linalg.eigh(a)
    except tf.errors.InvalidArgumentError:
        # Fallback for non-PSD numerical pathologies.
        return symmetrize_tf(tf.linalg.pinv(a, rcond=rcond))
    max_eval = tf.reduce_max(evals) if tf.size(evals) > 0 else tf.constant(0.0, tf.float64)
    thresh = tf.cast(rcond, tf.float64) * tf.maximum(max_eval, 1.0)
    # Hard-threshold tiny eigenvalues; equivalent to Moore-Penrose truncation.
    inv_evals = tf.where(evals > thresh, 1.0 / evals, tf.zeros_like(evals))
    a_pinv = tf.matmul(evecs * inv_evals[None, :], evecs, transpose_b=True)
    return symmetrize_tf(a_pinv)


def zcrit(alpha: float = 0.05) -> float:
    """Return the two-sided standard-normal critical value for ``alpha``."""
    try:
        import tensorflow_probability as tfp

        return float(
            tfp.distributions.Normal(0.0, 1.0).quantile(1.0 - alpha / 2.0).numpy()
        )
    except Exception:
        return 1.959963984540054


def chi2_sf(x: float, df: int) -> float:
    """Return ``P[Chi2(df) >= x]`` via the regularized incomplete gamma."""
    # Uses Q(a, x) = igammac(a, x), where a = df/2 and x = stat/2.
    x_tf = tf.constant(float(x), dtype=tf.float64)
    a_tf = tf.constant(0.5 * float(df), dtype=tf.float64)
    return float(tf.math.igammac(a_tf, 0.5 * x_tf).numpy())


def symmetrize_np(a):
    """NumPy-style wrapper around ``symmetrize_tf``."""
    return symmetrize_tf(tf.convert_to_tensor(a, dtype=tf.float64)).numpy()


def pinv_psd_np(a, rcond: float = 1e-10, ridge: float = 0.0):
    """NumPy-style wrapper around ``pinv_psd_tf``."""
    return pinv_psd_tf(tf.convert_to_tensor(a, dtype=tf.float64), rcond=rcond, ridge=ridge).numpy()
