"""Core economic primitives used in Euler-equation computations.

These functions keep the model algebra in one place so training losses and
estimators reference the same formulas.
"""

from __future__ import annotations

from dyninv.utils import DTYPE, tf

ONE = tf.constant(1.0, dtype=DTYPE)
HALF = tf.constant(0.5, dtype=DTYPE)


@tf.function
def profit_k(k: tf.Tensor, z: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """Compute marginal product of capital ``d pi / d k``."""
    return z * theta * tf.pow(k, theta - ONE)


@tf.function
def psi_i(iota: tf.Tensor, phi: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Derivative of adjustment-cost term with respect to ``iota = I/k``."""
    return phi * (iota - delta)


@tf.function
def psi_k(iota: tf.Tensor, phi: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Derivative of adjustment-cost contribution with respect to capital."""
    return HALF * phi * (delta * delta - tf.square(iota))


@tf.function
def euler_term(k: tf.Tensor, z: tf.Tensor, iota: tf.Tensor, theta: tf.Tensor, phi: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Compute the continuation-value term entering the Euler condition.

    This combines marginal product, envelope adjustment term, and depreciated
    capital continuation effects under the model's quadratic adjustment costs.
    """
    return profit_k(k, z, theta) - psi_k(iota, phi, delta) + (ONE - delta) * (ONE + psi_i(iota, phi, delta))
