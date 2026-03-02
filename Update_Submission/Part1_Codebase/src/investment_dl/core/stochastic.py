"""Stochastic process and quadrature helpers.

This module contains all reusable stochastic-state utilities:
- AR(1) transitions in ``ln z``,
- Gauss-Hermite nodes for normal expectations,
- Tauchen discretization for Markov-chain approximations.
"""

from __future__ import annotations

import math
from typing import Tuple

from investment_dl.config import BasicModelParams
from investment_dl.core.tf_env import DTYPE, tf

_GH_CACHE: dict[int, Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = {}


def ar1_step_ln_z(
    z: tf.Tensor,
    rho: tf.Tensor,
    eps: tf.Tensor,
    mu_ln_z: tf.Tensor,
) -> tf.Tensor:
    """Advance AR(1) in logs for productivity.

    The process is:
    ``ln z_{t+1} = mu_ln_z + rho * ln z_t + eps``.
    """
    # Guard against taking log(0) when upstream code clips at tiny floors.
    z_clipped = tf.maximum(z, tf.constant(1e-12, dtype=z.dtype))
    lnz = tf.math.log(z_clipped)
    lnz_next = mu_ln_z + rho * lnz + eps
    return tf.exp(lnz_next)


def get_gh_nodes(n: int, dtype=DTYPE) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return Gauss-Hermite nodes/weights and constants for normal expectations.

    Returns
    -------
    x, w, factor, sqrt2:
        Objects used for the approximation
        ``E[f(eps)] ~= factor * sum_i w_i f(sqrt2 * sigma * x_i)``.
    """
    # Cache by node count to avoid repeated eigendecompositions.
    if n in _GH_CACHE:
        return _GH_CACHE[n]

    n_int = int(n)
    # Build the symmetric tridiagonal Jacobi matrix for Hermite polynomials.
    idx = tf.range(1, n_int, dtype=tf.float64)
    off = tf.sqrt(idx / 2.0)
    diag = tf.zeros((n_int,), dtype=tf.float64)
    J = tf.linalg.diag(diag) + tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1)

    # Nodes are eigenvalues; weights are derived from first-row eigenvectors.
    eigvals, eigvecs = tf.linalg.eigh(J)
    v0 = eigvecs[0, :]
    x = tf.cast(eigvals, dtype=dtype)
    w = tf.cast(tf.square(v0) * tf.sqrt(tf.constant(math.pi, dtype=tf.float64)), dtype=dtype)
    factor = tf.constant(1.0 / math.sqrt(math.pi), dtype=dtype)
    sqrt2 = tf.constant(math.sqrt(2.0), dtype=dtype)
    _GH_CACHE[n] = (x, w, factor, sqrt2)
    return _GH_CACHE[n]


def tauchen_ln_z_grid(
    mp: BasicModelParams,
    n_z: int = 11,
    m_std: float = 3.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Discretize ``ln(z)`` AR(1) with Tauchen (1986).

    Parameters
    ----------
    mp:
        Model parameters (rho, sigma_eps) defining the AR(1) process.
    n_z:
        Number of grid points for productivity states.
    m_std:
        Number of stationary standard deviations used for grid bounds.

    Returns
    -------
    z_grid, P:
        Discrete productivity levels and row-stochastic transition matrix.
    """
    rho = tf.constant(mp.rho, dtype=DTYPE)
    sigma_eps = tf.constant(mp.sigma_eps, dtype=DTYPE)

    # Drift ensuring stationary E[z]=1 under lognormal shocks.
    mu_ln_z = tf.constant(
        -0.5 * (mp.sigma_eps**2) / (1.0 + mp.rho),
        dtype=DTYPE,
    )

    # Stationary moments of ln(z) for AR(1): x'=mu+rho*x+eps.
    sigma_x = sigma_eps / tf.sqrt(1.0 - rho * rho)
    mean_x = tf.constant(
        -0.5 * (mp.sigma_eps**2) / (1.0 - mp.rho * mp.rho),
        dtype=DTYPE,
    )

    # Evenly spaced log-state grid then exponentiate to z-levels.
    x_min = mean_x - m_std * sigma_x
    x_max = mean_x + m_std * sigma_x
    x_grid = tf.linspace(x_min, x_max, n_z)
    z_grid = tf.exp(x_grid)

    dx = x_grid[1] - x_grid[0]

    def normal_cdf(x: tf.Tensor) -> tf.Tensor:
        return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=DTYPE))))

    rows = []
    for i in range(n_z):
        x_i = x_grid[i]
        mean_next = mu_ln_z + rho * x_i

        # Endpoint bins use one-sided Gaussian tails.
        upper = (x_grid[0] - mean_next + dx / 2.0) / sigma_eps
        p0 = normal_cdf(upper)
        row = [p0]

        # Interior bins integrate Gaussian mass over each cell interval.
        for j in range(1, n_z - 1):
            upper = (x_grid[j] - mean_next + dx / 2.0) / sigma_eps
            lower = (x_grid[j] - mean_next - dx / 2.0) / sigma_eps
            row.append(normal_cdf(upper) - normal_cdf(lower))

        lower = (x_grid[-1] - mean_next - dx / 2.0) / sigma_eps
        plast = 1.0 - normal_cdf(lower)
        row.append(plast)

        rows.append(tf.stack(row))

    P = tf.stack(rows, axis=0)
    # Normalize rows for numerical safety.
    P = P / tf.reduce_sum(P, axis=1, keepdims=True)
    return z_grid, P
