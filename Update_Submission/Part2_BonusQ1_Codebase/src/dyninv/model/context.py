"""Build reusable scalar constants for the dynamic investment model.

This module centralizes "derived once, reused many times" quantities such as
discount factors, stationary productivity moments, and TensorFlow scalar
constants. Keeping these values in one context dictionary avoids duplicated
formula fragments across training, simulation, and estimation code paths.
"""

from __future__ import annotations

import math

from dyninv.config import BasicModelParams
from dyninv.utils import DTYPE, tf


def build_basic_model_context(mp: BasicModelParams | None = None, dtype=DTYPE) -> dict:
    """Construct a dictionary of Python and TensorFlow model constants.

    Args:
        mp: Optional model-parameter dataclass. If omitted, package defaults are
            used.
        dtype: TensorFlow dtype used for scalar constants consumed by graph code.

    Returns:
        A dictionary containing both Python floats (for host-side logic) and
        TensorFlow scalar tensors (for graph execution), plus commonly reused
        derived values such as ``beta``, ``mu_ln_z``, and capital floors.
    """
    if mp is None:
        mp = BasicModelParams()

    delta_f = float(mp.delta)
    rho_f = float(mp.rho)
    sigma_eps_f = float(mp.sigma_eps)
    r_f = float(mp.r)
    # Discrete-time discount factor implied by gross return (1 + r).
    beta_f = 1.0 / (1.0 + r_f)

    # AR(1) intercept chosen so stationary E[z]=1 normalization is respected.
    mu_ln_z_f = -0.5 * (sigma_eps_f**2) / (1.0 + rho_f)
    # Stationary ln z moments under AR(1): std and mean.
    sigma_ln_z_f = sigma_eps_f / math.sqrt(1.0 - rho_f * rho_f)
    m_ln_z_f = -0.5 * (sigma_eps_f**2) / (1.0 - rho_f * rho_f)

    # Build scalar tensors once so downstream code can reuse them without
    # repeatedly constructing constants inside loops or @tf.function graphs.
    one_tf = tf.constant(1.0, dtype=dtype)
    delta_tf = tf.constant(delta_f, dtype=dtype)
    rho_tf = tf.constant(rho_f, dtype=dtype)
    r_tf = tf.constant(r_f, dtype=dtype)
    beta_tf = tf.constant(beta_f, dtype=dtype)
    mu_ln_z_tf = tf.constant(mu_ln_z_f, dtype=dtype)
    k_floor_tf = tf.constant(1e-12, dtype=dtype)
    sigma_eps_tf = tf.constant(sigma_eps_f, dtype=dtype)

    # The context intentionally exposes both float and tensor variants to support
    # code that runs partly in eager Python and partly in traced TensorFlow.
    return dict(
        mp=mp,
        delta_f=delta_f,
        rho_f=rho_f,
        sigma_eps_f=sigma_eps_f,
        r_f=r_f,
        beta_f=beta_f,
        mu_ln_z_f=mu_ln_z_f,
        sigma_ln_z_f=sigma_ln_z_f,
        m_ln_z_f=m_ln_z_f,
        one_tf=one_tf,
        delta_tf=delta_tf,
        one_minus_delta_tf=one_tf - delta_tf,
        rho_tf=rho_tf,
        r_tf=r_tf,
        beta_tf=beta_tf,
        mu_ln_z_tf=mu_ln_z_tf,
        sigma_eps_tf=sigma_eps_tf,
        k_floor_tf=k_floor_tf,
    )


def iota_bounds(mp: BasicModelParams) -> tuple[float, float]:
    """Compute feasible investment-rate bounds used by the policy network.

    The lower bound permits disinvestment while keeping the implied capital-law
    update numerically safe. The upper bound is a direct model/configuration cap.
    """
    # Lower bound allows disinvestment but keeps capital update numerically stable.
    iota_min = float(-(mp.iota_lower_eps) * (1.0 - mp.delta))
    iota_max = float(mp.iota_upper)
    return iota_min, iota_max
