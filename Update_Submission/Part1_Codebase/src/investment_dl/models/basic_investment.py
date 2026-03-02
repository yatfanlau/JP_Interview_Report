"""Economic primitives and TensorFlow constants for the basic model.

This module centralizes the model equations and all scalar constants used in
training, simulation, and evaluation. Keeping these definitions in one class
avoids duplicated formulas across other modules and helps ensure consistency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from investment_dl.config import BasicModelParams
from investment_dl.core.math_utils import steady_state_k
from investment_dl.core.tf_env import DTYPE, set_global_seed, tf


@dataclass
class BasicInvestmentState:
    """Container for model state tensors.

    Attributes
    ----------
    k:
        Capital state for each simulated path.
    z:
        Productivity state for each simulated path.
    """

    k: tf.Tensor
    z: tf.Tensor


class BasicInvestmentModel:
    """Encapsulate economic primitives and reusable TensorFlow constants.

    Notes
    -----
    The class stores both Python scalars (for configuration/reporting) and
    TensorFlow scalars (for vectorized math in downstream code). The TensorFlow
    values are pre-cast to `DTYPE` to avoid repeated casts in tight loops.
    """

    def __init__(self, params: BasicModelParams | None = None) -> None:
        """Initialize model parameters and derived constants.

        Parameters
        ----------
        params:
            Optional model parameter object. If omitted, defaults from
            :class:`BasicModelParams` are used.
        """
        self.params = params or BasicModelParams()
        set_global_seed(self.params.seed)

        # Steady-state scale and policy bounds used by sampling and networks.
        self.k_star = steady_state_k(self.params.theta, self.params.delta, self.params.r)
        self.iota_min = -(self.params.iota_lower_eps) * (1.0 - self.params.delta)
        self.iota_max = self.params.iota_upper

        # Core primitives as TensorFlow scalars for broadcast-safe math.
        self.theta = tf.constant(self.params.theta, dtype=DTYPE)
        self.delta = tf.constant(self.params.delta, dtype=DTYPE)
        self.rho = tf.constant(self.params.rho, dtype=DTYPE)
        self.sigma_eps = tf.constant(self.params.sigma_eps, dtype=DTYPE)
        self.r = tf.constant(self.params.r, dtype=DTYPE)
        self.phi = tf.constant(self.params.phi, dtype=DTYPE)

        self.one = tf.constant(1.0, dtype=DTYPE)
        self.half = tf.constant(0.5, dtype=DTYPE)
        self.zero = tf.constant(0.0, dtype=DTYPE)
        self.beta = self.one / (self.one + self.r)
        self.k_floor = tf.constant(1e-12, dtype=DTYPE)

        # AR(1) intercept chosen so stationary E[z] = 1 under lognormal shocks.
        self.mu_ln_z = tf.constant(
            -0.5 * (self.params.sigma_eps**2) / (1.0 + self.params.rho),
            dtype=DTYPE,
        )

        # Moments of stationary log-z distribution used by coverage sampling.
        self.k_ss_tf = tf.constant(self.k_star, dtype=DTYPE)
        self.sigma_ln_z = tf.constant(
            self.params.sigma_eps / math.sqrt(1.0 - self.params.rho * self.params.rho),
            dtype=DTYPE,
        )
        self.m_ln_z = tf.constant(
            -0.5 * (self.params.sigma_eps**2)
            / (1.0 - self.params.rho * self.params.rho),
            dtype=DTYPE,
        )

    def profit_k(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Return marginal product of capital ``∂π/∂k``.

        Under Cobb-Douglas production ``π(k,z)=z*k^θ``, the derivative is
        ``z * θ * k^(θ-1)``.
        """
        return z * self.theta * tf.pow(k, self.theta - self.one)

    def psi_i(self, iota: tf.Tensor) -> tf.Tensor:
        """Return ``∂ψ/∂iota`` for quadratic adjustment costs.

        With ``ψ(iota)=0.5*phi*(iota-delta)^2``, the slope is linear in iota.
        """
        return self.phi * (iota - self.delta)

    def psi_k(self, iota: tf.Tensor) -> tf.Tensor:
        """Return envelope derivative of adjustment costs with respect to ``k``.

        This term enters the Euler continuation component after applying the
        envelope condition in capital.
        """
        return self.half * self.phi * (self.delta * self.delta - tf.square(iota))

    def policy_input(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Build policy-network input tensor ``[ln(k), ln(z)]``.

        Log states are used because the raw capital scale can be large and
        strongly skewed. The log transform stabilizes network optimization.
        """
        return tf.stack([tf.math.log(k), tf.math.log(z)], axis=1)

    def euler_term(self, k: tf.Tensor, z: tf.Tensor, iota: tf.Tensor) -> tf.Tensor:
        """Return continuation term used inside Euler expectations.

        The expression combines marginal product, adjustment-cost envelope
        effect, and depreciated continuation weight.
        """
        return (
            self.profit_k(k, z)
            - self.psi_k(iota)
            + (self.one - self.delta) * (self.one + self.psi_i(iota))
        )

    def next_capital(self, k: tf.Tensor, iota: tf.Tensor) -> tf.Tensor:
        """Apply the capital law of motion and enforce strict positivity.

        The floor prevents invalid log operations in downstream state transforms.
        """
        return tf.maximum(self.k_floor, (self.one - self.delta + iota) * k)

    def coverage_sampler(
        self,
        batch_size: int,
        m_minus: float = 0.2,
        m_plus: float = 5.0,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample broad-coverage ``(k, z)`` states for exploration.

        ``k`` is sampled log-uniformly on ``[m_minus * k*, m_plus * k*]``.
        ``ln z`` is sampled from a clipped stationary normal approximation.
        """
        k_min = tf.constant(m_minus, dtype=DTYPE) * self.k_ss_tf
        k_max = tf.constant(m_plus, dtype=DTYPE) * self.k_ss_tf

        # Log-uniform in k to avoid over-concentrating at high capital values.
        lnk = tf.random.uniform(
            (batch_size,),
            minval=tf.math.log(k_min),
            maxval=tf.math.log(k_max),
            dtype=DTYPE,
        )
        k = tf.exp(lnk)

        # Stationary-normal approximation for ln z, clipped to control tails.
        lnz = tf.random.normal(
            (batch_size,),
            mean=self.m_ln_z,
            stddev=self.sigma_ln_z,
            dtype=DTYPE,
        )
        lnz = tf.clip_by_value(
            lnz,
            self.m_ln_z - 3.0 * self.sigma_ln_z,
            self.m_ln_z + 3.0 * self.sigma_ln_z,
        )
        z = tf.exp(lnz)
        return k, z

    def initial_state(self, n_paths: int) -> BasicInvestmentState:
        """Return deterministic initial state for each simulated path.

        All paths start at ``(k*, z=1)`` to provide a neutral reference point
        before burn-in dynamics take over.
        """
        k = tf.ones((n_paths,), dtype=DTYPE) * self.k_ss_tf
        z = tf.ones((n_paths,), dtype=DTYPE)
        return BasicInvestmentState(k=k, z=z)
