"""Sampling utilities for structural parameters and coverage states.

These draws feed policy training, where broad support over states/parameters is
important for amortization quality.
"""

from __future__ import annotations

import math

from dyninv.config import BasicModelParams, ParamBoundsPart2
from dyninv.model.processes import steady_state_ln_k
from dyninv.utils import DTYPE, tf


class ParameterSampler:
    """Sample model parameters and state points for training batches."""

    def __init__(
        self,
        mp: BasicModelParams | None = None,
        bounds: ParamBoundsPart2 | None = None,
    ):
        """Cache bounds and stationary moments used by random samplers.

        Precomputing scalar tensors keeps sampling methods lightweight when
        called repeatedly from traced training steps.
        """
        self.mp = mp or BasicModelParams()
        self.bounds = bounds or ParamBoundsPart2()
        self.theta_min = tf.constant(float(self.bounds.theta_min), dtype=DTYPE)
        self.theta_max = tf.constant(float(self.bounds.theta_max), dtype=DTYPE)
        self.log_phi_min = tf.constant(math.log(float(self.bounds.phi_min)), dtype=DTYPE)
        self.log_phi_max = tf.constant(math.log(float(self.bounds.phi_max)), dtype=DTYPE)
        self.delta = tf.constant(float(self.mp.delta), dtype=DTYPE)
        self.r = tf.constant(float(self.mp.r), dtype=DTYPE)
        self.rho = tf.constant(float(self.mp.rho), dtype=DTYPE)
        self.sigma_eps = tf.constant(float(self.mp.sigma_eps), dtype=DTYPE)
        # Precompute stationary ln z moments once to avoid repeated Python math in graph calls.
        self.stationary_lnz_std = tf.constant(
            float(self.mp.sigma_eps / math.sqrt(1.0 - self.mp.rho * self.mp.rho)),
            dtype=DTYPE,
        )
        self.stationary_lnz_mean = tf.constant(
            float(-0.5 * (self.mp.sigma_eps**2) / (1.0 - self.mp.rho * self.mp.rho)),
            dtype=DTYPE,
        )

    @tf.function
    def sample_parameters(self, batch_size: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Draw independent ``(theta, phi)`` pairs from configured bounds."""
        # Uniform draws over configured structural bounds.
        theta = tf.random.uniform([batch_size], self.theta_min, self.theta_max, dtype=DTYPE)
        log_phi = tf.random.uniform([batch_size], self.log_phi_min, self.log_phi_max, dtype=DTYPE)
        phi = tf.exp(log_phi)
        return theta, phi

    @tf.function
    def sample_coverage(self, batch_size: int, m_minus: float = 0.2, m_plus: float = 5.0) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Draw ``(k, z, theta, phi)`` from the coverage distribution.

        ``k`` is sampled relative to steady-state capital and ``z`` from a
        clipped stationary approximation, producing diverse but controlled
        training support.
        """
        theta, phi = self.sample_parameters(batch_size)
        # Draw multiplicative deviations around steady-state capital in log space.
        ln_kk = tf.random.uniform(
            [batch_size],
            tf.math.log(tf.constant(m_minus, dtype=DTYPE)),
            tf.math.log(tf.constant(m_plus, dtype=DTYPE)),
            dtype=DTYPE,
        )
        ln_k_star = steady_state_ln_k(theta, self.delta, self.r)
        k = tf.exp(ln_k_star + ln_kk)

        # Draw productivity from stationary approximation and clip 3-sigma tails.
        lnz = tf.random.normal(
            [batch_size],
            mean=self.stationary_lnz_mean,
            stddev=self.stationary_lnz_std,
            dtype=DTYPE,
        )
        lnz = tf.clip_by_value(
            lnz,
            self.stationary_lnz_mean - 3.0 * self.stationary_lnz_std,
            self.stationary_lnz_mean + 3.0 * self.stationary_lnz_std,
        )
        z = tf.exp(lnz)
        return k, z, theta, phi
