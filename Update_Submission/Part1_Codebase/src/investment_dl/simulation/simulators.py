"""Policy simulation routines.

This module provides two simulation modes:
- flattened ergodic sample collection (`simulate_sample`),
- panel simulation with explicit time/path axes (`simulate_panel`).
"""

from __future__ import annotations

import math
from typing import Callable

from investment_dl.core.stochastic import ar1_step_ln_z
from investment_dl.core.tf_env import DTYPE, set_global_seed, tf
from investment_dl.models.basic_investment import BasicInvestmentModel


class PolicySimulator:
    """Simulate state and control paths under an arbitrary policy function.

    The simulator exposes both flattened-sample and panel-shaped outputs so
    downstream diagnostics can choose whichever layout is most convenient.
    """

    def __init__(self, model: BasicInvestmentModel, default_n_paths: int = 2048) -> None:
        self.model = model
        self.default_n_paths = default_n_paths

    def _init_state(self, n_paths: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Initialize ``(k, z)`` at the deterministic benchmark start state."""
        state = self.model.initial_state(n_paths)
        return state.k, state.z

    def simulate_sample(
        self,
        policy_iota_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        burn_in_steps: int,
        T: int,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Simulate an ergodic sample and return flattened arrays of length ``T``.

        The routine first burns in, then collects blocks of ``n_paths`` states
        per simulation step and truncates to exactly ``T`` observations.
        """
        # Allow callers to override path count without duplicating defaults.
        if n_paths is None:
            n_paths = self.default_n_paths
        if seed is not None:
            set_global_seed(seed)

        k, z = self._init_state(n_paths)

        print(f"  Burn-in under policy: steps = {burn_in_steps}, paths = {n_paths} ...")
        for _ in range(burn_in_steps):
            iota = policy_iota_fn(k, z)
            k = self.model.next_capital(k, iota)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=self.model.zero,
                stddev=self.model.sigma_eps,
                dtype=DTYPE,
            )
            z = ar1_step_ln_z(z, self.model.rho, eps, self.model.mu_ln_z)

        # Collect in full cross-sectional blocks, then truncate to exactly T.
        steps_collect = math.ceil(T / n_paths)
        print(
            f"  Collecting on-policy states: total = {T}, "
            f"via {steps_collect} steps x {n_paths} paths ...",
        )

        # TensorArray stores one full cross-section per simulation step.
        k_ta = tf.TensorArray(DTYPE, size=steps_collect)
        z_ta = tf.TensorArray(DTYPE, size=steps_collect)
        iota_ta = tf.TensorArray(DTYPE, size=steps_collect)

        for t in range(steps_collect):
            iota = policy_iota_fn(k, z)
            k_ta = k_ta.write(t, k)
            z_ta = z_ta.write(t, z)
            iota_ta = iota_ta.write(t, iota)

            k = self.model.next_capital(k, iota)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=self.model.zero,
                stddev=self.model.sigma_eps,
                dtype=DTYPE,
            )
            z = ar1_step_ln_z(z, self.model.rho, eps, self.model.mu_ln_z)

        k_arr = tf.reshape(k_ta.concat(), [-1])[:T]
        z_arr = tf.reshape(z_ta.concat(), [-1])[:T]
        iota_arr = tf.reshape(iota_ta.concat(), [-1])[:T]
        return k_arr, z_arr, iota_arr

    def simulate_panel(
        self,
        policy_iota_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        burn_in_steps: int,
        T: int,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Simulate a panel and return ``(k, z, iota)`` arrays of shape ``(T, n_paths)``.

        Unlike :meth:`simulate_sample`, no flattening is applied, so this output
        is directly usable for panel-moment and autocorrelation calculations.
        """
        # Panel output keeps explicit time x path structure.
        if n_paths is None:
            n_paths = self.default_n_paths
        if seed is not None:
            set_global_seed(seed)

        k, z = self._init_state(n_paths)

        print(f"  Burn-in under policy: steps = {burn_in_steps}, paths = {n_paths} ...")
        for _ in range(burn_in_steps):
            iota = policy_iota_fn(k, z)
            k = self.model.next_capital(k, iota)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=self.model.zero,
                stddev=self.model.sigma_eps,
                dtype=DTYPE,
            )
            z = ar1_step_ln_z(z, self.model.rho, eps, self.model.mu_ln_z)

        print(f"  Simulating panel: steps = {T}, paths = {n_paths} ...")
        # One row per period, one column per path in the final stacked arrays.
        k_ta = tf.TensorArray(DTYPE, size=T)
        z_ta = tf.TensorArray(DTYPE, size=T)
        iota_ta = tf.TensorArray(DTYPE, size=T)

        for t in range(T):
            iota = policy_iota_fn(k, z)
            k_ta = k_ta.write(t, k)
            z_ta = z_ta.write(t, z)
            iota_ta = iota_ta.write(t, iota)

            k = self.model.next_capital(k, iota)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=self.model.zero,
                stddev=self.model.sigma_eps,
                dtype=DTYPE,
            )
            z = ar1_step_ln_z(z, self.model.rho, eps, self.model.mu_ln_z)

        return k_ta.stack(), z_ta.stack(), iota_ta.stack()
