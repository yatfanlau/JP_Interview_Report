"""Euler-residual diagnostics using Gauss-Hermite quadrature.

The evaluator computes Euler residuals for a candidate policy function on a
state grid and summarizes errors with standard norms.
"""

from __future__ import annotations

from typing import Callable

from investment_dl.core.stochastic import ar1_step_ln_z, get_gh_nodes
from investment_dl.core.tf_env import DTYPE, tf
from investment_dl.models.basic_investment import BasicInvestmentModel


class EulerResidualEvaluator:
    """Evaluate Euler-equation errors for a candidate policy function.

    The evaluator keeps diagnostics separate from training code so any policy
    callable (DL network, VFI interpolator, or custom baseline) can be tested
    under a common integration and reporting convention.
    """

    def __init__(self, model: BasicInvestmentModel) -> None:
        self.model = model

    def residuals_gh(
        self,
        policy_iota_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: tf.Tensor,
        z: tf.Tensor,
        gh_nodes: int = 7,
    ) -> tf.Tensor:
        """Compute pointwise Euler residuals with Gauss-Hermite integration.

        The expectation over next-period shocks is approximated by
        tensorized Gauss-Hermite nodes evaluated in parallel.
        """
        k = tf.convert_to_tensor(k, dtype=DTYPE)
        z = tf.convert_to_tensor(z, dtype=DTYPE)

        iota_t = policy_iota_fn(k, z)
        k_next = self.model.next_capital(k, iota_t)
        psi_i_t = self.model.psi_i(iota_t)

        # Reusable nodes/weights are cached by order in the stochastic module.
        x, w, factor, sqrt2 = get_gh_nodes(gh_nodes, dtype=DTYPE)
        eps = sqrt2 * self.model.sigma_eps * x

        # Expand to [n_nodes, batch] for fully vectorized next-state evaluation.
        z_b = tf.expand_dims(z, 0)
        k_b = tf.expand_dims(k_next, 0)
        eps_b = tf.expand_dims(eps, 1)

        z_next = ar1_step_ln_z(z_b, self.model.rho, eps_b, self.model.mu_ln_z)
        k_next_b = tf.broadcast_to(k_b, tf.shape(z_next))

        k_flat = tf.reshape(k_next_b, [-1])
        z_flat = tf.reshape(z_next, [-1])
        iota_next_flat = policy_iota_fn(k_flat, z_flat)
        iota_next = tf.reshape(iota_next_flat, tf.shape(z_next))

        term_next = self.model.euler_term(k_next_b, z_next, iota_next)

        # Weighted average over nodes produces E[term | current state].
        w_b = tf.reshape(w, [-1, 1])
        exp_term = factor * tf.reduce_sum(w_b * term_next, axis=0)
        return self.model.one + psi_i_t - self.model.beta * exp_term

    def stats(
        self,
        policy_iota_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        k: tf.Tensor,
        z: tf.Tensor,
        gh_nodes: int = 7,
    ) -> dict[str, float]:
        """Return scalar error summaries from residuals on the input states."""
        resid = self.residuals_gh(policy_iota_fn, k, z, gh_nodes=gh_nodes)
        resid_flat = tf.reshape(resid, [-1])
        abs_resid = tf.abs(resid_flat)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(resid_flat)))
        return {
            "rmse": float(rmse.numpy()),
            "mae": float(tf.reduce_mean(abs_resid).numpy()),
            "max_abs": float(tf.reduce_max(abs_resid).numpy()),
        }
