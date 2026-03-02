"""Policy-value evaluation and Bellman residual diagnostics.

The utilities here evaluate a fixed policy on a grid and then compute
Bellman residuals using the same interpolation/transition conventions.
"""

from __future__ import annotations

from investment_dl.core.math_utils import tf_quantile_1d
from investment_dl.core.tf_env import tf
from investment_dl.models.basic_investment import BasicInvestmentModel


class PolicyValueEvaluator:
    """Evaluate fixed policies and Bellman consistency on a common grid.

    This class is designed for post-solution diagnostics: it takes policy
    objects as input and computes value/bellman quantities using one shared
    interpolation and transition convention.
    """

    def __init__(self, model: BasicInvestmentModel) -> None:
        self.model = model

    def evaluate_policy_value_on_grid(
        self,
        k_grid: tf.Tensor,
        z_grid: tf.Tensor,
        iota_policy: tf.Tensor,
        p_z: tf.Tensor,
        max_iter: int = 1000,
        tol: float = 1e-5,
        init_V: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Evaluate fixed-policy values on a ``(k,z)`` grid via interpolation in ``k``.

        This solves the linear fixed-point problem for a given policy:
        ``V = flow + beta * E[V(k', z')]``.
        """
        # Use float64 for this diagnostic step to reduce interpolation noise.
        k = tf.cast(k_grid, tf.float64)
        z = tf.cast(z_grid, tf.float64)
        iota = tf.cast(iota_policy, tf.float64)
        P = tf.cast(p_z, tf.float64)

        n_k = tf.shape(iota)[0]
        n_z = tf.shape(iota)[1]

        theta = tf.constant(self.model.params.theta, dtype=tf.float64)
        delta = tf.constant(self.model.params.delta, dtype=tf.float64)
        phi = tf.constant(self.model.params.phi, dtype=tf.float64)
        beta = tf.constant(1.0 / (1.0 + self.model.params.r), dtype=tf.float64)

        # Broadcasted state grids produce vectorized flow and transition terms.
        k_mat = tf.reshape(k, [-1, 1])
        z_mat = tf.reshape(z, [1, -1])

        profit = z_mat * tf.pow(k_mat, theta)
        I = iota * k_mat
        adj = 0.5 * phi * tf.square(iota - delta) * k_mat
        flow = profit - adj - I

        k_next = (1.0 - delta + iota) * k_mat
        k_next = tf.clip_by_value(k_next, k[0], k[-1])

        # Locate interpolation brackets for each k'(k,z) on the k-grid.
        idx = tf.searchsorted(k, tf.reshape(k_next, [-1]), side="right", out_type=tf.int32) - 1
        idx = tf.reshape(idx, tf.shape(k_next))
        idx = tf.clip_by_value(idx, 0, n_k - 2)
        k0 = tf.gather(k, idx)
        k1 = tf.gather(k, idx + 1)
        w = (k_next - k0) / (k1 - k0 + tf.constant(1e-12, dtype=tf.float64))

        # Warm starts reduce fixed-point iterations when a good baseline exists.
        if init_V is None:
            V = tf.zeros([n_k, n_z], dtype=tf.float64)
        else:
            V = tf.cast(init_V, tf.float64)

        for _ in range(max_iter):
            # For each current z, gather V at lower/upper k brackets and
            # interpolate linearly to approximate V(k'(k,z), z').
            n_z_dyn = tf.shape(P)[0]
            V_tile = tf.tile(tf.expand_dims(V, 0), [n_z_dyn, 1, 1])
            idx_T = tf.transpose(idx)
            idx1_T = tf.transpose(idx + 1)
            V0_T = tf.gather(V_tile, idx_T, axis=1, batch_dims=1)
            V1_T = tf.gather(V_tile, idx1_T, axis=1, batch_dims=1)
            w_T = tf.transpose(w)
            V_interp_T = (1.0 - w_T)[..., None] * V0_T + w_T[..., None] * V1_T

            # Transition-average over z' dimension.
            P_expand = tf.expand_dims(P, axis=1)
            EV_T = tf.reduce_sum(V_interp_T * P_expand, axis=2)
            EV = tf.transpose(EV_T)
            V_new = flow + beta * EV
            diff = tf.reduce_max(tf.abs(V_new - V))
            V = V_new
            if float(diff.numpy()) < tol:
                break

        return V

    def compute_bellman_residual_on_grid(
        self,
        k_grid: tf.Tensor,
        z_grid: tf.Tensor,
        iota_policy: tf.Tensor,
        V_policy: tf.Tensor,
        p_z: tf.Tensor,
    ) -> tf.Tensor:
        """Compute Bellman residuals for fixed policy/value on a ``(k,z)`` grid.

        Residual definition:
        ``R(k,z) = V(k,z) - [flow(k,z) + beta * E[V(k',z')]]``.
        """
        k = tf.cast(k_grid, tf.float64)
        z = tf.cast(z_grid, tf.float64)
        iota = tf.cast(iota_policy, tf.float64)
        V = tf.cast(V_policy, tf.float64)
        P = tf.cast(p_z, tf.float64)

        n_k = tf.shape(iota)[0]
        n_z = tf.shape(iota)[1]

        theta = tf.constant(self.model.params.theta, dtype=tf.float64)
        delta = tf.constant(self.model.params.delta, dtype=tf.float64)
        phi = tf.constant(self.model.params.phi, dtype=tf.float64)
        beta = tf.constant(1.0 / (1.0 + self.model.params.r), dtype=tf.float64)

        # Keep flow construction identical to policy-value evaluation.
        k_mat = tf.reshape(k, [-1, 1])
        z_mat = tf.reshape(z, [1, -1])

        profit = z_mat * tf.pow(k_mat, theta)
        I = iota * k_mat
        adj = 0.5 * phi * tf.square(iota - delta) * k_mat
        flow = profit - adj - I

        k_next = (1.0 - delta + iota) * k_mat
        k_next = tf.clip_by_value(k_next, k[0], k[-1])

        # Same interpolation brackets/weights as in policy evaluation.
        idx = tf.searchsorted(k, tf.reshape(k_next, [-1]), side="right", out_type=tf.int32) - 1
        idx = tf.reshape(idx, tf.shape(k_next))
        idx = tf.clip_by_value(idx, 0, n_k - 2)
        k0 = tf.gather(k, idx)
        k1 = tf.gather(k, idx + 1)
        w = (k_next - k0) / (k1 - k0 + tf.constant(1e-12, dtype=tf.float64))

        n_z_dyn = tf.shape(P)[0]
        V_tile = tf.tile(tf.expand_dims(V, 0), [n_z_dyn, 1, 1])
        idx_T = tf.transpose(idx)
        idx1_T = tf.transpose(idx + 1)
        V0_T = tf.gather(V_tile, idx_T, axis=1, batch_dims=1)
        V1_T = tf.gather(V_tile, idx1_T, axis=1, batch_dims=1)
        w_T = tf.transpose(w)
        V_interp_T = (1.0 - w_T)[..., None] * V0_T + w_T[..., None] * V1_T
        P_expand = tf.expand_dims(P, axis=1)
        EV_T = tf.reduce_sum(V_interp_T * P_expand, axis=2)
        EV = tf.transpose(EV_T)

        rhs = flow + beta * EV
        return V - rhs

    def bellman_residual_stats(self, resid: tf.Tensor) -> dict[str, float]:
        """Return summary statistics for Bellman residuals.

        The returned metrics are computed over all grid points after flattening.
        """
        # Flatten over all grid cells for global diagnostics.
        resid_flat = tf.reshape(tf.convert_to_tensor(resid), [-1])
        abs_resid = tf.abs(resid_flat)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(resid_flat)))
        median_abs = tf_quantile_1d(abs_resid, tf.constant(0.5))
        p90_abs = tf_quantile_1d(abs_resid, tf.constant(0.90))
        return {
            "rmse": float(rmse.numpy()),
            "mae": float(tf.reduce_mean(abs_resid).numpy()),
            "max_abs": float(tf.reduce_max(abs_resid).numpy()),
            "mean_abs": float(tf.reduce_mean(abs_resid).numpy()),
            "median_abs": float(median_abs.numpy()),
            "p90_abs": float(p90_abs.numpy()),
        }
