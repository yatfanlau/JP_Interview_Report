"""Value-function-iteration solver and policy interpolation utilities.

This module provides:
1) a discrete-grid Bellman solver for the benchmark policy/value,
2) a bilinear interpolator to evaluate the grid policy off-grid.
"""

from __future__ import annotations

from typing import Callable

from investment_dl.core.stochastic import tauchen_ln_z_grid
from investment_dl.core.tf_env import DTYPE, tf
from investment_dl.models.basic_investment import BasicInvestmentModel


@tf.function
def _bellman_update(
    V_old: tf.Tensor,
    flow: tf.Tensor,
    p_z: tf.Tensor,
    beta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Bellman update and return value and argmax policy index.

    Parameters are expected in the following shapes:
    - ``V_old``: ``(n_k, n_z)``
    - ``flow``: ``(n_k_current, n_k_next, n_z_current)``
    - ``p_z``: ``(n_z_current, n_z_next)``
    """
    # Continuation value EV(k', z) by integrating over z' transitions.
    ev = tf.transpose(tf.tensordot(p_z, V_old, axes=[[1], [1]]), perm=[1, 0])
    # Broadcast EV across current-k dimension to align with flow tensor.
    ev_expanded = tf.expand_dims(ev, axis=0)
    total = flow + beta * ev_expanded
    # Max over discrete next-capital choices.
    V_new = tf.reduce_max(total, axis=1)
    policy_idx = tf.argmax(total, axis=1, output_type=tf.int32)
    return V_new, policy_idx


class VFISolver:
    """Solve the basic model with grid-based value function iteration (VFI).

    The solver precomputes one-period payoffs, iterates Bellman updates until
    convergence, and finally extracts the implied optimal policy on the grid.
    """

    def __init__(self, model: BasicInvestmentModel) -> None:
        self.model = model
        self.params = model.params
        self.k_floor = tf.constant(1e-12, dtype=DTYPE)

    def build_flow_tensor(
        self,
        k_grid: tf.Tensor,
        z_grid: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Precompute flow payoff tensor and implied iota matrix.

        Returns
        -------
        flow_masked:
            One-period payoff array with infeasible controls replaced by a
            large negative number, shape ``(n_k, n_k, n_z)``.
        iota:
            Implied investment-rate tensor for each ``(k, k')`` pair.
        """
        # Grid sizes are static Python integers in this eager setup.
        n_k = k_grid.shape[0]
        n_z = z_grid.shape[0]

        # Broadcast structure:
        # k_cur  -> (n_k, 1,   1)
        # k_next -> (1,   n_k, 1)
        # z      -> (1,   1,   n_z)
        k_cur = tf.reshape(k_grid, [n_k, 1, 1])
        k_next = tf.reshape(k_grid, [1, n_k, 1])
        z = tf.reshape(z_grid, [1, 1, n_z])

        delta = tf.constant(self.params.delta, dtype=DTYPE)
        phi = tf.constant(self.params.phi, dtype=DTYPE)
        theta = tf.constant(self.params.theta, dtype=DTYPE)

        I = k_next - (1.0 - delta) * k_cur
        iota = I / tf.maximum(k_cur, self.k_floor)

        # Flow payoff e = production - adjustment cost - investment.
        profit = z * tf.pow(k_cur, theta)
        adj_per_k = 0.5 * phi * tf.square(iota - delta)
        adj_cost = adj_per_k * k_cur
        flow = profit - adj_cost - I

        iota_min = tf.constant(self.model.iota_min, dtype=DTYPE)
        iota_max = tf.constant(self.model.iota_max, dtype=DTYPE)
        admissible = tf.logical_and(iota >= iota_min, iota <= iota_max)
        neg_inf = tf.constant(-1.0e20, dtype=DTYPE)
        # Inadmissible choices are dominated in the Bellman max.
        flow_masked = tf.where(admissible, flow, neg_inf)
        return flow_masked, iota

    def solve(
        self,
        n_k: int = 201,
        n_z: int = 11,
        k_min_mul: float | None = None,
        k_max_mul: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = True,
        return_history: bool = False,
    ):
        """Solve VFI and return ``(k_grid, z_grid, V, iota_policy[, history])``.

        Notes
        -----
        Convergence is checked by the sup norm of consecutive value iterates.
        """
        if k_min_mul is None:
            k_min_mul = 0.2
        if k_max_mul is None:
            k_max_mul = 5.0

        # Build capital grid around steady state using user multipliers.
        k_min = k_min_mul * self.model.k_star
        k_max = k_max_mul * self.model.k_star

        k_grid = tf.linspace(
            tf.constant(k_min, dtype=DTYPE),
            tf.constant(k_max, dtype=DTYPE),
            n_k,
        )
        z_grid, p_z = tauchen_ln_z_grid(self.params, n_z=n_z, m_std=3.0)
        flow, iota_ij = self.build_flow_tensor(k_grid, z_grid)

        beta = tf.constant(1.0 / (1.0 + self.params.r), dtype=DTYPE)
        # Start from zero value function; contraction mapping handles transients.
        V = tf.zeros([n_k, n_z], dtype=DTYPE)

        history = {"iter": [], "sup_norm": []}
        diff_val = float("inf")
        policy_idx = tf.zeros([n_k, n_z], dtype=tf.int32)

        # Standard fixed-point iteration over value function.
        for it in range(max_iter):
            V_new, policy_idx = _bellman_update(V, flow, p_z, beta)
            diff = tf.reduce_max(tf.abs(V_new - V))
            diff_val = float(diff.numpy())
            V = V_new

            if return_history:
                history["iter"].append(it)
                history["sup_norm"].append(diff_val)

            if verbose and (it % 50 == 0 or it == max_iter - 1):
                print(f"[VFI] iter {it:4d}, sup-norm diff = {diff_val:.3e}")

            if diff_val < tol:
                if verbose:
                    print(f"[VFI] converged in {it + 1} iterations, diff = {diff_val:.3e}")
                break
        else:
            print(
                "[VFI] WARNING: did not converge "
                f"after {max_iter} iterations; last diff={diff_val:.3e}",
            )

        iota_ij_2d = tf.squeeze(iota_ij, axis=2)
        # Gather chosen iota(k,z) using policy indices over k' dimension.
        iota_policy = tf.gather(iota_ij_2d, policy_idx, axis=1, batch_dims=1)

        if return_history:
            history["converged_iter"] = it + 1
            history["converged"] = diff_val < tol
            return k_grid, z_grid, V, iota_policy, history
        return k_grid, z_grid, V, iota_policy

    def evaluate_policy_on_grid(
        self,
        k_grid: tf.Tensor,
        z_grid: tf.Tensor,
        policy_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        """Evaluate any policy function on the ``(k,z)`` grid.

        The policy callable is expected to accept flattened 1D tensors
        ``policy_fn(k_flat, z_flat)`` and return a 1D tensor of iota values.
        """
        n_k = k_grid.shape[0]
        n_z = z_grid.shape[0]

        K, Z = tf.meshgrid(k_grid, z_grid, indexing="ij")
        k_flat = tf.reshape(K, [-1])
        z_flat = tf.reshape(Z, [-1])
        iota_flat = policy_fn(k_flat, z_flat)
        return tf.reshape(iota_flat, [n_k, n_z])


class VFIInterpolatedPolicy:
    """Bilinear interpolator for a VFI policy on ``(k_grid, z_grid)``.

    This is used whenever policy queries fall off the discrete grid, such as
    during simulation, Euler diagnostics, or DL-vs-VFI comparisons.
    """

    def __init__(self, k_grid: tf.Tensor, z_grid: tf.Tensor, iota_vfi: tf.Tensor) -> None:
        self.k_grid = tf.convert_to_tensor(k_grid, dtype=DTYPE)
        self.z_grid = tf.convert_to_tensor(z_grid, dtype=DTYPE)
        self.iota_vfi = tf.convert_to_tensor(iota_vfi, dtype=DTYPE)

        self.k_min = self.k_grid[0]
        self.k_max = self.k_grid[-1]
        self.z_min = self.z_grid[0]
        self.z_max = self.z_grid[-1]

        self.n_k = tf.shape(self.k_grid)[0]
        self.n_z = tf.shape(self.z_grid)[0]

    def __call__(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Interpolate iota on arbitrary state points.

        Points outside the grid are clipped to boundary values before
        interpolation to avoid extrapolation artifacts.
        """
        k = tf.reshape(tf.convert_to_tensor(k, dtype=DTYPE), [-1])
        z = tf.reshape(tf.convert_to_tensor(z, dtype=DTYPE), [-1])

        k_c = tf.clip_by_value(k, self.k_min, self.k_max)
        z_c = tf.clip_by_value(z, self.z_min, self.z_max)

        # Locate containing cell indices for each query point.
        i = tf.searchsorted(self.k_grid, k_c, side="right") - 1
        j = tf.searchsorted(self.z_grid, z_c, side="right") - 1
        i = tf.clip_by_value(i, 0, self.n_k - 2)
        j = tf.clip_by_value(j, 0, self.n_z - 2)

        # Corner coordinates of each containing grid cell.
        k0 = tf.gather(self.k_grid, i)
        k1 = tf.gather(self.k_grid, i + 1)
        z0 = tf.gather(self.z_grid, j)
        z1 = tf.gather(self.z_grid, j + 1)

        wk = (k_c - k0) / (k1 - k0 + tf.constant(1e-12, dtype=DTYPE))
        wz = (z_c - z0) / (z1 - z0 + tf.constant(1e-12, dtype=DTYPE))

        idx00 = tf.stack([i, j], axis=1)
        idx10 = tf.stack([i + 1, j], axis=1)
        idx01 = tf.stack([i, j + 1], axis=1)
        idx11 = tf.stack([i + 1, j + 1], axis=1)

        v00 = tf.gather_nd(self.iota_vfi, idx00)
        v10 = tf.gather_nd(self.iota_vfi, idx10)
        v01 = tf.gather_nd(self.iota_vfi, idx01)
        v11 = tf.gather_nd(self.iota_vfi, idx11)

        # Two-stage bilinear interpolation: first in k, then in z.
        v0 = v00 + (v10 - v00) * wk
        v1 = v01 + (v11 - v01) * wk
        return v0 + (v1 - v0) * wz
