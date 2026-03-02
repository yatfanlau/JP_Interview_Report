"""TensorFlow simulation kernels for synthetic panel generation.

The simulator evolves firm-level capital and productivity states under a fixed
policy and returns tensors formatted for downstream panel-data conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

from dyninv.config import BasicModelParams
from dyninv.model.processes import ar1_step_ln_z, steady_state_k
from dyninv.policy.network import policy_iota
from dyninv.utils import DTYPE, set_global_seed, tf


@dataclass
class SimulationPanel:
    """Container for one simulated replication in panel-tensor format.

    All fields are shaped ``(n_firms, t_periods)`` after burn-in trimming.
    """

    k: tf.Tensor
    i: tf.Tensor
    iota: tf.Tensor
    z: tf.Tensor
    k_next: tf.Tensor
    z_next: tf.Tensor


class PanelSimulator:
    """Simulate firm-level panels under a trained amortized policy model."""

    def __init__(self, mp: BasicModelParams | None = None):
        """Cache simulation constants from model primitives.

        These constants are reused across replications and TensorFlow graph
        calls to keep the simulation loop minimal.
        """
        self.mp = mp or BasicModelParams()
        self.delta = tf.constant(float(self.mp.delta), dtype=DTYPE)
        self.rho = tf.constant(float(self.mp.rho), dtype=DTYPE)
        self.r = tf.constant(float(self.mp.r), dtype=DTYPE)
        self.sigma_eps = tf.constant(float(self.mp.sigma_eps), dtype=DTYPE)
        self.mu_ln_z = tf.constant(-0.5 * (self.mp.sigma_eps**2) / (1.0 + self.mp.rho), dtype=DTYPE)
        self.k_floor = tf.constant(1e-12, dtype=DTYPE)

    @tf.function
    def simulate_panel_hist_tf(
        self,
        policy,
        theta_true: tf.Tensor,
        phi_true: tf.Tensor,
        n_firms: tf.Tensor,
        t_periods: tf.Tensor,
        burn_in: tf.Tensor,
    ) -> SimulationPanel:
        """Simulate one panel path and return burn-in-trimmed histories.

        Args:
            policy: Trained policy model that maps states/params to ``iota``.
            theta_true: Replication-level structural ``theta``.
            phi_true: Replication-level structural ``phi``.
            n_firms: Number of firms to simulate.
            t_periods: Number of retained sample periods.
            burn_in: Number of initial periods to discard.

        Returns:
            ``SimulationPanel`` containing aligned current and next-period
            histories over retained windows.
        """
        # Full simulation length includes burn-in plus retained sample periods.
        t_total = burn_in + t_periods

        # Initialize all firms at deterministic steady-state capital and unit productivity.
        k = tf.fill([n_firms], tf.squeeze(steady_state_k(theta_true, self.delta, self.r)))
        z = tf.ones([n_firms], dtype=DTYPE)
        theta_vec = tf.fill([n_firms], tf.squeeze(theta_true))
        phi_vec = tf.fill([n_firms], tf.squeeze(phi_true))

        # Store full paths in TensorArray because graph loops need static containers.
        k_ta = tf.TensorArray(dtype=DTYPE, size=t_total + 1)
        z_ta = tf.TensorArray(dtype=DTYPE, size=t_total + 1)
        iota_ta = tf.TensorArray(dtype=DTYPE, size=t_total)
        i_ta = tf.TensorArray(dtype=DTYPE, size=t_total)

        k_ta = k_ta.write(0, k)
        z_ta = z_ta.write(0, z)

        t0 = tf.constant(0)

        def cond(t, *_):
            """Continue looping until all periods have been simulated."""
            return t < t_total

        def body(t, k_cur, z_cur, k_ta_v, z_ta_v, iota_ta_v, i_ta_v):
            """Advance one period and write current simulated quantities."""
            iota = policy_iota(policy, k_cur, z_cur, theta_vec, phi_vec, training=False)
            invest = iota * k_cur
            # Capital accumulation plus floor prevents invalid logs downstream.
            k_next = tf.maximum(self.k_floor, (1.0 - self.delta + iota) * k_cur)
            eps = tf.random.normal([n_firms], mean=0.0, stddev=self.sigma_eps, dtype=DTYPE)
            z_next = ar1_step_ln_z(z_cur, self.rho, eps, self.mu_ln_z)

            k_ta_v = k_ta_v.write(t + 1, k_next)
            z_ta_v = z_ta_v.write(t + 1, z_next)
            iota_ta_v = iota_ta_v.write(t, iota)
            i_ta_v = i_ta_v.write(t, invest)
            return t + 1, k_next, z_next, k_ta_v, z_ta_v, iota_ta_v, i_ta_v

        _, _, _, k_ta, z_ta, iota_ta, i_ta = tf.while_loop(
            cond,
            body,
            loop_vars=[t0, k, z, k_ta, z_ta, iota_ta, i_ta],
            # Deterministic sequential updates are preferred for reproducibility.
            parallel_iterations=1,
        )

        # Stacked arrays are `(time, firms)` and are transposed for panel format.
        k_all = k_ta.stack()
        z_all = z_ta.stack()
        iota_all = iota_ta.stack()
        i_all = i_ta.stack()

        # Slice away burn-in and align next-period variables with current-period rows.
        t0_idx = burn_in
        t1_idx = burn_in + t_periods

        return SimulationPanel(
            k=tf.transpose(k_all[t0_idx:t1_idx, :], [1, 0]),
            i=tf.transpose(i_all[t0_idx:t1_idx, :], [1, 0]),
            iota=tf.transpose(iota_all[t0_idx:t1_idx, :], [1, 0]),
            z=tf.transpose(z_all[t0_idx:t1_idx, :], [1, 0]),
            k_next=tf.transpose(k_all[t0_idx + 1 : t1_idx + 1, :], [1, 0]),
            z_next=tf.transpose(z_all[t0_idx + 1 : t1_idx + 1, :], [1, 0]),
        )

    def simulate_panel(
        self,
        policy,
        theta_true: float,
        phi_true: float,
        n_firms: int,
        t_periods: int,
        burn_in: int,
        base_seed: int,
    ) -> SimulationPanel:
        """Run one seeded replication from Python scalar inputs.

        This thin wrapper handles scalar-to-tensor conversion and global seeding
        before dispatching to the traced TensorFlow simulation kernel.
        """
        set_global_seed(base_seed)
        return self.simulate_panel_hist_tf(
            policy=policy,
            theta_true=tf.constant(theta_true, dtype=DTYPE),
            phi_true=tf.constant(phi_true, dtype=DTYPE),
            n_firms=tf.constant(n_firms, dtype=tf.int32),
            t_periods=tf.constant(t_periods, dtype=tf.int32),
            burn_in=tf.constant(burn_in, dtype=tf.int32),
        )
