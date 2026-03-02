"""Environment wrappers for one-step policy transitions.

The environment object packages model constants and the stochastic transition
rule so policy rollouts and simulators can call a single, consistent
"state/action to next state" interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from dyninv.config import BasicModelParams
from dyninv.model.context import build_basic_model_context
from dyninv.model.processes import ar1_step_ln_z
from dyninv.utils import DTYPE, tf


@dataclass
class TransitionOutput:
    """Typed return object for one simulated transition step.

    Attributes:
        k_next: Next-period capital after depreciation and investment.
        z_next: Next-period productivity after AR(1) shock propagation.
        iota: Current-period investment-rate choice produced by the policy.
    """

    k_next: tf.Tensor
    z_next: tf.Tensor
    iota: tf.Tensor


class EconomicEnvironment:
    """TensorFlow environment for single-period dynamics under a policy."""

    def __init__(self, mp: BasicModelParams | None = None):
        """Load model parameters and cache scalar tensors for fast reuse.

        Constants are stored as attributes so repeated transition calls do not
        allocate fresh tensors inside performance-critical loops.
        """
        self.mp = mp or BasicModelParams()
        self.ctx = build_basic_model_context(self.mp, dtype=DTYPE)
        self.delta = self.ctx["delta_tf"]
        self.rho = self.ctx["rho_tf"]
        self.sigma_eps = self.ctx["sigma_eps_tf"]
        self.mu_ln_z = self.ctx["mu_ln_z_tf"]
        self.k_floor = self.ctx["k_floor_tf"]
        self.zero = tf.constant(0.0, dtype=DTYPE)

    @tf.function
    def transition(self, policy, k: tf.Tensor, z: tf.Tensor, theta: tf.Tensor, phi: tf.Tensor) -> TransitionOutput:
        """Advance the economy by one step for a batch of firms.

        Args:
            policy: Trained policy model mapping state/features to ``iota``.
            k: Current capital levels.
            z: Current productivity levels.
            theta: Production-curvature parameter values.
            phi: Adjustment-cost curvature parameter values.

        Returns:
            ``TransitionOutput`` containing next-period ``k``, next-period ``z``,
            and the selected current ``iota``.
        """
        # Canonical feature order is shared across training, simulation, and
        # estimation components that call the same policy network.
        x = tf.stack([tf.math.log(k), tf.math.log(z), theta, tf.math.log(phi)], axis=1)
        iota_t = policy(x, training=False)[:, 0]
        k_next = tf.maximum(self.k_floor, (1.0 - self.delta + iota_t) * k)
        eps = tf.random.normal(tf.shape(k), mean=self.zero, stddev=self.sigma_eps, dtype=DTYPE)
        z_next = ar1_step_ln_z(z, self.rho, eps, self.mu_ln_z)
        return TransitionOutput(k_next=k_next, z_next=z_next, iota=iota_t)
