"""On-policy rollout helpers used during policy training.

Rollouts advance state distributions under the current policy and optionally
store visited states in replay memory for mixed on/off-policy minibatches.
"""

from __future__ import annotations

from dataclasses import dataclass

from dyninv.config import BasicModelParams
from dyninv.model.environment import EconomicEnvironment
from dyninv.policy.network import policy_iota
from dyninv.utils import DTYPE, ReplayBuffer, tf


@dataclass
class RolloutState:
    """State batch carried between rollout updates.

    Each field is a vector over paths/firms and is updated in lockstep during
    policy rollout steps.
    """

    k: tf.Tensor
    z: tf.Tensor
    theta: tf.Tensor
    phi: tf.Tensor


class PolicyRollout:
    """Roll out trajectories under a fixed policy and update replay memory."""

    def __init__(self, mp: BasicModelParams | None = None, replay_buffer: ReplayBuffer | None = None):
        """Initialize transition environment and optional replay buffer handle."""
        self.mp = mp or BasicModelParams()
        self.env = EconomicEnvironment(self.mp)
        self.replay_buffer = replay_buffer

    @tf.function
    def policy_step(self, policy, k: tf.Tensor, z: tf.Tensor, theta: tf.Tensor, phi: tf.Tensor, training: bool = False):
        """Advance one period for all paths under current policy dynamics.

        Returns the next-period states together with current ``iota`` so callers
        can log or reuse policy actions if needed.
        """
        iota_t = policy_iota(policy, k, z, theta, phi, training=training)
        # Capital update is clipped at a floor to prevent invalid log evaluations later.
        k_next = tf.maximum(self.env.k_floor, (1.0 - self.env.delta + iota_t) * k)
        eps = tf.random.normal(tf.shape(k), mean=0.0, stddev=self.env.sigma_eps, dtype=DTYPE)
        # Productivity follows AR(1) in logs, exponentiated back to levels.
        z_next = tf.exp(self.env.mu_ln_z + self.env.rho * tf.math.log(tf.maximum(z, 1e-12)) + eps)
        return k_next, z_next, iota_t

    def rollout(self, policy, state: RolloutState, n_steps: int) -> RolloutState:
        """Roll state forward for ``n_steps`` and optionally store visited states.

        The replay buffer receives pre-transition states in canonical feature
        order ``(k, z, theta, phi)``.
        """
        k = tf.cast(state.k, DTYPE)
        z = tf.cast(state.z, DTYPE)
        theta = tf.cast(state.theta, DTYPE)
        phi = tf.cast(state.phi, DTYPE)

        for _ in range(int(n_steps)):
            if self.replay_buffer is not None:
                # Buffer stores `(k, z, theta, phi)` state vectors used by trainer minibatches.
                self.replay_buffer.push_batch(tf.stack([k, z, theta, phi], axis=1))
            k, z, _ = self.policy_step(policy, k, z, theta, phi, training=False)

        return RolloutState(k=k, z=z, theta=theta, phi=phi)
