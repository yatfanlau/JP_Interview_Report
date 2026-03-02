"""Neural-network policy components used by the DL Euler solver.

The architecture is intentionally compact: a two-hidden-layer MLP followed by
a deterministic bounded output layer to enforce feasible ``iota=I/k`` values.
"""

from __future__ import annotations

from typing import Sequence

from investment_dl.core.tf_env import DTYPE, tf


class BoundedTanh(tf.keras.layers.Layer):
    """Map unconstrained activations into a fixed closed interval.

    This layer applies a tanh squash and affine rescaling so the network output
    always satisfies model feasibility bounds.
    """

    def __init__(self, min_val: float | tf.Tensor, max_val: float | tf.Tensor, name: str = "bounded_tanh") -> None:
        super().__init__(name=name)
        self.min_val = tf.cast(min_val, dtype=DTYPE)
        self.max_val = tf.cast(max_val, dtype=DTYPE)
        self.span = self.max_val - self.min_val

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply tanh squashing and rescale to ``[min_val, max_val]``."""
        half = tf.constant(0.5, dtype=DTYPE)
        one = tf.constant(1.0, dtype=DTYPE)
        return self.min_val + half * (tf.tanh(x) + one) * self.span


class PolicyNetwork(tf.keras.Model):
    """Map state features ``(ln k, ln z)`` to bounded investment rate ``iota``.

    The network keeps training simple and stable by:
    1) using dense hidden layers for nonlinear state interactions,
    2) delegating feasibility constraints to :class:`BoundedTanh`.
    """

    def __init__(
        self,
        iota_min: float,
        iota_max: float,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        name: str = "policy_network",
    ) -> None:
        super().__init__(name=name)
        if len(hidden_sizes) != 2:
            raise ValueError("hidden_sizes must have exactly two entries.")

        self.h1 = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)
        self.h2 = tf.keras.layers.Dense(hidden_sizes[1], activation=activation)
        self.out = tf.keras.layers.Dense(1, activation=None)
        self.bound = BoundedTanh(iota_min, iota_max)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return one-column tensor of bounded policy outputs."""
        del training
        # Hidden stack for representation learning over log-states.
        h = self.h1(x)
        h = self.h2(h)
        # Final projection plus hard feasibility transform.
        raw = self.out(h)
        return self.bound(raw)
