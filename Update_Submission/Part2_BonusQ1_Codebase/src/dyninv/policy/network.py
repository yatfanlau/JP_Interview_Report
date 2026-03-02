"""Policy-network components for amortized investment decisions.

The policy takes state variables and structural parameters as inputs and returns
the bounded investment rate ``iota`` used throughout simulation and estimation.
"""

from __future__ import annotations

import math

from dyninv.utils import DTYPE, tf


@tf.keras.utils.register_keras_serializable(package="dyninv")
class StateFeatureLayer(tf.keras.layers.Layer):
    """Deterministic feature transform for raw policy inputs.

    The layer recenters and rescales inputs so the downstream MLP sees features
    on comparable scales, which improves training stability.
    """

    def __init__(
        self,
        delta: float,
        r: float,
        theta_min: float,
        theta_max: float,
        log_phi_min: float,
        log_phi_max: float,
        **kwargs,
    ):
        """Store normalization constants used by the feature transform."""
        super().__init__(**kwargs)
        self.delta = float(delta)
        self.r = float(r)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.log_phi_min = float(log_phi_min)
        self.log_phi_max = float(log_phi_max)

    def get_config(self):
        """Return serializable configuration for model save/load cycles."""
        cfg = super().get_config()
        cfg.update(
            {
                "delta": self.delta,
                "r": self.r,
                "theta_min": self.theta_min,
                "theta_max": self.theta_max,
                "log_phi_min": self.log_phi_min,
                "log_phi_max": self.log_phi_max,
            }
        )
        return cfg

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Map raw ``(lnk, lnz, theta, log_phi)`` into normalized features.

        The output keeps economically meaningful coordinates (deviation from
        steady-state capital, productivity level, normalized parameters) while
        reducing scale disparities across channels.
        """
        x = tf.cast(x, DTYPE)
        one = tf.constant(1.0, dtype=DTYPE)
        lnk = x[:, 0:1]
        lnz = x[:, 1:2]
        theta = x[:, 2:3]
        log_phi = x[:, 3:4]

        # Center log capital around its steady-state level so the network learns deviations.
        ln_k_star = tf.math.log(theta / tf.constant(self.r + self.delta, dtype=DTYPE)) / (one - theta)
        lnk_rel = lnk - ln_k_star

        # Affine scaling to roughly [-1, 1] improves optimizer conditioning.
        theta_norm = (2.0 * (theta - self.theta_min) / (self.theta_max - self.theta_min)) - one
        log_phi_norm = (2.0 * (log_phi - self.log_phi_min) / (self.log_phi_max - self.log_phi_min)) - one
        return tf.concat([lnk_rel, lnz, theta_norm, log_phi_norm], axis=1)


@tf.keras.utils.register_keras_serializable(package="dyninv")
class ParamPolicyNet(tf.keras.Model):
    """Amortized MLP policy ``(ln k, ln z, theta, log phi) -> iota``."""

    def __init__(
        self,
        hidden_sizes=(64, 64),
        activation="tanh",
        iota_min: float = -0.891,
        iota_max: float = 0.5,
        delta: float = 0.1,
        r: float = 0.04,
        theta_min: float = 0.5,
        theta_max: float = 0.9,
        log_phi_min: float = math.log(0.5),
        log_phi_max: float = math.log(5.0),
        **kwargs,
    ):
        """Initialize architecture, feature layer, and output bounds.

        ``iota_min`` and ``iota_max`` are enforced exactly by an output
        squashing transform, so simulated dynamics remain inside configured
        feasibility limits.
        """
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation = str(activation)
        self.iota_min = float(iota_min)
        self.iota_max = float(iota_max)
        self.delta = float(delta)
        self.r = float(r)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.log_phi_min = float(log_phi_min)
        self.log_phi_max = float(log_phi_max)

        self.features = StateFeatureLayer(
            delta=self.delta,
            r=self.r,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            log_phi_min=self.log_phi_min,
            log_phi_max=self.log_phi_max,
            name="state_feature_layer",
        )
        self.hidden = [
            tf.keras.layers.Dense(self.hidden_sizes[0], activation=self.activation),
            tf.keras.layers.Dense(self.hidden_sizes[1], activation=self.activation),
        ]
        self.out = tf.keras.layers.Dense(1, activation=None)

    def get_config(self):
        """Return model configuration for Keras serialization."""
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_sizes": list(self.hidden_sizes),
                "activation": self.activation,
                "iota_min": self.iota_min,
                "iota_max": self.iota_max,
                "delta": self.delta,
                "r": self.r,
                "theta_min": self.theta_min,
                "theta_max": self.theta_max,
                "log_phi_min": self.log_phi_min,
                "log_phi_max": self.log_phi_max,
            }
        )
        return cfg

    @tf.function
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Predict bounded investment rates for a batch of inputs.

        The network predicts an unconstrained scalar and then applies
        ``tanh`` plus an affine map so the final action is always in
        ``[iota_min, iota_max]``.
        """
        feats = self.features(x)
        h = feats
        for layer in self.hidden:
            h = layer(h, training=training)
        raw = self.out(h, training=training)
        one = tf.constant(1.0, dtype=DTYPE)
        half = tf.constant(0.5, dtype=DTYPE)
        span = tf.constant(self.iota_max - self.iota_min, dtype=DTYPE)
        # Tanh squashes to [-1, 1], then affine map enforces exact policy bounds.
        return tf.constant(self.iota_min, dtype=DTYPE) + half * (tf.tanh(raw) + one) * span


@tf.function
def policy_iota(policy: tf.keras.Model, k: tf.Tensor, z: tf.Tensor, theta: tf.Tensor, phi: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Convenience wrapper to evaluate ``iota = I / k`` from a policy model.

    This helper keeps feature ordering consistent across all call sites.
    """
    # All estimators/simulators share this canonical feature packing order.
    x = tf.stack([tf.math.log(k), tf.math.log(z), theta, tf.math.log(phi)], axis=1)
    return policy(x, training=training)[:, 0]
