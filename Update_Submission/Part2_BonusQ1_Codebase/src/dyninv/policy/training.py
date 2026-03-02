"""Training workflow for the amortized investment policy.

This module defines the end-to-end optimization loop used to fit the policy
network with an Euler-equation objective, including rollout updates and replay
buffer integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import time

from dyninv.config import BasicModelParams, BasicTrainingParams
from dyninv.model.context import iota_bounds
from dyninv.model.primitives import euler_term, psi_i
from dyninv.model.processes import ar1_step_ln_z
from dyninv.policy.network import ParamPolicyNet, policy_iota
from dyninv.policy.rollout import PolicyRollout, RolloutState
from dyninv.policy.sampler import ParameterSampler
from dyninv.utils import DTYPE, ReplayBuffer, set_global_seed, tf


@dataclass
class TrainingResult:
    """Summary object returned by policy training.

    Attributes:
        final_loss: Final diagnostic loss on a fresh coverage sample.
        seconds: Total wall-clock training time.
        save_path: Path where the trained Keras model was written.
    """

    final_loss: float
    seconds: float
    save_path: str


class PolicyTrainer:
    """Train amortized policy with the antithetic-in-antithetic-out loss."""

    def __init__(
        self,
        mp: BasicModelParams | None = None,
        tp: BasicTrainingParams | None = None,
    ):
        """Initialize model components and state for iterative training.

        This includes policy network construction, optimizer setup, replay
        storage, parameter/state samplers, and an initial rollout state.
        """
        self.mp = mp or BasicModelParams()
        self.tp = tp or BasicTrainingParams()
        # Keep TensorFlow and Python RNGs synchronized for reproducible runs.
        set_global_seed(self.mp.seed)

        self.delta = tf.constant(float(self.mp.delta), dtype=DTYPE)
        self.rho = tf.constant(float(self.mp.rho), dtype=DTYPE)
        self.sigma_eps = tf.constant(float(self.mp.sigma_eps), dtype=DTYPE)
        self.r = tf.constant(float(self.mp.r), dtype=DTYPE)
        self.one = tf.constant(1.0, dtype=DTYPE)
        self.half = tf.constant(0.5, dtype=DTYPE)
        self.zero = tf.constant(0.0, dtype=DTYPE)
        self.beta = self.one / (self.one + self.r)
        self.k_floor = tf.constant(1e-12, dtype=DTYPE)
        self.mu_ln_z = tf.constant(-0.5 * (self.mp.sigma_eps**2) / (1.0 + self.mp.rho), dtype=DTYPE)

        iota_min, iota_max = iota_bounds(self.mp)
        self.policy = ParamPolicyNet(
            hidden_sizes=self.tp.hidden_sizes,
            activation=self.tp.activation,
            iota_min=iota_min,
            iota_max=iota_max,
            delta=self.mp.delta,
            r=self.mp.r,
            theta_min=0.5,
            theta_max=0.9,
            log_phi_min=math.log(0.5),
            log_phi_max=math.log(5.0),
            name="param_policy_net",
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.tp.lr)
        self.replay_buffer = ReplayBuffer(max_size=self.tp.buffer_size, state_dim=4, seed=self.mp.seed)
        self.sampler = ParameterSampler(self.mp)
        self.rollout_engine = PolicyRollout(self.mp, replay_buffer=self.replay_buffer)

        # Initialize rollout capital at steady state for each sampled theta.
        theta0, phi0 = self.sampler.sample_parameters(self.tp.n_paths)
        ln_k_star = tf.math.log(theta0 / (self.r + self.delta)) / (self.one - theta0)
        k0 = tf.exp(ln_k_star)
        z0 = tf.ones([self.tp.n_paths], dtype=DTYPE)
        self.current_state = RolloutState(k=k0, z=z0, theta=theta0, phi=phi0)

    @tf.function
    def euler_aio_loss(self, k_batch: tf.Tensor, z_batch: tf.Tensor, theta_batch: tf.Tensor, phi_batch: tf.Tensor) -> tf.Tensor:
        """Compute minibatch Euler loss using antithetic shock pairing.

        The objective forms two independent antithetic averages and multiplies
        them, reducing Monte Carlo noise relative to single-draw residual losses.
        """
        b = tf.shape(k_batch)[0]

        iota_t = policy_iota(self.policy, k_batch, z_batch, theta_batch, phi_batch, training=True)
        k_next = tf.maximum(self.k_floor, (self.one - self.delta + iota_t) * k_batch)
        psi_i_t = psi_i(iota_t, phi_batch, self.delta)

        # Two independent shock draws and their antithetic counterparts.
        eps1 = tf.random.normal([b], mean=self.zero, stddev=self.sigma_eps, dtype=DTYPE)
        eps2 = tf.random.normal([b], mean=self.zero, stddev=self.sigma_eps, dtype=DTYPE)

        z1_plus = ar1_step_ln_z(z_batch, self.rho, eps1, self.mu_ln_z)
        z1_minus = ar1_step_ln_z(z_batch, self.rho, -eps1, self.mu_ln_z)
        z2_plus = ar1_step_ln_z(z_batch, self.rho, eps2, self.mu_ln_z)
        z2_minus = ar1_step_ln_z(z_batch, self.rho, -eps2, self.mu_ln_z)

        # Evaluate four continuation states in one batched forward pass.
        z_all = tf.concat([z1_plus, z1_minus, z2_plus, z2_minus], axis=0)
        k_all = tf.tile(k_next, [4])
        theta_all = tf.tile(theta_batch, [4])
        phi_all = tf.tile(phi_batch, [4])

        iota_all = policy_iota(self.policy, k_all, z_all, theta_all, phi_all, training=True)
        term_all = euler_term(k_all, z_all, iota_all, theta_all, phi_all, self.delta)
        # Reshape back to `(4 shocks, batch)` to combine antithetic pairs.
        term_all = tf.reshape(term_all, [4, -1])

        g1_plus = self.one + psi_i_t - self.beta * term_all[0]
        g1_minus = self.one + psi_i_t - self.beta * term_all[1]
        g2_plus = self.one + psi_i_t - self.beta * term_all[2]
        g2_minus = self.one + psi_i_t - self.beta * term_all[3]

        # Antithetic averaging reduces Monte Carlo variance in the stochastic objective.
        g1_bar = self.half * (g1_plus + g1_minus)
        g2_bar = self.half * (g2_plus + g2_minus)
        return tf.reduce_mean(g1_bar * g2_bar)

    @tf.function
    def train_step(self, k_batch: tf.Tensor, z_batch: tf.Tensor, theta_batch: tf.Tensor, phi_batch: tf.Tensor) -> tf.Tensor:
        """Run one gradient update on the policy parameters.

        Args:
            k_batch: Capital states in the minibatch.
            z_batch: Productivity states in the minibatch.
            theta_batch: Sampled production-curve parameters.
            phi_batch: Sampled adjustment-cost parameters.

        Returns:
            Scalar loss value after applying one optimizer step.
        """
        with tf.GradientTape() as tape:
            loss = self.euler_aio_loss(k_batch, z_batch, theta_batch, phi_batch)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return loss

    def sample_minibatch_states(self, coverage_share: float):
        """Build one SGD minibatch from coverage and replay sources.

        ``coverage_share`` controls the portion of fresh coverage draws; the
        remaining share comes from replay when available.
        """
        n_cov = int(self.tp.batch_size * coverage_share)
        n_buf = int(self.tp.batch_size - n_cov)
        k_cov, z_cov, th_cov, ph_cov = self.sampler.sample_coverage(
            n_cov,
            m_minus=self.tp.k_cov_m_minus,
            m_plus=self.tp.k_cov_m_plus,
        )

        if len(self.replay_buffer) >= n_buf and n_buf > 0:
            buf = self.replay_buffer.sample(n_buf)
            # Replay samples are NumPy-backed; convert back to TensorFlow tensors.
            k_buf = tf.convert_to_tensor(buf[:, 0], dtype=DTYPE)
            z_buf = tf.convert_to_tensor(buf[:, 1], dtype=DTYPE)
            th_buf = tf.convert_to_tensor(buf[:, 2], dtype=DTYPE)
            ph_buf = tf.convert_to_tensor(buf[:, 3], dtype=DTYPE)
        else:
            k_buf, z_buf, th_buf, ph_buf = self.sampler.sample_coverage(
                n_buf,
                m_minus=self.tp.k_cov_m_minus,
                m_plus=self.tp.k_cov_m_plus,
            )

        if n_cov > 0 and n_buf > 0:
            return (
                tf.concat([k_cov, k_buf], axis=0),
                tf.concat([z_cov, z_buf], axis=0),
                tf.concat([th_cov, th_buf], axis=0),
                tf.concat([ph_cov, ph_buf], axis=0),
            )
        if n_cov > 0:
            return k_cov, z_cov, th_cov, ph_cov
        return k_buf, z_buf, th_buf, ph_buf

    def train(self, save_path: str = "param_policy_theta_phi.keras") -> TrainingResult:
        """Run the full two-phase training schedule and save the policy model.

        Phase 1 uses only coverage draws for stable warm start; Phase 2 blends
        replay and coverage samples while annealing toward replay-heavy updates.
        """
        self.current_state = self.rollout_engine.rollout(
            self.policy,
            self.current_state,
            n_steps=self.tp.roll_steps,
        )

        t0 = time()

        # Phase 1: pure coverage pretraining for stable initialization.
        for step in range(1, self.tp.pretrain_steps + 1):
            batches = self.sampler.sample_coverage(
                self.tp.batch_size,
                m_minus=self.tp.k_cov_m_minus,
                m_plus=self.tp.k_cov_m_plus,
            )
            loss = self.train_step(*batches)
            if step % self.tp.roll_steps == 0:
                self.current_state = self.rollout_engine.rollout(self.policy, self.current_state, n_steps=1)
            if step % self.tp.log_every == 0:
                print(f"[Pretrain {step}/{self.tp.pretrain_steps}] Loss={float(loss.numpy()):.4e}")

        # Phase 2: hybrid replay+coverage training for on-policy refinement.
        for step in range(1, self.tp.train_steps + 1):
            # Linearly anneal from full coverage to configured floor share.
            coverage_share = max(
                self.tp.coverage_final_share,
                1.0 - (1.0 - self.tp.coverage_final_share) * (step / self.tp.train_steps),
            )
            loss = self.train_step(*self.sample_minibatch_states(coverage_share))
            if step % self.tp.roll_steps == 0:
                self.current_state = self.rollout_engine.rollout(self.policy, self.current_state, n_steps=1)
            if step % self.tp.log_every == 0:
                print(
                    f"[Train {step}/{self.tp.train_steps}] Loss={float(loss.numpy()):.4e} "
                    f"| CoverageShare={coverage_share:.3f} | Buffer={len(self.replay_buffer)}"
                )

        # Report one final diagnostic on a fresh coverage batch for consistency
        # across runs regardless of the last SGD minibatch composition.
        final_batch = self.sampler.sample_coverage(
            self.tp.batch_size,
            m_minus=self.tp.k_cov_m_minus,
            m_plus=self.tp.k_cov_m_plus,
        )
        final_loss = float(self.euler_aio_loss(*final_batch).numpy())

        self.policy.save(save_path)
        t1 = time()
        return TrainingResult(final_loss=final_loss, seconds=t1 - t0, save_path=save_path)
