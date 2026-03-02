"""Deep-learning Euler-equation solver for the basic investment model.

The trainer in this module learns a policy function directly from Euler
conditions, without fitting a separate value network. The objective combines
Monte Carlo expectations and antithetic variates to reduce variance.
"""

from __future__ import annotations

import time

from investment_dl.config import BasicTrainingParams
from investment_dl.core.replay_buffer import ReplayBuffer
from investment_dl.core.stochastic import ar1_step_ln_z
from investment_dl.core.tf_env import DTYPE, tf
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.models.policies import PolicyNetwork


class EulerEquationTrainer:
    """Train a policy network by minimizing Euler-equation residual moments.

    The class owns all mutable training state (network weights, optimizer,
    replay buffer, and current on-policy states), which replaces the original
    global-state functional implementation.
    """

    def __init__(
        self,
        model: BasicInvestmentModel,
        training_params: BasicTrainingParams | None = None,
    ) -> None:
        """Build trainer components and initialize on-policy state ensemble.

        This wires together policy network, optimizer, replay memory, and the
        current simulation state used to refresh on-policy samples.
        """
        self.model = model
        self.training_params = training_params or BasicTrainingParams()

        # Policy and optimizer.
        self.policy = PolicyNetwork(
            iota_min=self.model.iota_min,
            iota_max=self.model.iota_max,
            hidden_sizes=self.training_params.hidden_sizes,
            activation=self.training_params.activation,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_params.lr)
        self.train_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

        # Replay memory stores [k, z] states collected from rollouts.
        self.replay_buffer = ReplayBuffer(
            max_size=self.training_params.buffer_size,
            state_dim=2,
            seed=self.model.params.seed,
        )
        # Deterministic initialization at the steady-state reference point.
        initial_state = self.model.initial_state(self.training_params.n_paths)
        self.current_k = initial_state.k
        self.current_z = initial_state.z

    def policy_iota(self, k: tf.Tensor, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Evaluate policy network and return ``iota=I/k`` controls.

        Inputs are raw states; feature transformation is delegated to the model.
        """
        x = self.model.policy_input(k, z)
        return self.policy(x, training=training)[:, 0]

    @tf.function
    def policy_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Advance one stochastic step under the current policy.

        Returns
        -------
        k_next, z_next, iota_t:
            Next-period state and current control for each simulated path.
        """
        iota_t = self.policy_iota(k, z, training=training)
        k_next = self.model.next_capital(k, iota_t)
        # Draw independent innovations for each simulated path.
        eps = tf.random.normal(
            shape=tf.shape(k),
            mean=self.model.zero,
            stddev=self.model.sigma_eps,
            dtype=DTYPE,
        )
        z_next = ar1_step_ln_z(z, self.model.rho, eps, self.model.mu_ln_z)
        return k_next, z_next, iota_t

    def rollout_on_policy(self, n_steps: int) -> None:
        """Collect on-policy states into the replay buffer.

        The buffer receives pre-transition states, so each recorded point
        corresponds to a state where the policy is actually queried.
        """
        k = tf.convert_to_tensor(self.current_k, dtype=DTYPE)
        z = tf.convert_to_tensor(self.current_z, dtype=DTYPE)

        for _ in range(n_steps):
            states = tf.stack([k, z], axis=1)
            self.replay_buffer.push_batch(states)
            k, z, _ = self.policy_step(k, z, training=False)

        self.current_k = k
        self.current_z = z

    @tf.function
    def euler_aio_loss(self, k_batch: tf.Tensor, z_batch: tf.Tensor) -> tf.Tensor:
        """Return AiO Euler loss with antithetic variates.

        This constructs two independent shock draws and their antithetic
        counterparts, then multiplies two antithetic-averaged Euler residuals
        (`g1_bar * g2_bar`) before averaging over the batch.
        """
        B = tf.shape(k_batch)[0]

        # Current-period control and implied next capital.
        iota_t = self.policy_iota(k_batch, z_batch, training=True)
        k_next = self.model.next_capital(k_batch, iota_t)
        psi_i_t = self.model.psi_i(iota_t)

        # Two independent Monte Carlo draws.
        eps1 = tf.random.normal(
            shape=(B,),
            mean=self.model.zero,
            stddev=self.model.sigma_eps,
            dtype=DTYPE,
        )
        eps2 = tf.random.normal(
            shape=(B,),
            mean=self.model.zero,
            stddev=self.model.sigma_eps,
            dtype=DTYPE,
        )

        # Antithetic pairing: (+eps, -eps) for each draw.
        z1_plus = ar1_step_ln_z(z_batch, self.model.rho, eps1, self.model.mu_ln_z)
        z1_minus = ar1_step_ln_z(z_batch, self.model.rho, -eps1, self.model.mu_ln_z)
        z2_plus = ar1_step_ln_z(z_batch, self.model.rho, eps2, self.model.mu_ln_z)
        z2_minus = ar1_step_ln_z(z_batch, self.model.rho, -eps2, self.model.mu_ln_z)

        # Evaluate continuation term at all four shock realizations in one pass.
        z_all = tf.concat([z1_plus, z1_minus, z2_plus, z2_minus], axis=0)
        k_all = tf.tile(k_next, multiples=[4])

        iota_all = self.policy_iota(k_all, z_all, training=True)
        term_all = self.model.euler_term(k_all, z_all, iota_all)

        term_all = tf.reshape(term_all, (4, -1))
        term1_plus = term_all[0]
        term1_minus = term_all[1]
        term2_plus = term_all[2]
        term2_minus = term_all[3]

        # Euler residuals g = 1 + psi_i(iota_t) - beta * continuation_term.
        g1_plus = self.model.one + psi_i_t - self.model.beta * term1_plus
        g1_minus = self.model.one + psi_i_t - self.model.beta * term1_minus
        g2_plus = self.model.one + psi_i_t - self.model.beta * term2_plus
        g2_minus = self.model.one + psi_i_t - self.model.beta * term2_minus

        # Antithetic averages remove odd-order noise terms.
        g1_bar = self.model.half * (g1_plus + g1_minus)
        g2_bar = self.model.half * (g2_plus + g2_minus)

        # AiO objective sample for each state.
        loss_sample = g1_bar * g2_bar
        return tf.reduce_mean(loss_sample)

    @tf.function
    def train_step(self, k_batch: tf.Tensor, z_batch: tf.Tensor) -> tf.Tensor:
        """Run one SGD step and return batch loss.

        Gradients are taken with respect to policy parameters only.
        """
        # Compute gradients of Euler residual objective w.r.t. policy weights.
        with tf.GradientTape() as tape:
            loss = self.euler_aio_loss(k_batch, z_batch)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        self.train_step_counter.assign_add(1)
        return loss

    def sample_minibatch_states(
        self,
        coverage_share: float,
        batch_size: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample mixed minibatch from coverage and replay buffer.

        `coverage_share` controls exploration states, while the complement is
        sampled from the replay buffer to emphasize on-policy regions.
        """
        # Split batch into exploration (coverage) and replay (on-policy) parts.
        n_cov = int(batch_size * coverage_share)
        n_buf = batch_size - n_cov

        k_cov, z_cov = self.model.coverage_sampler(
            n_cov,
            m_minus=self.training_params.k_cov_m_minus,
            m_plus=self.training_params.k_cov_m_plus,
        )

        if len(self.replay_buffer) >= n_buf and n_buf > 0:
            buf_states = self.replay_buffer.sample(n_buf)
            k_buf = tf.convert_to_tensor(buf_states[:, 0], dtype=DTYPE)
            z_buf = tf.convert_to_tensor(buf_states[:, 1], dtype=DTYPE)
        else:
            # Fallback to coverage draws when replay data is insufficient.
            k_buf, z_buf = self.model.coverage_sampler(
                n_buf,
                m_minus=self.training_params.k_cov_m_minus,
                m_plus=self.training_params.k_cov_m_plus,
            )

        # Return concatenated batch when replay component is non-empty.
        if n_buf > 0:
            k_batch = tf.concat([k_cov, k_buf], axis=0)
            z_batch = tf.concat([z_cov, z_buf], axis=0)
            return k_batch, z_batch
        return k_cov, z_cov

    def train(self, return_history: bool = False) -> dict | None:
        """Run pretraining and main training loops.

        The schedule consists of:
        1) optional coverage-only pretraining,
        2) hybrid training with linearly annealed coverage share.
        """
        history = None
        if return_history:
            history = {
                "pretrain_steps": [],
                "pretrain_loss": [],
                "train_steps": [],
                "train_loss": [],
                "train_cover_share": [],
            }

        # Warm up replay memory with a few policy-driven transitions.
        print("Initialization...")
        self.rollout_on_policy(self.training_params.roll_steps)
        print(f"Replay buffer size after warm-up: {len(self.replay_buffer)}")

        t0 = time.time()

        # Optional coverage-only phase for broad state-space conditioning.
        print(f"Pretrain {self.training_params.pretrain_steps} steps on coverage sampling...")
        for step in range(1, self.training_params.pretrain_steps + 1):
            k_b, z_b = self.model.coverage_sampler(
                self.training_params.batch_size,
                m_minus=self.training_params.k_cov_m_minus,
                m_plus=self.training_params.k_cov_m_plus,
            )
            loss = self.train_step(k_b, z_b)

            if return_history and (
                step % self.training_params.log_every == 0
                or step == 1
                or step == self.training_params.pretrain_steps
            ):
                history["pretrain_steps"].append(step)
                history["pretrain_loss"].append(float(loss.numpy()))

            if step % self.training_params.log_every == 0:
                print(
                    f"[Pretrain {step}/{self.training_params.pretrain_steps}] "
                    f"Loss={loss.numpy():.4e}",
                )

            if step % self.training_params.roll_steps == 0:
                self.rollout_on_policy(1)

        # Main phase blends coverage and replay with a decaying coverage share.
        print(f"Main training {self.training_params.train_steps} steps: hybrid sampling...")
        for step in range(1, self.training_params.train_steps + 1):
            # Start from mostly coverage data and anneal to final share.
            cover_share = max(
                self.training_params.coverage_final_share,
                1.0
                - (1.0 - self.training_params.coverage_final_share)
                * (step / self.training_params.train_steps),
            )

            k_b, z_b = self.sample_minibatch_states(
                coverage_share=cover_share,
                batch_size=self.training_params.batch_size,
            )
            loss = self.train_step(k_b, z_b)

            if step % self.training_params.roll_steps == 0:
                self.rollout_on_policy(1)

            if step % self.training_params.log_every == 0:
                print(
                    f"[Train {step}/{self.training_params.train_steps}] "
                    f"Loss={loss.numpy():.4e} | CoverageShare={cover_share:.3f} "
                    f"| Buffer={len(self.replay_buffer)}",
                )

            if return_history and (
                step % self.training_params.log_every == 0
                or step == 1
                or step == self.training_params.train_steps
            ):
                history["train_steps"].append(step)
                history["train_loss"].append(float(loss.numpy()))
                history["train_cover_share"].append(float(cover_share))

        t1 = time.time()
        print(f"Done. Total training time: {t1 - t0:.2f} sec")
        return history

    def make_policy_function(self):
        """Return an inference-only callable policy ``(k, z) -> iota``.

        This helper provides a lightweight function interface for evaluators
        and simulators that should not toggle training-specific behavior.
        """
        return lambda k, z: self.policy_iota(k, z, training=False)
