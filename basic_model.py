# Deep learning Euler-equation solver for the stochastic investment model.

import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

from config import BasicModelParams, BasicTrainingParams, BasicFinalTestParams
from common import (
    tf,
    DTYPE,
    ReplayBuffer,
    steady_state_k,
    ar1_step_ln_z,
    get_gh_nodes,
    set_global_seed,
)

# ---------------------------------------------------------------------
# 1. Model primitives
# ---------------------------------------------------------------------


def profit_k_tf(k: tf.Tensor, z: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """Marginal product of capital: ∂π/∂k = z * θ * k^(θ-1)."""
    one = tf.constant(1.0, dtype=DTYPE)
    return z * theta * tf.pow(k, theta - one)


def psi_I_tf(iota: tf.Tensor, phi: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """Investment adjustment cost wrt I/k: ψ_I = φ * (iota - δ)."""
    return phi * (iota - delta)


def psi_k_tf(iota: tf.Tensor, phi: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """
    Envelope-based derivative wrt capital:
        ψ_k = (φ/2) * (δ^2 - iota^2).
    """
    half = tf.constant(0.5, dtype=DTYPE)
    return half * phi * (delta * delta - tf.square(iota))


# ---------------------------------------------------------------------
# 2. Policy network (predicts iota = I/k)
# ---------------------------------------------------------------------


class PolicyNet(tf.keras.Model):
    """Simple feed-forward network mapping (ln k, ln z) -> iota = I/k."""

    def __init__(
        self,
        mp: BasicModelParams,
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        name: str = "policy_net",
    ):
        super().__init__(name=name)
        self.mp = mp

        self.h1 = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)
        self.h2 = tf.keras.layers.Dense(hidden_sizes[1], activation=activation)
        self.out = tf.keras.layers.Dense(1, activation=None)

        # Bounds as float32 tensors
        self.iota_min = tf.constant(
            -(mp.iota_lower_eps) * (1.0 - mp.delta), dtype=DTYPE
        )
        self.iota_max = tf.constant(mp.iota_upper, dtype=DTYPE)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        Args:
            x: state tensor with columns [ln k, ln z], shape (B, 2).
        Returns:
            iota: investment rate I/k, shape (B, 1) in [iota_min, iota_max].
        """
        h = self.h1(x)
        h = self.h2(h)
        raw = self.out(h)
        span = self.iota_max - self.iota_min
        half = tf.constant(0.5, dtype=DTYPE)
        one = tf.constant(1.0, dtype=DTYPE)
        iota = self.iota_min + half * (tf.tanh(raw) + one) * span
        return iota


# ---------------------------------------------------------------------
# 3. Coverage sampling and on-policy rollouts
# ---------------------------------------------------------------------


def coverage_sampler(
    batch_size: int,
    mp: BasicModelParams,
    m_minus: float = 0.2,
    m_plus: float = 5.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Coverage sampler for the basic model.

    - k is log-uniform over [m_minus * k*, m_plus * k*].
    - z is log-normal at the stationary distribution of ln z with E[z] = 1,
      truncated at ±3σ around its mean.

    Returns:
        k: shape (B,)
        z: shape (B,)
    """
    # steady state capital
    k_ss = steady_state_k(mp.theta, mp.delta, mp.r)
    k_ss_tf = tf.constant(k_ss, dtype=DTYPE)
    k_min = tf.constant(m_minus, dtype=DTYPE) * k_ss_tf
    k_max = tf.constant(m_plus, dtype=DTYPE) * k_ss_tf

    lnk = tf.random.uniform(
        (batch_size,),
        minval=tf.math.log(k_min),
        maxval=tf.math.log(k_max),
        dtype=DTYPE,
    )
    k = tf.exp(lnk)

    # Stationary distribution of ln z for AR(1) with E[z]=1
    sigma_ln_z = tf.constant(
        mp.sigma_eps / math.sqrt(1.0 - mp.rho * mp.rho), dtype=DTYPE
    )
    m_ln_z = tf.constant(
        -0.5 * (mp.sigma_eps ** 2) / (1.0 - mp.rho * mp.rho),
        dtype=DTYPE,
    )
    lnz = tf.random.normal(
        (batch_size,),
        mean=m_ln_z,
        stddev=sigma_ln_z,
        dtype=DTYPE,
    )
    lnz = tf.clip_by_value(lnz, m_ln_z - 3.0 * sigma_ln_z, m_ln_z + 3.0 * sigma_ln_z)
    z = tf.exp(lnz)
    return k, z


def rollout_on_policy(
    policy: PolicyNet,
    current_states: Tuple[np.ndarray, np.ndarray],
    mp: BasicModelParams,
    n_steps: int,
    buffer: ReplayBuffer,
):
    """
    Roll out the basic model forward under the current policy, and populate
    the replay buffer with visited states (k,z).

    Args:
        policy: PolicyNet
        current_states: tuple of NumPy arrays (k, z), each of shape (N,)
        mp: BasicModelParams
        n_steps: number of rollout steps
        buffer: ReplayBuffer storing states as [k, z]
    Returns:
        new_current_states: (k_np, z_np) after n_steps.
    """
    k_arr, z_arr = current_states
    k = tf.convert_to_tensor(k_arr, dtype=DTYPE)
    z = tf.convert_to_tensor(z_arr, dtype=DTYPE)

    rho = tf.constant(mp.rho, dtype=DTYPE)
    delta = tf.constant(mp.delta, dtype=DTYPE)
    one = tf.constant(1.0, dtype=DTYPE)
    eps_std = tf.constant(mp.sigma_eps, dtype=DTYPE)
    eps_mean = tf.constant(0.0, dtype=DTYPE)
    k_floor = tf.constant(1e-12, dtype=DTYPE)

    # AR(1) intercept mu such that E[z] = 1 in the stationary distribution
    mu_ln_z = tf.constant(
        -0.5 * (mp.sigma_eps ** 2) / (1.0 + mp.rho),
        dtype=DTYPE,
    )

    for _ in range(n_steps):
        # push current states to replay buffer
        states_np = np.stack([k.numpy(), z.numpy()], axis=1).astype(np.float32)
        buffer.push_batch(states_np)

        # policy action and next capital
        x = tf.stack([tf.math.log(k), tf.math.log(z)], axis=1)
        iota = policy(x, training=False)[:, 0]
        k_next = tf.maximum(k_floor, (one - delta + iota) * k)

        # shock and next productivity
        eps = tf.random.normal(
            shape=tf.shape(k),
            mean=eps_mean,
            stddev=eps_std,
            dtype=DTYPE,
        )
        z_next = ar1_step_ln_z(z, rho, eps, mu_ln_z)

        k, z = k_next, z_next

    return k.numpy(), z.numpy()


# ---------------------------------------------------------------------
# 4. Trainer for the basic model
# ---------------------------------------------------------------------


class BasicTrainer:
    """
    Trainer for the basic Euler-equation model using AiO operator and
    hybrid coverage + on-policy sampling.
    """

    def __init__(
        self,
        mp: BasicModelParams,
        tp: BasicTrainingParams,
    ):
        self.mp = mp
        self.tp = tp

        # Seed everything
        set_global_seed(mp.seed)

        # Policy network and optimizer
        self.policy = PolicyNet(
            mp=mp,
            hidden_sizes=tp.hidden_sizes,
            activation=tp.activation,
        )
        self.opt = tf.keras.optimizers.Adam(learning_rate=tp.lr)

        # Replay buffer for states (k,z) as [k, z]
        self.buffer = ReplayBuffer(
            max_size=tp.buffer_size,
            state_dim=2,
            seed=mp.seed,
        )

        # Training settings
        self.n_paths = tp.n_paths
        self.roll_steps = tp.roll_steps
        self.batch_size = tp.batch_size
        self.pretrain_steps = tp.pretrain_steps
        self.train_steps = tp.train_steps
        self.coverage_final_share = tp.coverage_final_share
        self.log_every = tp.log_every
        self.eval_every = tp.eval_every
        self.test_size = tp.test_size

        # Tensor parameters
        self.theta = tf.constant(mp.theta, dtype=DTYPE)
        self.delta = tf.constant(mp.delta, dtype=DTYPE)
        self.rho = tf.constant(mp.rho, dtype=DTYPE)
        self.sigma_eps = tf.constant(mp.sigma_eps, dtype=DTYPE)
        self.r = tf.constant(mp.r, dtype=DTYPE)
        self.phi = tf.constant(mp.phi, dtype=DTYPE)
        self.one = tf.constant(1.0, dtype=DTYPE)
        self.beta = self.one / (self.one + self.r)
        self.k_floor = tf.constant(1e-12, dtype=DTYPE)

        # AR(1) intercept mu for ln z so that E[z] = 1 in the stationary distribution
        self.mu_ln_z = tf.constant(
            -0.5 * (mp.sigma_eps ** 2) / (1.0 + mp.rho),
            dtype=DTYPE,
        )

        # Gauss–Hermite default order for evaluation
        self.default_gh_nodes = 10

        # Initialize on-policy state ensemble at steady state
        k0 = np.ones(self.n_paths, dtype=np.float32) * np.float32(
            steady_state_k(mp.theta, mp.delta, mp.r)
        )
        z0 = np.ones(self.n_paths, dtype=np.float32)
        self.curr_states = (k0, z0)

        # Pre-build a fixed coverage evaluation set for mid-training diagnostics
        self.eval_cov_k_np, self.eval_cov_z_np = self._build_fixed_eval_coverage(
            self.test_size,
            m_minus=tp.k_cov_m_minus,
            m_plus=tp.k_cov_m_plus,
        )

    # ------------------------------------------------------------------
    # AiO Euler loss and train step
    # ------------------------------------------------------------------

    @tf.function
    def _euler_aio_loss(self, k_batch: tf.Tensor, z_batch: tf.Tensor) -> tf.Tensor:
        """
        AiO Euler loss with antithetic variates.
        """
        B = tf.shape(k_batch)[0]

        # current policy and next capital
        x_curr = tf.stack([tf.math.log(k_batch), tf.math.log(z_batch)], axis=1)
        iota_t = self.policy(x_curr, training=True)[:, 0]
        k_next = tf.maximum(self.k_floor, (self.one - self.delta + iota_t) * k_batch)
        psiI_t = psi_I_tf(iota_t, self.phi, self.delta)

        # two independent innovations and antithetic pairs
        eps1 = tf.random.normal(
            shape=(B,),
            mean=0.0,
            stddev=self.sigma_eps,
            dtype=DTYPE,
        )
        eps2 = tf.random.normal(
            shape=(B,),
            mean=0.0,
            stddev=self.sigma_eps,
            dtype=DTYPE,
        )

        z1_plus = ar1_step_ln_z(z_batch, self.rho, eps1, self.mu_ln_z)
        z1_minus = ar1_step_ln_z(z_batch, self.rho, -eps1, self.mu_ln_z)
        z2_plus = ar1_step_ln_z(z_batch, self.rho, eps2, self.mu_ln_z)
        z2_minus = ar1_step_ln_z(z_batch, self.rho, -eps2, self.mu_ln_z)

        # vectorized evaluation at (k_next,z') for four shock combinations
        z_all = tf.concat([z1_plus, z1_minus, z2_plus, z2_minus], axis=0)
        k_all = tf.tile(k_next, multiples=[4])
        x_all = tf.stack([tf.math.log(k_all), tf.math.log(z_all)], axis=1)

        iota_all = self.policy(x_all, training=True)[:, 0]

        term_all = (
            profit_k_tf(k_all, z_all, self.theta)
            - psi_k_tf(iota_all, self.phi, self.delta)
            + (self.one - self.delta)
            * (self.one + psi_I_tf(iota_all, self.phi, self.delta))
        )

        term_all = tf.reshape(term_all, (4, -1))
        term1_plus, term1_minus, term2_plus, term2_minus = (
            term_all[0],
            term_all[1],
            term_all[2],
            term_all[3],
        )

        g1_plus = self.one + psiI_t - self.beta * term1_plus
        g1_minus = self.one + psiI_t - self.beta * term1_minus
        g2_plus = self.one + psiI_t - self.beta * term2_plus
        g2_minus = self.one + psiI_t - self.beta * term2_minus

        g1_bar = 0.5 * (g1_plus + g1_minus)
        g2_bar = 0.5 * (g2_plus + g2_minus)

        loss_sample = g1_bar * g2_bar
        return tf.reduce_mean(loss_sample)

    @tf.function
    def _train_step(self, k_batch: tf.Tensor, z_batch: tf.Tensor) -> tf.Tensor:
        """One SGD step on a batch of (k,z) states."""
        with tf.GradientTape() as tape:
            loss = self._euler_aio_loss(k_batch, z_batch)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy.trainable_variables))
        return loss

    # ------------------------------------------------------------------
    # Mid-training evaluation using a fixed coverage set and GH-10
    # ------------------------------------------------------------------

    def _euler_residuals_gh(
        self, k: tf.Tensor, z: tf.Tensor, n_nodes: int = 10
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute Euler residuals g_mean = E[g] via Gauss–Hermite nodes.

        Returns:
            g_mean: shape (B,)
            one_plus_psiI: shape (B,)
            beta_E_term: shape (B,)
        where g = 1 + ψ_I(iota_t) - β E[term].
        """
        gh_x, gh_w, gh_factor, sqrt2 = get_gh_nodes(n_nodes)

        x_curr = tf.stack([tf.math.log(k), tf.math.log(z)], axis=1)
        iota_t = self.policy(x_curr, training=False)[:, 0]
        k_next = tf.maximum(self.k_floor, (self.one - self.delta + iota_t) * k)
        psiI_t = psi_I_tf(iota_t, self.phi, self.delta)

        eps_nodes = sqrt2 * self.sigma_eps * gh_x  # shape (N,)

        B = tf.shape(k)[0]
        N = tf.shape(gh_x)[0]

        k_next_rep = tf.repeat(tf.expand_dims(k_next, axis=1), repeats=N, axis=1)
        z_next = tf.exp(
            self.rho * tf.expand_dims(tf.math.log(z), axis=1)
            + eps_nodes[tf.newaxis, :]
            + self.mu_ln_z
        )

        x_next = tf.concat(
            [
                tf.math.log(k_next_rep)[..., tf.newaxis],
                tf.math.log(z_next)[..., tf.newaxis],
            ],
            axis=2,
        )
        x_flat = tf.reshape(x_next, (-1, 2))
        iota_next_flat = self.policy(x_flat, training=False)[:, 0]

        term_next_flat = (
            profit_k_tf(tf.reshape(k_next_rep, (-1,)), tf.reshape(z_next, (-1,)), self.theta)
            - psi_k_tf(iota_next_flat, self.phi, self.delta)
            + (self.one - self.delta)
            * (self.one + psi_I_tf(iota_next_flat, self.phi, self.delta))
        )
        term_next = tf.reshape(term_next_flat, (B, -1))

        E_term = gh_factor * tf.reduce_sum(gh_w[tf.newaxis, :] * term_next, axis=1)
        g_nodes = self.one + tf.expand_dims(psiI_t, axis=1) - self.beta * term_next
        g_mean = gh_factor * tf.reduce_sum(gh_w[tf.newaxis, :] * g_nodes, axis=1)

        one_plus_psiI = self.one + psiI_t
        beta_E_term = self.beta * E_term
        return g_mean, one_plus_psiI, beta_E_term

    def _eval_batched(
        self,
        k_np: np.ndarray,
        z_np: np.ndarray,
        n_nodes: int = 10,
        batch: int = 16_384,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched GH evaluation to limit memory pressure.

        Returns:
            g_mean_all: array of shape (N,)
            denom_all: array of shape (N,), where
                       denom = |1+ψ_I| + |β E[term]|.
        """
        N = k_np.shape[0]
        g_list = []
        denom_list = []

        for start in range(0, N, batch):
            end = min(N, start + batch)
            k = tf.convert_to_tensor(k_np[start:end], dtype=DTYPE)
            z = tf.convert_to_tensor(z_np[start:end], dtype=DTYPE)
            g_mean, one_plus_psiI, beta_E_term = self._euler_residuals_gh(
                k, z, n_nodes=n_nodes
            )
            den = tf.abs(one_plus_psiI) + tf.abs(beta_E_term)
            den = tf.maximum(den, tf.constant(1e-12, dtype=DTYPE))

            g_list.append(g_mean.numpy())
            denom_list.append(den.numpy())

        g_all = np.concatenate(g_list, axis=0)
        den_all = np.concatenate(denom_list, axis=0)
        return g_all, den_all

    def _compute_stats(
        self, g: np.ndarray, denom: np.ndarray, tol_list=(1e-3, 1e-4)
    ) -> Dict[str, float]:
        """Compute absolute and relative residual statistics."""
        abs_g = np.abs(g)
        mae = abs_g.mean()
        rmse = np.sqrt((g ** 2).mean())
        med = np.quantile(abs_g, 0.5)
        p95 = np.quantile(abs_g, 0.95)
        mx = abs_g.max()

        abs_clip = np.maximum(abs_g, 1e-300)
        log10_abs = np.log10(abs_clip)
        log10_med = float(np.quantile(log10_abs, 0.5))
        log10_p95 = float(np.quantile(log10_abs, 0.95))

        rel = abs_g / np.maximum(denom, 1e-12)
        rel_mean = float(rel.mean())
        rel_med = float(np.quantile(rel, 0.5))
        rel_p95 = float(np.quantile(rel, 0.95))

        out = dict(
            N=len(g),
            Abs_MAE=float(mae),
            Abs_RMSE=float(rmse),
            Abs_Median=float(med),
            Abs_P95=float(p95),
            Abs_Max=float(mx),
            Log10Abs_Median=log10_med,
            Log10Abs_P95=log10_p95,
            Rel_Mean=rel_mean,
            Rel_Median=rel_med,
            Rel_P95=rel_p95,
        )
        for t in tol_list:
            out[f"Share(|E[g]|<= {t:.0e})"] = float((abs_g <= t).mean())
        return out

    def _print_stats(
        self, title: str, stats: Dict[str, float], tol_list=(1e-3, 1e-4)
    ) -> None:
        """Pretty-print residual statistics."""
        print(f"\n[{title}]")
        print(f"- N = {stats['N']}")
        print(
            f"- Absolute residual |E[g]|: MAE={stats['Abs_MAE']:.3e}, "
            f"RMSE={stats['Abs_RMSE']:.3e}, Median={stats['Abs_Median']:.3e}, "
            f"P95={stats['Abs_P95']:.3e}, Max={stats['Abs_Max']:.3e}"
        )
        print(
            f"- log10(|E[g]|): Median={stats['Log10Abs_Median']:.3f}, "
            f"P95={stats['Log10Abs_P95']:.3f}"
        )
        print(
            "- Relative residual |E[g]|/(|1+psi_I|+|beta*E[term]|): "
            f"Mean={stats['Rel_Mean']:.3e}, Median={stats['Rel_Median']:.3e}, "
            f"P95={stats['Rel_P95']:.3e}"
        )
        for t in tol_list:
            key = f"Share(|E[g]|<= {t:.0e})"
            print(f"- {key}: {stats[key]:.3f}")

    def _eval_accuracy(self) -> Dict[str, float]:
        """
        Stable out-of-sample Euler accuracy using a fixed coverage set
        and GH-10 quadrature.
        """
        g_mean, _ = self._eval_batched(
            self.eval_cov_k_np, self.eval_cov_z_np, n_nodes=self.default_gh_nodes
        )
        abs_g = np.abs(g_mean)
        mae = float(abs_g.mean())
        rmse = float((g_mean ** 2).mean() ** 0.5)
        max_abs = float(abs_g.max())
        tol = 1e-3
        share_ok = float((abs_g <= tol).mean())
        return dict(MAE=mae, RMSE=rmse, MAX=max_abs, ShareWithinTol=share_ok)

    # ------------------------------------------------------------------
    # Training loop (pretrain + main training)
    # ------------------------------------------------------------------

    def _sample_minibatch_states(self, coverage_share: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """Draw a minibatch mixing coverage and replay-buffer states."""
        n_cov = int(self.batch_size * coverage_share)
        n_buf = self.batch_size - n_cov

        k_cov, z_cov = coverage_sampler(n_cov, self.mp)

        if len(self.buffer) >= n_buf and n_buf > 0:
            buf_states = self.buffer.sample(n_buf)
            k_buf = tf.convert_to_tensor(buf_states[:, 0], dtype=DTYPE)
            z_buf = tf.convert_to_tensor(buf_states[:, 1], dtype=DTYPE)
        else:
            k_buf, z_buf = coverage_sampler(n_buf, self.mp)

        if n_buf > 0:
            k_batch = tf.concat([k_cov, k_buf], axis=0)
            z_batch = tf.concat([z_cov, z_buf], axis=0)
        else:
            k_batch, z_batch = k_cov, z_cov

        return k_batch, z_batch

    def train(self) -> None:
        """Run pretraining and main training for the basic model."""
        print("Initialization...")
        # Warm-up rollout to populate the buffer
        self.curr_states = rollout_on_policy(
            self.policy,
            self.curr_states,
            self.mp,
            n_steps=self.roll_steps,
            buffer=self.buffer,
        )
        print(f"Replay buffer size after warm-up: {len(self.buffer)}")

        t0 = time.time()

        # ---------------- Pretrain: coverage only ----------------
        print(f"Pretrain {self.pretrain_steps} steps on coverage sampling...")
        for step in range(1, self.pretrain_steps + 1):
            k_b, z_b = coverage_sampler(self.batch_size, self.mp)
            loss = self._train_step(k_b, z_b)

            if step % self.log_every == 0:
                print(f"[Pretrain {step}/{self.pretrain_steps}] Loss={loss.numpy():.4e}")

            if step % self.roll_steps == 0:
                self.curr_states = rollout_on_policy(
                    self.policy,
                    self.curr_states,
                    self.mp,
                    n_steps=1,
                    buffer=self.buffer,
                )

        # ---------------- Main training: hybrid sampling ----------------
        print(f"Main training {self.train_steps} steps: hybrid sampling...")
        for step in range(1, self.train_steps + 1):
            cover_share = max(
                self.coverage_final_share,
                1.0 - (1.0 - self.coverage_final_share) * (step / self.train_steps),
            )

            k_b, z_b = self._sample_minibatch_states(coverage_share=cover_share)
            loss = self._train_step(k_b, z_b)

            if step % self.roll_steps == 0:
                self.curr_states = rollout_on_policy(
                    self.policy,
                    self.curr_states,
                    self.mp,
                    n_steps=1,
                    buffer=self.buffer,
                )

            if step % self.log_every == 0:
                print(
                    f"[Train {step}/{self.train_steps}] Loss={loss.numpy():.4e} "
                    f"| CoverageShare={cover_share:.3f} | Buffer={len(self.buffer)}"
                )

            if step % self.eval_every == 0:
                acc = self._eval_accuracy()
                print(
                    "  >> Eval (GH-10 coverage): "
                    f"MAE={acc['MAE']:.3e} | RMSE={acc['RMSE']:.3e} | "
                    f"MAX={acc['MAX']:.3e} | Share(|E[g]|≤1e-3)={acc['ShareWithinTol']:.3f}"
                )

        t1 = time.time()
        print(f"Done. Total training time: {t1 - t0:.2f} sec")

    # ------------------------------------------------------------------
    # Final GH-based test (coverage + on-policy + stress)
    # ------------------------------------------------------------------

    def _simulate_on_policy_sample(
        self,
        burn_in_steps: int,
        T: int,
        n_paths: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate an ergodic on-policy sample under the trained policy.

        Returns:
            k_arr, z_arr: NumPy arrays of length T.
        """
        if n_paths is None:
            n_paths = self.n_paths

        k = tf.ones((n_paths,), dtype=DTYPE) * tf.constant(
            steady_state_k(self.mp.theta, self.mp.delta, self.mp.r),
            dtype=DTYPE,
        )
        z = tf.ones((n_paths,), dtype=DTYPE)

        rho = self.rho
        delta = self.delta
        one = self.one
        eps_std = self.sigma_eps
        k_floor = self.k_floor
        mu_ln_z = self.mu_ln_z

        print(
            f"  Burn-in under policy: steps = {burn_in_steps}, paths = {n_paths} ..."
        )
        for _ in range(burn_in_steps):
            x = tf.stack([tf.math.log(k), tf.math.log(z)], axis=1)
            iota = self.policy(x, training=False)[:, 0]
            k_next = tf.maximum(k_floor, (one - delta + iota) * k)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=0.0,
                stddev=eps_std,
                dtype=DTYPE,
            )
            z_next = ar1_step_ln_z(z, rho, eps, mu_ln_z)
            k, z = k_next, z_next

        steps_collect = math.ceil(T / n_paths)
        print(
            f"  Collecting on-policy states: total = {T}, via "
            f"{steps_collect} steps x {n_paths} paths ..."
        )
        collected_k = []
        collected_z = []

        for _ in range(steps_collect):
            collected_k.append(k.numpy())
            collected_z.append(z.numpy())

            x = tf.stack([tf.math.log(k), tf.math.log(z)], axis=1)
            iota = self.policy(x, training=False)[:, 0]
            k_next = tf.maximum(k_floor, (one - delta + iota) * k)
            eps = tf.random.normal(
                shape=tf.shape(k),
                mean=0.0,
                stddev=eps_std,
                dtype=DTYPE,
            )
            z_next = ar1_step_ln_z(z, rho, eps, mu_ln_z)
            k, z = k_next, z_next

        k_arr = np.concatenate(collected_k, axis=0)[:T].astype(np.float32)
        z_arr = np.concatenate(collected_z, axis=0)[:T].astype(np.float32)
        return k_arr, z_arr

    @staticmethod
    def _build_coverage_box_and_sample(
        lnk: np.ndarray,
        lnz: np.ndarray,
        q_low: float = 0.01,
        q_high: float = 0.99,
        expand: float = 0.05,
        M: int = 20_000,
    ):
        """
        Build a tight outer box from ergodic samples in (ln k, ln z) and draw
        M uniform points inside this box.
        """
        lnk_q1, lnk_q99 = np.quantile(lnk, [q_low, q_high])
        lnz_q1, lnz_q99 = np.quantile(lnz, [q_low, q_high])

        def expand_bounds(a, b, frac):
            width = max(b - a, 1e-8)
            return a - frac * width, b + frac * width

        lnk_min, lnk_max = expand_bounds(lnk_q1, lnk_q99, expand)
        lnz_min, lnz_max = expand_bounds(lnz_q1, lnz_q99, expand)

        lnk_u = np.random.uniform(lnk_min, lnk_max, size=M).astype(np.float32)
        lnz_u = np.random.uniform(lnz_min, lnz_max, size=M).astype(np.float32)

        k_cov = np.exp(lnk_u).astype(np.float32)
        z_cov = np.exp(lnz_u).astype(np.float32)

        box = dict(
            lnk_min=float(lnk_min),
            lnk_max=float(lnk_max),
            lnz_min=float(lnz_min),
            lnz_max=float(lnz_max),
        )
        return k_cov, z_cov, box

    @staticmethod
    def _edge_and_corner_points(box: Dict[str, float], n_edge: int = 50):
        """
        Construct edge and corner points on the coverage box in (ln k, ln z).
        """
        lnk_min, lnk_max = box["lnk_min"], box["lnk_max"]
        lnz_min, lnz_max = box["lnz_min"], box["lnz_max"]

        lnk_grid = np.linspace(lnk_min, lnk_max, n_edge, dtype=np.float32)
        lnz_grid = np.linspace(lnz_min, lnz_max, n_edge, dtype=np.float32)

        top = np.stack([lnk_grid, np.full_like(lnk_grid, lnz_max)], axis=1)[1:-1]
        bottom = np.stack([lnk_grid, np.full_like(lnk_grid, lnz_min)], axis=1)[1:-1]
        left = np.stack([np.full_like(lnz_grid, lnk_min), lnz_grid], axis=1)[1:-1]
        right = np.stack([np.full_like(lnz_grid, lnk_max), lnz_grid], axis=1)[1:-1]

        corners = np.array(
            [
                [lnk_min, lnz_min],
                [lnk_min, lnz_max],
                [lnk_max, lnz_min],
                [lnk_max, lnz_max],
            ],
            dtype=np.float32,
        )

        all_ln_pairs = np.concatenate([top, bottom, left, right, corners], axis=0)
        k_edge = np.exp(all_ln_pairs[:, 0]).astype(np.float32)
        z_edge = np.exp(all_ln_pairs[:, 1]).astype(np.float32)
        return k_edge, z_edge

    def _gh_robustness_check(
        self,
        k_np: np.ndarray,
        z_np: np.ndarray,
        base_nodes: int = 10,
        compare_nodes=(15, 20),
        n_sub: int = 5000,
        batch_eval: int = 16_384,
    ) -> Dict[str, float]:
        """
        Compare GH-{15,20} vs GH-10 on a random subsample to check robustness.
        """
        N = len(k_np)
        n_sub = min(n_sub, N)
        idx = np.random.choice(N, size=n_sub, replace=False)
        k_sub = k_np[idx]
        z_sub = z_np[idx]

        g_base, _ = self._eval_batched(k_sub, z_sub, n_nodes=base_nodes, batch=batch_eval)
        abs_base = np.abs(g_base)
        base_p50 = np.quantile(abs_base, 0.5)
        base_p95 = np.quantile(abs_base, 0.95)
        eps = 1e-16

        results = {}
        for n in compare_nodes:
            g_cmp, _ = self._eval_batched(k_sub, z_sub, n_nodes=n, batch=batch_eval)
            abs_cmp = np.abs(g_cmp)
            cmp_p50 = np.quantile(abs_cmp, 0.5)
            cmp_p95 = np.quantile(abs_cmp, 0.95)

            rel50 = float(abs(cmp_p50 - base_p50) / max(base_p50, eps))
            rel95 = float(abs(cmp_p95 - base_p95) / max(base_p95, eps))
            results[f"GH{n}_vs_GH{base_nodes}_RelChange_P50"] = rel50
            results[f"GH{n}_vs_GH{base_nodes}_RelChange_P95"] = rel95
        return results

    def final_test(self, fp: BasicFinalTestParams) -> None:
        """
        Comprehensive final evaluation with GH-10 (and GH-15/20 robustness)
        on both on-policy and coverage sets, including stress tests.
        """
        print("\n========================")
        print("Final Test: Begin")
        print("========================")

        # 1) On-policy ergodic sample
        t0 = time.time()
        k_ops, z_ops = self._simulate_on_policy_sample(
            burn_in_steps=fp.burn_in_steps,
            T=fp.T_on_policy,
            n_paths=self.n_paths,
        )
        lnk_ops = np.log(k_ops)
        lnz_ops = np.log(z_ops)
        print(
            f"On-policy sample ready. N = {len(k_ops)}. "
            f"Time = {time.time() - t0:.2f} sec."
        )

        # 2) Coverage box from ergodic sample
        k_cov, z_cov, box = self._build_coverage_box_and_sample(
            lnk_ops,
            lnz_ops,
            q_low=fp.q_low,
            q_high=fp.q_high,
            expand=fp.expand_frac,
            M=fp.M_coverage,
        )
        print("Coverage box (in log-space):")
        print(
            f"  ln k: [{box['lnk_min']:.3f}, {box['lnk_max']:.3f}] | "
            f"ln z: [{box['lnz_min']:.3f}, {box['lnz_max']:.3f}]"
        )
        print(f"Coverage sample ready. M = {len(k_cov)}")

        # 3) Euler residuals with GH-10
        print("Evaluating Euler residuals with GH-10 ...")
        t1 = time.time()
        g_ops, den_ops = self._eval_batched(
            k_ops, z_ops, n_nodes=self.default_gh_nodes, batch=fp.batch_eval
        )
        g_cov, den_cov = self._eval_batched(
            k_cov, z_cov, n_nodes=self.default_gh_nodes, batch=fp.batch_eval
        )
        print(f"Evaluation time: {time.time() - t1:.2f} sec.")

        # 4) Summary statistics
        stats_ops = self._compute_stats(g_ops, den_ops, tol_list=fp.tol_list)
        stats_cov = self._compute_stats(g_cov, den_cov, tol_list=fp.tol_list)

        self._print_stats("On-policy test set (GH-10)", stats_ops, tol_list=fp.tol_list)
        self._print_stats("Coverage test set (GH-10)", stats_cov, tol_list=fp.tol_list)

        # 4b) Edge and corner stress test
        print("\nEdge/Corner stress test on the coverage box ...")
        k_edge, z_edge = self._edge_and_corner_points(box, n_edge=fp.edge_points)
        g_edge, den_edge = self._eval_batched(
            k_edge, z_edge, n_nodes=self.default_gh_nodes, batch=fp.batch_eval
        )
        stats_edge = self._compute_stats(g_edge, den_edge, tol_list=fp.tol_list)
        self._print_stats(
            "Stress test (box edges & corners) (GH-10)",
            stats_edge,
            tol_list=fp.tol_list,
        )

        # 4c) GH-node robustness check
        print(
            "\nGH-node robustness check (subsample): compare GH-15/20 vs GH-10 on coverage set"
        )
        robust = self._gh_robustness_check(
            k_cov,
            z_cov,
            base_nodes=self.default_gh_nodes,
            compare_nodes=(15, 20),
            n_sub=min(5000, fp.M_coverage),
            batch_eval=fp.batch_eval,
        )
        for k, v in robust.items():
            print(f"- {k}: {v * 100:.1f}%")

        # 5) Informal pass/fail checks
        share_cov_1e3 = float((np.abs(g_cov) <= 1e-3).mean())
        share_ops_1e3 = float((np.abs(g_ops) <= 1e-3).mean())

        cov_pass = (
            stats_cov["Abs_MAE"] <= 1e-3
            and stats_cov["Abs_P95"] <= 1e-2
            and share_cov_1e3 >= 0.95
        )
        ops_pass = (
            stats_ops["Abs_Median"] <= 1e-3
            and stats_ops["Abs_P95"] <= 1e-2
        )
        gh15_ok = (
            robust.get("GH15_vs_GH10_RelChange_P50", 1.0) <= 0.20
            and robust.get("GH15_vs_GH10_RelChange_P95", 1.0) <= 0.20
        )
        gh20_ok = (
            robust.get("GH20_vs_GH10_RelChange_P50", 1.0) <= 0.20
            and robust.get("GH20_vs_GH10_RelChange_P95", 1.0) <= 0.20
        )

        print("\nInformal pass/fail against suggested thresholds:")
        print(
            f"- Coverage set: {'PASS' if cov_pass else 'FAIL'} "
            f"(share(|E[g]|<=1e-3)={share_cov_1e3:.3f})"
        )
        print(f"- On-policy set: {'PASS' if ops_pass else 'FAIL'}")
        print(
            f"- GH-node robustness (vs 10): GH-15={'PASS' if gh15_ok else 'FAIL'}, "
            f"GH-20={'PASS' if gh20_ok else 'FAIL'}"
        )

        print("\n========================")
        print("Final Test: End")
        print("========================\n")

    # ------------------------------------------------------------------
    # Helper to build a fixed coverage evaluation set for mid-training
    # ------------------------------------------------------------------

    def _build_fixed_eval_coverage(
        self,
        N: int,
        m_minus: float = 0.2,
        m_plus: float = 5.0,
    ):
        """
        Build a fixed coverage evaluation set (k,z) for mid-training diagnostics.
        Uses a dedicated NumPy RNG to keep it independent of training randomness.
        """
        rng = np.random.RandomState(self.mp.seed + 777)

        k_ss = steady_state_k(self.mp.theta, self.mp.delta, self.mp.r)
        lnk_min = math.log(m_minus * k_ss)
        lnk_max = math.log(m_plus * k_ss)
        lnk = rng.uniform(low=lnk_min, high=lnk_max, size=N).astype(np.float32)
        k = np.exp(lnk).astype(np.float32)

        sigma_ln_z = self.mp.sigma_eps / math.sqrt(1.0 - self.mp.rho * self.mp.rho)
        m_ln_z = -0.5 * (self.mp.sigma_eps ** 2) / (1.0 - self.mp.rho * self.mp.rho)
        lnz = rng.normal(loc=m_ln_z, scale=sigma_ln_z, size=N).astype(np.float32)
        lnz = np.clip(
            lnz,
            m_ln_z - 3.0 * sigma_ln_z,
            m_ln_z + 3.0 * sigma_ln_z,
        ).astype(np.float32)
        z = np.exp(lnz).astype(np.float32)
        return k, z


# ---------------------------------------------------------------------
# 5. Run basic model
# ---------------------------------------------------------------------

if __name__ == "__main__":
    mp = BasicModelParams()
    tp = BasicTrainingParams()
    fp = BasicFinalTestParams()

    print("Basic model parameters:")
    print(mp)

    k_star = steady_state_k(mp.theta, mp.delta, mp.r)
    print(f"Steady-state capital (no adj. cost): k* = {k_star:.6f}")
    iota_min = -(mp.iota_lower_eps) * (1.0 - mp.delta)
    iota_max = mp.iota_upper
    print(f"Policy output bounds for iota=I/k: [{iota_min:.4f}, {iota_max:.4f}]")

    trainer = BasicTrainer(mp=mp, tp=tp)

    # Train
    trainer.train()

    # Final test
    trainer.final_test(fp)