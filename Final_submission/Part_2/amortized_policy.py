"""
Train a self-contained amortized (parameter-conditioned) policy network:

    (ln k, ln z, theta, log(phi))  ->  iota = I/k

using an AiO Euler-equation loss with antithetic variates and a
coverage+replay sampler.

Keeps the original setting (delta, r, rho, sigma_eps fixed), but trains
one policy over a range of (theta, phi).

Output:
    - param_policy_theta_phi.keras
"""

import math
from time import time
import numpy as np

from utils_part2 import tf, DTYPE, ReplayBuffer, ar1_step_ln_z, set_global_seed
from config_part2 import BasicModelParams, BasicTrainingParams

# Short helper for scalar TensorFlow constants with the global DTYPE
C = lambda x: tf.constant(x, dtype=DTYPE)

# ----------------------------------------------------------------------
# Parameter setup
# ----------------------------------------------------------------------

mp = BasicModelParams()
tp = BasicTrainingParams()

print("Basic (fixed) model parameters:")
print(mp)

set_global_seed(mp.seed)

# Fixed parameters
DELTA = C(mp.delta)
RHO = C(mp.rho)
SIGMA_EPS = C(mp.sigma_eps)
R = C(mp.r)  # interest rate
ONE = C(1.0)
HALF = C(0.5)
ZERO = C(0.0)
BETA = ONE / (ONE + R)
K_FLOOR = C(1e-12)

# AR(1) intercept in ln z so that stationary E[z] = 1
MU_LN_Z = C(-0.5 * (mp.sigma_eps**2) / (1.0 + mp.rho))

# z coverage distribution constants (stationary ln z with E[z]=1, truncated)
SIGMA_LN_Z = C(mp.sigma_eps / math.sqrt(1.0 - mp.rho * mp.rho))
M_LN_Z = C(-0.5 * (mp.sigma_eps**2) / (1.0 - mp.rho * mp.rho))

# Policy output bounds for iota = I/k (same as before)
IOTA_MIN = float(-(mp.iota_lower_eps) * (1.0 - mp.delta))
IOTA_MAX = float(mp.iota_upper)
print(f"Policy output bounds for iota=I/k: [{IOTA_MIN:.4f}, {IOTA_MAX:.4f}]")

# ----------------------------------------------------------------------
# Training ranges for (theta, phi)  [EDIT THESE LATER IF YOU WANT]
# ----------------------------------------------------------------------

THETA_MIN_F = 0.5
THETA_MAX_F = 0.9

PHI_MIN_F = 0.5
PHI_MAX_F = 5

LOG_PHI_MIN_F = math.log(PHI_MIN_F)
LOG_PHI_MAX_F = math.log(PHI_MAX_F)

print("Amortization (training) ranges:")
print(f"  theta in [{THETA_MIN_F}, {THETA_MAX_F}]")
print(f"  phi   in [{PHI_MIN_F}, {PHI_MAX_F}]  (policy conditioned on log(phi))")

# Useful TF constants for normalization
THETA_MIN = C(THETA_MIN_F)
THETA_MAX = C(THETA_MAX_F)
LOG_PHI_MIN = C(LOG_PHI_MIN_F)
LOG_PHI_MAX = C(LOG_PHI_MAX_F)

# ----------------------------------------------------------------------
# Model primitives and helpers (now parameterized)
# ----------------------------------------------------------------------


def steady_state_ln_k_tf(theta, delta, r):
    """
    ln k*(theta) for the no-adjustment-cost steady state (z=1):
        MPK = theta * k^(theta-1) = r + delta
        => ln k* = (1/(1-theta)) * ln(theta/(r+delta))
    """
    return tf.math.log(theta / (r + delta)) / (ONE - theta)


def profit_k_tf(k, z, theta):
    """Marginal product of capital: ∂π/∂k = z * θ * k^(θ-1)."""
    return z * theta * tf.pow(k, theta - ONE)


def psi_I_tf(iota, phi, delta):
    """Adjustment-cost derivative wrt ι = I/k: ψ_I = φ * (ι - δ)."""
    return phi * (iota - delta)


def psi_k_tf(iota, phi, delta):
    """Envelope-based derivative wrt capital: ψ_k = (φ/2) * (δ² - ι²)."""
    return HALF * phi * (delta * delta - tf.square(iota))


def euler_term(k, z, iota, theta, phi):
    """Euler term used in the AiO loss (now depends on theta, phi)."""
    return (
        profit_k_tf(k, z, theta)
        - psi_k_tf(iota, phi, DELTA)
        + (ONE - DELTA) * (ONE + psi_I_tf(iota, phi, DELTA))
    )


# ----------------------------------------------------------------------
# Parametrized policy network
#   input:  (ln k, ln z, theta, log(phi))
#   internal features: (ln(k/k*(theta)), ln z, norm(theta), norm(log phi))
#   output: iota in [IOTA_MIN, IOTA_MAX]
# ----------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="amortized_policy")
class ParamPolicyNet(tf.keras.Model):
    def __init__(
        self,
        hidden_sizes=(64, 64),
        activation="tanh",
        iota_min=IOTA_MIN,
        iota_max=IOTA_MAX,
        delta=float(mp.delta),
        r=float(mp.r),
        theta_min=THETA_MIN_F,
        theta_max=THETA_MAX_F,
        log_phi_min=LOG_PHI_MIN_F,
        log_phi_max=LOG_PHI_MAX_F,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation = str(activation)

        # Save scalars as Python floats for serialization
        self.iota_min = float(iota_min)
        self.iota_max = float(iota_max)
        self.delta = float(delta)
        self.r = float(r)

        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.log_phi_min = float(log_phi_min)
        self.log_phi_max = float(log_phi_max)

        self.h1 = tf.keras.layers.Dense(self.hidden_sizes[0], activation=self.activation)
        self.h2 = tf.keras.layers.Dense(self.hidden_sizes[1], activation=self.activation)
        self.out = tf.keras.layers.Dense(1, activation=None)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                hidden_sizes=list(self.hidden_sizes),
                activation=self.activation,
                iota_min=self.iota_min,
                iota_max=self.iota_max,
                delta=self.delta,
                r=self.r,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                log_phi_min=self.log_phi_min,
                log_phi_max=self.log_phi_max,
            )
        )
        return config

    def call(self, x, training=False):
        """
        Args:
            x: shape (B, 4) with columns [ln k, ln z, theta, log(phi)]
        Returns:
            iota: shape (B, 1), constrained to [iota_min, iota_max]
        """
        x = tf.cast(x, DTYPE)

        lnk = x[:, 0:1]
        lnz = x[:, 1:2]
        theta = x[:, 2:3]
        log_phi = x[:, 3:4]

        # ln k*(theta) with fixed (delta, r)
        ln_k_star = tf.math.log(theta / C(self.r + self.delta)) / (ONE - theta)
        lnk_rel = lnk - ln_k_star  # ln(k/k*(theta))

        # Normalize parameters to roughly [-1, 1] using training ranges
        theta_norm = (2.0 * (theta - C(self.theta_min)) / C(self.theta_max - self.theta_min)) - ONE
        log_phi_norm = (2.0 * (log_phi - C(self.log_phi_min)) / C(self.log_phi_max - self.log_phi_min)) - ONE

        feats = tf.concat([lnk_rel, lnz, theta_norm, log_phi_norm], axis=1)

        h = self.h1(feats, training=training)
        h = self.h2(h, training=training)
        raw = self.out(h, training=training)

        span = C(self.iota_max - self.iota_min)
        iota = C(self.iota_min) + HALF * (tf.tanh(raw) + ONE) * span
        return iota


def policy_iota(policy, k, z, theta, phi, training):
    """Evaluate amortized policy and return iota = I/k, shape (B,)."""
    x = tf.stack(
        [
            tf.math.log(k),
            tf.math.log(z),
            theta,
            tf.math.log(phi),
        ],
        axis=1,
    )
    return policy(x, training=training)[:, 0]


# ----------------------------------------------------------------------
# Parameter sampling
# ----------------------------------------------------------------------

def sample_params_tf(batch_size):
    """Sample theta ~ U[THETA_MIN, THETA_MAX], log(phi) ~ U[logphi_min, logphi_max]."""
    theta = tf.random.uniform((batch_size,), minval=THETA_MIN, maxval=THETA_MAX, dtype=DTYPE)
    log_phi = tf.random.uniform((batch_size,), minval=LOG_PHI_MIN, maxval=LOG_PHI_MAX, dtype=DTYPE)
    phi = tf.exp(log_phi)
    return theta, phi


def steady_state_k_np(theta, delta, r):
    """Vectorized numpy steady state k*(theta) for z=1, MPK=r+delta."""
    theta = np.asarray(theta, dtype=np.float64)
    return (theta / (r + delta)) ** (1.0 / (1.0 - theta))


# ----------------------------------------------------------------------
# Coverage sampler for (k, z, theta, phi)
# ----------------------------------------------------------------------

def coverage_sampler(batch_size, m_minus=0.2, m_plus=5.0):
    """
    Sample training tuples (k, z, theta, phi).

    theta: uniform on [THETA_MIN, THETA_MAX]
    phi:   log-uniform on [PHI_MIN, PHI_MAX]
    k:     log-uniform on [m_minus * k*(theta), m_plus * k*(theta)]
           implemented by sampling ln(k/k*) uniform on [ln m_minus, ln m_plus]
    z:     log-normal from stationary ln z distribution with E[z]=1, truncated at ±3σ
    """
    theta, phi = sample_params_tf(batch_size)

    # sample ln(k/k*) uniformly, then map to k
    lnkk = tf.random.uniform(
        (batch_size,),
        minval=tf.math.log(C(m_minus)),
        maxval=tf.math.log(C(m_plus)),
        dtype=DTYPE,
    )
    ln_k_star = steady_state_ln_k_tf(theta, DELTA, R)
    k = tf.exp(ln_k_star + lnkk)

    lnz = tf.random.normal(
        (batch_size,),
        mean=M_LN_Z,
        stddev=SIGMA_LN_Z,
        dtype=DTYPE,
    )
    lnz = tf.clip_by_value(lnz, M_LN_Z - 3.0 * SIGMA_LN_Z, M_LN_Z + 3.0 * SIGMA_LN_Z)
    z = tf.exp(lnz)

    return k, z, theta, phi


# ----------------------------------------------------------------------
# On-policy rollouts and replay buffer  (store k,z,theta,phi)
# ----------------------------------------------------------------------

def policy_step(policy, k, z, theta, phi, training=False):
    """Single transition (k, z) -> (k_next, z_next) under amortized policy."""
    iota_t = policy_iota(policy, k, z, theta, phi, training=training)
    k_next = tf.maximum(K_FLOOR, (ONE - DELTA + iota_t) * k)

    eps = tf.random.normal(
        shape=tf.shape(k),
        mean=ZERO,
        stddev=SIGMA_EPS,
        dtype=DTYPE,
    )
    z_next = ar1_step_ln_z(z, RHO, eps, MU_LN_Z)
    return k_next, z_next, iota_t


def rollout_on_policy(policy, current_states, n_steps, buffer):
    """
    Roll the model forward under the current policy and add visited (k, z, theta, phi)
    to the replay buffer.

    current_states: tuple (k, z, theta, phi) as NumPy arrays of shape (N,).
                    theta, phi are fixed per path.
    """
    k_arr, z_arr, theta_arr, phi_arr = current_states

    k = tf.convert_to_tensor(k_arr, dtype=DTYPE)
    z = tf.convert_to_tensor(z_arr, dtype=DTYPE)
    theta = tf.convert_to_tensor(theta_arr, dtype=DTYPE)
    phi = tf.convert_to_tensor(phi_arr, dtype=DTYPE)

    for _ in range(n_steps):
        states_np = np.stack(
            [k.numpy(), z.numpy(), theta_arr, phi_arr],
            axis=1,
        ).astype(np.float32)
        buffer.push_batch(states_np)

        k, z, _ = policy_step(policy, k, z, theta, phi, training=False)

    return k.numpy(), z.numpy(), theta_arr, phi_arr


# ----------------------------------------------------------------------
# Initialize policy, optimizer, replay buffer
# ----------------------------------------------------------------------

policy = ParamPolicyNet(
    hidden_sizes=tp.hidden_sizes,
    activation=tp.activation,
    name="param_policy_net",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=tp.lr)

replay_buffer = ReplayBuffer(
    max_size=tp.buffer_size,
    state_dim=4,  # (k, z, theta, phi)
    seed=mp.seed,
)

# Initial on-policy paths: sample parameters per path, start at k*(theta), z=1
rng = np.random.default_rng(mp.seed)
theta0 = rng.uniform(THETA_MIN_F, THETA_MAX_F, size=tp.n_paths).astype(np.float32)
logphi0 = rng.uniform(LOG_PHI_MIN_F, LOG_PHI_MAX_F, size=tp.n_paths).astype(np.float32)
phi0 = np.exp(logphi0).astype(np.float32)

k0 = steady_state_k_np(theta0, mp.delta, mp.r).astype(np.float32)
z0 = np.ones(tp.n_paths, dtype=np.float32)

current_states = (k0, z0, theta0, phi0)

print("Initial on-policy state shapes:", k0.shape, z0.shape, theta0.shape, phi0.shape)


# ----------------------------------------------------------------------
# AiO Euler loss with antithetic variates (now parameterized)
# ----------------------------------------------------------------------

@tf.function
def euler_aio_loss(k_batch, z_batch, theta_batch, phi_batch):
    """AiO Euler loss using antithetic variates for variance reduction."""
    B = tf.shape(k_batch)[0]

    iota_t = policy_iota(policy, k_batch, z_batch, theta_batch, phi_batch, training=True)
    k_next = tf.maximum(K_FLOOR, (ONE - DELTA + iota_t) * k_batch)
    psiI_t = psi_I_tf(iota_t, phi_batch, DELTA)

    eps1 = tf.random.normal((B,), mean=ZERO, stddev=SIGMA_EPS, dtype=DTYPE)
    eps2 = tf.random.normal((B,), mean=ZERO, stddev=SIGMA_EPS, dtype=DTYPE)

    z1_plus = ar1_step_ln_z(z_batch, RHO, eps1, MU_LN_Z)
    z1_minus = ar1_step_ln_z(z_batch, RHO, -eps1, MU_LN_Z)
    z2_plus = ar1_step_ln_z(z_batch, RHO, eps2, MU_LN_Z)
    z2_minus = ar1_step_ln_z(z_batch, RHO, -eps2, MU_LN_Z)

    z_all = tf.concat([z1_plus, z1_minus, z2_plus, z2_minus], axis=0)
    k_all = tf.tile(k_next, multiples=[4])
    theta_all = tf.tile(theta_batch, multiples=[4])
    phi_all = tf.tile(phi_batch, multiples=[4])

    iota_all = policy_iota(policy, k_all, z_all, theta_all, phi_all, training=True)
    term_all = euler_term(k_all, z_all, iota_all, theta_all, phi_all)

    term_all = tf.reshape(term_all, (4, -1))
    term1_plus = term_all[0]
    term1_minus = term_all[1]
    term2_plus = term_all[2]
    term2_minus = term_all[3]

    g1_plus = ONE + psiI_t - BETA * term1_plus
    g1_minus = ONE + psiI_t - BETA * term1_minus
    g2_plus = ONE + psiI_t - BETA * term2_plus
    g2_minus = ONE + psiI_t - BETA * term2_minus

    g1_bar = HALF * (g1_plus + g1_minus)
    g2_bar = HALF * (g2_plus + g2_minus)

    loss_sample = g1_bar * g2_bar
    return tf.reduce_mean(loss_sample)


# ----------------------------------------------------------------------
# Single SGD training step
# ----------------------------------------------------------------------

@tf.function
def train_step(k_batch, z_batch, theta_batch, phi_batch):
    """Perform one SGD step on a batch of (k, z, theta, phi)."""
    with tf.GradientTape() as tape:
        loss = euler_aio_loss(k_batch, z_batch, theta_batch, phi_batch)
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss


# ----------------------------------------------------------------------
# Minibatch sampling (coverage + replay buffer)
# ----------------------------------------------------------------------

def sample_minibatch_states(coverage_share, batch_size, buffer):
    """Draw a minibatch mixing coverage samples and replay-buffer states."""
    n_cov = int(batch_size * coverage_share)
    n_buf = batch_size - n_cov

    k_cov, z_cov, th_cov, ph_cov = coverage_sampler(n_cov)

    if len(buffer) >= n_buf and n_buf > 0:
        buf_states = buffer.sample(n_buf)  # columns: k,z,theta,phi
        k_buf = tf.convert_to_tensor(buf_states[:, 0], dtype=DTYPE)
        z_buf = tf.convert_to_tensor(buf_states[:, 1], dtype=DTYPE)
        th_buf = tf.convert_to_tensor(buf_states[:, 2], dtype=DTYPE)
        ph_buf = tf.convert_to_tensor(buf_states[:, 3], dtype=DTYPE)
    else:
        k_buf, z_buf, th_buf, ph_buf = coverage_sampler(n_buf)

    if n_buf > 0 and n_cov > 0:
        k_b = tf.concat([k_cov, k_buf], axis=0)
        z_b = tf.concat([z_cov, z_buf], axis=0)
        th_b = tf.concat([th_cov, th_buf], axis=0)
        ph_b = tf.concat([ph_cov, ph_buf], axis=0)
        return k_b, z_b, th_b, ph_b

    if n_cov > 0:
        return k_cov, z_cov, th_cov, ph_cov

    return k_buf, z_buf, th_buf, ph_buf


# ----------------------------------------------------------------------
# Training loop (pretraining + main training)
# ----------------------------------------------------------------------

def train_model(save_path="param_policy_theta_phi.keras"):
    global current_states

    print("Initialization (warm-up rollouts)...")
    current_states = rollout_on_policy(
        policy,
        current_states,
        n_steps=tp.roll_steps,
        buffer=replay_buffer,
    )
    print(f"Replay buffer size after warm-up: {len(replay_buffer)}")

    t0 = time()

    # Pretraining phase: coverage sampler only
    print(f"Pretrain {tp.pretrain_steps} steps on coverage sampling...")
    for step in range(1, tp.pretrain_steps + 1):
        k_b, z_b, th_b, ph_b = coverage_sampler(tp.batch_size)
        loss = train_step(k_b, z_b, th_b, ph_b)

        if step % tp.log_every == 0:
            print(f"[Pretrain {step}/{tp.pretrain_steps}] Loss={loss.numpy():.4e}")

        if step % tp.roll_steps == 0:
            current_states = rollout_on_policy(
                policy,
                current_states,
                n_steps=1,
                buffer=replay_buffer,
            )

    # Main training phase: hybrid coverage + replay sampling
    print(f"Main training {tp.train_steps} steps: hybrid sampling...")
    for step in range(1, tp.train_steps + 1):
        cover_share = max(
            tp.coverage_final_share,
            1.0 - (1.0 - tp.coverage_final_share) * (step / tp.train_steps),
        )

        k_b, z_b, th_b, ph_b = sample_minibatch_states(
            coverage_share=cover_share,
            batch_size=tp.batch_size,
            buffer=replay_buffer,
        )
        loss = train_step(k_b, z_b, th_b, ph_b)

        if step % tp.roll_steps == 0:
            current_states = rollout_on_policy(
                policy,
                current_states,
                n_steps=1,
                buffer=replay_buffer,
            )

        if step % tp.log_every == 0:
            print(
                f"[Train {step}/{tp.train_steps}] Loss={loss.numpy():.4e} "
                f"| CoverageShare={cover_share:.3f} | Buffer={len(replay_buffer)}"
            )

    t1 = time()
    print(f"Done. Total training time: {t1 - t0:.2f} sec")

    k_eval, z_eval, th_eval, ph_eval = coverage_sampler(tp.batch_size)
    final_loss = euler_aio_loss(k_eval, z_eval, th_eval, ph_eval).numpy()
    print(f"[Final diagnostic] AiO loss on fresh batch = {final_loss:.4e}")

    policy.save(save_path)
    print(f"Saved trained amortized policy to: {save_path}")


if __name__ == "__main__":
    train_model()