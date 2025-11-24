# Risky debt model

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from config import (
    RiskyDebtParams,
    RiskyDebtTrainingParams,
    RiskyDebtFinalTestParams,
)
from common import (
    tf,
    DTYPE,
    set_global_seed,
    risky_coverage_sampler,
    ar1_step_ln_z,
)


class RiskyDebtModel(tf.keras.Model):
    """
    Risky-debt model with:
      - Policy network for (k', b')
      - Value network for continuation value C(k, b, z)
      - Soft limited liability V = max(0, C) via softplus
      - Endogenous risky-debt pricing via a zero-profit condition
      - AiO loss combining Bellman residuals and FOC information
    """

    def __init__(
        self,
        params: RiskyDebtParams,
        train_params: RiskyDebtTrainingParams,
    ):
        super(RiskyDebtModel, self).__init__()

        self.params = params
        self.train_params = train_params

        # --- 1. Economic parameters (from config) ---
        self.theta = params.theta          # Capital elasticity in production
        self.delta = params.delta          # Depreciation
        self.r = params.r                  # Risk-free rate
        self.beta = 1.0 / (1.0 + self.r)
        self.tau = params.tau              # Corporate tax rate
        self.bankruptcy_cost = params.bankruptcy_cost  # Deadweight loss in default

        # Shock process: AR(1) in ln z
        self.rho_z = params.rho_z
        self.sigma_z = params.sigma_z

        # Costly external finance: eta0 * I(e<0) + eta1 * |e| * I(e<0)
        self.eta_0 = params.eta_0
        self.eta_1 = params.eta_1

        # --- 2. Numerical / training hyperparameters ---
        self.batch_size = train_params.batch_size
        self.M_pricing = params.M_pricing          # Inner MC draws for pricing
        self.eps_b = params.eps_b                  # Threshold for "no debt" / savings

        # Soft logic temperatures (initialized from config, will be annealed)
        self.tau_V = params.tau_V_init            # for softplus in V = max(0, C)
        self.tau_D = params.tau_D_init            # for sigmoid default indicator

        # FOC weight (set dynamically in the training loop)
        self.lambda_foc = 0.0

        # --- 3. Neural networks ---

        # Policy network: state -> (k', b')
        # State input: [log k, b/k, log z]
        n_hidden = params.n_hidden
        policy_in = tf.keras.Input(shape=(3,), dtype=DTYPE)
        x = tf.keras.layers.Dense(n_hidden, activation="elu")(policy_in)
        x = tf.keras.layers.Dense(n_hidden, activation="elu")(x)

        # Capital k' must be positive: softplus plus a small floor
        raw_kp = tf.keras.layers.Dense(1, activation="softplus")(x)
        kp_out = tf.keras.layers.Lambda(lambda t: t + 0.1)(raw_kp)  # k' >= 0.1

        # Debt b' can be positive (borrowing) or negative (savings)
        bp_out = tf.keras.layers.Dense(1, activation="linear")(x)

        self.policy_net = tf.keras.Model(inputs=policy_in, outputs=[kp_out, bp_out])

        # Value network: state -> continuation value C(k,b,z)
        # State input: [log k, b/k, log z]
        val_in = tf.keras.Input(shape=(3,), dtype=DTYPE)
        xv = tf.keras.layers.Dense(n_hidden, activation="elu")(val_in)
        xv = tf.keras.layers.Dense(n_hidden, activation="elu")(xv)
        Cv_out = tf.keras.layers.Dense(1, activation="linear")(xv)

        self.value_net = tf.keras.Model(inputs=val_in, outputs=Cv_out)

        # Target value network for pricing and continuation (Polyak averaging)
        self.target_value_net = tf.keras.models.clone_model(self.value_net)
        self.target_value_net.set_weights(self.value_net.get_weights())

        # Optimizer (learning rate from RiskyDebtTrainingParams)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=train_params.lr)

    # ------------------------------------------------------------------
    # Shock and technology
    # ------------------------------------------------------------------

    def get_next_z(self, z: tf.Tensor, n_samples: int) -> tf.Tensor:
        """
        AR(1) transition for productivity z using the shared ar1_step_ln_z helper.

        ln z' = rho_z * ln z + sigma_z * eps,    eps ~ N(0, 1),
        by passing eps_scaled = sigma_z * eps and zero intercept to ar1_step_ln_z.

        Args:
            z: tensor of shape [batch, 1] with current productivity in levels.
            n_samples: number of Monte Carlo draws per state.

        Returns:
            z_prime: tensor of shape [batch, n_samples] with next-period
                     productivity draws in levels.
        """
        batch_size = tf.shape(z)[0]

        # Tile current z across MC draws: shape [batch, n_samples]
        z_tiled = tf.tile(z, multiples=[1, n_samples])

        # Gaussian innovations in ln z with std = sigma_z
        eps = tf.random.normal(
            shape=(batch_size, n_samples),
            mean=0.0,
            stddev=self.sigma_z,
            dtype=DTYPE,
        )

        # Intercept mu_ln_z = 0.0
        mu_ln_z = tf.constant(0.0, dtype=DTYPE)

        z_next = ar1_step_ln_z(
            z_tiled,
            rho=tf.constant(self.rho_z, dtype=DTYPE),
            eps=eps,
            mu_ln_z=mu_ln_z,
        )
        return z_next

    def production(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Production function: z * k^theta with theta from the paper."""
        return z * tf.pow(k, self.theta)

    def adjustment_cost(self, k: tf.Tensor, k_prime: tf.Tensor) -> tf.Tensor:
        """
        Quadratic adjustment cost in investment:
            I = k' - (1 - delta) k,
            cost = 0.01 * I^2.
        """
        i = k_prime - (1.0 - self.delta) * k
        return 0.01 * tf.square(i)

    # ------------------------------------------------------------------
    # Recovery and value
    # ------------------------------------------------------------------

    def get_recovery(self, k_prime: tf.Tensor, z_prime: tf.Tensor) -> tf.Tensor:
        """
        Recovery value upon default:
            (1 - bankruptcy_cost) * [ (1 - tau) * pi + (1 - delta) * k' ].
        """
        pi = self.production(k_prime, z_prime)
        val = (1.0 - self.tau) * pi + (1.0 - self.delta) * k_prime
        return (1.0 - self.bankruptcy_cost) * val

    def compute_equity_value(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        use_target: bool = False,
    ):
        """
        Compute continuation value C(k,b,z) and equity value V ~ max(0, C)
        via a softplus approximation.

        Inputs k, b, z are in levels; the network state is [log k, b/k, log z].
        """
        # Normalize inputs: log k, leverage b/k, and log z
        k_safe = tf.maximum(k, 1e-3)
        norm_k = tf.math.log(k_safe)
        norm_b = b / k_safe
        z_safe = tf.maximum(z, 1e-6)
        log_z = tf.math.log(z_safe)
        state = tf.concat([norm_k, norm_b, log_z], axis=-1)

        if use_target:
            C = self.target_value_net(state)
        else:
            C = self.value_net(state)

        # Soft limited liability: V ~ max(0, C)
        V = self.tau_V * tf.math.softplus(C / self.tau_V)
        return C, V

    # ------------------------------------------------------------------
    # Default probability (helper)
    # ------------------------------------------------------------------

    def default_probability(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        n_draws: int = 64,
        use_target: bool = True,
    ) -> tf.Tensor:
        """
        One-step-ahead default probability under the current policy.

        For each current state (k,b,z), this function:
        1. Applies the policy network to obtain (k', b'), using [log k, b/k, log z].
        2. Draws n_draws shocks for z'.
        3. Computes C'(k', b', z') using the (target) value network.
        4. Uses soft default indicator:
               D' = sigmoid(-C'/tau_D)
        5. Returns p_default = E[D'].
        """
        # 1. Current policy actions a(s) = (k', b'), NN input uses log z
        k_safe = tf.maximum(k, 1e-3)
        norm_k = tf.math.log(k_safe)
        norm_b = b / k_safe
        z_safe = tf.maximum(z, 1e-6)
        log_z = tf.math.log(z_safe)
        policy_input = tf.concat([norm_k, norm_b, log_z], axis=-1)
        k_prime, b_prime = self.policy_net(policy_input)

        # 2. Simulate future productivity z'
        z_prime = self.get_next_z(z, n_draws)  # [N, n_draws]

        # 3. Broadcast controls to match z' draws
        k_prime_exp = tf.tile(k_prime, [1, n_draws])   # [N, n_draws]
        b_prime_exp = tf.tile(b_prime, [1, n_draws])   # [N, n_draws]

        kp_flat = tf.reshape(k_prime_exp, [-1, 1])     # [N*n_draws, 1]
        bp_flat = tf.reshape(b_prime_exp, [-1, 1])
        zp_flat = tf.reshape(z_prime, [-1, 1])

        # 4. Future continuation value C'(k',b',z')
        C_prime, _ = self.compute_equity_value(
            kp_flat, bp_flat, zp_flat, use_target=use_target
        )

        # 5. Soft default indicator and expected default probability
        D_prime = tf.sigmoid(-C_prime / self.tau_D)           # [N*n_draws, 1]
        D_prime = tf.reshape(D_prime, [-1, n_draws])          # [N, n_draws]
        p_default = tf.reduce_mean(D_prime, axis=1, keepdims=True)  # [N, 1]

        return p_default

    # ------------------------------------------------------------------
    # Pricing kernel for risky debt
    # ------------------------------------------------------------------

    def pricing_kernel(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_current: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the risky bond price q(k', b', z) given controls (k', b') and current z.

        For true debt positions (b' > eps_b), we implement:

            b' (1 + r)
              = E[ D' R'(k', Z')
                    + (1 - D') (1 + \\tilde r) b' ],

        which implies

            1 + \\tilde r
              = ( 1 + r - E[D' R'/b'] ) / E[1 - D'],

        and q = 1 / (1 + \\tilde r).

        For b' <= eps_b (net saving or tiny positions), we treat the position as risk-free
        and set q = 1 / (1 + r).
        """
        # 1. Draw future shocks Z'
        z_prime = self.get_next_z(z_current, self.M_pricing)  # [batch, M]

        # 2. Broadcast controls to match shock draws
        k_prime_exp = tf.tile(k_prime, [1, self.M_pricing])  # [batch, M]
        b_prime_exp = tf.tile(b_prime, [1, self.M_pricing])  # [batch, M]

        kp_flat = tf.reshape(k_prime_exp, [-1, 1])  # [batch*M, 1]
        bp_flat = tf.reshape(b_prime_exp, [-1, 1])  # [batch*M, 1]
        zp_flat = tf.reshape(z_prime, [-1, 1])      # [batch*M, 1]

        # 3. Future continuation value C'(k',b',Z') from the target net
        C_prime, _ = self.compute_equity_value(
            kp_flat, bp_flat, zp_flat, use_target=True
        )

        # Soft default indicator: ~1 if C'<0, 0 otherwise
        D_prime = tf.sigmoid(-C_prime / self.tau_D)  # [batch*M, 1]

        # 4. Recovery value R'(k',Z')
        R_prime = self.get_recovery(kp_flat, zp_flat)  # [batch*M, 1]

        # 5. Compute A = E[D' R'/b'] and p_solv = E[1 - D'] across the M draws
        #    Use a safe division for small |b'|; this is only relevant for b' > eps_b
        #    but we keep it generic for numerical robustness.
        bp_safe = tf.where(
            tf.abs(bp_flat) < self.eps_b,
            tf.sign(bp_flat + 1e-6) * self.eps_b,
            bp_flat,
        )
        ratio_R = R_prime / bp_safe  # [batch*M, 1]
        ratio_R = tf.clip_by_value(ratio_R, -5.0, 5.0)  # optional clipping

        # Reshape to [batch, M] to take expectations across draws
        D_2d = tf.reshape(D_prime, [-1, self.M_pricing])  # [batch, M]
        DR_over_b_2d = tf.reshape(D_prime * ratio_R, [-1, self.M_pricing])

        A = tf.reduce_mean(DR_over_b_2d, axis=1, keepdims=True)  # E[D' R'/b']
        p_solv = tf.reduce_mean(1.0 - D_2d, axis=1, keepdims=True)  # E[1 - D']

        # 6. Gross risky rate 1 + tilde_r and price q for debt positions
        eps_p = 1e-4
        denom = tf.maximum(p_solv, eps_p)  # guard against p_solv ~ 0
        one_plus_tilde = (1.0 + self.r - A) / denom  # [batch, 1]

        # Ensure positivity and avoid extreme values
        one_plus_tilde = tf.clip_by_value(one_plus_tilde, 1e-3, 1e3)

        q_raw = 1.0 / one_plus_tilde  # [batch, 1]

        # 7. Debt vs savings: if b' <= eps_b, treat as risk-free asset
        q_riskfree = 1.0 / (1.0 + self.r)
        mask_debt = tf.cast(b_prime > self.eps_b, DTYPE)  # [batch,1]

        q_final = mask_debt * q_raw + (1.0 - mask_debt) * q_riskfree

        # Numerical clamp: keep q in a reasonable range
        q_final = tf.clip_by_value(q_final, 0.01, q_riskfree)
        return q_final

    # ------------------------------------------------------------------
    # Bellman RHS
    # ------------------------------------------------------------------

    def compute_rhs(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
    ) -> tf.Tensor:
        """
        Bellman RHS:
            RHS = dividend(k,b,z,k',b') + beta * E[V(k',b',z')]
        using the target value net for V'.

        k, b, z are in levels; networks called inside handle log transforms.
        """
        # Pricing of new debt
        q = self.pricing_kernel(k_prime, b_prime, z)

        # Cash flows
        pi = self.production(k, z)
        inv_cost = k_prime - (1.0 - self.delta) * k + self.adjustment_cost(k, k_prime)

        debt_proceeds = q * b_prime

        # Tax shield (discrete approximation)
        tax_shield = self.tau * (1.0 - q) * b_prime * self.beta

        # Equity payout before issuance costs
        e = (1.0 - self.tau) * pi - inv_cost + debt_proceeds + tax_shield - b

        # Costly external finance if e < 0
        is_negative = tf.cast(e < 0.0, DTYPE)
        financing_cost = is_negative * (self.eta_0 + self.eta_1 * tf.abs(e))

        div = e - financing_cost

        # Future value V'(k',b',z'): continuation under limited liability
        z_prime = self.get_next_z(z, 1)  # [batch,1]
        _, V_prime = self.compute_equity_value(
            k_prime, b_prime, z_prime, use_target=True
        )

        rhs = div + self.beta * V_prime
        return rhs

    # ------------------------------------------------------------------
    # Training: Bellman + FOC (AiO)
    # ------------------------------------------------------------------

    @tf.function
    def train_step(self, states: tf.Tensor):
        """
        One training step on a batch of states.
        - Bellman AiO loss trains the value_net.
        - FOC AiO loss trains the policy_net (via nested GradientTape).

        states is [k, b, z] in levels; NN inputs use [log k, b/k, log z].
        """
        k = states[:, 0:1]
        b = states[:, 1:2]
        z = states[:, 2:3]

        lambda_foc = self.lambda_foc

        # Outer tape: w.r.t. policy_net and value_net parameters
        with tf.GradientTape() as outer_tape:
            # 1. Current policy actions a(s) = (k', b'), NN input uses log z
            k_safe = tf.maximum(k, 1e-3)
            norm_k = tf.math.log(k_safe)
            norm_b = b / k_safe
            z_safe = tf.maximum(z, 1e-6)
            log_z = tf.math.log(z_safe)
            policy_input = tf.concat([norm_k, norm_b, log_z], axis=-1)

            k_prime, b_prime = self.policy_net(policy_input)

            # 2. Current continuation value C(k,b,z)
            C_theta, _ = self.compute_equity_value(
                k, b, z, use_target=False
            )

            # 3. Nested tape: gradients of RHS w.r.t. actions (FOCs)
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([k_prime, b_prime])

                rhs_a = self.compute_rhs(k, b, z, k_prime, b_prime)
                rhs_b = self.compute_rhs(k, b, z, k_prime, b_prime)

            # ---- Bellman AiO loss ----
            # Use stop_gradient on RHS to make Bellman purely a value_net evaluation loss.
            bellman_residuals = (
                C_theta - tf.stop_gradient(rhs_a)
            ) * (C_theta - tf.stop_gradient(rhs_b))
            loss_bellman = tf.reduce_mean(bellman_residuals)

            # ---- FOC AiO loss ----
            # g_A = d RHS_A / d a, g_B = d RHS_B / d a
            grads_a = inner_tape.gradient(rhs_a, [k_prime, b_prime])
            grads_b = inner_tape.gradient(rhs_b, [k_prime, b_prime])
            del inner_tape  # free resources

            def safe_grad(g):
                """Replace non-finite gradients with zeros."""
                return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))

            g_a_k = safe_grad(grads_a[0])
            g_a_b = safe_grad(grads_a[1])
            g_b_k = safe_grad(grads_b[0])
            g_b_b = safe_grad(grads_b[1])

            # AiO structure: E[g_A * g_B], BUT only let gradients flow through g_B.
            foc_k = tf.stop_gradient(g_a_k) * g_b_k
            foc_b = tf.stop_gradient(g_a_b) * g_b_b
            loss_foc = tf.reduce_mean(foc_k + foc_b)

            # Total loss
            total_loss = loss_bellman + lambda_foc * loss_foc

        # 4. Apply gradients to policy and value networks
        trainable_vars = (
            self.policy_net.trainable_variables + self.value_net.trainable_variables
        )

        grads = outer_tape.gradient(total_loss, trainable_vars)
        del outer_tape

        # Gradient clipping
        clipped_grads = []
        for g in grads:
            if g is None:
                clipped_grads.append(None)
            else:
                clipped_grads.append(tf.clip_by_norm(g, 1.0))

        self.optimizer.apply_gradients(zip(clipped_grads, trainable_vars))

        # 5. Polyak averaging for target value net
        tau_polyak = 0.005
        for v_main, v_target in zip(
            self.value_net.variables, self.target_value_net.variables
        ):
            v_target.assign(tau_polyak * v_main + (1.0 - tau_polyak) * v_target)

        return total_loss, loss_bellman, loss_foc, k_prime, b_prime

    # ------------------------------------------------------------------
    # Residual diagnostics (Bellman, FOC, limited liability)
    # ------------------------------------------------------------------

    def compute_residuals(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        n_mc_rhs: int = 50,
    ):
        """
        Compute multiple residual diagnostics on a given batch of states:
        - Bellman residual: relative mean & max
        - FOC residual: mean squared, max absolute
        - Limited-liability residual: |V - max(C,0)|, relative mean & max

        Inputs k, b, z are in levels; NN inputs use [log k, b/k, log z].
        """
        # Current policy actions
        k_safe = tf.maximum(k, 1e-3)
        norm_k = tf.math.log(k_safe)
        norm_b = b / k_safe
        z_safe = tf.maximum(z, 1e-6)
        log_z = tf.math.log(z_safe)
        policy_input = tf.concat([norm_k, norm_b, log_z], axis=-1)
        k_prime, b_prime = self.policy_net(policy_input)

        # Bellman: C_pred vs averaged RHS
        C_pred, V_pred = self.compute_equity_value(
            k, b, z, use_target=False
        )

        rhs_accum = 0.0
        for _ in range(n_mc_rhs):
            rhs_accum += self.compute_rhs(k, b, z, k_prime, b_prime)
        rhs_true = rhs_accum / n_mc_rhs

        bell_abs = tf.abs(C_pred - rhs_true)
        bell_scale = tf.abs(rhs_true) + 1e-3
        bell_rel = bell_abs / bell_scale

        bell_rel_mean = tf.reduce_mean(bell_rel)
        bell_rel_max = tf.reduce_max(bell_rel)

        # FOC: gradients of single-draw RHS w.r.t. (k',b')
        with tf.GradientTape() as tape:
            tape.watch([k_prime, b_prime])
            rhs_single = self.compute_rhs(k, b, z, k_prime, b_prime)

        g_k, g_b = tape.gradient(rhs_single, [k_prime, b_prime])

        def safe(g):
            return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))

        g_k = safe(g_k)
        g_b = safe(g_b)

        foc_sq = tf.square(g_k) + tf.square(g_b)
        foc_mean_sq = tf.reduce_mean(foc_sq)
        foc_max_abs = tf.reduce_max(tf.sqrt(foc_sq))

        # Limited-liability residual: V vs relu(C)
        relu_C = tf.nn.relu(C_pred)
        ll_abs = tf.abs(V_pred - relu_C)
        ll_scale = tf.abs(relu_C) + 1e-3
        ll_rel = ll_abs / ll_scale

        ll_rel_mean = tf.reduce_mean(ll_rel)
        ll_rel_max = tf.reduce_max(ll_rel)

        return {
            "bell_rel_mean": float(bell_rel_mean.numpy()),
            "bell_rel_max": float(bell_rel_max.numpy()),
            "foc_mean_sq": float(foc_mean_sq.numpy()),
            "foc_max_abs": float(foc_max_abs.numpy()),
            "ll_rel_mean": float(ll_rel_mean.numpy()),
            "ll_rel_max": float(ll_rel_max.numpy()),
        }

    def test_coverage_residuals(
        self,
        n_samples: int = 500,
        n_mc_rhs: int = 50,
    ):
        """
        Coverage test: random states in (k,b,z) space.

        Here we reuse the shared risky_coverage_sampler from common.py,
        so the coverage region is driven by RiskyDebtParams.
        """
        states = risky_coverage_sampler(n_samples, self.params)  # [N, 3]
        k = states[:, 0:1]
        b = states[:, 1:2]
        z = states[:, 2:3]
        return self.compute_residuals(k, b, z, n_mc_rhs=n_mc_rhs)

    def test_onpolicy_residuals(
        self,
        states: tf.Tensor,
        n_samples: int = 500,
        n_mc_rhs: int = 50,
    ):
        """
        On-policy test: sample from given ergodic/on-policy states.
        states are [k, b, z] in levels.
        """
        N = tf.shape(states)[0]
        n = tf.minimum(N, n_samples)
        perm = tf.random.shuffle(tf.range(N))[:n]
        batch = tf.gather(states, perm, axis=0)
        k = batch[:, 0:1]
        b = batch[:, 1:2]
        z = batch[:, 2:3]
        return self.compute_residuals(k, b, z, n_mc_rhs=n_mc_rhs)


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------


def run_training(
    params: Optional[RiskyDebtParams] = None,
    train_params: Optional[RiskyDebtTrainingParams] = None,
    final_test_params: Optional[RiskyDebtFinalTestParams] = None,
):
    """
    Train the risky-debt model,
    while reusing parameters from config.py and utilities from common.py.
    """
    # Instantiate default configs if not provided
    if params is None:
        params = RiskyDebtParams()
    if train_params is None:
        train_params = RiskyDebtTrainingParams()
    if final_test_params is None:
        final_test_params = RiskyDebtFinalTestParams()

    # Global seed for reproducibility (Python, NumPy, TF)
    set_global_seed(params.seed)

    model = RiskyDebtModel(params=params, train_params=train_params)

    # Training schedule
    epochs = 2000
    warmup_epochs = 400          # grid-training phase
    # FOC weights from RiskyDebtTrainingParams
    lambda_foc_final = train_params.lambda_foc_final
    lambda_foc_warmup = train_params.lambda_foc_warmup

    # On-policy mixing schedule
    onpolicy_ramp_epochs = 500   # number of epochs over which on-policy share ramps up
    max_onpolicy_frac = 0.85     # upper bound on on-policy share (keep some grid forever)

    loss_history = []
    bell_history = []
    foc_history = []
    oos_history = []             # list of (epoch, cov_bell, onp_bell)

    print(
        f"Starting training: {epochs} epochs, warm-up = {warmup_epochs} epochs, "
        f"on-policy ramp = {onpolicy_ramp_epochs} epochs..."
    )
    start_time = time.time()

    # Initial on-policy state panel (used after warm-up)
    # States are [k, b, z] in levels.
    current_states = tf.concat(
        [
            tf.random.uniform(
                (model.batch_size, 1), 0.5, 1.5, dtype=DTYPE
            ),  # k
            tf.random.uniform(
                (model.batch_size, 1), 0.0, 1.0, dtype=DTYPE
            ),  # b
            tf.ones((model.batch_size, 1), dtype=DTYPE),  # z = 1
        ],
        axis=1,
    )

    def sample_mixed_states(
        current_states_: tf.Tensor,
        epoch: int,
    ):
        """
        Sample a batch that mixes random coverage states and on-policy states.

        During warm-up (epoch < warmup_epochs), frac_on = 0 and we use only
        coverage states. After warm-up, the on-policy share increases
        linearly up to max_onpolicy_frac over onpolicy_ramp_epochs.

        States are [k, b, z] in levels.
        """
        batch_size = model.batch_size

        if epoch < warmup_epochs:
            frac_on = 0.0
        else:
            t = min(1.0, float(epoch - warmup_epochs) / float(onpolicy_ramp_epochs))
            frac_on = max_onpolicy_frac * t

        n_on = int(batch_size * frac_on)
        n_cov = batch_size - n_on

        # Coverage states from the shared sampler
        if n_cov > 0:
            states_cov = risky_coverage_sampler(n_cov, params)
        else:
            states_cov = None

        # On-policy states sampled from current_states
        if n_on > 0:
            idx = tf.random.shuffle(tf.range(tf.shape(current_states_)[0]))[:n_on]
            states_on = tf.gather(current_states_, idx, axis=0)
        else:
            states_on = None

        if states_cov is None:
            states_batch = states_on
        elif states_on is None:
            states_batch = states_cov
        else:
            states_batch = tf.concat([states_cov, states_on], axis=0)

        return states_batch, frac_on

    for epoch in range(epochs):
        # 1. Choose training states (coverage + on-policy mix)
        states_to_train, frac_on = sample_mixed_states(current_states, epoch)

        # Set FOC weight: small during warm-up to pre-train policy, larger afterwards
        if epoch < warmup_epochs:
            model.lambda_foc = lambda_foc_warmup
        else:
            model.lambda_foc = lambda_foc_final

        # 2. Training step
        total_loss, loss_bell, loss_foc, kp, bp = model.train_step(states_to_train)

        loss_history.append(float(total_loss.numpy()))
        bell_history.append(float(loss_bell.numpy()))
        foc_history.append(float(loss_foc.numpy()))

        # 3. Ergodic update for on-policy state panel (using the same batch)
        z_curr = states_to_train[:, 2:3]   # z in levels
        z_next = model.get_next_z(z_curr, 1)  # [batch,1]
        next_states = tf.concat([kp, bp, z_next], axis=1)

        # Clamp states to a reasonable box
        next_states = tf.clip_by_value(
            next_states,
            clip_value_min=tf.constant([0.1, -0.5, 0.1], dtype=DTYPE),
            clip_value_max=tf.constant([5.0, 5.0, 5.0], dtype=DTYPE),
        )

        # Random resets for exploration (5%)
        mask_reset = tf.random.uniform(
            (model.batch_size, 1), dtype=DTYPE
        ) < 0.05
        rand_states = tf.concat(
            [
                tf.random.uniform(
                    (model.batch_size, 1), 0.5, 3.0, dtype=DTYPE
                ),  # k
                tf.random.uniform(
                    (model.batch_size, 1), -0.2, 2.0, dtype=DTYPE
                ),  # b
                tf.math.exp(
                    tf.random.normal(
                        (model.batch_size, 1), 0.0, 0.1, dtype=DTYPE
                    )
                ),
            ],
            axis=1,
        )

        current_states = tf.where(mask_reset, rand_states, next_states)

        # 4. Monitoring (every 50 epochs)
        if epoch % 50 == 0:
            # Anneal soft logic temperatures
            model.tau_V = max(0.5, model.tau_V * 0.98)
            model.tau_D = max(0.9, model.tau_D * 0.98)

            # Coverage OOS diagnostics
            cov_diag = model.test_coverage_residuals(
                n_samples=512, n_mc_rhs=20
            )
            # On-policy OOS diagnostics (using current_states)
            onp_diag = model.test_onpolicy_residuals(
                current_states, n_samples=512, n_mc_rhs=20
            )

            oos_history.append(
                (epoch, cov_diag["bell_rel_mean"], onp_diag["bell_rel_mean"])
            )

            phase = "warmup" if epoch < warmup_epochs else "mixed/on-policy"
            print(
                f"Ep {epoch:4d} [{phase}] "
                f"(on-policy frac ≈ {frac_on:.2f}): "
                f"Total={total_loss.numpy():.4f}, "
                f"Bell={loss_bell.numpy():.4f}, "
                f"FOC={loss_foc.numpy():.4f}, "
                f"CovBell={cov_diag['bell_rel_mean']:.3e}, "
                f"CovFOC²={cov_diag['foc_mean_sq']:.3e}, "
                f"OnPolBell={onp_diag['bell_rel_mean']:.3e}, "
                f"OnPolFOC²={onp_diag['foc_mean_sq']:.3e}"
            )

    print(f"Training done in {time.time() - start_time:.1f} seconds.")

    # ------------------------------------------------------------------
    # Plots: training diagnostics
    # ------------------------------------------------------------------

    epochs_arr = np.arange(epochs)

    plt.figure(figsize=(12, 5))

    # Loss decomposition
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, loss_history, label="Total loss")
    #plt.plot(epochs_arr, bell_history, label="Bellman loss")
    #plt.plot(epochs_arr, foc_history, label="FOC loss")
    plt.yscale("log")
    plt.axvline(warmup_epochs, color="r", linestyle="--", label="End warm-up")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training losses")
    plt.legend()

    # OOS Bellman error (coverage vs on-policy)
    plt.subplot(1, 2, 2)
    if len(oos_history) > 0:
        epochs_oos, cov_vals, onp_vals = zip(*oos_history)
        plt.plot(epochs_oos, cov_vals, "o-", label="Coverage Bellman OOS")
        plt.plot(epochs_oos, onp_vals, "s-", label="On-policy Bellman OOS")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Bellman error")
    plt.title("Out-of-sample Bellman errors")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Ergodic simulation helper
    # ------------------------------------------------------------------

    def simulate_ergodic_panel(
        model_: RiskyDebtModel,
        n_firms: int = 256,
        T: int = 400,
        burn_in: int = 100,
    ):
        """
        Simulate an ergodic panel of firm states under the learned policy.

        Args:
            model_: trained RiskyDebtModel
            n_firms: number of parallel firms in the cross-section
            T: total number of simulated periods
            burn_in: number of initial periods to discard

        Returns:
            k_panel, b_panel, z_panel: numpy arrays of shape [N_obs, 1]
                where N_obs = (T - burn_in) * n_firms

        States are [k, b, z] in levels; NN inputs use [log k, b/k, log z].
        """
        # Initial states: moderate values for k and b, z = 1
        k0 = tf.random.uniform((n_firms, 1), 0.5, 2.0, dtype=DTYPE)
        b0 = tf.random.uniform((n_firms, 1), 0.0, 1.5, dtype=DTYPE)
        z0 = tf.ones((n_firms, 1), dtype=DTYPE)
        states = tf.concat([k0, b0, z0], axis=1)

        states_list = []
        clip_min = tf.constant([0.1, -0.5, 0.1], dtype=DTYPE)
        clip_max = tf.constant([5.0, 5.0, 5.0], dtype=DTYPE)

        for t in range(T):
            k = states[:, 0:1]
            b = states[:, 1:2]
            z = states[:, 2:3]

            # Apply policy: NN input uses log z
            k_safe = tf.maximum(k, 1e-3)
            norm_k = tf.math.log(k_safe)
            norm_b = b / k_safe
            z_safe = tf.maximum(z, 1e-6)
            log_z = tf.math.log(z_safe)
            policy_input = tf.concat([norm_k, norm_b, log_z], axis=-1)
            k_prime, b_prime = model_.policy_net(policy_input)

            # Shock for z
            z_next = model_.get_next_z(z, 1)  # [n_firms, 1]

            # Next-period states (levels)
            states = tf.concat([k_prime, b_prime, z_next], axis=1)
            states = tf.clip_by_value(states, clip_min, clip_max)

            if t >= burn_in:
                # Collect states after burn-in as approximate ergodic panel
                states_list.append(states.numpy())

        states_all = np.vstack(states_list)  # [N_obs, 3]
        k_panel = states_all[:, 0:1]
        b_panel = states_all[:, 1:2]
        z_panel = states_all[:, 2:3]
        return k_panel, b_panel, z_panel

    # ------------------------------------------------------------------
    # Final diagnostics: coverage
    # ------------------------------------------------------------------

    print("\n========== Final coverage diagnostics ==========")
    cov_final = model.test_coverage_residuals(
        n_samples=final_test_params.n_coverage,
        n_mc_rhs=final_test_params.n_mc_rhs,
    )
    for key, val in cov_final.items():
        print(f"{key:>15s}: {val:.4e}")

    # ------------------------------------------------------------------
    # Simulate ergodic panel
    # ------------------------------------------------------------------

    k_panel, b_panel, z_panel = simulate_ergodic_panel(model)

    # ------------------------------------------------------------------
    # Final diagnostics: on-policy (ergodic)
    # ------------------------------------------------------------------

    print("\n========== Final on-policy (ergodic) diagnostics ==========")
    states_erg_np = np.concatenate([k_panel, b_panel, z_panel], axis=1)
    states_erg = tf.convert_to_tensor(states_erg_np, dtype=DTYPE)
    onp_final = model.test_onpolicy_residuals(
        states_erg,
        n_samples=final_test_params.n_onpolicy,
        n_mc_rhs=final_test_params.n_mc_rhs,
    )
    for key, val in onp_final.items():
        print(f"{key:>15s}: {val:.4e}")

    # ------------------------------------------------------------------
    # Ergodic leverage statistics
    # ------------------------------------------------------------------

    k_safe_np = np.maximum(k_panel, 1e-6)
    leverage = b_panel / k_safe_np

    print("\n========== Ergodic leverage statistics (b/k) ==========")
    print(f"Number of observations: {leverage.size}")
    print(f"Mean leverage          : {np.mean(leverage):.4f}")
    print(f"Median leverage        : {np.median(leverage):.4f}")
    print(f"Std of leverage        : {np.std(leverage):.4f}")
    print(f"10th percentile        : {np.percentile(leverage, 10):.4f}")
    print(f"50th percentile        : {np.percentile(leverage, 50):.4f}")
    print(f"90th percentile        : {np.percentile(leverage, 90):.4f}")
    print(f"Max leverage           : {np.max(leverage):.4f}")

    # Compute default probabilities for the ergodic panel (one-step ahead)
    print("\nComputing default probability distribution on ergodic panel...")
    batch_size_stats = 1024
    n_obs = k_panel.shape[0]
    pd_list = []

    for start in range(0, n_obs, batch_size_stats):
        end = min(start + batch_size_stats, n_obs)
        k_batch = tf.convert_to_tensor(
            k_panel[start:end, :], dtype=DTYPE
        )
        b_batch = tf.convert_to_tensor(
            b_panel[start:end, :], dtype=DTYPE
        )
        z_batch = tf.convert_to_tensor(
            z_panel[start:end, :], dtype=DTYPE
        )

        p_default_batch = model.default_probability(
            k_batch, b_batch, z_batch, n_draws=64, use_target=True
        )
        pd_list.append(p_default_batch.numpy())

    pd_all = np.vstack(pd_list)  # [n_obs, 1]

    print(
        "\n========== Ergodic default probability statistics "
        "(one-step-ahead) =========="
    )
    print(f"Mean default probability : {np.mean(pd_all):.4f}")
    print(f"Median default probability: {np.median(pd_all):.4f}")
    print(f"Std of default probability: {np.std(pd_all):.4f}")
    print(f"10th percentile           : {np.percentile(pd_all, 10):.4f}")
    print(f"50th percentile           : {np.percentile(pd_all, 50):.4f}")
    print(f"90th percentile           : {np.percentile(pd_all, 90):.4f}")
    print(f"Max default probability   : {np.max(pd_all):.4f}")

    # ------------------------------------------------------------------
    # Histograms: leverage and default probability
    # ------------------------------------------------------------------

    plt.figure(figsize=(12, 4))

    # Histogram of leverage
    plt.subplot(1, 2, 1)
    plt.hist(leverage, bins=40, density=True, alpha=0.7, color="tab:blue")
    plt.xlabel("Leverage b/k")
    plt.ylabel("Density")
    plt.title("Ergodic distribution of leverage (b/k)")

    # Histogram of default probability
    plt.subplot(1, 2, 2)
    plt.hist(pd_all, bins=40, density=True, alpha=0.7, color="tab:red")
    plt.xlabel("One-step default probability")
    plt.ylabel("Density")
    plt.title("Ergodic distribution of default probability")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Heat maps on (k, b) grid (z fixed)
    # ------------------------------------------------------------------

    print("\nGenerating heat maps on (k,b) grid...")

    # Grid for k and b (z fixed to 1, within coverage range from config)
    k_vals = np.linspace(params.k_cov_min, params.k_cov_max, 50)
    b_vals = np.linspace(params.b_cov_min, params.b_cov_max, 50)
    K_grid, B_grid = np.meshgrid(
        k_vals, b_vals, indexing="xy"
    )  # shape [Nb, Nk] after reshape

    k_flat = K_grid.reshape(-1, 1).astype(np.float32)
    b_flat = B_grid.reshape(-1, 1).astype(np.float32)
    z_flat = np.ones_like(k_flat, dtype=np.float32)  # z in levels

    # 1) Value function V(k,b,z=1)
    k_tf = tf.convert_to_tensor(k_flat, dtype=DTYPE)
    b_tf = tf.convert_to_tensor(b_flat, dtype=DTYPE)
    z_tf = tf.convert_to_tensor(z_flat, dtype=DTYPE)

    _, V_tf = model.compute_equity_value(
        k_tf, b_tf, z_tf, use_target=False
    )
    V_np = V_tf.numpy().reshape(len(b_vals), len(k_vals))  # [Nb, Nk]

    # 2) One-step-ahead default probability under policy
    pd_grid_tf = model.default_probability(
        k_tf, b_tf, z_tf, n_draws=64, use_target=True
    )
    pd_grid_np = pd_grid_tf.numpy().reshape(len(b_vals), len(k_vals))

    # 3) Next-period leverage b'/k' under policy
    k_safe_flat = np.maximum(k_flat, 1e-3)
    norm_k_flat = np.log(k_safe_flat)
    norm_b_flat = b_flat / k_safe_flat
    z_safe_flat = np.maximum(z_flat, 1e-6)
    log_z_flat = np.log(z_safe_flat)
    policy_input_np = np.concatenate(
        [norm_k_flat, norm_b_flat, log_z_flat], axis=1
    )
    policy_input_tf = tf.convert_to_tensor(
        policy_input_np, dtype=DTYPE
    )

    k_prime_tf, b_prime_tf = model.policy_net(policy_input_tf)
    k_prime_np = k_prime_tf.numpy()
    b_prime_np = b_prime_tf.numpy()
    k_prime_safe = np.maximum(k_prime_np, 1e-6)
    leverage_next_np = (b_prime_np / k_prime_safe).reshape(
        len(b_vals), len(k_vals)
    )

    # Plot heat maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Value function heat map
    im0 = axes[0].imshow(
        V_np,
        origin="lower",
        extent=[k_vals[0], k_vals[-1], b_vals[0], b_vals[-1]],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("Capital k")
    axes[0].set_ylabel("Debt b")
    axes[0].set_title("Equity value V(k,b,z=1)")
    fig.colorbar(im0, ax=axes[0])

    # Default probability heat map
    im1 = axes[1].imshow(
        pd_grid_np,
        origin="lower",
        extent=[k_vals[0], k_vals[-1], b_vals[0], b_vals[-1]],
        aspect="auto",
        cmap="magma",
    )
    axes[1].set_xlabel("Capital k")
    axes[1].set_ylabel("Debt b")
    axes[1].set_title("One-step default probability")
    fig.colorbar(im1, ax=axes[1])

    # Next-period leverage heat map
    im2 = axes[2].imshow(
        leverage_next_np,
        origin="lower",
        extent=[k_vals[0], k_vals[-1], b_vals[0], b_vals[-1]],
        aspect="auto",
        cmap="plasma",
    )
    axes[2].set_xlabel("Capital k")
    axes[2].set_ylabel("Debt b")
    axes[2].set_title("Next-period leverage b'/k'")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Policy function slice: b' as a function of b
    # ------------------------------------------------------------------

    k_fix = tf.ones((100, 1), dtype=DTYPE) * 1.5
    b_grid = tf.reshape(tf.linspace(0.0, 2.0, 100), (-1, 1))
    z_fix = tf.ones((100, 1), dtype=DTYPE)  # z in levels

    k_safe = tf.maximum(k_fix, 1e-3)
    norm_k = tf.math.log(k_safe)
    norm_b = b_grid / k_safe
    z_safe = tf.maximum(z_fix, 1e-6)
    log_z = tf.math.log(z_safe)
    inputs = tf.concat([norm_k, norm_b, log_z], axis=1)

    kp_slice, bp_slice = model.policy_net(inputs)

    plt.figure(figsize=(6, 4))
    plt.plot(b_grid.numpy(), bp_slice.numpy(), label="b'(b)")
    plt.plot(b_grid.numpy(), b_grid.numpy(), "k--", alpha=0.3, label="45° line")
    plt.xlabel("Current debt b")
    plt.ylabel("Next-period debt b'")
    plt.title("Debt policy (k=1.5, z=1)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Use central configuration objects from config.py
    rp = RiskyDebtParams()
    tp = RiskyDebtTrainingParams()
    fp = RiskyDebtFinalTestParams()

    run_training(params=rp, train_params=tp, final_test_params=fp)