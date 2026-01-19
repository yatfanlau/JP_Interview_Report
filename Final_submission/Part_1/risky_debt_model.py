"""
Deep-learning solver for the risky-debt model.

This module contains:
- RiskyDebtModel (policy/value networks, pricing, Bellman RHS, training step)
- Training loop that reproduces the original runtime behavior
"""

import time
import numpy as np

from config_part1 import RiskyDebtParams, RiskyDebtTrainingParams, RiskyDebtFinalTestParams
from model_core import tf, DTYPE, set_global_seed, ar1_step_ln_z


class RiskyDebtModel(tf.keras.Model):
    """
    Risky-debt model with:
    - Policy network producing (k', b')
    - Value network approximating raw equity value C
    - Target value network for stabilized bootstrapping
    """

    def __init__(self, params: RiskyDebtParams, train_params: RiskyDebtTrainingParams):
        """
        Initialize the model, networks, and optimizer.

        Args:
            params: Economic parameters.
            train_params: Training hyperparameters and schedules.
        """
        super().__init__()
        self.params = params
        self.train_params = train_params

        # Model parameters
        self.theta = params.theta
        self.delta = params.delta
        self.phi = params.phi
        self.r = params.r
        self.beta = 1.0 / (1.0 + self.r)
        self.tau = params.tau
        self.bankruptcy_cost = params.bankruptcy_cost
        self.rho_z = params.rho_z
        self.sigma_z = params.sigma_z
        self.mu_ln_z = tf.constant(params.mu_ln_z, dtype=DTYPE)
        self.eta_0 = params.eta_0
        self.eta_1 = params.eta_1

        # Training parameters / numerics
        self.batch_size = train_params.batch_size
        self.M_pricing = params.M_pricing
        self.eps_b = params.eps_b
        self.tau_V = params.tau_V_init
        self.tau_D = params.tau_D_init

        n_hidden = params.n_hidden

        # Policy network: inputs are (log k, b/k, log z)
        policy_in = tf.keras.Input(shape=(3,), dtype=DTYPE)
        x = tf.keras.layers.Dense(n_hidden, activation="elu")(policy_in)
        x = tf.keras.layers.Dense(n_hidden, activation="elu")(x)
        raw_kp = tf.keras.layers.Dense(1, activation="softplus")(x)
        # Ensure positive next-period capital
        kp_out = tf.keras.layers.Lambda(lambda t: t + params.k_cov_min)(raw_kp)
        bp_out = tf.keras.layers.Dense(1)(x)
        self.policy_net = tf.keras.Model(inputs=policy_in, outputs=[kp_out, bp_out])

        # Value network: predicts raw equity value C
        val_in = tf.keras.Input(shape=(3,), dtype=DTYPE)
        xv = tf.keras.layers.Dense(n_hidden, activation="elu")(val_in)
        xv = tf.keras.layers.Dense(n_hidden, activation="elu")(xv)
        Cv_out = tf.keras.layers.Dense(1)(xv)
        self.value_net = tf.keras.Model(inputs=val_in, outputs=Cv_out)

        # Target value network
        self.target_value_net = tf.keras.models.clone_model(self.value_net)
        self.target_value_net.set_weights(self.value_net.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(train_params.lr)

    def coverage_sampler(self, batch_size: int) -> tf.Tensor:
        """
        Coverage sampler for risky-debt model states.

        Draws:
            k ~ Uniform[k_cov_min, k_cov_max]
            b ~ Uniform[b_cov_min, b_cov_max]
            ln z ~ Normal(lnz_cov_mean, lnz_cov_std^2)
            z = exp(ln z)

        Args:
            batch_size: Number of states to draw.

        Returns:
            Tensor of shape (batch_size, 3) with columns [k, b, z].
        """
        rp = self.params
        k = tf.random.uniform(
            (batch_size, 1),
            minval=rp.k_cov_min,
            maxval=rp.k_cov_max,
            dtype=DTYPE,
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=rp.b_cov_min,
            maxval=rp.b_cov_max,
            dtype=DTYPE,
        )
        lnz = tf.random.normal(
            (batch_size, 1),
            mean=rp.lnz_cov_mean,
            stddev=rp.lnz_cov_std,
            dtype=DTYPE,
        )
        z = tf.exp(lnz)
        return tf.concat([k, b, z], axis=1)

    def state_features(self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """
        Build normalized state features for policy/value networks.

        Features are:
            [log(k_safe), b/k_safe, log(z_safe)]

        Args:
            k: Capital, shape (B,1).
            b: Debt, shape (B,1).
            z: Productivity, shape (B,1).

        Returns:
            Tensor of shape (B,3).
        """
        k_safe = tf.maximum(k, 1e-3)
        norm_k = tf.math.log(k_safe)
        norm_b = b / k_safe
        z_safe = tf.maximum(z, 1e-6)
        log_z = tf.math.log(z_safe)
        return tf.concat([norm_k, norm_b, log_z], axis=-1)

    def get_next_z(self, z: tf.Tensor, n_samples: int) -> tf.Tensor:
        """
        Draw next-period productivity shocks given current z.

        Args:
            z: Current z, shape (B,1).
            n_samples: Number of shock draws per state.

        Returns:
            z_next: Tensor of shape (B, n_samples).
        """
        batch_size = tf.shape(z)[0]
        z_tiled = tf.tile(z, [1, n_samples])
        eps = tf.random.normal(
            shape=(batch_size, n_samples),
            mean=0.0,
            stddev=self.sigma_z,
            dtype=DTYPE,
        )
        z_next = ar1_step_ln_z(
            z_tiled,
            rho=tf.constant(self.rho_z, dtype=DTYPE),
            eps=eps,
            mu_ln_z=self.mu_ln_z,
        )
        return z_next

    def production(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Production function π(k,z) = z * k^theta."""
        return z * tf.pow(k, self.theta)

    def adjustment_cost(self, k: tf.Tensor, k_prime: tf.Tensor) -> tf.Tensor:
        """
        Quadratic adjustment cost.

        Args:
            k: Current capital, shape (B,1).
            k_prime: Next capital, shape (B,1).

        Returns:
            Adjustment cost, shape (B,1).
        """
        k_safe = tf.maximum(k, tf.constant(1e-8, dtype=DTYPE))
        i = k_prime - (1.0 - self.delta) * k_safe
        return 0.5 * self.phi * tf.square(i - self.delta * k_safe) / k_safe

    def get_recovery(self, k_prime: tf.Tensor, z_prime: tf.Tensor) -> tf.Tensor:
        """
        Recovery value in default states.

        Args:
            k_prime: Next-period capital, shape (B,1).
            z_prime: Next-period productivity, shape (B,1).

        Returns:
            Recovery payoff before capping by face value, shape (B,1).
        """
        pi = self.production(k_prime, z_prime)
        val = (1.0 - self.tau) * pi + (1.0 - self.delta) * k_prime
        return (1.0 - self.bankruptcy_cost) * val

    def compute_equity_value(self, k, b, z, use_target: bool = False):
        """
        Compute raw equity value C and smoothed limited-liability value V.

        Args:
            k: Capital, shape (B,1).
            b: Debt, shape (B,1).
            z: Productivity, shape (B,1).
            use_target: If True, use the target value network.

        Returns:
            (C, V), both shape (B,1).
        """
        state = self.state_features(k, b, z)

        if use_target:
            C = self.target_value_net(state)
        else:
            C = self.value_net(state)

        # Softplus smoothing of max(C, 0)
        V = self.tau_V * tf.math.softplus(C / self.tau_V)
        return C, V

    def default_probability(self, k, b, z, n_draws: int = 1500, use_target: bool = True):
        """
        One-step-ahead default probability.

        Default is interpreted as: next-period raw equity value C' < 0
        AND next-period debt b' > 0.

        Args:
            k: Capital, shape (B,1).
            b: Debt, shape (B,1).
            z: Productivity, shape (B,1).
            n_draws: Monte Carlo draws for z'.
            use_target: If True, compute C' using the target value network.

        Returns:
            p_default: Tensor of shape (B,1).
        """
        policy_input = self.state_features(k, b, z)

        # Policy gives (k', b') at current state
        k_prime, b_prime = self.policy_net(policy_input)

        # Draw future shocks z'
        z_prime = self.get_next_z(z, n_draws)
        k_prime_exp = tf.tile(k_prime, [1, n_draws])
        b_prime_exp = tf.tile(b_prime, [1, n_draws])

        # Flatten to feed into value network
        kp_flat = tf.reshape(k_prime_exp, [-1, 1])
        bp_flat = tf.reshape(b_prime_exp, [-1, 1])
        zp_flat = tf.reshape(z_prime, [-1, 1])

        # Compute next-period raw equity value C'
        C_prime, _ = self.compute_equity_value(
            kp_flat, bp_flat, zp_flat, use_target=use_target
        )

        # Smooth indicator 1{C' < 0}
        D_prime = tf.sigmoid(-C_prime / self.tau_D)

        # Only treat states with strictly positive next-period debt as credit default
        is_debt = tf.cast(bp_flat > self.eps_b, DTYPE)
        D_prime = D_prime * is_debt

        # Average over z' draws
        D_prime = tf.reshape(D_prime, [-1, n_draws])
        p_default = tf.reduce_mean(D_prime, axis=1, keepdims=True)
        return p_default

    def pricing_kernel(self, k_prime, b_prime, z_current):
        """
        Risky bond pricing kernel.

        Default is defined as (C' < 0 and b' > 0).
        When b' <= eps_b, price at the risk-free rate.

        Args:
            k_prime: Next-period capital, shape (B,1).
            b_prime: Next-period debt, shape (B,1).
            z_current: Current productivity, shape (B,1).

        Returns:
            q_final: Bond price, shape (B,1).
        """
        z_prime = self.get_next_z(z_current, self.M_pricing)
        k_prime_exp = tf.tile(k_prime, [1, self.M_pricing])
        b_prime_exp = tf.tile(b_prime, [1, self.M_pricing])

        kp_flat = tf.reshape(k_prime_exp, [-1, 1])
        bp_flat = tf.reshape(b_prime_exp, [-1, 1])
        zp_flat = tf.reshape(z_prime, [-1, 1])

        # Next-period equity raw value
        C_prime, _ = self.compute_equity_value(
            kp_flat, bp_flat, zp_flat, use_target=True
        )

        # Smooth indicator for C' < 0
        D_prime = tf.sigmoid(-C_prime / self.tau_D)

        # Mask out states with no positive debt: no credit default without debt
        is_debt_bool = bp_flat > self.eps_b
        is_debt_float = tf.cast(is_debt_bool, DTYPE)
        D_prime = D_prime * is_debt_float

        # Recovery payoff in default (before capping by face value)
        R_prime = self.get_recovery(kp_flat, zp_flat)

        # Effective face value only where there is debt
        b_pos = tf.where(is_debt_bool, bp_flat, tf.ones_like(bp_flat))
        R_eff = tf.minimum(R_prime, b_pos)

        # R / b only in debt states; zero otherwise
        ratio_R = tf.where(is_debt_bool, R_eff / b_pos, tf.zeros_like(R_eff))
        ratio_R = tf.clip_by_value(ratio_R, 0.0, 1.0)

        D_2d = tf.reshape(D_prime, [-1, self.M_pricing])
        DR_over_b_2d = tf.reshape(D_prime * ratio_R, [-1, self.M_pricing])

        # A: expected default payoff (recovery / b)
        A = tf.reduce_mean(DR_over_b_2d, axis=1, keepdims=True)
        # p_solv: probability of solvency (no default)
        p_solv = tf.reduce_mean(1.0 - D_2d, axis=1, keepdims=True)

        # Avoid division by very small values
        eps_p = 1e-4
        exp_term = A + p_solv
        denom = tf.maximum(exp_term, eps_p)

        # One plus risky gross return
        one_plus_tilde = (1.0 + self.r) / denom
        one_plus_tilde = tf.clip_by_value(one_plus_tilde, 1e-3, 1e3)
        q_raw = 1.0 / one_plus_tilde

        q_riskfree = 1.0 / (1.0 + self.r)

        # For states with no positive debt, price at risk-free
        mask_debt = tf.cast(b_prime > self.eps_b, DTYPE)
        q_final = mask_debt * q_raw + (1.0 - mask_debt) * q_riskfree
        q_final = tf.clip_by_value(q_final, 0.01, q_riskfree)
        return q_final

    def compute_rhs(self, k, b, z, k_prime, b_prime, n_z_draws: int = 1):
        """
        Monte Carlo approximation of the Bellman RHS:

            RHS = dividends + beta * E_z'[V'(k',b',z')].

        Args:
            k: Current capital, shape (B,1).
            b: Current debt, shape (B,1).
            z: Current productivity, shape (B,1).
            k_prime: Next capital (control), shape (B,1).
            b_prime: Next debt (control), shape (B,1).
            n_z_draws: Number of z' draws for the continuation value.

        Returns:
            rhs: Tensor of shape (B,1).
        """
        q = self.pricing_kernel(k_prime, b_prime, z)
        pi = self.production(k, z)
        inv_cost = k_prime - (1.0 - self.delta) * k + self.adjustment_cost(k, k_prime)
        debt_proceeds = q * b_prime
        tax_shield = self.tau * (1.0 - q) * b_prime * self.beta

        e = (1.0 - self.tau) * pi - inv_cost + debt_proceeds + tax_shield - b

        is_negative = tf.cast(e < 0.0, DTYPE)
        financing_cost = is_negative * (self.eta_0 + self.eta_1 * tf.abs(e))
        div = e - financing_cost

        # Multiple z' draws
        z_prime = self.get_next_z(z, n_z_draws)

        # Tile k' and b' to match z' draws
        k_prime_exp = tf.tile(k_prime, [1, n_z_draws])
        b_prime_exp = tf.tile(b_prime, [1, n_z_draws])

        # Flatten to feed into value network
        kp_flat = tf.reshape(k_prime_exp, [-1, 1])
        bp_flat = tf.reshape(b_prime_exp, [-1, 1])
        zp_flat = tf.reshape(z_prime, [-1, 1])

        # Compute V' for each draw using target network
        _, V_prime_all = self.compute_equity_value(
            kp_flat, bp_flat, zp_flat, use_target=True
        )

        # Average over z' draws
        V_prime_all = tf.reshape(V_prime_all, [-1, n_z_draws])
        V_prime_mean = tf.reduce_mean(V_prime_all, axis=1, keepdims=True)

        rhs = div + self.beta * V_prime_mean
        return rhs

    @tf.function
    def train_step(self, states):
        """
        Single training step based on a batch of states.

        Args:
            states: Tensor of shape (B,3) with columns [k, b, z].

        Returns:
            total_loss, loss_bellman, loss_foc, k_prime, b_prime
        """
        k = states[:, 0:1]
        b = states[:, 1:2]
        z = states[:, 2:3]

        lambda_foc = self.lambda_foc

        with tf.GradientTape() as outer_tape:
            # Normalize states for policy network
            policy_input = self.state_features(k, b, z)

            # Policy actions
            k_prime, b_prime = self.policy_net(policy_input)

            # Current raw equity value
            C_theta, _ = self.compute_equity_value(k, b, z, use_target=False)

            # Double-sampling scheme for Bellman residuals and FOCs
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([k_prime, b_prime])
                rhs_a = self.compute_rhs(
                    k,
                    b,
                    z,
                    k_prime,
                    b_prime,
                    n_z_draws=self.train_params.rhs_n_z_draws_train,
                )
                rhs_b = self.compute_rhs(
                    k,
                    b,
                    z,
                    k_prime,
                    b_prime,
                    n_z_draws=self.train_params.rhs_n_z_draws_train,
                )

            # Squared Bellman residual term (double sampling)
            bellman_residuals = (C_theta - tf.stop_gradient(rhs_a)) * (
                C_theta - tf.stop_gradient(rhs_b)
            )
            loss_bellman = tf.reduce_mean(bellman_residuals)

            # FOC penalties based on gradients of RHS wrt controls
            grads_a = inner_tape.gradient(rhs_a, [k_prime, b_prime])
            grads_b = inner_tape.gradient(rhs_b, [k_prime, b_prime])
            del inner_tape

            def safe_grad(g):
                """Replace non-finite gradients with zeros for numerical stability."""
                return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))

            g_a_k = safe_grad(grads_a[0])
            g_a_b = safe_grad(grads_a[1])
            g_b_k = safe_grad(grads_b[0])
            g_b_b = safe_grad(grads_b[1])

            foc_k = tf.stop_gradient(g_a_k) * g_b_k
            foc_b = tf.stop_gradient(g_a_b) * g_b_b
            loss_foc = tf.reduce_mean(foc_k + foc_b)

            total_loss = loss_bellman + lambda_foc * loss_foc

        trainable_vars = (
            self.policy_net.trainable_variables + self.value_net.trainable_variables
        )
        grads = outer_tape.gradient(total_loss, trainable_vars)
        del outer_tape

        # Gradient clipping for stability
        clipped_grads = []
        for g in grads:
            if g is None:
                clipped_grads.append(None)
            else:
                clipped_grads.append(tf.clip_by_norm(g, self.train_params.grad_clip_norm))

        self.optimizer.apply_gradients(zip(clipped_grads, trainable_vars))

        # Polyak averaging for target network
        tau_polyak = self.train_params.polyak_tau
        for v_main, v_target in zip(
            self.value_net.variables, self.target_value_net.variables
        ):
            v_target.assign(tau_polyak * v_main + (1.0 - tau_polyak) * v_target)

        return total_loss, loss_bellman, loss_foc, k_prime, b_prime


def run_training(
    params: RiskyDebtParams | None = None,
    train_params: RiskyDebtTrainingParams | None = None,
    final_test_params: RiskyDebtFinalTestParams | None = None,
):
    """
    Train the risky-debt model and run diagnostics/plots.

    Defaults reproduce the original training and reporting behavior.
    """
    if params is None:
        params = RiskyDebtParams()
    if train_params is None:
        train_params = RiskyDebtTrainingParams()
    if final_test_params is None:
        final_test_params = RiskyDebtFinalTestParams()

    # Import here to keep the model module lightweight for ones who only want the class.
    from risky_debt_diagnostic import (
        plot_training_diagnostics,
        test_coverage_residuals,
        test_onpolicy_residuals,
        run_final_diagnostics_and_plots,
    )

    set_global_seed(params.seed)
    model = RiskyDebtModel(params=params, train_params=train_params)

    epochs = train_params.epochs
    warmup_epochs = train_params.warmup_epochs
    lambda_foc_final = train_params.lambda_foc_final
    lambda_foc_warmup = train_params.lambda_foc_warmup
    onpolicy_ramp_epochs = train_params.onpolicy_ramp_epochs
    max_onpolicy_frac = train_params.max_onpolicy_frac

    loss_history = []
    bell_history = []
    foc_history = []
    oos_history = []

    print(
        f"Starting training: {epochs} epochs, warm-up = {warmup_epochs} epochs, "
        f"on-policy ramp = {onpolicy_ramp_epochs} epochs..."
    )

    start_time = time.time()

    # Initial state distribution for training
    current_states = tf.concat(
        [
            tf.random.uniform(
                (model.batch_size, 1),
                params.k_cov_min,
                params.k_cov_max,
                dtype=DTYPE,
            ),
            tf.random.uniform(
                (model.batch_size, 1),
                params.b_cov_min,
                params.b_cov_max,
                dtype=DTYPE,
            ),
            tf.ones((model.batch_size, 1), dtype=DTYPE),
        ],
        axis=1,
    )

    def sample_mixed_states(current_states_, epoch):
        """
        Mix coverage states and on-policy states; ramp up on-policy fraction over training.

        Args:
            current_states_: Tensor of candidate on-policy states.
            epoch: Current epoch index.

        Returns:
            (states_batch, frac_on) where frac_on is the on-policy share used.
        """
        batch_size = model.batch_size

        if epoch < warmup_epochs:
            frac_on = 0.0
        else:
            t = min(1.0, float(epoch - warmup_epochs) / float(onpolicy_ramp_epochs))
            frac_on = max_onpolicy_frac * t

        n_on = int(batch_size * frac_on)
        n_cov = batch_size - n_on

        if n_cov > 0:
            states_cov = model.coverage_sampler(n_cov)
        else:
            states_cov = None

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
        states_to_train, frac_on = sample_mixed_states(current_states, epoch)

        # FOC weight schedule
        if epoch < warmup_epochs:
            model.lambda_foc = lambda_foc_warmup
        else:
            model.lambda_foc = lambda_foc_final

        total_loss, loss_bell, loss_foc, kp, bp = model.train_step(states_to_train)

        loss_history.append(float(total_loss.numpy()))
        bell_history.append(float(loss_bell.numpy()))
        foc_history.append(float(loss_foc.numpy()))

        # Roll states forward using learned policy + z dynamics
        z_curr = states_to_train[:, 2:3]
        z_next = model.get_next_z(z_curr, 1)

        next_states = tf.concat([kp, bp, z_next], axis=1)
        next_states = tf.clip_by_value(
            next_states,
            clip_value_min=tf.constant(
                [params.k_cov_min, params.b_cov_min, train_params.z_clip_min], dtype=DTYPE
            ),
            clip_value_max=tf.constant(
                [params.k_cov_max, params.b_cov_max, train_params.z_clip_max], dtype=DTYPE
            ),
        )

        # Occasional random resets to keep exploration
        mask_reset = tf.random.uniform((model.batch_size, 1), dtype=DTYPE) < train_params.reset_prob
        rand_states = tf.concat(
            [
                tf.random.uniform(
                    (model.batch_size, 1),
                    params.k_cov_min,
                    params.k_cov_max,
                    dtype=DTYPE,
                ),
                tf.random.uniform(
                    (model.batch_size, 1),
                    params.b_cov_min,
                    params.b_cov_max,
                    dtype=DTYPE,
                ),
                tf.math.exp(
                    tf.random.normal(
                        (model.batch_size, 1), 0.0, train_params.reset_z_log_std, dtype=DTYPE
                    )
                ),
            ],
            axis=1,
        )
        current_states = tf.where(mask_reset, rand_states, next_states)

        if epoch % train_params.diag_every == 0:
            # Anneal smoothing temperatures
            model.tau_V = max(train_params.tau_min, model.tau_V * train_params.tau_anneal_factor)
            model.tau_D = max(train_params.tau_min, model.tau_D * train_params.tau_anneal_factor)
            print("tau_D=", model.tau_D)

            cov_diag = test_coverage_residuals(
                model, n_samples=train_params.diag_n_samples, n_mc_rhs=train_params.diag_n_mc_rhs
            )
            onp_diag = test_onpolicy_residuals(
                model, current_states, n_samples=train_params.diag_n_samples, n_mc_rhs=train_params.diag_n_mc_rhs
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

    # Plot training diagnostics (same figures as before)
    plot_training_diagnostics(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        loss_history=loss_history,
        bell_history=bell_history,
        foc_history=foc_history,
        oos_history=oos_history,
    )

    # Final diagnostics, ergodic simulation, and plots
    run_final_diagnostics_and_plots(
        model=model,
        params=params,
        train_params=train_params,
        final_test_params=final_test_params,
    )


if __name__ == "__main__":
    run_training(RiskyDebtParams(), RiskyDebtTrainingParams(), RiskyDebtFinalTestParams())