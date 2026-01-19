import os
import numpy as np
import pandas as pd

from utils_part2 import tf, DTYPE, set_global_seed, logit_np, log_sigmoid_prime_tf
from config_part2 import (
    PathsPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    BasicModelParams,
    HMCConfigPart2,
    HMCColumnsPart2,
)

from amortized_policy import policy_iota

try:
    import tensorflow_probability as tfp
except Exception as e:
    raise ImportError(
        "tensorflow_probability is required for HMC estimation. "
        "Install it via `pip install tensorflow-probability`."
    ) from e


LOG2PI = tf.constant(np.log(2.0 * np.pi), dtype=DTYPE)


def load_amortized_policy(path: str):
    """Load a trained amortized policy network from disk.

    Args:
      path: Filesystem path to the saved Keras model.

    Returns:
      Loaded tf.keras.Model policy network.
    """
    print(f"Loading amortized policy from: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    print("Loaded policy model:")
    try:
        model.summary()
    except Exception:
        pass
    return model


def _df_to_balanced_matrix(
    df: pd.DataFrame,
    value_col: str,
    firm_col: str,
    time_col: str,
) -> np.ndarray:
    """Convert a balanced panel DataFrame column into a [N_firms, T] matrix.

    Args:
      df: Input DataFrame containing panel data.
      value_col: Column name to reshape (measurement).
      firm_col: Firm identifier column name.
      time_col: Time identifier column name.

    Returns:
      2D NumPy array of shape [n_firms, t_periods] with values from value_col.

    Raises:
      ValueError: If the panel is not balanced.
    """
    df = df.sort_values([firm_col, time_col])
    n_firms = int(df[firm_col].nunique())
    t_periods = int(df[time_col].nunique())
    arr = df[value_col].to_numpy(dtype=np.float32)

    # Check that we have exactly one row per (firm, time).
    if arr.size != n_firms * t_periods:
        raise ValueError(
            f"Panel not balanced for col={value_col}: got {arr.size} rows, "
            f"expected {n_firms}*{t_periods}={n_firms * t_periods}."
        )
    return arr.reshape(n_firms, t_periods)


def _make_observations_from_truth(
    df_rep: pd.DataFrame,
    cols: PanelColumnsPart2,
    theta_true: float,
    cfg: HMCConfigPart2,
    seed: int,
):
    """Generate noisy observations (y_obs, logk_obs) from true (k, z).

    Uses fixed measurement noise scales from cfg:

      y_obs = ln z + theta_true * ln k + sigma_y * eta
      logk_obs = ln k + sigma_logk_obs * xi

    Args:
      df_rep: Data for a single replication.
      cols: Column names for the panel.
      theta_true: True production elasticity parameter.
      cfg: HMC configuration with measurement noise scales.
      seed: Random seed for measurement noise.

    Returns:
      Tuple (y_obs, logk_obs), each a float32 array of shape [n_firms, T].
    """
    k_mat = _df_to_balanced_matrix(df_rep, cols.k, cols.firm, cols.time)
    z_mat = _df_to_balanced_matrix(df_rep, cols.z, cols.firm, cols.time)

    logk = np.log(np.maximum(k_mat, 1e-32)).astype(np.float32)
    lnz = np.log(np.maximum(z_mat, 1e-32)).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    eta = rng.normal(0.0, 1.0, size=logk.shape).astype(np.float32)
    xi = rng.normal(0.0, 1.0, size=logk.shape).astype(np.float32)

    y_obs = lnz + float(theta_true) * logk + float(cfg.sigma_y) * eta
    logk_obs = logk + float(cfg.sigma_logk_obs) * xi
    return y_obs.astype(np.float32), logk_obs.astype(np.float32)


def build_hmc_context(mp: BasicModelParams, cfg: HMCConfigPart2):
    """Build static tensors and moments used by the UKF+HMC estimator.

    Computes stationary moments for x_t = ln z_t and constructs transition and
    observation noise covariance matrices.

    Args:
      mp: Basic model parameters for the productivity process and capital law.
      cfg: HMC configuration (only noise scales are used here).

    Returns:
      Dictionary of TensorFlow constants used throughout the UKF+HMC code.
    """
    rho = float(mp.rho)
    sigma_eps = float(mp.sigma_eps)

    # Choose mu_ln_z so that E[z] = 1 in the stationary distribution.
    mu_ln_z = float(-0.5 * (sigma_eps**2) / (1.0 + rho))
    x_mean = float(mu_ln_z / (1.0 - rho))
    x_var = float((sigma_eps**2) / (1.0 - rho**2))

    k_floor = tf.constant(1e-12, dtype=DTYPE)
    logk_floor = tf.math.log(k_floor)

    ctx = dict(
        delta=tf.constant(mp.delta, dtype=DTYPE),
        rho=tf.constant(mp.rho, dtype=DTYPE),
        sigma_eps=tf.constant(mp.sigma_eps, dtype=DTYPE),
        mu_ln_z=tf.constant(mu_ln_z, dtype=DTYPE),
        k_floor=k_floor,
        logk_floor=logk_floor,
        # Stationary moments for initial x distribution.
        x_mean=tf.constant(x_mean, dtype=DTYPE),
        x_var=tf.constant(x_var, dtype=DTYPE),
        # Process noise covariance for (x, log k).
        Q=tf.linalg.diag(
            tf.constant(
                [sigma_eps**2, float(cfg.sigma_logk_trans)**2],
                dtype=DTYPE,
            )
        ),
        # Measurement noise covariance for (y, log k_obs).
        R=tf.linalg.diag(
            tf.constant(
                [float(cfg.sigma_y)**2, float(cfg.sigma_logk_obs)**2],
                dtype=DTYPE,
            )
        ),
    )
    return ctx


def _ukf_weights(n: int, alpha: float, beta: float, kappa: float):
    """Compute UKF sigma-point weights.

    Args:
      n: State dimension.
      alpha: Primary spread parameter for sigma points.
      beta: Prior knowledge of distribution (2 is optimal for Gaussian).
      kappa: Secondary spread parameter.

    Returns:
      Tuple (Wm, Wc, gamma) where:
        Wm: Mean weights, shape [2n + 1].
        Wc: Covariance weights, shape [2n + 1].
        gamma: Scaling factor for Cholesky factor of covariance.
    """
    n_f = tf.constant(float(n), dtype=DTYPE)
    alpha_f = tf.constant(float(alpha), dtype=DTYPE)
    beta_f = tf.constant(float(beta), dtype=DTYPE)
    kappa_f = tf.constant(float(kappa), dtype=DTYPE)

    lam = alpha_f**2 * (n_f + kappa_f) - n_f
    c = n_f + lam
    gamma = tf.sqrt(c)

    Wm0 = lam / c
    Wc0 = Wm0 + (1.0 - alpha_f**2 + beta_f)
    Wi = 1.0 / (2.0 * c)

    # Length 2n + 1.
    Wm = tf.concat([[Wm0], tf.fill([2 * n], Wi)], axis=0)
    Wc = tf.concat([[Wc0], tf.fill([2 * n], Wi)], axis=0)
    return Wm, Wc, gamma


def _sigma_points(
    mean_b2: tf.Tensor,
    cov_b22: tf.Tensor,
    gamma: tf.Tensor,
    jitter: float,
) -> tf.Tensor:
    """Generate sigma points for each batch element.

    Args:
      mean_b2: State mean, shape [B, 2].
      cov_b22: State covariance, shape [B, 2, 2].
      gamma: UKF scaling factor.
      jitter: Diagonal jitter added to covariances for numerical stability.

    Returns:
      Sigma points tensor of shape [B, 2n + 1, 2] (here 2n + 1 = 5).
    """
    B = tf.shape(mean_b2)[0]
    I = tf.eye(2, dtype=DTYPE)[None, :, :]

    # Symmetrize covariance and add jitter to ensure positive definiteness.
    cov = 0.5 * (cov_b22 + tf.transpose(cov_b22, [0, 2, 1])) + tf.constant(
        float(jitter),
        dtype=DTYPE,
    ) * I

    chol = tf.linalg.cholesky(cov)

    # scaled has shape [B, 2, 2]; columns are direction vectors.
    scaled = gamma * chol
    cols = tf.transpose(scaled, [0, 2, 1])  # [B, 2, 2] with columns in last dim.
    m = mean_b2[:, None, :]  # [B, 1, 2]

    plus = m + cols  # [B, 2, 2]
    minus = m - cols  # [B, 2, 2]
    X = tf.concat([m, plus, minus], axis=1)  # [B, 5, 2]
    return X


def _weighted_mean(X_bj2: tf.Tensor, Wm_j: tf.Tensor) -> tf.Tensor:
    """Compute weighted mean over sigma points.

    Args:
      X_bj2: Sigma points, shape [B, J, 2].
      Wm_j: Mean weights, shape [J].

    Returns:
      Weighted mean over sigma points, shape [B, 2].
    """
    return tf.einsum("j,bjv->bv", Wm_j, X_bj2)


def _weighted_cov(dX_bj2: tf.Tensor, Wc_j: tf.Tensor) -> tf.Tensor:
    """Compute weighted covariance over centered sigma points.

    Args:
      dX_bj2: Centered sigma points, shape [B, J, 2].
      Wc_j: Covariance weights, shape [J].

    Returns:
      Covariance matrices, shape [B, 2, 2].
    """
    return tf.einsum("j,bjv,bjw->bvw", Wc_j, dX_bj2, dX_bj2)


def _weighted_cross_cov(
    dX_bj2: tf.Tensor,
    dY_bj2: tf.Tensor,
    Wc_j: tf.Tensor,
) -> tf.Tensor:
    """Compute weighted cross-covariance between two sets of sigma points.

    Args:
      dX_bj2: Centered sigma points for X, shape [B, J, 2].
      dY_bj2: Centered sigma points for Y, shape [B, J, 2].
      Wc_j: Covariance weights, shape [J].

    Returns:
      Cross-covariance matrices, shape [B, 2, 2].
    """
    return tf.einsum("j,bjv,bjw->bvw", Wc_j, dX_bj2, dY_bj2)


def _measurement_fn(X_bj2: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """Apply measurement equation to sigma points.

    Observations:
      o_t = [y_t, logk_obs_t]
      y_t = x_t + theta * logk_t
      logk_obs_t = logk_t

    Args:
      X_bj2: Sigma points for state (x, logk), shape [B, J, 2].
      theta: Production elasticity, scalar tensor.

    Returns:
      Predicted observations for each sigma point, shape [B, J, 2].
    """
    x = X_bj2[..., 0]
    logk = X_bj2[..., 1]
    y = x + theta * logk
    return tf.stack([y, logk], axis=-1)


def _transition_fn(
    X_bj2: tf.Tensor,
    theta: tf.Tensor,
    phi: tf.Tensor,
    policy,
    ctx,
) -> tf.Tensor:
    """Apply nonlinear state transition for s_t = (x_t, logk_t).

    Dynamics:
      x_{t+1} = mu + rho * x_t
      logk_{t+1} = log(max(k_floor, (1 - delta + iota) * k_t))

    The policy network determines investment iota as a function of
    (ln k_t, ln z_t, theta, log phi).

    Args:
      X_bj2: Sigma points for state (x, logk), shape [B, J, 2].
      theta: Production elasticity, scalar tensor.
      phi: Adjustment cost parameter, scalar tensor.
      policy: Amortized policy network mapping state/params to investment.
      ctx: Dictionary of model constants built by build_hmc_context().

    Returns:
      Next-period sigma points for state, shape [B, J, 2].

    Notes:
      Only the deterministic part of the transition is returned. The UKF
      adds process noise Q separately.
    """
    x = X_bj2[..., 0]       # ln z_t
    logk = X_bj2[..., 1]    # ln k_t

    x_next = ctx["mu_ln_z"] + ctx["rho"] * x

    # Flatten sigma points to a single batch before feeding the policy network.
    lnk_flat = tf.reshape(logk, [-1])
    lnz_flat = tf.reshape(x, [-1])

    theta_vec = tf.ones_like(lnk_flat) * theta
    logphi = tf.math.log(phi)
    logphi_vec = tf.ones_like(lnk_flat) * logphi

    inp = tf.stack([lnk_flat, lnz_flat, theta_vec, logphi_vec], axis=1)
    iota_flat = policy(inp, training=False)[:, 0]
    iota = tf.reshape(iota_flat, tf.shape(logk))

    g = 1.0 - ctx["delta"] + iota  # Gross investment factor (1 - delta + iota).
    g_pos = g > 0.0

    # For non-positive g, temporarily replace with 1.0 to avoid log issues;
    # the resulting logk_next is then forced to logk_floor below.
    g_safe = tf.where(g_pos, g, tf.ones_like(g))

    logk_det = logk + tf.math.log(g_safe)
    logk_next = tf.maximum(ctx["logk_floor"], logk_det)
    logk_next = tf.where(g_pos, logk_next, ctx["logk_floor"])

    return tf.stack([x_next, logk_next], axis=-1)


def _mvn_logpdf_from_chol(
    obs_b2: tf.Tensor,
    mean_b2: tf.Tensor,
    chol_b22: tf.Tensor,
) -> tf.Tensor:
    """Compute log-density of multivariate normal N(mean, S) per batch.

    Args:
      obs_b2: Observations, shape [B, 2].
      mean_b2: Means, shape [B, 2].
      chol_b22: Cholesky factors of covariance S, shape [B, 2, 2].

    Returns:
      Log-density for each batch element, shape [B].
    """
    v = obs_b2 - mean_b2
    v_col = v[..., None]  # [B, 2, 1]
    # Solve L w = v (where L is lower-triangular Cholesky factor).
    w = tf.linalg.triangular_solve(chol_b22, v_col, lower=True)
    quad = tf.reduce_sum(w**2, axis=[1, 2])
    logdet = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(chol_b22)),
        axis=1,
    )
    return -0.5 * (2.0 * LOG2PI + logdet + quad)


def make_ukf_loglik_fn(
    y_obs_tf: tf.Tensor,
    logk_obs_tf: tf.Tensor,
    policy,
    ctx,
    cfg: HMCConfigPart2,
):
    """Construct UKF-based log-likelihood function for (theta, phi).

    The returned function evaluates the joint log-likelihood of the observed
    panel (y_obs, logk_obs) given structural parameters (theta, phi). It uses:

      * An exact Kalman update for the linear measurement equation:
          o_t = [y_t, logk_obs_t]
          y_t = x_t + theta * logk_t + noise
          logk_obs_t = logk_t + noise

      * A UKF step only for the nonlinear transition, where the policy
        network enters.

    Args:
      y_obs_tf: Observed output log-levels, shape [B, T].
      logk_obs_tf: Observed log capital, shape [B, T].
      policy: Amortized policy network.
      ctx: Dictionary with model constants and noise matrices.
      cfg: HMC and UKF configuration.

    Returns:
      A tf.function f(theta, phi) -> scalar log-likelihood.
    """
    # UKF weights for the transition step.
    Wm, Wc, gamma = _ukf_weights(
        n=2,
        alpha=cfg.ukf_alpha,
        beta=cfg.ukf_beta,
        kappa=cfg.ukf_kappa,
    )

    R = ctx["R"]
    Q = ctx["Q"]
    jitter = float(cfg.ukf_jitter)

    # Initial moments for x_t.
    init_x_mean = (
        ctx["x_mean"]
        if cfg.init_x_mean is None
        else tf.constant(float(cfg.init_x_mean), dtype=DTYPE)
    )
    init_x_var = (
        ctx["x_var"]
        if cfg.init_x_var is None
        else tf.constant(float(cfg.init_x_var), dtype=DTYPE)
    )
    init_logk_var = tf.constant(float(cfg.init_logk_var), dtype=DTYPE)

    # Stack observations: obs_tf[b, t, :] = [y_obs, logk_obs].
    obs_tf = tf.stack([y_obs_tf, logk_obs_tf], axis=-1)
    I2 = tf.eye(2, dtype=DTYPE)

    @tf.function
    def _loglik(theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        """Compute log-likelihood for given (theta, phi).

        Args:
          theta: Production elasticity parameter.
          phi: Adjustment cost parameter.

        Returns:
          Scalar log-likelihood (sum over firms and time).
        """
        obs = obs_tf
        B = tf.shape(obs)[0]  # Number of firms.
        T = tf.shape(obs)[1]  # Number of periods.

        # Prior at t = 0.
        m0_x = tf.fill([B], init_x_mean)
        m0_logk = logk_obs_tf[:, 0]
        m = tf.stack([m0_x, m0_logk], axis=1)  # [B, 2]

        P0 = tf.linalg.diag(tf.stack([init_x_var, init_logk_var], axis=0))
        P = tf.broadcast_to(P0[None, :, :], [B, 2, 2])  # [B, 2, 2]

        ll = tf.constant(0.0, dtype=DTYPE)
        jitterI = tf.constant(jitter, dtype=DTYPE) * I2[None, :, :]

        # Extract diagonal elements of R for convenience.
        R11 = R[0, 0]  # var(y)
        R22 = R[1, 1]  # var(logk_obs)

        for t in tf.range(T):
            obs_t = obs[:, t, :]  # [B, 2]

            # Exact measurement prediction/update (Kalman).
            # H(theta) = [[1, theta],
            #             [0, 1]]
            y_hat = tf.stack(
                [m[:, 0] + theta * m[:, 1], m[:, 1]],
                axis=1,
            )  # [B, 2]
            v = obs_t - y_hat  # Innovation, [B, 2].

            # Pull symmetric elements of P for efficiency.
            p11 = P[:, 0, 0]
            p12 = P[:, 0, 1]
            p22 = P[:, 1, 1]

            # S = H P H' + R, expanded analytically.
            S11 = p11 + 2.0 * theta * p12 + (theta * theta) * p22 + R11
            S12 = p12 + theta * p22
            S22 = p22 + R22

            S = tf.stack(
                [
                    tf.stack([S11, S12], axis=1),
                    tf.stack([S12, S22], axis=1),
                ],
                axis=1,
            )  # [B, 2, 2]

            # Ensure S is symmetric and PD.
            S = 0.5 * (S + tf.transpose(S, [0, 2, 1])) + jitterI
            cholS = tf.linalg.cholesky(S)

            # Log-likelihood contribution for period t.
            ll_t = _mvn_logpdf_from_chol(obs_t, y_hat, cholS)  # [B]
            ll += tf.reduce_sum(ll_t)

            # PHT = P H' expanded analytically.
            PHT = tf.stack(
                [
                    tf.stack([p11 + theta * p12, p12], axis=1),
                    tf.stack([p12 + theta * p22, p22], axis=1),
                ],
                axis=1,
            )  # [B, 2, 2]

            # Kalman gain K = PHT S^{-1} via Cholesky solve.
            K = tf.transpose(
                tf.linalg.cholesky_solve(
                    cholS,
                    tf.transpose(PHT, [0, 2, 1]),
                ),
                [0, 2, 1],
            )  # [B, 2, 2]

            # Update mean and covariance.
            m = m + tf.einsum("bij,bj->bi", K, v)

            KS = tf.linalg.matmul(K, S)
            P = P - tf.linalg.matmul(KS, K, transpose_b=True)
            P = 0.5 * (P + tf.transpose(P, [0, 2, 1])) + jitterI

            # Predict to next period using UKF transition step.
            if t < T - 1:
                Xp = _sigma_points(m, P, gamma, jitter=jitter)  # [B, 5, 2]
                Xn = _transition_fn(
                    Xp,
                    theta,
                    phi,
                    policy,
                    ctx,
                )  # [B, 5, 2]
                m = _weighted_mean(Xn, Wm)
                dXn = Xn - m[:, None, :]
                P = _weighted_cov(dXn, Wc) + Q
                P = 0.5 * (P + tf.transpose(P, [0, 2, 1])) + jitterI

        return ll

    return _loglik


def _u_to_params(u_vec2: tf.Tensor, bounds: ParamBoundsPart2):
    """Map unconstrained parameters u to (theta, phi) within bounds.

    A sigmoid transform enforces the bounds, and the log prior is the
    Jacobian term corresponding to a uniform prior on (theta, phi).

    Args:
      u_vec2: Unconstrained parameters [u_theta, u_phi].
      bounds: Parameter bounds object with theta_min/max and phi_min/max.

    Returns:
      Tuple (theta, phi, log_prior) where log_prior is the log-density
      in u-space implied by a uniform prior in (theta, phi)-space.
    """
    u_theta = u_vec2[0]
    u_phi = u_vec2[1]

    s_theta = tf.math.sigmoid(u_theta)
    s_phi = tf.math.sigmoid(u_phi)

    theta = tf.constant(bounds.theta_min, dtype=DTYPE) + tf.constant(
        bounds.theta_max - bounds.theta_min,
        dtype=DTYPE,
    ) * s_theta
    phi = tf.constant(bounds.phi_min, dtype=DTYPE) + tf.constant(
        bounds.phi_max - bounds.phi_min,
        dtype=DTYPE,
    ) * s_phi

    # Uniform prior on (theta, phi) within bounds -> in u-space the
    # density includes the sigmoid Jacobian terms.
    log_prior = log_sigmoid_prime_tf(u_theta) + log_sigmoid_prime_tf(u_phi)

    return theta, phi, log_prior


def _init_u_from_guess(
    theta_guess: float,
    phi_guess: float,
    bounds: ParamBoundsPart2,
) -> np.ndarray:
    """Initialize unconstrained parameters u from guesses for (theta, phi).

    Args:
      theta_guess: Initial guess for theta within bounds.
      phi_guess: Initial guess for phi within bounds.
      bounds: Parameter bounds object with theta_min/max and phi_min/max.

    Returns:
      NumPy array [u_theta, u_phi] in unconstrained space (float32).
    """
    theta01 = (
        float(theta_guess) - bounds.theta_min
    ) / (bounds.theta_max - bounds.theta_min)
    phi01 = (
        float(phi_guess) - bounds.phi_min
    ) / (bounds.phi_max - bounds.phi_min)
    u_theta = float(logit_np(theta01))
    u_phi = float(logit_np(phi01))
    return np.array([u_theta, u_phi], dtype=np.float32)


def run_hmc_single_rep(
    y_obs: np.ndarray,
    logk_obs: np.ndarray,
    policy,
    ctx,
    bounds: ParamBoundsPart2,
    cfg: HMCConfigPart2,
    seed: int,
):
    """Run HMC for one replication and return posterior summaries.

    This function:
      * Builds a UKF-based log-likelihood for the given replication.
      * Defines an HMC target in unconstrained space u = (u_theta, u_phi).
      * Runs an HMC chain with dual averaging step-size adaptation.
      * Transforms samples back to (theta, phi) and summarizes posteriors.

    Args:
      y_obs: Observed output logs, shape [n_firms, T].
      logk_obs: Observed log capital, shape [n_firms, T].
      policy: Amortized policy network.
      ctx: Dictionary of model constants produced by build_hmc_context().
      bounds: Parameter bounds for (theta, phi).
      cfg: HMC configuration (step size, burn-in, etc.).
      seed: Random seed for the HMC chain.

    Returns:
      Dictionary with posterior means, credible intervals, acceptance
      rate, final step size, and effective sample sizes.
    """
    y_obs_tf = tf.constant(y_obs, dtype=DTYPE)
    logk_obs_tf = tf.constant(logk_obs, dtype=DTYPE)

    loglik_fn = make_ukf_loglik_fn(y_obs_tf, logk_obs_tf, policy, ctx, cfg)

    @tf.function
    def target_log_prob(u_vec2: tf.Tensor) -> tf.Tensor:
        """Compute log posterior in unconstrained space.

        Args:
          u_vec2: Unconstrained parameters [u_theta, u_phi].

        Returns:
          Scalar log posterior evaluated at u_vec2.
        """
        theta, phi, log_prior = _u_to_params(u_vec2, bounds)
        ll = loglik_fn(theta, phi)
        return ll + log_prior

    init_u = _init_u_from_guess(
        cfg.theta_init_guess,
        cfg.phi_init_guess,
        bounds,
    )
    current_state = tf.constant(init_u, dtype=DTYPE)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(float(cfg.step_size), dtype=DTYPE),
        num_leapfrog_steps=int(cfg.num_leapfrog_steps),
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(cfg.num_adaptation_steps),
        target_accept_prob=float(cfg.target_accept),
    )

    @tf.function
    def _run_chain():
        """Run the HMC chain with burn-in and adaptation."""
        return tfp.mcmc.sample_chain(
            num_results=int(cfg.num_results),
            num_burnin_steps=int(cfg.num_burnin),
            current_state=current_state,
            kernel=kernel,
            trace_fn=lambda _, kr: (
                kr.inner_results.is_accepted,
                kr.new_step_size,
            ),
            seed=int(seed),
        )

    samples_u, (is_acc, step_sizes) = _run_chain()

    u_np = samples_u.numpy().astype(np.float64)
    acc_rate = float(np.mean(is_acc.numpy().astype(np.float64)))
    step_size_final = float(step_sizes.numpy()[-1])

    # Transform samples to (theta, phi).
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    theta_s = bounds.theta_min + (bounds.theta_max - bounds.theta_min) * sig(
        u_np[:, 0]
    )
    phi_s = bounds.phi_min + (bounds.phi_max - bounds.phi_min) * sig(u_np[:, 1])

    theta_hat = float(theta_s.mean())
    phi_hat = float(phi_s.mean())

    alpha = (1.0 - float(cfg.cred_level)) / 2.0
    theta_lo, theta_hi = float(np.quantile(theta_s, alpha)), float(
        np.quantile(theta_s, 1.0 - alpha)
    )
    phi_lo, phi_hi = float(np.quantile(phi_s, alpha)), float(
        np.quantile(phi_s, 1.0 - alpha)
    )

    # Effective sample size in u-space (can fail for older TFP versions).
    try:
        ess = (
            tfp.mcmc.effective_sample_size(samples_u)
            .numpy()
            .astype(np.float64)
        )
        ess_u_theta, ess_u_phi = float(ess[0]), float(ess[1])
    except Exception:
        ess_u_theta, ess_u_phi = float("nan"), float("nan")

    return dict(
        theta_hat=theta_hat,
        phi_hat=phi_hat,
        theta_lo=theta_lo,
        theta_hi=theta_hi,
        phi_lo=phi_lo,
        phi_hi=phi_hi,
        acc_rate=acc_rate,
        step_size_final=step_size_final,
        ess_u_theta=ess_u_theta,
        ess_u_phi=ess_u_phi,
    )


def run_hmc_estimation(
    paths: PathsPart2 = None,
    cols: PanelColumnsPart2 = None,
    hmc_cols: HMCColumnsPart2 = None,
    bounds: ParamBoundsPart2 = None,
    cfg: HMCConfigPart2 = None,
    mp: BasicModelParams = None,
):
    """Run UKF+HMC estimator over selected replications.

    Observations are constructed from the synthetic truth in paths.data_csv by
    adding measurement noise using cfg.sigma_y and cfg.sigma_logk_obs. Only
    (theta, phi) are estimated.

    Args:
      paths: PathsPart2 configuration (data and model paths).
      cols: PanelColumnsPart2 specifying column names in the data.
      hmc_cols: Unused placeholder for HMC-related column names.
      bounds: Parameter bounds for (theta, phi).
      cfg: HMC configuration (replications, noise scales, MCMC settings).
      mp: BasicModelParams with structural primitives.

    Returns:
      Dictionary with:
        res_df: DataFrame of posterior summaries by replication.
        theta_true: True theta from the data (or mp.theta if absent).
        phi_true: True phi from the data (or mp.phi if absent).
        policy: Loaded amortized policy network.
    """
    paths = paths or PathsPart2()
    cols = cols or PanelColumnsPart2()
    hmc_cols = hmc_cols or HMCColumnsPart2()
    bounds = bounds or ParamBoundsPart2()
    cfg = cfg or HMCConfigPart2()
    mp = mp or BasicModelParams()

    set_global_seed(cfg.seed)

    policy = load_amortized_policy(paths.policy_path)
    ctx = build_hmc_context(mp, cfg)

    df_all = pd.read_csv(paths.data_csv)

    # True parameters from synthetic data (if present).
    theta_true = (
        float(df_all["theta_true"].iloc[0])
        if "theta_true" in df_all.columns
        else float(mp.theta)
    )
    phi_true = (
        float(df_all["phi_true"].iloc[0])
        if "phi_true" in df_all.columns
        else float(mp.phi)
    )

    rep_list = sorted(df_all[cols.rep].unique().tolist())
    if cfg.rep_ids:
        rep_ids = list(cfg.rep_ids)
    else:
        rep_ids = rep_list[: int(cfg.n_reps_eval)]

    rows = []
    print("\n=== Running UKF+HMC estimation (theta, phi only) ===")
    print(f"Replications: {rep_ids}")
    print(
        f"Obs noise: sigma_y={cfg.sigma_y}, "
        f"sigma_logk_obs={cfg.sigma_logk_obs}"
    )
    print(f"Trans noise: sigma_logk_trans={cfg.sigma_logk_trans}")
    print(
        "HMC: burnin={b}, results={r}, L={L}, step={s}".format(
            b=cfg.num_burnin,
            r=cfg.num_results,
            L=cfg.num_leapfrog_steps,
            s=cfg.step_size,
        )
    )

    for rep_id in rep_ids:
        df_rep = df_all[df_all[cols.rep] == rep_id].copy()

        # Allow per-rep truth (usually constant across reps).
        theta_true_rep = (
            float(df_rep["theta_true"].iloc[0])
            if "theta_true" in df_rep.columns
            else theta_true
        )
        phi_true_rep = (
            float(df_rep["phi_true"].iloc[0])
            if "phi_true" in df_rep.columns
            else phi_true
        )

        y_obs, logk_obs = _make_observations_from_truth(
            df_rep=df_rep,
            cols=cols,
            theta_true=theta_true_rep,
            cfg=cfg,
            seed=int(cfg.obs_seed_base + rep_id),
        )

        out = run_hmc_single_rep(
            y_obs=y_obs,
            logk_obs=logk_obs,
            policy=policy,
            ctx=ctx,
            bounds=bounds,
            cfg=cfg,
            seed=int(cfg.seed + rep_id),
        )

        rows.append(
            dict(
                rep=int(rep_id),
                theta_hat=out["theta_hat"],
                phi_hat=out["phi_hat"],
                theta_lo=out["theta_lo"],
                theta_hi=out["theta_hi"],
                phi_lo=out["phi_lo"],
                phi_hi=out["phi_hi"],
                acc_rate=out["acc_rate"],
                step_size_final=out["step_size_final"],
                ess_u_theta=out["ess_u_theta"],
                ess_u_phi=out["ess_u_phi"],
                theta_true=theta_true_rep,
                phi_true=phi_true_rep,
            )
        )
        print(
            "rep={rep} | theta={th:.8f} [{tlo:.8f},{thi:.8f}] "
            "| phi={ph:.8f} [{plo:.8f},{phi:.8f}] "
            "| acc={acc:.3f} | step={step:.4g}".format(
                rep=rep_id,
                th=out["theta_hat"],
                tlo=out["theta_lo"],
                thi=out["theta_hi"],
                ph=out["phi_hat"],
                plo=out["phi_lo"],
                phi=out["phi_hi"],
                acc=out["acc_rate"],
                step=out["step_size_final"],
            )
        )

    res_df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)

    return dict(
        res_df=res_df,
        theta_true=theta_true,
        phi_true=phi_true,
        policy=policy,
    )