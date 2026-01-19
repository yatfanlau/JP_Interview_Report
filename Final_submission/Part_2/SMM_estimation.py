"""Simulation-based method of moments (SMM) estimation utilities for part 2."""

import math
import numpy as np
import pandas as pd

from config_part2 import (
    BasicModelParams,
    BasicTrainingParams,
    PathsPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    SMMConfigPart2,
)
from utils_part2 import (
    DTYPE,
    safe_corr_np,
    ols_slopes_2reg_with_intercept_np,
    symmetrize_np,
    pinv_psd_np,
    chi2_sf,
)
import tensorflow as tf

from amortized_policy import ParamPolicyNet  # noqa: F401

from gmm_smm_core import (
    build_basic_model_context,
    policy_iota_tf,
    steady_state_k_tf,
    load_policy_model,
    load_synthetic_panels,
    select_rep_ids,
    iota_bounds,
)

PATHS = PathsPart2()
COLS = PanelColumnsPart2()
BOUNDS = ParamBoundsPart2()
SMM_CFG = SMMConfigPart2()
MP = BasicModelParams()
MODEL = build_basic_model_context(MP, dtype=DTYPE)

COL_REP = COLS.rep
COL_FIRM = COLS.firm
COL_TIME = COLS.time
COL_K = COLS.k
COL_Z = COLS.z
COL_IOTA = COLS.iota

THETA_MIN_F, THETA_MAX_F = float(BOUNDS.theta_min), float(BOUNDS.theta_max)
PHI_MIN_F, PHI_MAX_F = float(BOUNDS.phi_min), float(BOUNDS.phi_max)
LOG_PHI_MIN_F, LOG_PHI_MAX_F = math.log(PHI_MIN_F), math.log(PHI_MAX_F)

THETA_MIN_TF = tf.constant(THETA_MIN_F, dtype=DTYPE)
THETA_MAX_TF = tf.constant(THETA_MAX_F, dtype=DTYPE)
LOG_PHI_MIN_TF = tf.constant(LOG_PHI_MIN_F, dtype=DTYPE)
LOG_PHI_MAX_TF = tf.constant(LOG_PHI_MAX_F, dtype=DTYPE)

THETA_INIT_GUESS = float(SMM_CFG.theta_init_guess)
PHI_INIT_GUESS = float(SMM_CFG.phi_init_guess)

N_FIRMS_SIM = int(SMM_CFG.n_firms_sim)
T_DATA = int(SMM_CFG.t_data)
T_BURNIN = int(SMM_CFG.t_burnin)
T_SIM = int(T_DATA)
T_TOTAL = int(T_BURNIN + T_SIM)

ONE = MODEL["one_tf"]
DELTA_F = MODEL["delta_f"]
RHO_F = MODEL["rho_f"]
SIGMA_EPS_F = MODEL["sigma_eps_f"]
R_F = MODEL["r_f"]
MU_LN_Z_F = MODEL["mu_ln_z_f"]
SIGMA_LN_Z_F = MODEL["sigma_ln_z_f"]
M_LN_Z_F = MODEL["m_ln_z_f"]

DELTA_TF = MODEL["delta_tf"]
RHO_TF = MODEL["rho_tf"]
R_TF = MODEL["r_tf"]
MU_LN_Z_TF = MODEL["mu_ln_z_tf"]
K_FLOOR_TF = MODEL["k_floor_tf"]

IOTA_MIN, IOTA_MAX = iota_bounds(MP)

# Global context for sharing estimation results across helper functions.
SMM_CTX = {}

TARGET_MOMENT_NAMES = [
    "m1_mean_logk",
    "m2_corr_iota_lnz",
    "m3_ar1_iota",
    "m4_mean_sq_iota_minus_delta",
    "m5_var_dlogk",
    "m6_corr_iota_lead_lnz",
    "m7_mean_iota",
    "m8_std_iota",
    "m9_var_diota",
    "m10_corr_iota_innov",
    "m11_corr_iota_lead_innov",
    "m12_ols_b_iota_lag",
    "m13_ols_b_innov",
]
N_TARGET_MOMENTS = len(TARGET_MOMENT_NAMES)


def compute_moments_from_df(df_rep):
    """Compute target moments from one replication DataFrame.

    Args:

      df_rep: Pandas DataFrame for a single replication with firm-time panels.

    Returns:

      np.ndarray of shape (N_TARGET_MOMENTS,) with sample moments.
    """
    df_sorted = df_rep.sort_values([COL_FIRM, COL_TIME])

    k = df_sorted[COL_K].to_numpy(dtype=np.float64)
    z = df_sorted[COL_Z].to_numpy(dtype=np.float64)
    iota = df_sorted[COL_IOTA].to_numpy(dtype=np.float64)

    logk = np.log(k + 1e-32)
    lnz = np.log(z + 1e-32)

    m1 = float(np.mean(logk))
    m2 = safe_corr_np(iota, lnz)

    iota_lag_s = df_sorted.groupby(COL_FIRM)[COL_IOTA].shift(1)
    mask_ar1 = iota_lag_s.notna().to_numpy()
    iota_t_ar1 = iota[mask_ar1]  # Current-period iota where lag is available.
    iota_lag_ar1 = iota_lag_s.to_numpy(dtype=np.float64)[mask_ar1]
    m3 = safe_corr_np(iota_t_ar1, iota_lag_ar1)

    delta = float(DELTA_F)
    m4 = float(np.mean((iota - delta) ** 2))

    k_lag_s = df_sorted.groupby(COL_FIRM)[COL_K].shift(1)
    k_lag = k_lag_s.to_numpy(dtype=np.float64)
    mask_dk = ~np.isnan(k_lag)
    dlogk = logk[mask_dk] - np.log(k_lag[mask_dk] + 1e-32)
    m5 = float(dlogk.var(ddof=0)) if dlogk.size > 1 else 0.0

    iota_lead_s = df_sorted.groupby(COL_FIRM)[COL_IOTA].shift(-1)
    mask_lead = iota_lead_s.notna().to_numpy()
    iota_lead = iota_lead_s.to_numpy(dtype=np.float64)[mask_lead]
    lnz_t = lnz[mask_lead]
    m6 = safe_corr_np(iota_lead, lnz_t)

    m7 = float(iota.mean())
    m8 = float(iota.std(ddof=0))

    diota = iota_t_ar1 - iota_lag_ar1
    m9 = float(diota.var(ddof=0)) if diota.size > 1 else 0.0

    lnz_lag_s = df_sorted.groupby(COL_FIRM)[COL_Z].shift(1)
    lnz_lag = np.log(lnz_lag_s.to_numpy(dtype=np.float64) + 1e-32)
    innov = lnz - (MU_LN_Z_F + RHO_F * lnz_lag)  # AR(1) innovation in log z.
    mask_innov = ~np.isnan(innov)

    m10 = safe_corr_np(iota[mask_innov], innov[mask_innov])

    mask_lead_innov = mask_lead & mask_innov
    m11 = safe_corr_np(
        iota_lead_s.to_numpy(dtype=np.float64)[mask_lead_innov],
        innov[mask_lead_innov],
    )

    mask_reg = mask_ar1 & mask_innov
    y_reg = iota[mask_reg]
    x1_reg = iota_lag_s.to_numpy(dtype=np.float64)[mask_reg]
    x2_reg = innov[mask_reg]
    m12, m13 = ols_slopes_2reg_with_intercept_np(y_reg, x1_reg, x2_reg)

    out = np.array(
        [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13],
        dtype=np.float64,
    )
    assert out.size == N_TARGET_MOMENTS
    return out


def compute_moments_tf(k_hist, z_hist, iota_hist):
    """Compute target moments from simulated histories in TensorFlow.

    Args:

      k_hist: Tensor of capital histories with shape (N, T).
      z_hist: Tensor of productivity histories with shape (N, T).
      iota_hist: Tensor of investment-to-capital ratios with shape (N, T).

    Returns:

      tf.Tensor of shape (N_TARGET_MOMENTS,) with simulated moments.
    """
    eps = tf.constant(1e-12, dtype=DTYPE)

    def corr_tf(x, y):
        """Compute correlation between two tensors with numerical stabilizer."""
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        mx, my = tf.reduce_mean(x), tf.reduce_mean(y)
        cx, cy = x - mx, y - my
        cov = tf.reduce_mean(cx * cy)
        sx = tf.sqrt(tf.reduce_mean(cx * cx) + eps)
        sy = tf.sqrt(tf.reduce_mean(cy * cy) + eps)
        return cov / (sx * sy)

    k_flat = tf.reshape(k_hist, [-1])
    iota_flat = tf.reshape(iota_hist, [-1])

    logk_flat = tf.math.log(k_flat + 1e-32)
    lnz_hist = tf.math.log(z_hist + 1e-32)
    lnz_flat = tf.reshape(lnz_hist, [-1])

    m1 = tf.reduce_mean(logk_flat)
    m2 = corr_tf(iota_flat, lnz_flat)

    iota_t = iota_hist[:, 1:]
    iota_lag = iota_hist[:, :-1]
    m3 = corr_tf(iota_t, iota_lag)

    m4 = tf.reduce_mean((iota_flat - DELTA_TF) ** 2)

    logk_hist = tf.math.log(k_hist + 1e-32)
    dlogk = logk_hist[:, 1:] - logk_hist[:, :-1]
    dlogk_flat = tf.reshape(dlogk, [-1])
    mdlogk = tf.reduce_mean(dlogk_flat)
    m5 = tf.reduce_mean((dlogk_flat - mdlogk) ** 2)

    iota_lead = iota_hist[:, 1:]
    lnz_t = lnz_hist[:, :-1]
    m6 = corr_tf(iota_lead, lnz_t)

    m7 = tf.reduce_mean(iota_flat)
    mi = tf.reduce_mean(iota_flat)
    var_i = tf.reduce_mean((iota_flat - mi) ** 2)
    m8 = tf.sqrt(var_i + eps)

    diota = iota_hist[:, 1:] - iota_hist[:, :-1]
    diota_flat = tf.reshape(diota, [-1])
    mdi = tf.reduce_mean(diota_flat)
    m9 = tf.reduce_mean((diota_flat - mdi) ** 2)

    innov = lnz_hist[:, 1:] - (MU_LN_Z_TF + RHO_TF * lnz_hist[:, :-1])
    m10 = corr_tf(iota_hist[:, 1:], innov)
    m11 = corr_tf(iota_hist[:, 2:], innov[:, :-1])

    # OLS slopes from regression y on x1 and x2 with intercept using moments.
    y = tf.reshape(iota_hist[:, 1:], [-1])
    x1 = tf.reshape(iota_hist[:, :-1], [-1])
    x2 = tf.reshape(innov, [-1])
    my, mx1, mx2 = tf.reduce_mean(y), tf.reduce_mean(x1), tf.reduce_mean(x2)
    dy, dx1, dx2 = y - my, x1 - mx1, x2 - mx2
    S11 = tf.reduce_mean(dx1 * dx1)
    S22 = tf.reduce_mean(dx2 * dx2)
    S12 = tf.reduce_mean(dx1 * dx2)
    C1y = tf.reduce_mean(dx1 * dy)
    C2y = tf.reduce_mean(dx2 * dy)
    den = S11 * S22 - S12 * S12  # Determinant of 2x2 covariance matrix.
    den_ok = tf.abs(den) > eps
    b1 = tf.where(den_ok, (S22 * C1y - S12 * C2y) / den, tf.zeros([], dtype=DTYPE))
    b2 = tf.where(den_ok, (-S12 * C1y + S11 * C2y) / den, tf.zeros([], dtype=DTYPE))

    out = tf.stack(
        [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, b1, b2],
        axis=0,
    )
    return out


@tf.function
def simulate_panel_moments_tf(policy, theta, log_phi, eps_all_tf, lnz_init_tf):
    """Simulate panel data and compute moments for one parameter vector.

    Args:

      policy: Trained policy network mapping states and parameters to iota.
      theta: Scalar TensorFlow variable for structural parameter theta.
      log_phi: Scalar TensorFlow variable for log(phi).
      eps_all_tf: Tensor of shocks with shape (N, T_TOTAL).
      lnz_init_tf: Tensor of initial log z with shape (N,).

    Returns:

      tf.Tensor of shape (N_TARGET_MOMENTS,) with simulated moments.
    """
    phi = tf.exp(log_phi)
    N = tf.shape(eps_all_tf)[0]

    k_ss = steady_state_k_tf(theta, DELTA_TF, R_TF)
    k = tf.fill([N], tf.squeeze(k_ss))
    lnz = lnz_init_tf

    # TensorArrays store time series across simulation horizon.
    k_hist_ta = tf.TensorArray(DTYPE, size=T_SIM)
    z_hist_ta = tf.TensorArray(DTYPE, size=T_SIM)
    iota_hist_ta = tf.TensorArray(DTYPE, size=T_SIM)

    for t in tf.range(T_TOTAL):
        eps_t = eps_all_tf[:, t]
        lnz = MU_LN_Z_TF + RHO_TF * lnz + eps_t
        z = tf.exp(lnz)
        iota_t = policy_iota_tf(policy, k, z, theta, phi)
        k_next = tf.maximum(K_FLOOR_TF, (ONE - DELTA_TF + iota_t) * k)

        # Only store observations after burn-in.
        if t >= T_BURNIN:
            idx = t - T_BURNIN
            k_hist_ta = k_hist_ta.write(idx, k)
            z_hist_ta = z_hist_ta.write(idx, z)
            iota_hist_ta = iota_hist_ta.write(idx, iota_t)

        k = k_next

    # Stack over time and transpose so shape is (N, T_SIM).
    k_hist = tf.transpose(k_hist_ta.stack(), [1, 0])
    z_hist = tf.transpose(z_hist_ta.stack(), [1, 0])
    iota_hist = tf.transpose(iota_hist_ta.stack(), [1, 0])
    return compute_moments_tf(k_hist, z_hist, iota_hist)


def make_crn_draws(rep_id, base_seed=1234, n_sims=1):
    """Generate common random number draws for a replication.

    Args:

      rep_id: Replication identifier used to shift the base seed.
      base_seed: Base seed for RNG; rep_id is added to this.
      n_sims: Number of independent Monte Carlo simulations.

    Returns:

      Tuple (eps_all_tf, lnz0_tf) of TensorFlow tensors with shocks and initial ln z.
    """
    rng = np.random.default_rng(int(base_seed + int(rep_id)))
    n_sims = int(n_sims)

    if n_sims == 1:
        # Shape (N_FIRMS_SIM, T_TOTAL) for single simulation.
        eps_all = rng.normal(
            0.0,
            SIGMA_EPS_F,
            size=(N_FIRMS_SIM, T_TOTAL),
        ).astype(np.float32)
        lnz0 = rng.normal(
            M_LN_Z_F,
            SIGMA_LN_Z_F,
            size=(N_FIRMS_SIM,),
        ).astype(np.float32)
    else:
        # Shape (n_sims, N_FIRMS_SIM, T_TOTAL) when averaging over multiple sims.
        eps_all = rng.normal(
            0.0,
            SIGMA_EPS_F,
            size=(n_sims, N_FIRMS_SIM, T_TOTAL),
        ).astype(np.float32)
        lnz0 = rng.normal(
            M_LN_Z_F,
            SIGMA_LN_Z_F,
            size=(n_sims, N_FIRMS_SIM),
        ).astype(np.float32)

    return tf.constant(eps_all, dtype=DTYPE), tf.constant(lnz0, dtype=DTYPE)


def bootstrap_moment_cov_by_firm(df_rep, n_boot=50, seed=0):
    """Estimate covariance of moments by firm-level bootstrap.

    Args:

      df_rep: DataFrame for a single replication.
      n_boot: Number of bootstrap resamples.
      seed: Random seed for the bootstrap.

    Returns:

      np.ndarray covariance matrix of shape (q, q) for the moments.
    """
    rng = np.random.default_rng(int(seed))
    firm_ids = df_rep[COL_FIRM].unique()
    n_firms = len(firm_ids)
    q = N_TARGET_MOMENTS
    moms = np.empty((n_boot, q), dtype=np.float64)
    # Cache firm-specific chunks for fast resampling.
    firm_chunks = {fid: df_rep[df_rep[COL_FIRM] == fid].copy() for fid in firm_ids}

    for b in range(n_boot):
        draw = rng.choice(firm_ids, size=n_firms, replace=True)
        chunks = []
        for j, fid in enumerate(draw):
            tmp = firm_chunks[fid].copy()
            tmp[COL_FIRM] = j  # Reindex firms to avoid collisions.
            chunks.append(tmp)
        df_b = pd.concat(chunks, ignore_index=True)
        moms[b] = compute_moments_from_df(df_b)

    return np.cov(moms, rowvar=False, ddof=1)


def simulate_moments_mc(policy, theta, phi, n_sims=20, seed=50000):
    """Simulate moments via Monte Carlo averaging over parameter values.

    Args:

      policy: Policy network used in simulation.
      theta: Scalar theta parameter (float).
      phi: Scalar phi parameter (float).
      n_sims: Number of Monte Carlo simulations.
      seed: Random seed for simulations.

    Returns:

      Tuple (mbar, Vsim) of mean moment vector and sample covariance matrix.
    """
    rng = np.random.default_rng(int(seed))
    q = N_TARGET_MOMENTS
    moms = np.empty((n_sims, q), dtype=np.float64)

    theta_tf = tf.constant(float(theta), dtype=DTYPE)
    log_phi_tf = tf.constant(float(np.log(phi)), dtype=DTYPE)

    for s in range(int(n_sims)):
        eps = rng.normal(
            0.0,
            SIGMA_EPS_F,
            size=(N_FIRMS_SIM, T_TOTAL),
        ).astype(np.float32)
        lnz0 = rng.normal(
            M_LN_Z_F,
            SIGMA_LN_Z_F,
            size=(N_FIRMS_SIM,),
        ).astype(np.float32)
        m_tf = simulate_panel_moments_tf(
            policy,
            theta=theta_tf,
            log_phi=log_phi_tf,
            eps_all_tf=tf.constant(eps, dtype=DTYPE),
            lnz_init_tf=tf.constant(lnz0, dtype=DTYPE),
        )
        moms[s] = m_tf.numpy().astype(np.float64)

    mbar = moms.mean(axis=0)
    Vsim = (
        np.cov(moms, rowvar=False, ddof=1)
        if n_sims > 1
        else np.zeros((q, q), dtype=np.float64)
    )
    return mbar, Vsim


def simulate_panel_moments_avg_tf(policy, theta, log_phi, eps_all_tf, lnz_init_tf):
    """Simulate moments, averaging over multiple shock paths if provided.

    Args:

      policy: Policy network used in simulation.
      theta: Scalar TensorFlow variable for theta.
      log_phi: Scalar TensorFlow variable for log(phi).
      eps_all_tf: Shocks tensor with rank 2 or 3.
      lnz_init_tf: Initial ln z tensor with matching leading dimensions.

    Returns:

      tf.Tensor of shape (N_TARGET_MOMENTS,) with averaged moments.
    """
    if eps_all_tf.shape.ndims == 2:
        # Single Monte Carlo draw.
        return simulate_panel_moments_tf(
            policy,
            theta,
            log_phi,
            eps_all_tf,
            lnz_init_tf,
        )

    if eps_all_tf.shape.ndims != 3:
        raise ValueError(
            f"eps_all_tf must be rank-2 or rank-3, got shape {eps_all_tf.shape}"
        )

    S = int(eps_all_tf.shape[0])
    moms = []
    for s in range(S):
        moms.append(
            simulate_panel_moments_tf(
                policy,
                theta,
                log_phi,
                eps_all_tf[s],
                lnz_init_tf[s],
            )
        )
    return tf.reduce_mean(tf.stack(moms, axis=0), axis=0)


def smm_objective_weighted(
    policy,
    theta,
    log_phi,
    eps_all_tf,
    lnz_init_tf,
    data_moments_tf,
    W_tf,
):
    """Compute weighted SMM loss and simulated moments.

    Args:

      policy: Policy network used in simulation.
      theta: Scalar TensorFlow variable for theta.
      log_phi: Scalar TensorFlow variable for log(phi).
      eps_all_tf: Shocks tensor used in simulation.
      lnz_init_tf: Initial ln z tensor.
      data_moments_tf: Tensor of empirical target moments.
      W_tf: Weighting matrix tensor of shape (q, q).

    Returns:

      Tuple (loss, sim_moments) with scalar loss and moment tensor.
    """
    sim_moments = simulate_panel_moments_avg_tf(
        policy,
        theta,
        log_phi,
        eps_all_tf,
        lnz_init_tf,
    )
    diff = sim_moments - data_moments_tf
    # Quadratic form diff' W diff.
    loss = tf.tensordot(diff, tf.linalg.matvec(W_tf, diff), axes=1)
    return loss, sim_moments


def estimate_weight_matrix_for_rep(
    policy,
    df_rep,
    theta_ref,
    phi_ref,
    rep_id,
    n_boot=50,
    n_sims=20,
    sims_per_obj=1,
    include_sim_var=True,
    boot_seed_base=10000,
    sim_seed_base=50000,
    ridge=1e-8,
    rcond=1e-10,
):
    """Estimate optimal SMM weight matrix for one replication.

    Args:

      policy: Policy network used in simulations.
      df_rep: DataFrame with panel data for one replication.
      theta_ref: Reference theta used for simulation-based variance.
      phi_ref: Reference phi used for simulation-based variance.
      rep_id: Replication identifier.
      n_boot: Number of bootstrap draws for data variance.
      n_sims: Number of Monte Carlo simulations for sim variance.
      sims_per_obj: Number of sims per objective evaluation.
      include_sim_var: Whether to include simulation variance in Sigma_g.
      boot_seed_base: Base seed for bootstrap RNG.
      sim_seed_base: Base seed for simulation RNG.
      ridge: Ridge regularization added to covariance diagonal.
      rcond: Relative condition number for pseudo-inverse.

    Returns:

      dict with keys Sigma_data, Sigma_sim, Sigma_g, and W (inverse).
    """
    q = N_TARGET_MOMENTS
    Sigma_data = bootstrap_moment_cov_by_firm(
        df_rep,
        n_boot=n_boot,
        seed=boot_seed_base + int(rep_id),
    )

    if include_sim_var:
        _, Sigma_sim = simulate_moments_mc(
            policy,
            theta_ref,
            phi_ref,
            n_sims=n_sims,
            seed=sim_seed_base + int(rep_id),
        )
        # Divide simulation variance by sims_per_obj per SMM theory.
        Sigma_g = Sigma_data + (Sigma_sim / max(int(sims_per_obj), 1))
    else:
        Sigma_sim = np.zeros((q, q), dtype=np.float64)
        Sigma_g = Sigma_data.copy()

    # Symmetrize and regularize for numerical stability.
    Sigma_g = symmetrize_np(Sigma_g) + float(ridge) * np.eye(q, dtype=np.float64)
    W = pinv_psd_np(Sigma_g, rcond=rcond)
    return dict(Sigma_data=Sigma_data, Sigma_sim=Sigma_sim, Sigma_g=Sigma_g, W=W)


def _run_smm_optim(
    policy,
    data_moments_tf,
    eps_all_tf,
    lnz_init_tf,
    theta_init,
    log_phi_init,
    W_np=None,
    learning_rate=0.01,
    train_steps=300,
    print_every=50,
):
    """Run gradient-based SMM optimization for (theta, phi).

    Args:

      policy: Policy network used for simulations.
      data_moments_tf: Tensor of empirical target moments.
      eps_all_tf: Shocks tensor (2D or 3D) for simulations.
      lnz_init_tf: Initial ln z tensor.
      theta_init: Initial value for theta.
      log_phi_init: Initial value for log(phi).
      W_np: Optional numpy weighting matrix; identity if None.
      learning_rate: Adam optimizer learning rate.
      train_steps: Number of gradient steps.
      print_every: Frequency of progress printing.

    Returns:

      dict with estimates theta_hat, phi_hat, logphi_hat, and final loss.
    """
    q = N_TARGET_MOMENTS
    if W_np is None:
        W_tf = tf.eye(q, dtype=DTYPE)
    else:
        W_tf = tf.constant(W_np.astype(np.float32), dtype=DTYPE)

    theta_var = tf.Variable(float(theta_init), dtype=DTYPE)
    log_phi_var = tf.Variable(float(log_phi_init), dtype=DTYPE)

    # Project initial guesses into parameter bounds.
    theta_var.assign(tf.clip_by_value(theta_var, THETA_MIN_TF, THETA_MAX_TF))
    log_phi_var.assign(
        tf.clip_by_value(log_phi_var, LOG_PHI_MIN_TF, LOG_PHI_MAX_TF)
    )

    opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    loss = tf.constant(np.nan, dtype=DTYPE)

    for step in range(1, int(train_steps) + 1):
        with tf.GradientTape() as tape:
            theta_t = theta_var.read_value()
            log_phi_t = log_phi_var.read_value()
            loss, _ = smm_objective_weighted(
                policy,
                theta_t,
                log_phi_t,
                eps_all_tf,
                lnz_init_tf,
                data_moments_tf,
                W_tf,
            )

        grads = tape.gradient(loss, [theta_var, log_phi_var])
        opt.apply_gradients(zip(grads, [theta_var, log_phi_var]))

        # Enforce bounds after each gradient step.
        theta_var.assign(tf.clip_by_value(theta_var, THETA_MIN_TF, THETA_MAX_TF))
        log_phi_var.assign(
            tf.clip_by_value(log_phi_var, LOG_PHI_MIN_TF, LOG_PHI_MAX_TF)
        )

        if step == 1 or step % int(print_every) == 0 or step == int(train_steps):
            print(
                f"    step {step:4d}: loss={loss.numpy():.4e}, "
                f"theta={theta_var.numpy():.4f}, "
                f"phi={float(np.exp(log_phi_var.numpy())):.4f}"
            )

    theta_hat = float(theta_var.numpy())
    logphi_hat = float(log_phi_var.numpy())
    phi_hat = float(np.exp(logphi_hat))
    final_loss = float(loss.numpy())
    return dict(
        theta_hat=theta_hat,
        phi_hat=phi_hat,
        logphi_hat=logphi_hat,
        loss=final_loss,
    )


def estimate_one_replication_2step(
    policy,
    rep_id,
    df_rep,
    base_seed=1234,
    lr1=0.01,
    steps1=300,
    sims_per_obj1=1,
    w_n_boot=50,
    w_n_sims=20,
    include_sim_var_in_W=True,
    w_ridge=1e-8,
    w_rcond=1e-10,
    lr2=0.005,
    steps2=200,
    sims_per_obj2=1,
    boot_seed_base=10000,
    sim_seed_base=50000,
):
    """Estimate parameters for one replication using 2-step SMM.

    Args:

      policy: Policy network used for simulations.
      rep_id: Replication identifier.
      df_rep: DataFrame with panel data for this replication.
      base_seed: Base seed for CRN draws within this replication.
      lr1: Learning rate for step-1 optimization (W = I).
      steps1: Number of gradient steps in step 1.
      sims_per_obj1: Sims per objective evaluation in step 1.
      w_n_boot: Number of bootstrap draws for weight matrix.
      w_n_sims: Number of sims for simulation variance in weight matrix.
      include_sim_var_in_W: Whether to include simulation variance in W.
      w_ridge: Ridge term for regularizing weight matrix.
      w_rcond: rcond for pseudo-inverse in weight matrix.
      lr2: Learning rate for step-2 optimization (optimal W).
      steps2: Number of gradient steps in step 2.
      sims_per_obj2: Sims per objective evaluation in step 2.
      boot_seed_base: Base seed for bootstrap used in W estimation.
      sim_seed_base: Base seed for Monte Carlo used in W estimation.

    Returns:

      dict with step-1 and step-2 estimates and configuration details.
    """
    m_data_np = compute_moments_from_df(df_rep)
    data_moments_tf = tf.constant(m_data_np, dtype=DTYPE)

    print(f"  Step 1 (W=I), sims_per_obj={sims_per_obj1}:")
    eps1_tf, lnz1_tf = make_crn_draws(
        rep_id,
        base_seed=base_seed,
        n_sims=sims_per_obj1,
    )
    step1 = _run_smm_optim(
        policy,
        data_moments_tf=data_moments_tf,
        eps_all_tf=eps1_tf,
        lnz_init_tf=lnz1_tf,
        theta_init=THETA_INIT_GUESS,
        log_phi_init=np.log(PHI_INIT_GUESS),
        W_np=None,
        learning_rate=lr1,
        train_steps=steps1,
        print_every=50,
    )

    print(
        "  Estimating W at step-1 estimate: "
        f"theta={step1['theta_hat']:.4f}, phi={step1['phi_hat']:.4f}"
    )
    wout = estimate_weight_matrix_for_rep(
        policy=policy,
        df_rep=df_rep,
        theta_ref=step1["theta_hat"],
        phi_ref=step1["phi_hat"],
        rep_id=rep_id,
        n_boot=w_n_boot,
        n_sims=w_n_sims,
        sims_per_obj=sims_per_obj2,
        include_sim_var=include_sim_var_in_W,
        boot_seed_base=boot_seed_base,
        sim_seed_base=sim_seed_base,
        ridge=w_ridge,
        rcond=w_rcond,
    )
    W2 = wout["W"]

    print(f"  Step 2 (W=Sigma_g^-1), sims_per_obj={sims_per_obj2}:")
    eps2_tf, lnz2_tf = make_crn_draws(
        rep_id,
        base_seed=base_seed,
        n_sims=sims_per_obj2,
    )
    step2 = _run_smm_optim(
        policy,
        data_moments_tf=data_moments_tf,
        eps_all_tf=eps2_tf,
        lnz_init_tf=lnz2_tf,
        theta_init=step1["theta_hat"],
        log_phi_init=np.log(step1["phi_hat"]),
        W_np=W2,
        learning_rate=lr2,
        train_steps=steps2,
        print_every=50,
    )

    return dict(
        rep=int(rep_id),
        theta1_hat=step1["theta_hat"],
        phi1_hat=step1["phi_hat"],
        loss1=step1["loss"],
        theta_hat=step2["theta_hat"],
        phi_hat=step2["phi_hat"],
        loss=step2["loss"],
        sims_per_obj1=int(sims_per_obj1),
        sims_per_obj2=int(sims_per_obj2),
        w_n_boot=int(w_n_boot),
        w_n_sims=int(w_n_sims),
        include_sim_var_in_W=bool(include_sim_var_in_W),
    )


def run_smm_estimation(paths=None, cols=None, bounds=None, cfg=None, mp=None):
    """Run SMM estimation pipeline over selected replications.

    Args:

      paths: Optional PathsPart2 instance; defaults to global PATHS.
      cols: Optional PanelColumnsPart2 instance; defaults to global COLS.
      bounds: Optional ParamBoundsPart2 instance; defaults to global BOUNDS.
      cfg: Optional SMMConfigPart2 instance; defaults to global SMM_CFG.
      mp: Optional BasicModelParams instance; defaults to global MP.

    Returns:

      dict context with policy, true parameters, replication results, and flags.
    """
    paths = paths or PATHS
    cols = cols or COLS
    bounds = bounds or BOUNDS
    cfg = cfg or SMM_CFG
    mp = mp or MP

    tp = BasicTrainingParams()

    print("Basic (fixed) model parameters:")
    print(mp)
    print(f"Policy output bounds for iota=I/k: [{IOTA_MIN:.4f}, {IOTA_MAX:.4f}]")
    print("Amortization (training) ranges:")
    print(f"  theta in [{bounds.theta_min}, {bounds.theta_max}]")
    print(
        "  phi   in "
        f"[{bounds.phi_min}, {bounds.phi_max}]  (policy conditioned on log(phi))"
    )
    print(
        "Initial on-policy state shapes: "
        f"({tp.n_paths},) ({tp.n_paths},) ({tp.n_paths},) ({tp.n_paths},)"
    )

    print("Loading amortized policy network...")
    policy = load_policy_model(paths.policy_path)
    print("Loaded policy from", paths.policy_path)

    print("Loading synthetic panels from", paths.data_csv)
    df, theta_true, phi_true = load_synthetic_panels(paths.data_csv)
    print(
        "Detected true parameters from CSV: "
        f"theta_true={theta_true}, phi_true={phi_true}"
    )

    rep_ids_eval = select_rep_ids(df, cols.rep, cfg.n_reps_eval)
    rep_ids_all = sorted(df[cols.rep].unique())
    print(
        "Total replications in data: "
        f"{len(rep_ids_all)}; using {len(rep_ids_eval)} for SMM/Metric 1.\n"
    )

    results = []
    for idx, rep_id in enumerate(rep_ids_eval, start=1):
        print(f"=== Replication {rep_id} ({idx}/{len(rep_ids_eval)}) ===")
        df_rep = df[df[cols.rep] == rep_id].copy()
        out = estimate_one_replication_2step(
            policy,
            rep_id=rep_id,
            df_rep=df_rep,
            base_seed=cfg.crn_base_seed,
            lr1=cfg.lr_1,
            steps1=cfg.steps_1,
            sims_per_obj1=cfg.sims_per_obj_1,
            w_n_boot=cfg.w_n_boot,
            w_n_sims=cfg.w_n_sims,
            include_sim_var_in_W=cfg.include_sim_var_in_W,
            w_ridge=cfg.w_ridge,
            w_rcond=cfg.w_rcond,
            lr2=cfg.lr_2,
            steps2=cfg.steps_2,
            sims_per_obj2=cfg.sims_per_obj_2,
            boot_seed_base=cfg.boot_seed_base,
            sim_seed_base=cfg.sim_seed_base,
        )
        results.append(out)
        print(
            "  Final (step-2) estimates: "
            f"theta_hat={out['theta_hat']:.4f}, "
            f"phi_hat={out['phi_hat']:.4f}, loss={out['loss']:.4e}\n"
        )

    res_df = pd.DataFrame(results).sort_values("rep").reset_index(drop=True)
    print("Estimation results (first few rows):")
    print(res_df.head())

    ctx = dict(
        policy=policy,
        theta_true=theta_true,
        phi_true=phi_true,
        rep_ids_eval=rep_ids_eval,
        res_df=res_df,
        sims_per_obj2=int(cfg.sims_per_obj_2),
        include_sim_var_in_W=bool(cfg.include_sim_var_in_W),
    )
    SMM_CTX.clear()
    SMM_CTX.update(ctx)
    return ctx


def jacobian_sim_moments_avg(
    policy,
    theta_hat,
    phi_hat,
    rep_id,
    base_seed=1234,
    sims_per_obj=1,
):
    """Compute Jacobian of simulated moments w.r.t. (theta, log phi).

    Args:

      policy: Policy network used for simulations.
      theta_hat: Point estimate of theta around which to linearize.
      phi_hat: Point estimate of phi around which to linearize.
      rep_id: Replication identifier for common shocks.
      base_seed: Base seed for shock draws.
      sims_per_obj: Number of sims per objective evaluation.

    Returns:

      np.ndarray of shape (q, 2) with derivatives wrt (theta, log phi).
    """
    eps_tf, lnz_tf = make_crn_draws(
        rep_id,
        base_seed=base_seed,
        n_sims=sims_per_obj,
    )
    theta = tf.constant(float(theta_hat), dtype=DTYPE)
    log_phi = tf.constant(float(np.log(phi_hat)), dtype=DTYPE)

    with tf.GradientTape() as tape:
        tape.watch([theta, log_phi])
        sim_m = simulate_panel_moments_avg_tf(
            policy,
            theta,
            log_phi,
            eps_tf,
            lnz_tf,
        )

    dtheta, dlogphi = tape.jacobian(sim_m, [theta, log_phi])
    J = tf.stack([dtheta, dlogphi], axis=1).numpy().astype(np.float64)
    return J


def j_test_one_rep(
    policy,
    df_rep,
    theta_hat,
    phi_hat,
    rep_id,
    alpha=0.05,
    n_boot=50,
    n_sims=30,
    ridge=1e-8,
):
    """Compute J test of overidentifying restrictions for one replication.

    Args:

      policy: Policy network used for simulations.
      df_rep: Data for one replication.
      theta_hat: Estimated theta for this replication.
      phi_hat: Estimated phi for this replication.
      rep_id: Replication identifier.
      alpha: Significance level for the test.
      n_boot: Number of bootstrap draws for data variance.
      n_sims: Number of Monte Carlo sims for simulation variance.
      ridge: Ridge term added to covariance diagonal.

    Returns:

      dict with rep, J statistic, degrees of freedom, p-value, and reject flag.
    """
    q = N_TARGET_MOMENTS
    k_params = 2
    df = q - k_params

    m_data = compute_moments_from_df(df_rep)
    Sigma_data = bootstrap_moment_cov_by_firm(
        df_rep,
        n_boot=n_boot,
        seed=SMM_CFG.boot_seed_base + int(rep_id),
    )
    m_sim_bar, Sigma_sim = simulate_moments_mc(
        policy,
        theta_hat,
        phi_hat,
        n_sims=n_sims,
        seed=SMM_CFG.sim_seed_base + int(rep_id),
    )

    g = (m_sim_bar - m_data).reshape(-1, 1)
    Vg = symmetrize_np(Sigma_data + Sigma_sim / max(int(n_sims), 1)) + float(
        ridge
    ) * np.eye(q)
    W = pinv_psd_np(Vg, rcond=1e-10)
    J = float((g.T @ W @ g).squeeze())
    pval = chi2_sf(J, df)
    reject = bool(pval < alpha)
    return dict(rep=int(rep_id), J=J, df=int(df), pval=pval, reject=reject)


def run_j_test_overid(
    res_df,
    df_all,
    policy=None,
    alpha=0.05,
    n_boot=50,
    n_sims=30,
    ridge=1e-8,
):
    """Run J test overidentification check for all replications.

    Args:

      res_df: DataFrame with parameter estimates by replication.
      df_all: Full panel DataFrame containing all replications.
      policy: Optional policy network; loaded from disk if None.
      alpha: Significance level for J test.
      n_boot: Number of bootstrap draws for data variance.
      n_sims: Number of sims for simulation variance.
      ridge: Ridge term for covariance regularization.

    Returns:

      pd.DataFrame with J test results by replication.
    """
    if policy is None:
        policy = SMM_CTX.get("policy")
        if policy is None:
            policy = load_policy_model(PATHS.policy_path)

    rows = []
    for _, r in res_df.iterrows():
        rep_id = int(r["rep"])
        df_rep = df_all[df_all[COL_REP] == rep_id].copy()
        rows.append(
            j_test_one_rep(
                policy=policy,
                df_rep=df_rep,
                theta_hat=float(r["theta_hat"]),
                phi_hat=float(r["phi_hat"]),
                rep_id=rep_id,
                alpha=alpha,
                n_boot=n_boot,
                n_sims=n_sims,
                ridge=ridge,
            )
        )

    out = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    q = N_TARGET_MOMENTS
    print("=== J test (overidentifying restrictions) ===")
    print(f"moments q={q} | params k=2 | df={q-2} | alpha={alpha:.2f}")
    print(
        "Rejection rate over reps: "
        f"{out['reject'].mean():.3f}   (n_boot={n_boot}, n_sims={n_sims})"
    )
    return out


def diagnostic_step2_loss_vs_chi2(
    res_df,
    df_chi,
    n_mc=100000,
    random_seed=12345,
):
    """Compare distribution of step-2 loss to Chi-square(df_chi).

    Args:

      res_df: DataFrame with step-2 losses by replication.
      df_chi: Degrees of freedom for reference Chi-square.
      n_mc: Number of Monte Carlo draws from Chi-square.
      random_seed: Seed for Monte Carlo generation.

    Returns:

      None. Prints diagnostics and produces a Q–Q plot when possible.
    """
    import matplotlib.pyplot as plt

    losses = res_df["loss"].to_numpy(dtype=float)
    R = len(losses)

    loss_mean = losses.mean()
    loss_std = losses.std(ddof=1) if R > 1 else float("nan")
    loss_q = (
        np.quantile(losses, [0.05, 0.25, 0.50, 0.75, 0.95]) if R else np.zeros(5)
    )

    rng = np.random.default_rng(int(random_seed))
    chi_mc = rng.chisquare(df_chi, size=n_mc)
    chi_mean = chi_mc.mean()
    chi_std = chi_mc.std(ddof=1)
    chi_q = np.quantile(chi_mc, [0.05, 0.25, 0.50, 0.75, 0.95])

    print("=== Diagnostic 1: Step-2 loss vs Chi-square(df = q - k) ===")
    print(f"# of replications: {R}")
    print(f"df (q - k) = {df_chi}")
    print(
        f"Loss mean = {loss_mean:.3f},   Chi2 mean ≈ {chi_mean:.3f}  "
        f"(theory mean = df = {df_chi})"
    )
    print(
        f"Loss std  = {loss_std:.3f},   Chi2 std  ≈ {chi_std:.3f}  "
        f"(theory std = sqrt(2*df) ≈ {np.sqrt(2*df_chi):.3f})"
    )
    print("Quantiles (loss vs Chi-square MC):")
    for p, L, C in zip([0.05, 0.25, 0.50, 0.75, 0.95], loss_q, chi_q):
        print(f"  q{int(100*p):2d}:  loss = {L:7.3f},   chi2 ≈ {C:7.3f}")

    if R >= 2:
        # Generate a Chi-square sample of the same size for Q–Q plot.
        chi_sim = rng.chisquare(df_chi, size=R)
        chi_sorted = np.sort(chi_sim)
        losses_sorted = np.sort(losses)

        plt.figure(figsize=(5, 5))
        plt.scatter(chi_sorted, losses_sorted, s=25, alpha=0.7)
        maxv = max(chi_sorted.max(), losses_sorted.max())
        plt.plot([0, maxv], [0, maxv], "r--", lw=1, label="45-degree line")
        plt.xlabel(f"Chi-square({df_chi}) quantiles (MC)")
        plt.ylabel("Step-2 loss quantiles")
        plt.title("Q–Q plot: Step-2 loss vs Chi-square")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()