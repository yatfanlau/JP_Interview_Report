"""GMM estimation routines for Euler-equation-based structural parameters.

This module implements two-step GMM estimation for a structural investment model,
including construction of instruments, estimation of weighting matrices, and
diagnostics such as the Hansen J test and comparison to the chi-square
distribution.
"""

import math
import numpy as np
import pandas as pd
import tensorflow as tf

from config_part2 import (
    BasicModelParams,
    PathsPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    GMMConfigPart2,
)
from utils_part2 import DTYPE, symmetrize_np, pinv_psd_np, chi2_sf

from amortized_policy import ParamPolicyNet  # noqa: F401

from gmm_smm_core import build_basic_model_context, load_synthetic_panels, select_rep_ids

PATHS = PathsPart2()
COLS = PanelColumnsPart2()
BOUNDS = ParamBoundsPart2()
GMM_CFG = GMMConfigPart2()
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

DELTA_TF = MODEL["delta_tf"]
ONE_MINUS_DELTA_TF = MODEL["one_minus_delta_tf"]
BETA_TF = MODEL["beta_tf"]
K_FLOOR_TF = MODEL["k_floor_tf"]
ONE = MODEL["one_tf"]

INSTRUMENT_NAMES = list(GMM_CFG.instrument_names)

# Global context storing results from the last GMM estimation run.
GMM_CTX = {}


def build_instruments_np(k_t, z_t, iota_t, standardize=True):
    """Build the instrument matrix from current-period states and controls.

    Args:

      k_t:
        Array-like of current-period capital.
      z_t:
        Array-like of current-period productivity.
      iota_t:
        Array-like of current-period investment rate.
      standardize:
        If True, standardize non-constant instruments to zero mean
        and unit variance.

    Returns:

      A float32 NumPy array of shape (T, q) with instrument values.
    """
    k_t = np.asarray(k_t, dtype=np.float64)
    z_t = np.asarray(z_t, dtype=np.float64)
    iota_t = np.asarray(iota_t, dtype=np.float64)

    # Use logs of k and z as instruments; add small epsilon to avoid log(0).
    logk = np.log(k_t + 1e-32)
    lnz = np.log(z_t + 1e-32)

    Z = np.stack(
        [
            np.ones_like(logk),
            logk,
            lnz,
            iota_t,
            logk**2,
            lnz**2,
            iota_t**2,
        ],
        axis=1,
    )

    if standardize:
        # Standardize non-constant columns; keep the intercept as is.
        X = Z[:, 1:]
        m = X.mean(axis=0)
        s = X.std(axis=0, ddof=0)
        # Avoid division by ~0; treat near-constant columns as unstandardized.
        s = np.where(s > 1e-12, s, 1.0)
        Z[:, 1:] = (X - m) / s

    # Replace NaN/inf (from extreme values) by 0.0.
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z.astype(np.float32)


def prepare_gmm_data_from_df(df_rep, standardize_instr=True):
    """Prepare tensor inputs for GMM from a replication-specific DataFrame.

    Args:

      df_rep:
        DataFrame containing a single replication of panel data.
      standardize_instr:
        If True, standardize instruments before converting to tensors.

    Returns:

      A dictionary with TensorFlow tensors and metadata keys:
      - k_t_tf, z_t_tf, iota_t_tf: current-period variables.
      - k_tp1_tf, z_tp1_tf, iota_tp1_tf: next-period variables.
      - Z_tf: instrument matrix.
      - firm_idx_tf: integer index per observation (0..N_firms-1).
      - N_firms: number of unique firms.
      - q: number of instruments (columns in Z_tf).
    """
    df_sorted = df_rep.sort_values([COL_FIRM, COL_TIME]).copy()

    k = df_sorted[COL_K].to_numpy(dtype=np.float64)
    z = df_sorted[COL_Z].to_numpy(dtype=np.float64)
    iota = df_sorted[COL_IOTA].to_numpy(dtype=np.float64)
    firm = df_sorted[COL_FIRM].to_numpy()

    # Shift by -1 to create t+1 variables within each firm.
    k1 = df_sorted.groupby(COL_FIRM)[COL_K].shift(-1).to_numpy(dtype=np.float64)
    z1 = df_sorted.groupby(COL_FIRM)[COL_Z].shift(-1).to_numpy(dtype=np.float64)
    iota1 = df_sorted.groupby(COL_FIRM)[COL_IOTA].shift(-1).to_numpy(dtype=np.float64)

    # Require finite data in all t+1 variables.
    mask = np.isfinite(k1) & np.isfinite(z1) & np.isfinite(iota1)

    k_t = k[mask]
    z_t = z[mask]
    iota_t = iota[mask]
    k_tp1 = k1[mask]
    z_tp1 = z1[mask]
    iota_tp1 = iota1[mask]

    firm_obs = firm[mask]
    # firm_idx is a compact [0..N_firms-1] index per firm.
    uniq_firms, firm_idx = np.unique(firm_obs, return_inverse=True)
    N_firms = int(len(uniq_firms))

    # Sort by firm index to make segment-based operations efficient.
    order = np.argsort(firm_idx, kind="mergesort")
    firm_idx = firm_idx[order]
    k_t = k_t[order]
    z_t = z_t[order]
    iota_t = iota_t[order]
    k_tp1 = k_tp1[order]
    z_tp1 = z_tp1[order]
    iota_tp1 = iota_tp1[order]

    Z_np = build_instruments_np(k_t, z_t, iota_t, standardize=standardize_instr)
    q = int(Z_np.shape[1])

    return dict(
        k_t_tf=tf.constant(k_t.astype(np.float32), dtype=DTYPE),
        z_t_tf=tf.constant(z_t.astype(np.float32), dtype=DTYPE),
        iota_t_tf=tf.constant(iota_t.astype(np.float32), dtype=DTYPE),
        k_tp1_tf=tf.constant(k_tp1.astype(np.float32), dtype=DTYPE),
        z_tp1_tf=tf.constant(z_tp1.astype(np.float32), dtype=DTYPE),
        iota_tp1_tf=tf.constant(iota_tp1.astype(np.float32), dtype=DTYPE),
        Z_tf=tf.constant(Z_np, dtype=DTYPE),
        firm_idx_tf=tf.constant(firm_idx.astype(np.int32)),
        N_firms=N_firms,
        q=q,
    )


@tf.function
def euler_residual_tf(theta, log_phi, k_t, iota_t, k_tp1, z_tp1, iota_tp1):
    """Compute the Euler equation residual for given parameters and data.

    Args:

      theta:
        Output elasticity of capital (scalar tensor).
      log_phi:
        Log of adjustment cost parameter phi (scalar tensor).
      k_t:
        Current-period capital tensor.
      iota_t:
        Current-period investment rate tensor.
      k_tp1:
        Next-period capital tensor.
      z_tp1:
        Next-period productivity tensor.
      iota_tp1:
        Next-period investment rate tensor.

    Returns:

      A 1D tensor of Euler residuals u_t for each observation.
    """
    phi = tf.exp(log_phi)
    # Impose a floor on capital to avoid numerical issues in k^{theta-1}.
    k_tp1 = tf.maximum(k_tp1, K_FLOOR_TF)

    psiI_t = phi * (iota_t - DELTA_TF)
    psiI_tp1 = phi * (iota_tp1 - DELTA_TF)

    psiK_tp1 = 0.5 * phi * (DELTA_TF * DELTA_TF - iota_tp1 * iota_tp1)
    piK_tp1 = theta * z_tp1 * tf.pow(k_tp1, theta - ONE)

    rhs = piK_tp1 - psiK_tp1 + ONE_MINUS_DELTA_TF * (ONE + psiI_tp1)
    u = (ONE + psiI_t) - BETA_TF * rhs
    return u


@tf.function
def gmm_g_and_mbar_tf(theta, log_phi, k_t, iota_t, k_tp1, z_tp1, iota_tp1, Z, firm_idx):
    """Compute GMM moment conditions and firm-level averaged moments.

    Args:

      theta:
        Output elasticity of capital (scalar tensor).
      log_phi:
        Log of adjustment cost parameter phi (scalar tensor).
      k_t:
        Current-period capital tensor.
      iota_t:
        Current-period investment rate tensor.
      k_tp1:
        Next-period capital tensor.
      z_tp1:
        Next-period productivity tensor.
      iota_tp1:
        Next-period investment rate tensor.
      Z:
        Instrument matrix tensor with shape (T, q).
      firm_idx:
        1D integer tensor of firm indices (0..N_firms-1).

    Returns:

      A tuple (g, mbar_i) where:
      - g is a 1D tensor of sample-average moments (length q).
      - mbar_i is a 2D tensor of firm-level average moments (N_firms, q).
    """
    u = euler_residual_tf(theta, log_phi, k_t, iota_t, k_tp1, z_tp1, iota_tp1)
    m = Z * tf.expand_dims(u, axis=1)
    # Average moments by firm using segment_mean over contiguous groups.
    mbar_i = tf.math.segment_mean(m, firm_idx)
    g = tf.reduce_mean(mbar_i, axis=0)
    return g, mbar_i


@tf.function
def gmm_objective_tf(theta, log_phi, W, k_t, iota_t, k_tp1, z_tp1, iota_tp1, Z, firm_idx):
    """Compute the GMM quadratic objective and the moment vector.

    Args:

      theta:
        Output elasticity of capital (scalar tensor).
      log_phi:
        Log of adjustment cost parameter phi (scalar tensor).
      W:
        Weighting matrix tensor of shape (q, q).
      k_t:
        Current-period capital tensor.
      iota_t:
        Current-period investment rate tensor.
      k_tp1:
        Next-period capital tensor.
      z_tp1:
        Next-period productivity tensor.
      iota_tp1:
        Next-period investment rate tensor.
      Z:
        Instrument matrix tensor with shape (T, q).
      firm_idx:
        1D integer tensor of firm indices (0..N_firms-1).

    Returns:

      A tuple (J, g) where:
      - J is the scalar GMM objective value g' W g.
      - g is the 1D tensor of sample-average moments.
    """
    g, _ = gmm_g_and_mbar_tf(theta, log_phi, k_t, iota_t, k_tp1, z_tp1, iota_tp1, Z, firm_idx)
    # Compute quadratic form g' W g.
    return tf.tensordot(g, tf.linalg.matvec(W, g), axes=1), g


def estimate_S_and_W_np(theta_hat, logphi_hat, data, ridge=1e-8, rcond=1e-10):
    """Estimate the moment covariance matrix S and its pseudo-inverse W.

    Args:

      theta_hat:
        Scalar estimate of theta used to evaluate moments.
      logphi_hat:
        Scalar estimate of log(phi) used to evaluate moments.
      data:
        Dictionary returned by prepare_gmm_data_from_df.
      ridge:
        Small positive constant added to the diagonal of S for stabilization.
      rcond:
        Relative condition number passed to pinv_psd_np.

    Returns:

      A tuple (S, W) where:
      - S is the estimated covariance matrix of moments (q, q).
      - W is the pseudo-inverse weighting matrix (q, q).
    """
    g_tf, mbar_tf = gmm_g_and_mbar_tf(
        tf.constant(theta_hat, DTYPE),
        tf.constant(logphi_hat, DTYPE),
        data["k_t_tf"],
        data["iota_t_tf"],
        data["k_tp1_tf"],
        data["z_tp1_tf"],
        data["iota_tp1_tf"],
        data["Z_tf"],
        data["firm_idx_tf"],
    )
    g = g_tf.numpy().astype(np.float64)
    mbar = mbar_tf.numpy().astype(np.float64)
    N = data["N_firms"]

    # Center firm-level moments and form sample covariance.
    dm = mbar - g.reshape(1, -1)
    S = (dm.T @ dm) / max(N, 1)
    # Symmetrize and add ridge to ensure positive semi-definiteness.
    S = symmetrize_np(S) + float(ridge) * np.eye(S.shape[0], dtype=np.float64)
    W = pinv_psd_np(S, rcond=rcond)
    return S, W


def _run_gmm_optim(
    data,
    theta_init,
    log_phi_init,
    W_np=None,
    learning_rate=0.01,
    train_steps=300,
    print_every=50,
):
    """Run gradient-based optimization of the GMM objective for one dataset.

    Args:

      data:
        Dictionary returned by prepare_gmm_data_from_df.
      theta_init:
        Initial value for theta.
      log_phi_init:
        Initial value for log(phi).
      W_np:
        Optional NumPy weighting matrix. If None, uses identity.
      learning_rate:
        Learning rate for the Adam optimizer.
      train_steps:
        Number of gradient steps to run.
      print_every:
        Interval (in steps) at which progress is printed.

    Returns:

      A dictionary with keys:
      - theta_hat: optimized theta.
      - logphi_hat: optimized log(phi).
      - phi_hat: optimized phi in levels.
      - loss: final GMM objective value.
    """
    q = data["q"]
    if W_np is None:
        W_tf = tf.eye(q, dtype=DTYPE)
    else:
        W_tf = tf.constant(W_np.astype(np.float32), dtype=DTYPE)

    theta_var = tf.Variable(float(theta_init), dtype=DTYPE)
    log_phi_var = tf.Variable(float(log_phi_init), dtype=DTYPE)

    # Enforce parameter bounds before starting optimization.
    theta_var.assign(tf.clip_by_value(theta_var, THETA_MIN_TF, THETA_MAX_TF))
    log_phi_var.assign(tf.clip_by_value(log_phi_var, LOG_PHI_MIN_TF, LOG_PHI_MAX_TF))

    opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))

    last_loss = None
    for step in range(1, int(train_steps) + 1):
        with tf.GradientTape() as tape:
            loss, g = gmm_objective_tf(
                theta_var.read_value(),
                log_phi_var.read_value(),
                W_tf,
                data["k_t_tf"],
                data["iota_t_tf"],
                data["k_tp1_tf"],
                data["z_tp1_tf"],
                data["iota_tp1_tf"],
                data["Z_tf"],
                data["firm_idx_tf"],
            )

        grads = tape.gradient(loss, [theta_var, log_phi_var])
        opt.apply_gradients(zip(grads, [theta_var, log_phi_var]))

        # Project updated parameters back into admissible bounds.
        theta_var.assign(tf.clip_by_value(theta_var, THETA_MIN_TF, THETA_MAX_TF))
        log_phi_var.assign(tf.clip_by_value(log_phi_var, LOG_PHI_MIN_TF, LOG_PHI_MAX_TF))

        last_loss = loss
        if step == 1 or step % int(print_every) == 0 or step == int(train_steps):
            g_norm = float(tf.linalg.norm(g).numpy())
            print(
                f"    step {step:4d}: J={float(loss.numpy()):.4e}, "
                f"||g||={g_norm:.3e}, theta={float(theta_var.numpy()):.4f}, "
                f"phi={float(np.exp(log_phi_var.numpy())):.4f}"
            )

    theta_hat = float(theta_var.numpy())
    logphi_hat = float(log_phi_var.numpy())
    phi_hat = float(np.exp(logphi_hat))
    final_loss = float(last_loss.numpy())
    return dict(theta_hat=theta_hat, logphi_hat=logphi_hat, phi_hat=phi_hat, loss=final_loss)


def estimate_one_replication_2step_gmm(rep_id, df_rep, standardize_instr=True, cfg=None):
    """Estimate theta and phi for one replication using two-step GMM.

    Args:

      rep_id:
        Replication identifier (scalar, used for reporting).
      df_rep:
        DataFrame with data for a single replication.
      standardize_instr:
        If True, standardize instruments before GMM.
      cfg:
        Optional GMMConfigPart2 instance with tuning parameters. If None,
        uses the default global configuration GMM_CFG.

    Returns:

      A dictionary with step-1 and step-2 estimates and test statistics:
      - rep: replication id.
      - N_firms, q, df: sample size, instruments, degrees of freedom.
      - theta1_hat, phi1_hat, J1: step-1 estimates and objective.
      - theta_hat, phi_hat, JN: step-2 estimates and objective.
      - Jstat: N * JN (Hansen J statistic).
      - pval: p-value of J test.
      - reject: boolean for rejection at cfg.hansen_alpha.
      - standardize_instr: whether instruments were standardized.
    """
    cfg = cfg or GMM_CFG

    data = prepare_gmm_data_from_df(df_rep, standardize_instr=standardize_instr)
    q = data["q"]

    print(f"  Step 1 (W=I), q={q}:")
    step1 = _run_gmm_optim(
        data=data,
        theta_init=cfg.theta_init_guess,
        log_phi_init=np.log(cfg.phi_init_guess),
        W_np=None,
        learning_rate=cfg.lr_1,
        train_steps=cfg.steps_1,
        print_every=50,
    )

    print(
        f"  Estimating W=S^-1 at step-1 estimate: "
        f"theta={step1['theta_hat']:.4f}, phi={step1['phi_hat']:.4f}"
    )
    _, W2 = estimate_S_and_W_np(
        theta_hat=step1["theta_hat"],
        logphi_hat=step1["logphi_hat"],
        data=data,
        ridge=cfg.w_ridge,
        rcond=cfg.pinv_rcond,
    )

    print(f"  Step 2 (W=S^-1), q={q}:")
    step2 = _run_gmm_optim(
        data=data,
        theta_init=step1["theta_hat"],
        log_phi_init=step1["logphi_hat"],
        W_np=W2,
        learning_rate=cfg.lr_2,
        train_steps=cfg.steps_2,
        print_every=50,
    )

    N = data["N_firms"]
    # Hansen J-statistic: N times the minimized GMM objective.
    J_stat = float(N * step2["loss"])
    # Degrees of freedom: q moments - 2 parameters (theta, phi).
    df_chi = int(q - 2)
    pval = chi2_sf(J_stat, df_chi) if df_chi > 0 else float("nan")
    reject = bool((df_chi > 0) and (pval < float(cfg.hansen_alpha)))

    return dict(
        rep=int(rep_id),
        N_firms=int(N),
        q=int(q),
        df=int(df_chi),
        theta1_hat=step1["theta_hat"],
        phi1_hat=step1["phi_hat"],
        J1=step1["loss"],
        theta_hat=step2["theta_hat"],
        phi_hat=step2["phi_hat"],
        JN=step2["loss"],
        Jstat=J_stat,
        pval=pval,
        reject=reject,
        standardize_instr=bool(standardize_instr),
    )


def run_gmm_estimation(paths=None, cols=None, bounds=None, cfg=None):
    """Run two-step GMM estimation over a subset of replications.

    Args:

      paths:
        Optional PathsPart2 instance specifying data_csv path. If None,
        uses the global PATHS.
      cols:
        Optional PanelColumnsPart2 instance. If None, uses global COLS.
      bounds:
        Unused placeholder for parameter bounds (kept for API symmetry).
      cfg:
        Optional GMMConfigPart2 instance. If None, uses global GMM_CFG.

    Returns:

      A context dictionary with keys:
      - theta_true, phi_true: true parameter values from the CSV.
      - rep_ids_eval: list of replication ids used in estimation.
      - res_df: DataFrame with estimation results by replication.
      The same context is also stored in the global GMM_CTX.
    """
    paths = paths or PATHS
    cols = cols or COLS
    _ = bounds or BOUNDS
    cfg = cfg or GMM_CFG

    print("Loading synthetic panels from", paths.data_csv)
    df, theta_true, phi_true = load_synthetic_panels(paths.data_csv)
    print(f"Detected true parameters from CSV: theta_true={theta_true}, phi_true={phi_true}")

    rep_ids_all = sorted(df[cols.rep].unique())
    rep_ids_eval = select_rep_ids(df, cols.rep, cfg.n_reps_eval)
    print(f"Total replications in data: {len(rep_ids_all)}; using {len(rep_ids_eval)} for GMM/Metric 1.\n")

    results = []
    for idx, rep_id in enumerate(rep_ids_eval, start=1):
        print(f"=== Replication {rep_id} ({idx}/{len(rep_ids_eval)}) ===")
        df_rep = df[df[cols.rep] == rep_id].copy()
        out = estimate_one_replication_2step_gmm(
            rep_id=rep_id,
            df_rep=df_rep,
            standardize_instr=cfg.standardize_instr,
            cfg=cfg,
        )
        results.append(out)
        print(
            f"  Final (step-2) estimates: theta_hat={out['theta_hat']:.4f}, "
            f"phi_hat={out['phi_hat']:.4f}, J_N={out['JN']:.4e}, "
            f"N*J_N={out['Jstat']:.3f}, pval={out['pval']:.3g}\n"
        )

    res_df = pd.DataFrame(results).sort_values("rep").reset_index(drop=True)
    print("Estimation results (first few rows):")
    print(res_df.head())

    ctx = dict(
        theta_true=theta_true,
        phi_true=phi_true,
        rep_ids_eval=rep_ids_eval,
        res_df=res_df,
    )
    GMM_CTX.clear()
    GMM_CTX.update(ctx)
    return ctx


def jacobian_g_of_params_np(theta_hat, phi_hat, df_rep, standardize_instr=True):
    """Compute Jacobian of the moment vector g with respect to parameters.

    Args:

      theta_hat:
        Scalar value of theta at which to evaluate the Jacobian.
      phi_hat:
        Scalar value of phi at which to evaluate the Jacobian.
      df_rep:
        DataFrame for a single replication.
      standardize_instr:
        If True, standardize instruments before computing moments.

    Returns:

      A tuple (D, data) where:
      - D is a NumPy array with shape (q, 2) containing partial derivatives
        of g with respect to (theta, log_phi).
      - data is the prepared data dictionary from prepare_gmm_data_from_df.
    """
    data = prepare_gmm_data_from_df(df_rep, standardize_instr=standardize_instr)
    theta = tf.constant(float(theta_hat), dtype=DTYPE)
    log_phi = tf.constant(float(np.log(phi_hat)), dtype=DTYPE)

    with tf.GradientTape() as tape:
        tape.watch([theta, log_phi])
        g, _ = gmm_g_and_mbar_tf(
            theta,
            log_phi,
            data["k_t_tf"],
            data["iota_t_tf"],
            data["k_tp1_tf"],
            data["z_tp1_tf"],
            data["iota_tp1_tf"],
            data["Z_tf"],
            data["firm_idx_tf"],
        )

    # Each of dtheta and dlogphi has shape (q,); stack to get (q, 2).
    dtheta, dlogphi = tape.jacobian(g, [theta, log_phi])
    D = tf.stack([dtheta, dlogphi], axis=1).numpy().astype(np.float64)
    return D, data


def run_j_test_overid_gmm(res_df, alpha=0.05):
    """Summarize Hansen J overidentification tests across replications.

    Args:

      res_df:
        DataFrame returned by run_gmm_estimation()["res_df"].
      alpha:
        Significance level used for rejection indicator.

    Returns:

      A DataFrame with columns:
      ["rep", "N_firms", "q", "df", "Jstat", "pval", "reject"] and
      prints a brief summary of rejection rates.
    """
    out = res_df[["rep", "N_firms", "q", "df", "Jstat", "pval", "reject"]].copy()
    q = int(out["q"].iloc[0]) if len(out) else None
    df_chi = int(out["df"].iloc[0]) if len(out) else None
    print("\n=== Hansen J test (Euler GMM overidentifying restrictions) ===")
    print(f"moments q={q} | params p=2 | df={df_chi} | alpha={alpha:.2f}")
    print(f"Rejection rate over reps: {out['reject'].mean():.3f}")
    return out


def diagnostic_Jstat_vs_chi2(res_df, df_chi, n_mc=100000, random_seed=12345):
    """Compare empirical J statistics to a chi-square(df) distribution.

    Args:

      res_df:
        DataFrame with a "Jstat" column from GMM estimation.
      df_chi:
        Degrees of freedom (q - p) for the chi-square distribution.
      n_mc:
        Number of Monte Carlo draws for reference chi-square distribution.
      random_seed:
        Seed for NumPy's random generator.

    Returns:

      None. Prints summary statistics and, when possible, shows a Q–Q plot
      of empirical Jstat versus simulated chi-square quantiles.
    """
    import matplotlib.pyplot as plt

    Jstat = res_df["Jstat"].to_numpy(dtype=float)
    R = len(Jstat)

    js_mean = Jstat.mean()
    js_std = Jstat.std(ddof=1) if R > 1 else float("nan")
    js_q = np.quantile(Jstat, [0.05, 0.25, 0.50, 0.75, 0.95]) if R else np.zeros(5)

    rng = np.random.default_rng(int(random_seed))
    chi_mc = rng.chisquare(df_chi, size=n_mc)
    chi_mean = chi_mc.mean()
    chi_std = chi_mc.std(ddof=1)
    chi_q = np.quantile(chi_mc, [0.05, 0.25, 0.50, 0.75, 0.95])

    print("\n=== Diagnostic: N*J_N vs Chi-square(df=q-p) ===")
    print(f"# of replications: {R}")
    print(f"df (q - p) = {df_chi}")
    print(f"Jstat mean = {js_mean:.3f},   Chi2 mean ≈ {chi_mean:.3f}  (theory mean = df = {df_chi})")
    print(
        f"Jstat std  = {js_std:.3f},   Chi2 std  ≈ {chi_std:.3f}  "
        f"(theory std = sqrt(2*df) ≈ {np.sqrt(2*df_chi):.3f})"
    )
    print("Quantiles (Jstat vs Chi-square MC):")
    for p, L, C in zip([0.05, 0.25, 0.50, 0.75, 0.95], js_q, chi_q):
        print(f"  q{int(100*p):2d}:  Jstat = {L:7.3f},   chi2 ≈ {C:7.3f}")

    if R >= 2:
        # Match the number of simulated chi-square draws to the number of J statistics
        # to make the Q–Q comparison visually straightforward.
        chi_sim = rng.chisquare(df_chi, size=R)
        chi_sorted = np.sort(chi_sim)
        js_sorted = np.sort(Jstat)

        plt.figure(figsize=(5, 5))
        plt.scatter(chi_sorted, js_sorted, s=25, alpha=0.7)
        maxv = max(chi_sorted.max(), js_sorted.max())
        plt.plot([0, maxv], [0, maxv], "r--", lw=1, label="45-degree line")
        plt.xlabel(f"Chi-square({df_chi}) quantiles (MC)")
        plt.ylabel("N * J_N quantiles")
        plt.title("Q–Q plot: N*J_N vs Chi-square")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()