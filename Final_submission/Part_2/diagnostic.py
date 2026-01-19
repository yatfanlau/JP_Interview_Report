"""Metrics and auxiliary-moment evaluation utilities for SMM, GMM, and HMC."""

import numpy as np
import pandas as pd
import tensorflow as tf

from config_part2 import (
    PathsPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    SMMConfigPart2,
    GMMConfigPart2,
    HMCConfigPart2,
    BasicModelParams,
)
from utils_part2 import zcrit, symmetrize_np, pinv_psd_np
from gmm_smm_core import build_basic_model_context, load_policy_model, simulate_panel_hist_tf

import SMM_estimation as smm
import GMM_estimation as gmm
from amortized_policy import ParamPolicyNet  # noqa: F401


def metric1_bias_sd_rmse(estimates, true_value):
    """Compute mean, bias, standard deviation, and RMSE of an estimator.

    Args:
      estimates:
        1D array-like of estimates from Monte Carlo replications.
      true_value:
        True scalar parameter value.

    Returns:
      A tuple (mean_hat, bias, sd, rmse).
    """
    est = np.asarray(estimates, dtype=float)
    mean_hat = est.mean()
    bias = mean_hat - true_value
    sd = est.std(ddof=1) if est.size > 1 else float("nan")
    rmse = np.sqrt(((est - true_value) ** 2).mean())
    return mean_hat, bias, sd, rmse


def run_metric_1(res_df, theta_true, phi_true, label):
    """Print bias/SD/RMSE for theta and phi over Monte Carlo replications.

    Args:
      res_df:
        DataFrame with columns 'theta_hat' and 'phi_hat' for each replication.
      theta_true:
        True value of theta.
      phi_true:
        True value of phi.
      label:
        Text label identifying the estimator (e.g., "SMM", "GMM").
    """
    print(f"\n=== Metric 1: {label} estimator performance over replications ===")
    for name, est, true in [
        ("theta", res_df["theta_hat"].to_numpy(), theta_true),
        ("phi", res_df["phi_hat"].to_numpy(), phi_true),
    ]:
        mean_hat, bias, sd, rmse = metric1_bias_sd_rmse(est, true)
        print(
            f"{name}: mean={mean_hat:.8f}, true={true:.8f}, "
            f"bias={bias:.3e}, SD={sd:.3e}, RMSE={rmse:.3e}"
        )


def run_metric_2_coverage_smm(
    res_df,
    df_all,
    theta_true,
    phi_true,
    policy=None,
    paths=None,
    cols=None,
    bounds=None,
    cfg=None,
):
    """Compute delta-method coverage of SMM confidence intervals.

    Uses a two-step or optimal-weighting SMM delta method based on simulated
    moments and (optionally) simulation variance in the weight matrix.

    Args:
      res_df:
        DataFrame with columns ['rep', 'theta_hat', 'phi_hat'] for each
        replication.
      df_all:
        Full panel DataFrame stacked over replications; must contain a
        replication identifier column (cols.rep).
      theta_true:
        True value of theta.
      phi_true:
        True value of phi.
      policy:
        TensorFlow policy model; if None, it is loaded from paths.policy_path.
      paths:
        PathsPart2 configuration. If None, a default PathsPart2() is used.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      bounds:
        ParamBoundsPart2 object with parameter bounds for clipping intervals.
      cfg:
        SMMConfigPart2 configuration for Metric 2.

    Returns:
      A DataFrame with per-replication standard errors, confidence intervals,
      and coverage indicators for theta and phi.
    """
    paths = paths or PathsPart2()
    cols = cols or PanelColumnsPart2()
    bounds = bounds or ParamBoundsPart2()
    cfg = cfg or SMMConfigPart2()

    if policy is None:
        policy = load_policy_model(paths.policy_path)

    z = zcrit(cfg.metric2_alpha)
    q = smm.N_TARGET_MOMENTS

    rows = []
    for _, row in res_df.iterrows():
        rep_id = int(row["rep"])
        theta_hat = float(row["theta_hat"])
        phi_hat = float(row["phi_hat"])
        logphi_hat = float(np.log(phi_hat))

        # Extract data for a single Monte Carlo replication.
        df_rep = df_all[df_all[cols.rep] == rep_id].copy()

        # Bootstrap covariance of data moments across firms.
        Sigma_data = smm.bootstrap_moment_cov_by_firm(
            df_rep, n_boot=cfg.metric2_n_boot, seed=cfg.boot_seed_base + rep_id
        )

        if cfg.include_sim_var_in_W:
            # Monte Carlo covariance of simulated moments at (theta_hat, phi_hat).
            _, Sigma_sim = smm.simulate_moments_mc(
                policy,
                theta_hat,
                phi_hat,
                n_sims=cfg.metric2_n_sims,
                seed=cfg.sim_seed_base + rep_id,
            )
            # Combine data and simulation variance; average over sims_per_obj_2.
            Sigma_g = Sigma_data + (Sigma_sim / max(int(cfg.sims_per_obj_2), 1))
        else:
            Sigma_g = Sigma_data.copy()

        # Enforce symmetry and ridge regularization for numerical stability.
        Sigma_g = symmetrize_np(Sigma_g) + float(cfg.metric2_ridge) * np.eye(
            q, dtype=np.float64
        )
        if cfg.metric2_use_optimal_weight:
            # Pseudo-inverse to handle near-singular covariance matrices.
            W = pinv_psd_np(Sigma_g, rcond=1e-10)
        else:
            W = np.eye(q, dtype=np.float64)

        # Jacobian of simulated average moments w.r.t. (theta, log phi).
        J = smm.jacobian_sim_moments_avg(
            policy,
            theta_hat,
            phi_hat,
            rep_id,
            base_seed=cfg.crn_base_seed,
            sims_per_obj=cfg.sims_per_obj_2,
        )

        # Hessian of the SMM criterion (up to scaling).
        H = J.T @ W @ J
        Hinv = np.linalg.pinv(H, rcond=1e-12)
        if cfg.metric2_use_optimal_weight:
            # Under optimal W, the delta-method covariance simplifies to H^{-1}.
            V = Hinv
        else:
            # Generalized sandwich formula for (theta, log phi).
            V = Hinv @ (J.T @ W @ Sigma_g @ W @ J) @ Hinv

        se_theta = float(np.sqrt(max(V[0, 0], 0.0)))
        se_logphi = float(np.sqrt(max(V[1, 1], 0.0)))

        theta_lo, theta_hi = theta_hat - z * se_theta, theta_hat + z * se_theta
        # Clip confidence interval to parameter bounds.
        theta_lo = float(np.clip(theta_lo, bounds.theta_min, bounds.theta_max))
        theta_hi = float(np.clip(theta_hi, bounds.theta_min, bounds.theta_max))

        logphi_lo, logphi_hi = logphi_hat - z * se_logphi, logphi_hat + z * se_logphi
        phi_lo, phi_hi = float(np.exp(logphi_lo)), float(np.exp(logphi_hi))
        phi_lo = float(np.clip(phi_lo, bounds.phi_min, bounds.phi_max))
        phi_hi = float(np.clip(phi_hi, bounds.phi_min, bounds.phi_max))

        rows.append(
            dict(
                rep=rep_id,
                theta_hat=theta_hat,
                theta_se=se_theta,
                theta_lo=theta_lo,
                theta_hi=theta_hi,
                cover_theta=(theta_lo <= theta_true <= theta_hi),
                phi_hat=phi_hat,
                logphi_se=se_logphi,
                phi_lo=phi_lo,
                phi_hi=phi_hi,
                cover_phi=(phi_lo <= phi_true <= phi_hi),
            )
        )

    out = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)

    print("\n=== Metric 2: Coverage (delta method, 2-step/optimal weighting) ===")
    print(
        f"Nominal level: {100 * (1 - cfg.metric2_alpha):.1f}% | n_boot={cfg.metric2_n_boot} | "
        f"n_sims={cfg.metric2_n_sims} | sims_per_obj={cfg.sims_per_obj_2}"
    )
    print(
        f"include_sim_var={cfg.include_sim_var_in_W} | "
        f"use_optimal_weight={cfg.metric2_use_optimal_weight}"
    )
    print(f"Theta coverage: {out['cover_theta'].mean():.3f}")
    print(f"Phi coverage:   {out['cover_phi'].mean():.3f}")
    return out


def run_metric_2_coverage_gmm(
    res_df,
    df_all,
    theta_true,
    phi_true,
    cols=None,
    bounds=None,
    cfg=None,
):
    """Compute GMM sandwich-based coverage of confidence intervals.

    Uses a cluster-by-firm robust covariance estimator and the usual GMM
    sandwich formula for (theta, log phi).

    Args:
      res_df:
        DataFrame with columns ['rep', 'theta_hat', 'phi_hat'] for each
        replication.
      df_all:
        Full panel DataFrame stacked over replications; must contain a
        replication identifier column (cols.rep).
      theta_true:
        True value of theta.
      phi_true:
        True value of phi.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      bounds:
        ParamBoundsPart2 object with parameter bounds for clipping intervals.
      cfg:
        GMMConfigPart2 configuration for Metric 2.

    Returns:
      A DataFrame with per-replication standard errors, confidence intervals,
      and coverage indicators for theta and phi.
    """
    cols = cols or PanelColumnsPart2()
    bounds = bounds or ParamBoundsPart2()
    cfg = cfg or GMMConfigPart2()

    z = zcrit(cfg.metric2_alpha)
    rows = []

    for _, row in res_df.iterrows():
        rep_id = int(row["rep"])
        theta_hat = float(row["theta_hat"])
        phi_hat = float(row["phi_hat"])
        logphi_hat = float(np.log(phi_hat))

        # Data for the current replication.
        df_rep = df_all[df_all[cols.rep] == rep_id].copy()

        # Jacobian of GMM moments w.r.t. (theta, log phi) and cached data.
        D, data = gmm.jacobian_g_of_params_np(
            theta_hat, phi_hat, df_rep, standardize_instr=cfg.standardize_instr
        )
        N = int(data["N_firms"])
        q = int(data["q"])

        # Cluster-robust S_hat and optimal weight matrix W_opt.
        S_hat, W_opt = gmm.estimate_S_and_W_np(
            theta_hat,
            logphi_hat,
            data,
            ridge=cfg.metric2_ridge,
            rcond=cfg.pinv_rcond,
        )
        W = W_opt if cfg.metric2_use_optimal_weight else np.eye(
            q, dtype=np.float64
        )

        # Hessian of the GMM criterion (up to scaling).
        H = D.T @ W @ D
        Hinv = np.linalg.pinv(H, rcond=1e-12)

        # sqrt(N)-asymptotic covariance via sandwich formula.
        V_sqrtN = Hinv @ (D.T @ W @ S_hat @ W @ D) @ Hinv
        # Convert from sqrt(N)-scale to finite-sample covariance.
        V = V_sqrtN / max(N, 1)

        se_theta = float(np.sqrt(max(V[0, 0], 0.0)))
        se_logphi = float(np.sqrt(max(V[1, 1], 0.0)))

        theta_lo, theta_hi = theta_hat - z * se_theta, theta_hat + z * se_theta
        theta_lo = float(np.clip(theta_lo, bounds.theta_min, bounds.theta_max))
        theta_hi = float(np.clip(theta_hi, bounds.theta_min, bounds.theta_max))

        logphi_lo, logphi_hi = logphi_hat - z * se_logphi, logphi_hat + z * se_logphi
        phi_lo, phi_hi = float(np.exp(logphi_lo)), float(np.exp(logphi_hi))
        phi_lo = float(np.clip(phi_lo, bounds.phi_min, bounds.phi_max))
        phi_hi = float(np.clip(phi_hi, bounds.phi_min, bounds.phi_max))

        rows.append(
            dict(
                rep=rep_id,
                N_firms=N,
                theta_hat=theta_hat,
                theta_se=se_theta,
                theta_lo=theta_lo,
                theta_hi=theta_hi,
                cover_theta=(theta_lo <= theta_true <= theta_hi),
                phi_hat=phi_hat,
                logphi_se=se_logphi,
                phi_lo=phi_lo,
                phi_hi=phi_hi,
                cover_phi=(phi_lo <= phi_true <= phi_hi),
            )
        )

    out = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)

    print("\n=== Metric 2: Coverage (GMM sandwich, cluster-by-firm) ===")
    print(
        f"Nominal level: {100 * (1 - cfg.metric2_alpha):.1f}% | "
        f"use_optimal_weight={cfg.metric2_use_optimal_weight}"
    )
    print(f"Theta coverage: {out['cover_theta'].mean():.3f}")
    print(f"Phi coverage:   {out['cover_phi'].mean():.3f}")
    return out


def run_metric_2_coverage_hmc(res_df, theta_true, phi_true, cfg=None):
    """Compute coverage of Bayesian credible intervals from HMC output.

    Args:
      res_df:
        DataFrame with columns ['rep', 'theta_lo', 'theta_hi',
        'phi_lo', 'phi_hi'] giving credible intervals for each
        replication.
      theta_true:
        True value of theta.
      phi_true:
        True value of phi.
      cfg:
        HMCConfigPart2 configuration carrying the nominal cred_level.

    Returns:
      A DataFrame with added 'cover_theta' and 'cover_phi' columns,
      sorted by replication.
    """
    cfg = cfg or HMCConfigPart2()
    out = res_df.copy()
    if not {"theta_lo", "theta_hi", "phi_lo", "phi_hi"}.issubset(out.columns):
        raise ValueError(
            "HMC res_df must contain theta_lo/theta_hi/phi_lo/phi_hi "
            "credible interval columns."
        )

    out["cover_theta"] = (out["theta_lo"] <= theta_true) & (
        theta_true <= out["theta_hi"]
    )
    out["cover_phi"] = (out["phi_lo"] <= phi_true) & (phi_true <= out["phi_hi"])

    print("\n=== Metric 2: Coverage (Bayesian credible intervals) ===")
    print(f"Nominal level: {100 * cfg.cred_level:.1f}%")
    print(f"Theta coverage: {out['cover_theta'].mean():.3f}")
    print(f"Phi coverage:   {out['cover_phi'].mean():.3f}")
    return out.sort_values("rep").reset_index(drop=True)


def aux_moments_from_df(df_rep, cols: PanelColumnsPart2):
    """Compute auxiliary moments from a single replication's panel data.

    Moments: mean(z), std(ln z), corr(log k, z), AR(1) of log k.

    Args:
      df_rep:
        DataFrame for a single replication, identified by cols.firm and
        cols.time.
      cols:
        PanelColumnsPart2 describing column names for k, z, firm, and time.

    Returns:
      A NumPy array of shape (4,) with the auxiliary moments.
    """
    df = df_rep.sort_values([cols.firm, cols.time])
    k = df[cols.k].to_numpy(dtype=np.float64)
    z = df[cols.z].to_numpy(dtype=np.float64)

    logk = np.log(k + 1e-32)
    lnz = np.log(z + 1e-32)

    # Lag of capital by firm; first period per firm becomes NaN.
    k_lag = df.groupby(cols.firm)[cols.k].shift(1).to_numpy(dtype=np.float64)
    logk_lag = np.log(k_lag + 1e-32)
    mask = np.isfinite(logk_lag)

    ar1_logk = (
        float(np.corrcoef(logk[mask], logk_lag[mask])[0, 1]) if mask.any() else 0.0
    )
    corr_logk_z = (
        float(np.corrcoef(logk, z)[0, 1])
        if (logk.std(ddof=0) > 0 and z.std(ddof=0) > 0)
        else 0.0
    )

    return np.array(
        [
            float(z.mean()),
            float(lnz.std(ddof=0)),
            float(corr_logk_z),
            float(ar1_logk),
        ],
        dtype=np.float64,
    )


def aux_moments_tf(k_hist, z_hist):
    """Compute auxiliary moments from simulated histories in TensorFlow.

    Moments: mean(z), std(ln z), corr(log k, z), AR(1) of log k.

    Args:
      k_hist:
        Tensor of shape [n_firms, T] containing capital histories.
      z_hist:
        Tensor of shape [n_firms, T] containing productivity histories.

    Returns:
      A rank-1 Tensor of length 4 with the auxiliary moments.
    """
    # Flatten across firms and time.
    k_flat = tf.reshape(k_hist, [-1])
    z_flat = tf.reshape(z_hist, [-1])

    logk = tf.math.log(k_flat + 1e-32)
    lnz = tf.math.log(z_flat + 1e-32)

    # 1) Mean of z.
    a1 = tf.reduce_mean(z_flat)

    # 2) Standard deviation of ln z.
    a2 = tf.sqrt(tf.reduce_mean((lnz - tf.reduce_mean(lnz)) ** 2) + 1e-12)

    # 3) Correlation between log k and z.
    mlogk, mz = tf.reduce_mean(logk), tf.reduce_mean(z_flat)
    cov = tf.reduce_mean((logk - mlogk) * (z_flat - mz))
    std_logk = tf.sqrt(tf.reduce_mean((logk - mlogk) ** 2) + 1e-12)
    std_z = tf.sqrt(tf.reduce_mean((z_flat - mz) ** 2) + 1e-12)
    a3 = cov / (std_logk * std_z)

    # 4) AR(1) of log k using lag-1 within-firm pairs.
    logk_hist = tf.math.log(k_hist + 1e-32)
    x = tf.reshape(logk_hist[:, 1:], [-1])   # k_t
    y = tf.reshape(logk_hist[:, :-1], [-1])  # k_{t-1}
    mx, my = tf.reduce_mean(x), tf.reduce_mean(y)
    cov_xy = tf.reduce_mean((x - mx) * (y - my))
    stdx = tf.sqrt(tf.reduce_mean((x - mx) ** 2) + 1e-12)
    stdy = tf.sqrt(tf.reduce_mean((y - my) ** 2) + 1e-12)
    a4 = cov_xy / (stdx * stdy)

    return tf.stack([a1, a2, a3, a4])


def _run_metric_3_aux_fit_common(
    res_df, df_all, policy, cols, burnin, sim_len, n_firms, seed_base, ctx, title
):
    """Common helper for Metric 3: auxiliary-moment fit diagnostics.

    For each replication, simulates a panel from the policy at the estimated
    parameters and compares auxiliary moments between data and model.

    Args:
      res_df:
        DataFrame with columns ['rep', 'theta_hat', 'phi_hat'] for each
        replication.
      df_all:
        Full panel DataFrame stacked over replications; must contain a
        replication identifier column (cols.rep).
      policy:
        TensorFlow policy network returning investment decisions.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      burnin:
        Number of initial simulated periods to discard.
      sim_len:
        Number of simulated periods used for computing moments.
      n_firms:
        Number of firms simulated in the auxiliary panel.
      seed_base:
        Integer base seed; per-replication seed is (seed_base + rep_id).
      ctx:
        Model context dictionary produced by build_basic_model_context.
      title:
        String used in the printed title for this metric.

    Returns:
      A DataFrame with per-replication auxiliary-moment discrepancies.
      Columns include 'D_aux' (mean absolute scaled deviation) and d1–d4.
    """
    rows = []
    for _, row in res_df.iterrows():
        rep_id = int(row["rep"])
        theta_hat = float(row["theta_hat"])
        phi_hat = float(row["phi_hat"])

        # Data auxiliary moments for this replication.
        df_rep = df_all[df_all[cols.rep] == rep_id].copy()
        mu_data = aux_moments_from_df(df_rep, cols)

        rng = np.random.default_rng(int(seed_base + rep_id))
        # Pre-draw productivity shocks and initial ln z for all firms and times.
        eps = rng.normal(
            0.0,
            ctx["sigma_eps_f"],
            size=(n_firms, burnin + sim_len),
        ).astype(np.float32)
        lnz0 = rng.normal(
            ctx["m_ln_z_f"],
            ctx["sigma_ln_z_f"],
            size=(n_firms,),
        ).astype(np.float32)

        # Simulate model-implied panel at estimated parameters.
        k_hist, z_hist, _ = simulate_panel_hist_tf(
            policy,
            theta=tf.constant(theta_hat, dtype=tf.float32),
            log_phi=tf.constant(np.log(phi_hat), dtype=tf.float32),
            eps_all_tf=tf.constant(eps, dtype=tf.float32),
            lnz_init_tf=tf.constant(lnz0, dtype=tf.float32),
            burnin=int(burnin),
            sim_len=int(sim_len),
            ctx=ctx,
        )
        mu_model = aux_moments_tf(k_hist, z_hist).numpy().astype(np.float64)

        # Scale each moment by its magnitude to make deviations comparable.
        scale = np.maximum(np.abs(mu_data), 1e-6)
        d_scaled = (mu_model - mu_data) / scale
        # D_aux is the average absolute scaled deviation across moments.
        D_aux = float(np.mean(np.abs(d_scaled)))

        rows.append(
            dict(
                rep=rep_id,
                D_aux=D_aux,
                **{f"d{j + 1}": d_scaled[j] for j in range(len(d_scaled))},
            )
        )

    out = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)

    print(f"\n=== Metric 3: {title} ===")
    print("Aux moments: [mean z, std lnz, corr(logk,z), AR1 logk]")
    print(f"Mean D_aux over reps: {out['D_aux'].mean():.4f}")
    return out


def run_metric_3_aux_fit_smm(res_df, df_all, policy, cols=None, cfg=None, mp=None):
    """Run Metric 3 (auxiliary-moment fit) for SMM estimates.

    Args:
      res_df:
        DataFrame with SMM estimates ('rep', 'theta_hat', 'phi_hat').
      df_all:
        Full panel DataFrame stacked over replications.
      policy:
        TensorFlow policy model used in the SMM estimation.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      cfg:
        SMMConfigPart2 configuration for Metric 3.
      mp:
        BasicModelParams object for constructing the model context.

    Returns:
      A DataFrame with D_aux and per-moment discrepancies for each replication.
    """
    cols = cols or PanelColumnsPart2()
    cfg = cfg or SMMConfigPart2()
    mp = mp or BasicModelParams()
    ctx = build_basic_model_context(mp)

    return _run_metric_3_aux_fit_common(
        res_df=res_df,
        df_all=df_all,
        policy=policy,
        cols=cols,
        burnin=cfg.metric3_burnin,
        sim_len=cfg.metric3_sim_len,
        n_firms=cfg.metric3_n_firms_sim,
        seed_base=cfg.metric3_seed_base,
        ctx=ctx,
        title="Auxiliary-moment fit",
    )


def run_metric_3_aux_fit_gmm(res_df, df_all, paths=None, cols=None, cfg=None, mp=None):
    """Run Metric 3 (auxiliary-moment fit) for GMM estimates.

    Loads the amortized policy network from disk, simulates panels at the
    GMM point estimates, and compares auxiliary moments.

    Args:
      res_df:
        DataFrame with GMM estimates ('rep', 'theta_hat', 'phi_hat').
      df_all:
        Full panel DataFrame stacked over replications.
      paths:
        PathsPart2 configuration used to locate the policy network.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      cfg:
        GMMConfigPart2 configuration for Metric 3.
      mp:
        BasicModelParams object for constructing the model context.

    Returns:
      A DataFrame with D_aux and per-moment discrepancies for each replication.
    """
    paths = paths or PathsPart2()
    cols = cols or PanelColumnsPart2()
    cfg = cfg or GMMConfigPart2()
    mp = mp or BasicModelParams()
    ctx = build_basic_model_context(mp)

    print("\nLoading amortized policy network for Metric 3...")
    policy = load_policy_model(paths.policy_path)
    print("Loaded policy from", paths.policy_path)

    return _run_metric_3_aux_fit_common(
        res_df=res_df,
        df_all=df_all,
        policy=policy,
        cols=cols,
        burnin=cfg.metric3_burnin,
        sim_len=cfg.metric3_sim_len,
        n_firms=cfg.metric3_n_firms_sim,
        seed_base=cfg.metric3_seed_base,
        ctx=ctx,
        title="Auxiliary-moment fit (GMM estimates)",
    )


def run_metric_3_aux_fit_hmc(res_df, df_all, policy, cols=None, cfg=None, mp=None):
    """Run Metric 3 (auxiliary-moment fit) for HMC-based estimates.

    Args:
      res_df:
        DataFrame with HMC posterior summaries ('rep', 'theta_hat', 'phi_hat').
      df_all:
        Full panel DataFrame stacked over replications.
      policy:
        TensorFlow policy model used when simulating from the posterior mode
        or representative point estimates.
      cols:
        PanelColumnsPart2 configuration describing DataFrame column names.
      cfg:
        HMCConfigPart2 configuration for Metric 3.
      mp:
        BasicModelParams object for constructing the model context.

    Returns:
      A DataFrame with D_aux and per-moment discrepancies for each replication.
    """
    cols = cols or PanelColumnsPart2()
    cfg = cfg or HMCConfigPart2()
    mp = mp or BasicModelParams()
    ctx = build_basic_model_context(mp)

    return _run_metric_3_aux_fit_common(
        res_df=res_df,
        df_all=df_all,
        policy=policy,
        cols=cols,
        burnin=cfg.metric3_burnin,
        sim_len=cfg.metric3_sim_len,
        n_firms=cfg.metric3_n_firms_sim,
        seed_base=cfg.metric3_seed_base,
        ctx=ctx,
        title="Auxiliary-moment fit (HMC estimates)",
    )