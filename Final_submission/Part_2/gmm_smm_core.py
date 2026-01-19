"""Common functions for GMM/SMM"""

import math
import numpy as np
import pandas as pd

from utils_part2 import DTYPE
import tensorflow as tf

from config_part2 import BasicModelParams


def C(x, dtype=DTYPE):
    """Return a TensorFlow constant with the given value and dtype."""
    return tf.constant(x, dtype=dtype)


def build_basic_model_context(mp=None, dtype=DTYPE):
    """Build derived parameters and TensorFlow constants for the model.

    Args:
      mp: Optional BasicModelParams instance with primitive parameters. If
        None, a new BasicModelParams object is constructed.
      dtype: TensorFlow dtype to use for created constants.

    Returns:
      A dictionary containing scalar floats and TensorFlow constants used
      across the model and simulation code.
    """
    if mp is None:
        mp = BasicModelParams()

    delta_f = float(mp.delta)
    rho_f = float(mp.rho)
    sigma_eps_f = float(mp.sigma_eps)
    r_f = float(mp.r)
    beta_f = 1.0 / (1.0 + r_f)

    # Moments of the stationary AR(1) process for ln(z).
    mu_ln_z_f = -0.5 * (sigma_eps_f**2) / (1.0 + rho_f)
    sigma_ln_z_f = sigma_eps_f / math.sqrt(1.0 - rho_f * rho_f)
    m_ln_z_f = -0.5 * (sigma_eps_f**2) / (1.0 - rho_f * rho_f)

    one_tf = tf.constant(1.0, dtype=dtype)
    delta_tf = tf.constant(delta_f, dtype=dtype)
    rho_tf = tf.constant(rho_f, dtype=dtype)
    r_tf = tf.constant(r_f, dtype=dtype)
    beta_tf = tf.constant(beta_f, dtype=dtype)
    mu_ln_z_tf = tf.constant(mu_ln_z_f, dtype=dtype)
    k_floor_tf = tf.constant(1e-12, dtype=dtype)
    one_minus_delta_tf = one_tf - delta_tf

    return dict(
        mp=mp,
        delta_f=delta_f,
        rho_f=rho_f,
        sigma_eps_f=sigma_eps_f,
        r_f=r_f,
        beta_f=beta_f,
        mu_ln_z_f=mu_ln_z_f,
        sigma_ln_z_f=sigma_ln_z_f,
        m_ln_z_f=m_ln_z_f,
        one_tf=one_tf,
        delta_tf=delta_tf,
        one_minus_delta_tf=one_minus_delta_tf,
        rho_tf=rho_tf,
        r_tf=r_tf,
        beta_tf=beta_tf,
        mu_ln_z_tf=mu_ln_z_tf,
        k_floor_tf=k_floor_tf,
    )


def iota_bounds(mp: BasicModelParams):
    """Compute lower and upper bounds for the investment rate iota.

    Args:
      mp: BasicModelParams instance providing iota and delta parameters.

    Returns:
      A tuple (iota_min, iota_max) as floats.
    """
    # Allow for small negative investment, scaled by (1 - delta).
    iota_min = float(-(mp.iota_lower_eps) * (1.0 - mp.delta))
    iota_max = float(mp.iota_upper)
    return iota_min, iota_max


def load_policy_model(policy_path: str):
    """Load a Keras policy model from disk without compiling it."""
    return tf.keras.models.load_model(policy_path, compile=False)


def load_synthetic_panels(data_csv: str):
    """Load synthetic panel data and extract the true (theta, phi).

    The CSV is expected to contain columns 'theta_true' and 'phi_true'
    with a single unique value for each across the file.

    Args:
      data_csv: Path to the CSV file containing simulated panel data.

    Returns:
      A tuple (df, theta_true, phi_true) where:
        df: A pandas DataFrame with the loaded data.
        theta_true: The unique true theta value (float).
        phi_true: The unique true phi value (float).

    Raises:
      ValueError: If the required columns are missing or if there is not
        exactly one unique (theta_true, phi_true) pair.
    """
    df = pd.read_csv(data_csv)

    if "theta_true" in df.columns and "phi_true" in df.columns:
        theta_vals = df["theta_true"].unique()
        phi_vals = df["phi_true"].unique()

        # Enforce that the dataset has a single underlying (theta, phi).
        if len(theta_vals) != 1 or len(phi_vals) != 1:
            raise ValueError(
                f"Expected single true (theta, phi) in CSV, "
                f"got theta_true={theta_vals}, phi_true={phi_vals}"
            )
        theta_true = float(theta_vals[0])
        phi_true = float(phi_vals[0])
    else:
        raise ValueError("CSV is missing 'theta_true' or 'phi_true' columns.")

    return df, theta_true, phi_true


def select_rep_ids(df, rep_col: str, n_reps_eval: int):
    """Select a subset of representative IDs from a DataFrame.

    Args:
      df: pandas DataFrame containing panel data.
      rep_col: Column name identifying cross-sectional units (e.g., firm id).
      n_reps_eval: Maximum number of representative IDs to return. If None
        or larger than the number of unique IDs, all IDs are returned.

    Returns:
      A sorted list of representative IDs.
    """
    rep_ids = sorted(df[rep_col].unique())
    if n_reps_eval is None or int(n_reps_eval) >= len(rep_ids):
        return rep_ids
    return rep_ids[: int(n_reps_eval)]


def policy_iota_tf(policy, k, z, theta, phi):
    """Evaluate the policy network to obtain iota given state variables.

    Args:
      policy: A TensorFlow/Keras model mapping inputs to policy outputs.
      k: 1D Tensor of capital stocks for each unit.
      z: 1D Tensor of productivity shocks for each unit.
      theta: Scalar parameter theta (broadcast across units).
      phi: Scalar parameter phi (broadcast across units).

    Returns:
      A 1D Tensor of the first output dimension of the policy, interpreted
      as the investment rate iota for each unit.
    """
    # Broadcast scalar parameters theta and phi to match the shape of k.
    theta_vec = tf.fill(tf.shape(k), theta)
    phi_vec = tf.fill(tf.shape(k), phi)

    # Stack log-state variables and parameters to form the policy input X:
    # X = [log(k), log(z), theta, log(phi)].
    x = tf.stack(
        [
            tf.math.log(k),
            tf.math.log(z),
            theta_vec,
            tf.math.log(phi_vec),
        ],
        axis=1,
    )
    # Assume the policy outputs at least one dimension; take the first.
    return policy(x, training=False)[:, 0]


def steady_state_ln_k_tf(theta, delta, r):
    """Compute the log of the steady-state capital stock.

    The formula corresponds to the steady state of a standard growth model
    with constant parameters.

    Args:
      theta: Output elasticity parameter (Tensor).
      delta: Depreciation rate (Tensor).
      r: Interest rate (Tensor).

    Returns:
      A Tensor with the log of the steady-state capital stock.
    """
    one = tf.constant(1.0, dtype=theta.dtype)
    return tf.math.log(theta / (r + delta)) / (one - theta)


def steady_state_k_tf(theta, delta, r):
    """Compute the steady-state capital stock."""
    return tf.exp(steady_state_ln_k_tf(theta, delta, r))


def simulate_panel_hist_tf(
    policy,
    theta,
    log_phi,
    eps_all_tf,
    lnz_init_tf,
    burnin,
    sim_len,
    ctx,
):
    """Simulate panel histories of (k, z, iota) using a policy function.

    The simulation draws on a pre-specified sequence of shocks eps_all_tf
    and an initial log productivity lnz_init_tf, applying a burn-in period
    before recording data.

    Args:
      policy: TensorFlow/Keras model implementing the investment policy.
      theta: Scalar parameter theta (Tensor or float).
      log_phi: Log of scalar parameter phi.
      eps_all_tf: 2D Tensor of shocks with shape [N, T_total].
      lnz_init_tf: 1D Tensor of initial log productivity with shape [N].
      burnin: Number of initial periods to discard.
      sim_len: Number of periods to keep after burn-in.
      ctx: Dictionary of constants as constructed by build_basic_model_context.

    Returns:
      A tuple of three Tensors (k_hist, z_hist, iota_hist), each of shape
      [N, sim_len], containing simulated capital, productivity, and policy
      (iota) histories for each unit.
    """
    phi = tf.exp(log_phi)
    N = tf.shape(eps_all_tf)[0]
    # Use static shape for the time dimension for iteration.
    T_total = int(eps_all_tf.shape[1])

    # Start each unit at the deterministic steady-state capital level.
    k_ss = steady_state_k_tf(theta, ctx["delta_tf"], ctx["r_tf"])
    k = tf.fill([N], tf.squeeze(k_ss))
    lnz = lnz_init_tf

    k_list, z_list, i_list = [], [], []
    for t in range(T_total):
        # AR(1) process for ln(z): ln(z_t) = mu + rho * ln(z_{t-1}) + eps_t.
        lnz = ctx["mu_ln_z_tf"] + ctx["rho_tf"] * lnz + eps_all_tf[:, t]
        z = tf.exp(lnz)

        # Evaluate policy to obtain iota_t for each unit.
        iota = policy_iota_tf(policy, k, z, theta, phi)

        # Law of motion for capital with a lower bound to avoid numerical issues.
        k_next = tf.maximum(
            ctx["k_floor_tf"],
            (ctx["one_minus_delta_tf"] + iota) * k,
        )

        if t >= int(burnin):
            k_list.append(k)
            z_list.append(z)
            i_list.append(iota)
            # Stop once we have collected sim_len periods after burn-in.
            if len(k_list) >= int(sim_len):
                break

        k = k_next

    # Stack lists along the time axis: resulting shape [N, sim_len].
    return (
        tf.stack(k_list, axis=1),
        tf.stack(z_list, axis=1),
        tf.stack(i_list, axis=1),
    )