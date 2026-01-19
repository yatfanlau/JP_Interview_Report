"""Simulate synthetic firm-level investment panels using an amortized policy.

The script loads a trained policy network, simulates firm dynamics under the
policy, converts the simulated panel into a long-format DataFrame, and writes
replications to a CSV file.
"""

import os
import numpy as np
import pandas as pd

from utils_part2 import tf, DTYPE, ar1_step_ln_z, set_global_seed
from config_part2 import BasicModelParams
from amortized_policy import policy_iota


# Global simulation settings.
N_FIRMS, T_PERIODS, BURN_IN = 100, 40, 100  # Example: 200, 80, 200
N_REPS = 200
THETA_TRUE, PHI_TRUE = 0.7, 2.0
POLICY_PATH = "param_policy_theta_phi.keras"
OUTPUT_DIR = "synthetic_panels"
OUTPUT_CSV_BASENAME = "synthetic_panels_all.csv"

mp = BasicModelParams()
print("Basic (fixed) model parameters used in simulation:")
print(mp)
set_global_seed(mp.seed)

DELTA = tf.constant(mp.delta, dtype=DTYPE)
RHO = tf.constant(mp.rho, dtype=DTYPE)
SIGMA_EPS = tf.constant(mp.sigma_eps, dtype=DTYPE)
ONE = tf.constant(1.0, dtype=DTYPE)
ZERO = tf.constant(0.0, dtype=DTYPE)

# Mean of log productivity in the stationary distribution of the AR(1) in ln z.
MU_LN_Z = tf.constant(-0.5 * (mp.sigma_eps ** 2) / (1.0 + mp.rho), dtype=DTYPE)

# Numerical lower bound on capital to avoid exactly zero or negative k.
K_FLOOR = tf.constant(1e-12, dtype=DTYPE)


def steady_state_k(theta, delta, r):
    """Compute steady-state capital in a simple neoclassical model.

    Args:
        theta: Capital share in production (scalar or array-like).
        delta: Depreciation rate.
        r: Interest rate.

    Returns:
        numpy.ndarray: Steady-state capital corresponding to each theta.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return (theta / (r + delta)) ** (1.0 / (1.0 - theta))


def load_amortized_policy(path):
    """Load a pre-trained amortized policy network from disk.

    Args:
        path: File path to the saved Keras model.

    Returns:
        tf.keras.Model: Loaded policy network.
    """
    print(f"Loading amortized policy from: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    print("Loaded model:")
    model.summary()
    return model


def simulate_panel(policy, theta_true, phi_true, n_firms, t_periods, burn_in, base_seed):
    """Simulate firm-level panel data under a given policy.

    The function simulates a panel of firms evolving over time according to
    the investment policy and an AR(1) process for productivity. A burn-in
    phase is discarded so that the retained sample is closer to stationarity.

    Args:
        policy: Trained policy model mapping (k, z, theta, phi) to iota.
        theta_true: True capital share used in the simulation.
        phi_true: True adjustment cost parameter used in the simulation.
        n_firms: Number of firms to simulate.
        t_periods: Number of periods to keep after burn-in.
        burn_in: Number of initial periods to discard.
        base_seed: Base random seed used for this replication.

    Returns:
        dict: Dictionary with simulated arrays (time × firm):
            - "k": Capital stock.
            - "I": Investment level.
            - "iota": Investment rate I/k.
            - "z": Productivity.
            - "k_next": Next-period capital.
            - "z_next": Next-period productivity.
    """
    set_global_seed(base_seed)

    # Total simulated periods including burn-in.
    T_total = burn_in + t_periods

    # Arrays store the full simulation including the final next-period state.
    k = np.empty((T_total + 1, n_firms), dtype=np.float64)
    z = np.empty((T_total + 1, n_firms), dtype=np.float64)
    iota = np.empty((T_total, n_firms), dtype=np.float64)
    invest = np.empty((T_total, n_firms), dtype=np.float64)

    # Initialize all firms at the steady state with z = 1.
    k_star = steady_state_k(theta_true, mp.delta, mp.r)
    k[0, :], z[0, :] = k_star, 1.0

    # Broadcast structural parameters across firms as tensors.
    theta_vec = tf.fill([n_firms], tf.constant(theta_true, dtype=DTYPE))
    phi_vec = tf.fill([n_firms], tf.constant(phi_true, dtype=DTYPE))

    for t in range(T_total):
        # Current state as tensors.
        k_t_tf = tf.convert_to_tensor(k[t, :], dtype=DTYPE)
        z_t_tf = tf.convert_to_tensor(z[t, :], dtype=DTYPE)

        # Policy-implied investment rate iota_t = I_t / k_t.
        iota_t_tf = policy_iota(
            policy,
            k_t_tf,
            z_t_tf,
            theta_vec,
            phi_vec,
            training=False,
        )
        iota_t = iota_t_tf.numpy()

        # Level of investment I_t.
        I_t = iota_t * k[t, :]

        # Law of motion for capital with a floor to avoid degenerate values.
        k_next = np.maximum(
            K_FLOOR.numpy(),
            (1.0 - mp.delta + iota_t) * k[t, :],
        )

        # Innovation to log-productivity.
        eps_t = tf.random.normal(
            shape=(n_firms,),
            mean=ZERO,
            stddev=SIGMA_EPS,
            dtype=DTYPE,
        )

        # One step of AR(1) in ln z: ln z_{t+1} = rho ln z_t + eps_t + mu.
        z_next_tf = ar1_step_ln_z(z_t_tf, RHO, eps_t, MU_LN_Z)
        z_next = z_next_tf.numpy()

        # Store controls and next-period states.
        iota[t, :], invest[t, :], k[t + 1, :], z[t + 1, :] = (
            iota_t,
            I_t,
            k_next,
            z_next,
        )

    # Discard burn-in and keep only the sample window [burn_in, burn_in + t_periods).
    t0, t1 = burn_in, burn_in + t_periods

    return {
        "k": k[t0:t1, :],
        "I": invest[t0:t1, :],
        "iota": iota[t0:t1, :],
        "z": z[t0:t1, :],
        "k_next": k[t0 + 1 : t1 + 1, :],
        "z_next": z[t0 + 1 : t1 + 1, :],
    }


def panel_to_dataframe(panel, rep_index, theta_true, phi_true):
    """Convert a panel of simulated arrays into a long-format DataFrame.

    Args:
        panel: Dictionary of numpy arrays returned by `simulate_panel`.
        rep_index: Integer index of the current replication.
        theta_true: True capital share used in the simulation.
        phi_true: True adjustment cost parameter used in the simulation.

    Returns:
        pandas.DataFrame: Long-format DataFrame with one row per
        (replication, firm, time) observation.
    """
    # Unpack and keep the order consistent with CSV output.
    k, I, iota, z, k_next, z_next = (
        panel[x] for x in ("k", "I", "iota", "z", "k_next", "z_next")
    )

    # T: number of periods, N: number of firms.
    T, N = k.shape
    size = T * N

    # Flatten the state and control variables to 1D arrays.
    k_flat, I_flat, iota_flat, z_flat = (arr.reshape(-1) for arr in (k, I, iota, z))
    k_next_flat, z_next_flat = (arr.reshape(-1) for arr in (k_next, z_next))

    # Time index repeats 0..T-1 for each firm; firm index cycles 0..N-1 each period.
    t_idx = np.repeat(np.arange(T, dtype=int), N)
    firm_idx = np.tile(np.arange(N, dtype=int), T)

    # Replication index is constant within a replication.
    rep_idx = np.full(size, rep_index, dtype=int)

    return pd.DataFrame(
        {
            "rep": rep_idx,
            "firm": firm_idx,
            "t": t_idx,
            "k": k_flat,
            "I": I_flat,
            "iota": iota_flat,
            "z": z_flat,
            "k_next": k_next_flat,
            "z_next": z_next_flat,
            "theta_true": theta_true,
            "phi_true": phi_true,
        }
    )


def main():
    """Run the full simulation pipeline and write all replications to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_BASENAME)

    policy = load_amortized_policy(POLICY_PATH)

    print("\nSimulation settings:")
    print(f"  N_FIRMS    = {N_FIRMS}")
    print(f"  T_PERIODS  = {T_PERIODS}")
    print(f"  BURN_IN    = {BURN_IN}")
    print(f"  N_REPS     = {N_REPS}")
    print(f"  theta_true = {THETA_TRUE}")
    print(f"  phi_true   = {PHI_TRUE}")
    print(f"  Output CSV = {out_path}\n")

    for rep in range(N_REPS):
        print(f"Simulating replication {rep + 1}/{N_REPS} ...")

        # Shift the base seed so each replication has independent randomness.
        base_seed = mp.seed + rep

        panel = simulate_panel(
            policy,
            THETA_TRUE,
            PHI_TRUE,
            N_FIRMS,
            T_PERIODS,
            BURN_IN,
            base_seed,
        )
        df = panel_to_dataframe(panel, rep, THETA_TRUE, PHI_TRUE)

        # First replication creates the file and header; subsequent ones append.
        mode, header = ("w", True) if rep == 0 else ("a", False)
        df.to_csv(out_path, index=False, mode=mode, header=header)

        print(f"  -> appended {len(df):,} rows to {out_path}")

    print("\nDone. All replications written to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()