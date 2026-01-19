"""
Diagnostics and plotting utilities for the risky-debt model.

This module is model-specific (RiskyDebtModel) and contains:
- Residual diagnostics (Bellman, FOC, limited liability smoothing)
- Training history plots
- Ergodic simulation utilities
- Final reporting and policy/value slice plots
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config_part1 import RiskyDebtFinalTestParams, RiskyDebtParams, RiskyDebtTrainingParams
from model_core import tf, DTYPE


def compute_residuals(model, k, b, z, n_mc_rhs: int = 50):
    """
    Diagnostic residuals for:
    - Bellman equation (relative error)
    - FOCs (squared gradient norm)
    - Limited liability smoothing (V vs ReLU(C))

    Args:
        model: Trained RiskyDebtModel.
        k: Capital, shape (B,1).
        b: Debt, shape (B,1).
        z: Productivity, shape (B,1).
        n_mc_rhs: Number of MC replications for RHS averaging.

    Returns:
        Dict of scalar diagnostic metrics (Python floats).
    """
    # Policy
    policy_input = model.state_features(k, b, z)
    k_prime, b_prime = model.policy_net(policy_input)

    # Predicted C and V
    C_pred, V_pred = model.compute_equity_value(k, b, z, use_target=False)

    # Monte Carlo RHS estimate
    rhs_accum = 0.0
    for _ in range(n_mc_rhs):
        rhs_accum += model.compute_rhs(k, b, z, k_prime, b_prime)
    rhs_true = rhs_accum / n_mc_rhs

    # Relative Bellman errors
    bell_abs = tf.abs(C_pred - rhs_true)
    bell_scale = tf.abs(rhs_true) + 1e-3
    bell_rel = bell_abs / bell_scale
    bell_rel_mean = tf.reduce_mean(bell_rel)
    bell_rel_max = tf.reduce_max(bell_rel)

    # FOC diagnostics
    with tf.GradientTape() as tape:
        tape.watch([k_prime, b_prime])
        rhs_single = model.compute_rhs(k, b, z, k_prime, b_prime)
    g_k, g_b = tape.gradient(rhs_single, [k_prime, b_prime])

    def safe(g):
        """Replace non-finite values with zeros for stable summaries."""
        return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))

    g_k = safe(g_k)
    g_b = safe(g_b)

    foc_sq = tf.square(g_k) + tf.square(g_b)
    foc_mean_sq = tf.reduce_mean(foc_sq)
    foc_max_abs = tf.reduce_max(tf.sqrt(foc_sq))

    # Limited liability smoothing diagnostics
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


def test_coverage_residuals(model, n_samples: int = 500, n_mc_rhs: int = 50):
    """
    Diagnostics on an exogenous coverage sampler.

    Args:
        model: Trained RiskyDebtModel.
        n_samples: Number of coverage states.
        n_mc_rhs: Number of MC replications for RHS averaging.

    Returns:
        Dict of diagnostic metrics.
    """
    states = model.coverage_sampler(n_samples)
    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]
    return compute_residuals(model, k, b, z, n_mc_rhs=n_mc_rhs)


def test_onpolicy_residuals(model, states, n_samples: int = 500, n_mc_rhs: int = 50):
    """
    Diagnostics on an on-policy (possibly ergodic) sample.

    Args:
        model: Trained RiskyDebtModel.
        states: Tensor of shape (N,3) containing [k,b,z].
        n_samples: Maximum number of states to test.
        n_mc_rhs: Number of MC replications for RHS averaging.

    Returns:
        Dict of diagnostic metrics.
    """
    N = tf.shape(states)[0]
    n = tf.minimum(N, n_samples)
    perm = tf.random.shuffle(tf.range(N))[:n]
    batch = tf.gather(states, perm, axis=0)
    k = batch[:, 0:1]
    b = batch[:, 1:2]
    z = batch[:, 2:3]
    return compute_residuals(model, k, b, z, n_mc_rhs=n_mc_rhs)


def plot_training_diagnostics(
    epochs: int,
    warmup_epochs: int,
    loss_history,
    bell_history,
    foc_history,
    oos_history,
) -> None:
    """
    Plot training losses and out-of-sample Bellman errors.

    This reproduces the same figures and scales as the original training script.
    """
    epochs_arr = np.arange(epochs)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, loss_history, label="Total loss")
    plt.yscale("log")
    plt.axvline(warmup_epochs, color="r", linestyle="--", label="End warm-up")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training losses")
    plt.legend()

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


def simulate_ergodic_panel(
    model,
    params: RiskyDebtParams,
    train_params: RiskyDebtTrainingParams,
    n_firms: int = 256,
    T: int = 400,
    burn_in: int = 100,
    k0_min: float = 0.5,
    k0_max: float = 2.0,
):
    """
    Simulate a panel of firms under the learned policy to approximate the ergodic distribution.

    Default is triggered only when C_next < 0 and b_prime > 0.

    Args:
        model: Trained RiskyDebtModel.
        params: RiskyDebtParams (used for state clipping bounds).
        train_params: RiskyDebtTrainingParams (used for z clipping bounds).
        n_firms: Number of simulated firms.
        T: Total simulation length.
        burn_in: Burn-in periods (not recorded).
        k0_min: Lower bound for new-firm initial capital.
        k0_max: Upper bound for new-firm initial capital.

    Returns:
        (k_panel, b_panel, z_panel) as NumPy arrays.
    """

    def sample_new_firms(n: int) -> tf.Tensor:
        """Sample fresh firms used for initialization and restarts after default."""
        k0 = tf.random.uniform((n, 1), k0_min, k0_max, dtype=DTYPE)
        b0 = tf.zeros_like(k0, dtype=DTYPE)
        z0 = tf.ones((n, 1), dtype=DTYPE)
        return tf.concat([k0, b0, z0], axis=1)

    states = sample_new_firms(n_firms)
    states_list = []

    clip_min = tf.constant(
        [params.k_cov_min, params.b_cov_min, train_params.z_clip_min], dtype=DTYPE
    )
    clip_max = tf.constant(
        [params.k_cov_max, params.b_cov_max, train_params.z_clip_max], dtype=DTYPE
    )

    total_defaults = 0

    for t in range(T):
        k = states[:, 0:1]
        b = states[:, 1:2]
        z = states[:, 2:3]

        policy_input = model.state_features(k, b, z)

        # Next-period policy
        k_prime, b_prime = model.policy_net(policy_input)
        z_next = model.get_next_z(z, 1)

        states_next = tf.concat([k_prime, b_prime, z_next], axis=1)
        states_next = tf.clip_by_value(states_next, clip_min, clip_max)

        # Equity raw value next period
        C_next, _ = model.compute_equity_value(
            k_prime, b_prime, z_next, use_target=False
        )

        # Default only if equity raw value is negative AND there is positive debt
        has_debt = b_prime > model.eps_b
        default_mask = tf.logical_and(C_next < 0.0, has_debt)

        num_defaults = int(
            tf.reduce_sum(tf.cast(default_mask, tf.int32)).numpy()
        )
        total_defaults += num_defaults

        # Restart defaulted firms
        restart_states = sample_new_firms(n_firms)
        states = tf.where(default_mask, restart_states, states_next)

        if t >= burn_in:
            states_list.append(states.numpy())

    print(
        f"Total defaults over simulation: {total_defaults}, "
        f"per firm per period ≈ {total_defaults / (n_firms * T):.4e}"
    )

    states_all = np.vstack(states_list)
    k_panel = states_all[:, 0:1]
    b_panel = states_all[:, 1:2]
    z_panel = states_all[:, 2:3]
    return k_panel, b_panel, z_panel


def run_final_diagnostics_and_plots(
    model,
    params: RiskyDebtParams,
    train_params: RiskyDebtTrainingParams,
    final_test_params: RiskyDebtFinalTestParams,
) -> None:
    """
    Run final coverage/on-policy diagnostics and produce the full set of plots.

    The printouts and plots are produced in the same order as the training script.
    """
    print("\n========== Final coverage diagnostics ==========")
    cov_final = test_coverage_residuals(
        model,
        n_samples=final_test_params.n_coverage,
        n_mc_rhs=final_test_params.n_mc_rhs,
    )
    for key, val in cov_final.items():
        print(f"{key:>15s}: {val:.4e}")

    k_panel, b_panel, z_panel = simulate_ergodic_panel(
        model,
        params=params,
        train_params=train_params,
        n_firms=final_test_params.ergodic_n_firms,
        T=final_test_params.ergodic_T,
        burn_in=final_test_params.ergodic_burn_in,
        k0_min=final_test_params.ergodic_k0_min,
        k0_max=final_test_params.ergodic_k0_max,
    )

    print("\n========== Final on-policy (ergodic) diagnostics ==========")
    states_erg_np = np.concatenate([k_panel, b_panel, z_panel], axis=1)
    states_erg = tf.convert_to_tensor(states_erg_np, dtype=DTYPE)
    onp_final = test_onpolicy_residuals(
        model,
        states_erg,
        n_samples=final_test_params.n_onpolicy,
        n_mc_rhs=final_test_params.n_mc_rhs,
    )
    for key, val in onp_final.items():
        print(f"{key:>15s}: {val:.4e}")

    # Ergodic leverage statistics
    k_safe_np = np.maximum(k_panel, 1e-6)
    leverage = b_panel / k_safe_np

    print("\n========== Ergodic leverage statistics (b/k) ==========")
    print(f"Number of observations: {leverage.size}")
    print(f"Mean leverage          : {np.mean(leverage):.4f}")
    print(f"Median leverage        : {np.median(leverage):.4f}")
    print(f"Std of leverage        : {np.std(leverage):.4f}")
    print(f"10th percentile        : {np.percentile(leverage,10):.4f}")
    print(f"50th percentile        : {np.percentile(leverage,50):.4f}")
    print(f"90th percentile        : {np.percentile(leverage,90):.4f}")
    print(f"Max leverage           : {np.max(leverage):.4f}")

    # Default probability distribution on ergodic panel
    print("\nComputing default probability distribution on ergodic panel...")
    batch_size_stats = final_test_params.stats_batch_size
    n_obs = k_panel.shape[0]
    pd_list = []

    for start in range(0, n_obs, batch_size_stats):
        end = min(start + batch_size_stats, n_obs)
        k_batch = tf.convert_to_tensor(k_panel[start:end, :], dtype=DTYPE)
        b_batch = tf.convert_to_tensor(b_panel[start:end, :], dtype=DTYPE)
        z_batch = tf.convert_to_tensor(z_panel[start:end, :], dtype=DTYPE)

        p_default_batch = model.default_probability(
            k_batch,
            b_batch,
            z_batch,
            n_draws=final_test_params.default_prob_draws,
            use_target=True,
        )
        pd_list.append(p_default_batch.numpy())

    pd_all = np.vstack(pd_list)

    print(
        "\n========== Ergodic default probability statistics (one-step-ahead) =========="
    )
    print(f"Mean default probability : {np.mean(pd_all):.4f}")
    print(f"Median default probability: {np.median(pd_all):.4f}")
    print(f"Std of default probability: {np.std(pd_all):.4f}")
    print(f"10th percentile           : {np.percentile(pd_all,10):.4f}")
    print(f"50th percentile           : {np.percentile(pd_all,50):.4f}")
    print(f"90th percentile           : {np.percentile(pd_all,90):.4f}")
    print(f"Max default probability   : {np.max(pd_all):.4f}")

    # Histograms of leverage and default probability
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(leverage, bins=final_test_params.hist_bins, density=True, alpha=0.7, color="tab:blue")
    plt.xlabel("Leverage b/k")
    plt.ylabel("Density")
    plt.title("Ergodic distribution of leverage (b/k)")

    plt.subplot(1, 2, 2)
    plt.hist(pd_all, bins=final_test_params.hist_bins, density=True, alpha=0.7, color="tab:red")
    plt.xlabel("One-step default probability")
    plt.ylabel("Density")
    plt.title("Ergodic distribution of default probability")

    plt.tight_layout()
    plt.show()

    # Policy/value slices on (k, b) grid at fixed z
    print(f"\nGenerating policy/value slices on (k,b) grid (z={final_test_params.grid_z_fixed})...")
    k_vals = np.linspace(params.k_cov_min, params.k_cov_max, final_test_params.grid_k_points)
    b_vals = np.linspace(params.b_cov_min, params.b_cov_max, final_test_params.grid_b_points)
    K_grid, B_grid = np.meshgrid(k_vals, b_vals, indexing="xy")

    k_flat = K_grid.reshape(-1, 1).astype(np.float32)
    b_flat = B_grid.reshape(-1, 1).astype(np.float32)
    z_flat = float(final_test_params.grid_z_fixed) * np.ones_like(k_flat, dtype=np.float32)

    k_tf = tf.convert_to_tensor(k_flat, dtype=DTYPE)
    b_tf = tf.convert_to_tensor(b_flat, dtype=DTYPE)
    z_tf = tf.convert_to_tensor(z_flat, dtype=DTYPE)

    # Equity value
    _, V_tf = model.compute_equity_value(k_tf, b_tf, z_tf, use_target=False)
    V_np = V_tf.numpy().reshape(len(b_vals), len(k_vals))

    # Default probability grid
    pd_grid_tf = model.default_probability(
        k_tf, b_tf, z_tf, n_draws=final_test_params.default_prob_draws, use_target=True
    )
    pd_grid_np = pd_grid_tf.numpy().reshape(len(b_vals), len(k_vals))

    # Policy inputs (log k, b/k, log z) via NumPy for plotting convenience
    k_safe_flat = np.maximum(k_flat, 1e-3)
    norm_k_flat = np.log(k_safe_flat)
    norm_b_flat = b_flat / k_safe_flat
    z_safe_flat = np.maximum(z_flat, 1e-6)
    log_z_flat = np.log(z_safe_flat)

    policy_input_np = np.concatenate(
        [norm_k_flat, norm_b_flat, log_z_flat], axis=1
    )
    policy_input_tf = tf.convert_to_tensor(policy_input_np, dtype=DTYPE)

    # Policy k', b'
    k_prime_tf, b_prime_tf = model.policy_net(policy_input_tf)
    k_prime_np = k_prime_tf.numpy()
    b_prime_np = b_prime_tf.numpy()

    k_prime_safe = np.maximum(k_prime_np, 1e-6)
    leverage_next_np = (b_prime_np / k_prime_safe).reshape(
        len(b_vals), len(k_vals)
    )

    # Choose a few representative b-levels to plot slices
    b_indices = list(final_test_params.grid_b_indices)

    # Figure 1: Equity value, default probability, and next-period leverage slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # (a) Equity value slices
    for idx in b_indices:
        axes[0].plot(
            k_vals,
            V_np[idx, :],
            label=f"b={b_vals[idx]:.2f}",
        )
    axes[0].set_xlabel("Capital k")
    axes[0].set_ylabel("Equity value V")
    axes[0].set_title(f"Equity value V(k,b,z={final_test_params.grid_z_fixed}) for selected b")
    axes[0].legend(fontsize=8)

    # (b) Default probability slices
    for idx in b_indices:
        axes[1].plot(
            k_vals,
            pd_grid_np[idx, :],
            label=f"b={b_vals[idx]:.2f}",
        )
    axes[1].set_xlabel("Capital k")
    axes[1].set_ylabel("One-step default probability")
    axes[1].set_title(f"Default probability(k,b,z={final_test_params.grid_z_fixed}) for selected b")
    axes[1].legend(fontsize=8)

    # (c) Next-period leverage slices
    for idx in b_indices:
        axes[2].plot(
            k_vals,
            leverage_next_np[idx, :],
            label=f"b={b_vals[idx]:.2f}",
        )
    axes[2].set_xlabel("Capital k")
    axes[2].set_ylabel("Next-period leverage b'/k'")
    axes[2].set_title("Next-period leverage b'/k' for selected b")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # State-dependent default regions on (k, b) grid at fixed z
    # ------------------------------------------------------------------
    print(
        f"\nGenerating state-dependent default regions on (k,b) grid (z={final_test_params.grid_z_fixed})..."
    )

    fig_reg, ax_reg = plt.subplots(figsize=(7, 5))

    # Filled contour of default probability over (k, b)
    orig_min = float(pd_grid_np.min())
    orig_max = float(pd_grid_np.max())

    # Start from the true data range
    vmin, vmax = orig_min, orig_max
    n_levels = max(2, final_test_params.default_region_n_levels)

    # If the surface is (almost) constant, slightly widen the range so that
    # contourf gets strictly increasing levels and does not crash.
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmin == 0.0 else 1e-6 * abs(vmin)
        vmin -= eps
        vmax += eps

    levels_prob = np.linspace(vmin, vmax, n_levels)
    cs = ax_reg.contourf(K_grid, B_grid, pd_grid_np, levels=levels_prob, cmap="Reds")
    cbar = plt.colorbar(cs, ax=ax_reg)
    cbar.set_label("One-step default probability")

    # Overlay selected probability contours to delineate "default regions"
    contour_levels = [
        lev
        for lev in final_test_params.default_region_contour_levels
        if orig_min < lev < orig_max
    ]
    contour_levels = sorted(set(contour_levels))  # ensure increasing & unique

    if contour_levels:
        cs_lines = ax_reg.contour(
            K_grid,
            B_grid,
            pd_grid_np,
            levels=contour_levels,
            colors="k",
            linewidths=0.8,
        )
        ax_reg.clabel(
            cs_lines,
            fmt={lev: f"{lev:.2f}" for lev in contour_levels},
            fontsize=8,
        )

    ax_reg.set_xlabel("Capital k")
    ax_reg.set_ylabel("Debt b")
    ax_reg.set_title(
        f"State-dependent default regions (z = {final_test_params.grid_z_fixed})"
    )

    plt.tight_layout()
    plt.show()