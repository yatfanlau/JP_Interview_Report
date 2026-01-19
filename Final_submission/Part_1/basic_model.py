"""
Diagnostics, tests, and plotting utilities for the Basic Investment Model.

This module depends on `basic_model` for:

- Global parameters (mp, tp, fp)
- Trained policy network and TensorFlow constants
- Euler residual evaluation helpers
"""

import math
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config_part1 import BasicFinalTestParams
from utils_part1 import compute_stats, print_stats
from basic_model import (
    DTYPE,
    K_SS_TF,
    eval_batched,
    fp,
    mp,
    policy,
    policy_iota,
    policy_step,
    steady_state_k,
    tf,
    tp,
    euler_residuals_gh,  # useful for some plots
)


# ---------------------------------------------------------------------------
# On-policy ergodic simulation
# ---------------------------------------------------------------------------


def simulate_on_policy_sample(
    burn_in_steps: int,
    T: int,
    n_paths: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate an ergodic on-policy sample under the trained policy.

    Args:
        burn_in_steps: Number of steps discarded before collecting samples.
        T: Total number of (k, z) observations to return.
        n_paths: Number of parallel simulated paths. Defaults to `tp.n_paths`
            if None.

    Returns:
        A tuple (k_arr, z_arr), each an array of shape (T,) with simulated
        capital and productivity states.
    """
    if n_paths is None:
        n_paths = tp.n_paths

    # Start all paths at steady-state capital and normalized productivity.
    k = tf.ones((n_paths,), dtype=DTYPE) * K_SS_TF
    z = tf.ones((n_paths,), dtype=DTYPE)

    print(
        f"  Burn-in under policy: steps = {burn_in_steps}, paths = {n_paths} ...",
    )
    for _ in range(burn_in_steps):
        k, z, _ = policy_step(policy, k, z, training=False)

    # Number of simulation steps needed to collect at least T observations.
    steps_collect = math.ceil(T / n_paths)
    print(
        f"  Collecting on-policy states: total = {T}, via "
        f"{steps_collect} steps x {n_paths} paths ...",
    )
    collected_k: list[np.ndarray] = []
    collected_z: list[np.ndarray] = []

    for _ in range(steps_collect):
        collected_k.append(k.numpy())
        collected_z.append(z.numpy())
        k, z, _ = policy_step(policy, k, z, training=False)

    # Concatenate paths and truncate to exactly T observations.
    k_arr = np.concatenate(collected_k, axis=0)[:T].astype(np.float32)
    z_arr = np.concatenate(collected_z, axis=0)[:T].astype(np.float32)
    return k_arr, z_arr


# ---------------------------------------------------------------------------
# Coverage box, edge/corner sampling, GH robustness
# ---------------------------------------------------------------------------


def build_coverage_box_and_sample(
    lnk: np.ndarray,
    lnz: np.ndarray,
    q_low: float = 0.01,
    q_high: float = 0.99,
    expand: float = 0.05,
    M: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build a coverage box in log-space and draw uniform samples inside it.

    The box is defined by (ln k, ln z) quantiles of an ergodic sample and
    then slightly expanded. Uniform draws inside the box are mapped back
    to (k, z).

    Args:
        lnk: One-dimensional array of log capital samples.
        lnz: One-dimensional array of log productivity samples.
        q_low: Lower quantile for the coverage box.
        q_high: Upper quantile for the coverage box.
        expand: Fraction by which to widen the box on each side.
        M: Number of coverage sample points to draw.

    Returns:
        A tuple (k_cov, z_cov, box) where:
            k_cov: Array of coverage capital samples of length M.
            z_cov: Array of coverage productivity samples of length M.
            box: Dictionary with keys 'lnk_min', 'lnk_max', 'lnz_min',
                and 'lnz_max' describing the box in log-space.
    """
    lnk_q1, lnk_q99 = np.quantile(lnk, [q_low, q_high])
    lnz_q1, lnz_q99 = np.quantile(lnz, [q_low, q_high])

    def expand_bounds(a: float, b: float, frac: float) -> Tuple[float, float]:
        """Expand [a, b] symmetrically by a fraction of its width."""
        width = max(b - a, 1e-8)  # Avoid zero-width intervals.
        return a - frac * width, b + frac * width

    lnk_min, lnk_max = expand_bounds(lnk_q1, lnk_q99, expand)
    lnz_min, lnz_max = expand_bounds(lnz_q1, lnz_q99, expand)

    # Draw uniformly in (ln k, ln z), then exponentiate back to (k, z).
    lnk_u = np.random.uniform(lnk_min, lnk_max, size=M).astype(np.float32)
    lnz_u = np.random.uniform(lnz_min, lnz_max, size=M).astype(np.float32)

    k_cov = np.exp(lnk_u).astype(np.float32)
    z_cov = np.exp(lnz_u).astype(np.float32)

    box: Dict[str, float] = dict(
        lnk_min=float(lnk_min),
        lnk_max=float(lnk_max),
        lnz_min=float(lnz_min),
        lnz_max=float(lnz_max),
    )
    return k_cov, z_cov, box


def edge_and_corner_points(
    box: Dict[str, float],
    n_edge: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct edge and corner points on a coverage box in (ln k, ln z).

    Args:
        box: Dictionary with coverage bounds in log-space.
        n_edge: Number of grid points to place along each edge.

    Returns:
        A tuple (k_edge, z_edge) of arrays containing the exponentiated
        edge and corner points in levels.
    """
    lnk_min, lnk_max = box["lnk_min"], box["lnk_max"]
    lnz_min, lnz_max = box["lnz_min"], box["lnz_max"]

    lnk_grid = np.linspace(lnk_min, lnk_max, n_edge, dtype=np.float32)
    lnz_grid = np.linspace(lnz_min, lnz_max, n_edge, dtype=np.float32)

    # Exclude corners in edge segments to avoid duplicates; corners are added below.
    top = np.stack([lnk_grid, np.full_like(lnk_grid, lnz_max)], axis=1)[1:-1]
    bottom = np.stack([lnk_grid, np.full_like(lnk_grid, lnz_min)], axis=1)[1:-1]
    left = np.stack([np.full_like(lnz_grid, lnk_min), lnz_grid], axis=1)[1:-1]
    right = np.stack([np.full_like(lnz_grid, lnk_max), lnz_grid], axis=1)[1:-1]

    corners = np.array(
        [
            [lnk_min, lnz_min],
            [lnk_min, lnz_max],
            [lnk_max, lnz_min],
            [lnk_max, lnz_max],
        ],
        dtype=np.float32,
    )

    all_ln_pairs = np.concatenate([top, bottom, left, right, corners], axis=0)
    k_edge = np.exp(all_ln_pairs[:, 0]).astype(np.float32)
    z_edge = np.exp(all_ln_pairs[:, 1]).astype(np.float32)
    return k_edge, z_edge


def gh_robustness_check(
    k_np: np.ndarray,
    z_np: np.ndarray,
    base_nodes: int = 10,
    compare_nodes: Tuple[int, ...] = (15, 20),
    n_sub: int = 5000,
    batch_eval: int = 16_384,
) -> Dict[str, float]:
    """Compare Euler residuals across different Gauss–Hermite node counts.

    The function evaluates Euler residuals on a random subsample using a
    base number of GH nodes and several alternative node counts, and
    reports relative changes in key quantiles.

    Args:
        k_np: One-dimensional array of capital values.
        z_np: One-dimensional array of productivity values.
        base_nodes: Number of GH nodes for the baseline evaluation.
        compare_nodes: Tuple of alternative GH node counts to compare.
        n_sub: Maximum number of subsample points to evaluate.
        batch_eval: Batch size for `eval_batched`.

    Returns:
        A dictionary mapping metric names to relative changes in the 50th
        and 95th percentiles of |g| between each alternative and baseline.
    """
    N = len(k_np)
    n_sub = min(n_sub, N)
    idx = np.random.choice(N, size=n_sub, replace=False)
    k_sub = k_np[idx]
    z_sub = z_np[idx]

    # Baseline residuals with base_nodes GH nodes.
    g_base, _ = eval_batched(k_sub, z_sub, n_nodes=base_nodes, batch=batch_eval)
    abs_base = np.abs(g_base)
    base_p50 = np.quantile(abs_base, 0.5)
    base_p95 = np.quantile(abs_base, 0.95)
    eps = 1e-16  # Guard against division by zero.

    results: Dict[str, float] = {}
    for n in compare_nodes:
        g_cmp, _ = eval_batched(k_sub, z_sub, n_nodes=n, batch=batch_eval)
        abs_cmp = np.abs(g_cmp)
        cmp_p50 = np.quantile(abs_cmp, 0.5)
        cmp_p95 = np.quantile(abs_cmp, 0.95)

        rel50 = float(abs(cmp_p50 - base_p50) / max(base_p50, eps))
        rel95 = float(abs(cmp_p95 - base_p95) / max(base_p95, eps))
        results[f"GH{n}_vs_GH{base_nodes}_RelChange_P50"] = rel50
        results[f"GH{n}_vs_GH{base_nodes}_RelChange_P95"] = rel95
    return results


# ---------------------------------------------------------------------------
# Final Gauss–Hermite-based test
# ---------------------------------------------------------------------------


def final_test(params: BasicFinalTestParams) -> None:
    """Run a comprehensive Gauss–Hermite-based final evaluation.

    Args:
        params: Configuration object with final test hyperparameters.

    Returns:
        None
    """
    print("\n========================")
    print("Final Test: Begin")
    print("========================")

    # ------------------------------------------------------------------
    # 1. On-policy ergodic sample
    # ------------------------------------------------------------------
    t0 = time.time()
    k_ops, z_ops = simulate_on_policy_sample(
        burn_in_steps=params.burn_in_steps,
        T=params.T_on_policy,
        n_paths=tp.n_paths,
    )
    lnk_ops = np.log(k_ops)
    lnz_ops = np.log(z_ops)
    print(
        f"On-policy sample ready. N = {len(k_ops)}. "
        f"Time = {time.time() - t0:.2f} sec.",
    )

    # ------------------------------------------------------------------
    # 2. Coverage box and off-policy coverage sample
    # ------------------------------------------------------------------
    k_cov, z_cov, box = build_coverage_box_and_sample(
        lnk_ops,
        lnz_ops,
        q_low=params.q_low,
        q_high=params.q_high,
        expand=params.expand_frac,
        M=params.M_coverage,
    )
    print("Coverage box (in log-space):")
    print(
        f"  ln k: [{box['lnk_min']:.3f}, {box['lnk_max']:.3f}] | "
        f"ln z: [{box['lnz_min']:.3f}, {box['lnz_max']:.3f}]",
    )
    print(f"Coverage sample ready. M = {len(k_cov)}")

    # ------------------------------------------------------------------
    # 3. Euler residuals on on-policy and coverage sets
    # ------------------------------------------------------------------
    print(f"Evaluating Euler residuals with GH-{params.gh_nodes} ...")
    t1 = time.time()
    g_ops, den_ops = eval_batched(
        k_ops,
        z_ops,
        n_nodes=params.gh_nodes,
        batch=params.batch_eval,
    )
    g_cov, den_cov = eval_batched(
        k_cov,
        z_cov,
        n_nodes=params.gh_nodes,
        batch=params.batch_eval,
    )
    print(f"Evaluation time: {time.time() - t1:.2f} sec.")

    stats_ops = compute_stats(g_ops, den_ops, tol_list=params.tol_list)
    stats_cov = compute_stats(g_cov, den_cov, tol_list=params.tol_list)

    print_stats(
        f"On-policy test set (GH-{params.gh_nodes})",
        stats_ops,
        tol_list=params.tol_list,
    )
    print_stats(
        f"Coverage test set (GH-{params.gh_nodes})",
        stats_cov,
        tol_list=params.tol_list,
    )

    # ------------------------------------------------------------------
    # 4. Edge/corner stress test within the coverage box
    # ------------------------------------------------------------------
    print("\nEdge/Corner stress test on the coverage box ...")
    k_edge, z_edge = edge_and_corner_points(box, n_edge=params.edge_points)
    g_edge, den_edge = eval_batched(
        k_edge,
        z_edge,
        n_nodes=params.gh_nodes,
        batch=params.batch_eval,
    )
    stats_edge = compute_stats(g_edge, den_edge, tol_list=params.tol_list)
    print_stats(
        f"Stress test (box edges & corners) (GH-{params.gh_nodes})",
        stats_edge,
        tol_list=params.tol_list,
    )

    # ------------------------------------------------------------------
    # 5. Robustness to GH node count (on coverage set)
    # ------------------------------------------------------------------
    print(
        "\nGH-node robustness check (subsample): "
        f"compare GH-15/20 vs GH-{params.gh_nodes} on coverage set",
    )
    robust = gh_robustness_check(
        k_cov,
        z_cov,
        base_nodes=params.gh_nodes,
        compare_nodes=(15, 20),
        n_sub=min(5000, params.M_coverage),
        batch_eval=params.batch_eval,
    )
    for k, v in robust.items():
        print(f"- {k}: {v * 100:.1f}%")

    # Share of points with very small residuals in absolute value.
    share_cov_1e3 = float((np.abs(g_cov) <= 1e-3).mean())
    share_ops_1e3 = float((np.abs(g_ops) <= 1e-3).mean())

    # ------------------------------------------------------------------
    # 6. Informal pass/fail rules
    # ------------------------------------------------------------------
    cov_pass = (
        stats_cov["Abs_MAE"] <= 1e-3
        and stats_cov["Abs_P95"] <= 1e-2
        and share_cov_1e3 >= 0.95
    )
    ops_pass = (
        stats_ops["Abs_Median"] <= 1e-3
        and stats_ops["Abs_P95"] <= 1e-2
    )
    gh15_ok = (
        robust.get(f"GH15_vs_GH{params.gh_nodes}_RelChange_P50", 1.0) <= 0.20
        and robust.get(f"GH15_vs_GH{params.gh_nodes}_RelChange_P95", 1.0) <= 0.20
    )
    gh20_ok = (
        robust.get(f"GH20_vs_GH{params.gh_nodes}_RelChange_P50", 1.0) <= 0.20
        and robust.get(f"GH20_vs_GH{params.gh_nodes}_RelChange_P95", 1.0) <= 0.20
    )

    print("\nInformal pass/fail against suggested thresholds:")
    print(
        f"- Coverage set: {'PASS' if cov_pass else 'FAIL'} "
        f"(share(|E[g]|<=1e-3)={share_cov_1e3:.3f})",
    )
    print(f"- On-policy set: {'PASS' if ops_pass else 'FAIL'}")
    print(
        f"- GH-node robustness (vs {params.gh_nodes}): GH-15="
        f"{'PASS' if gh15_ok else 'FAIL'}, "
        f"GH-20={'PASS' if gh20_ok else 'FAIL'}",
    )

    print("\n========================")
    print("Final Test: End")
    print("========================\n")


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_deterministic_convergence(
    k0_list: list[float] | None = None,
    T: int = 100,
    z_fixed: float = 1.0,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot deterministic convergence paths of capital and investment rate.

    Convergence is computed for a fixed productivity level z_t ≡ z_fixed.

    Args:
        k0_list: Optional list of initial capital levels. If None, a
            default set around the deterministic steady state is used.
        T: Number of periods to simulate.
        z_fixed: Fixed productivity level used for all dates.
        save_path: Optional path to save the figure. If None, the figure
            is not saved.
        show: Whether to display the figure via `plt.show()`.

    Returns:
        None
    """
    # Deterministic steady-state capital (without adjustment costs).
    k_star_det = steady_state_k(mp.theta, mp.delta, mp.r)

    if k0_list is None:
        k0_list = [0.2 * k_star_det, 1.0 * k_star_det, 5.0 * k_star_det]

    z_val = float(z_fixed)
    delta = mp.delta

    all_k_paths: list[list[float]] = []
    all_iota_paths: list[list[float]] = []
    labels: list[str] = []

    for k0 in k0_list:
        k_t = float(k0)
        ks = [k_t]
        iotas: list[float] = []

        for _ in range(T):
            k_tf = tf.constant([k_t], dtype=DTYPE)
            z_tf = tf.constant([z_val], dtype=DTYPE)
            # Policy returns investment rate iota_t = I_t / k_t.
            iota_t = float(
                policy_iota(policy, k_tf, z_tf, training=False)[0].numpy(),
            )
            iotas.append(iota_t)
            # Capital law of motion under deterministic z_t.
            k_next = (1.0 - delta + iota_t) * k_t
            k_t = max(1e-12, k_next)  # Prevent degenerate zero or negative k.
            ks.append(k_t)

        all_k_paths.append(ks)
        all_iota_paths.append(iotas)
        labels.append(f"k0 = {k0:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    t_k = range(T + 1)
    for ks, label in zip(all_k_paths, labels, strict=True):
        axes[0].plot(t_k, ks, label=label)
    axes[0].axhline(
        k_star_det,
        color="k",
        linestyle="--",
        label="k* (no adj. cost)",
    )
    axes[0].set_ylabel("Capital $k_t$")
    axes[0].set_title(f"Deterministic convergence of $k_t$ (z_t ≡ {z_val})")
    axes[0].legend()

    t_iota = range(T)
    for iotas, label in zip(all_iota_paths, labels, strict=True):
        axes[1].plot(t_iota, iotas, label=label)
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel(r"Investment rate $\iota_t = I_t / k_t$")
    axes[1].set_title(
        f"Policy path $\\iota_t$ along deterministic transition (z_t ≡ {z_val})",
    )

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _build_k_z_grid(
    n_k: int,
    n_z: int,
    m_minus: float,
    m_plus: float,
    z_std_range: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Build a rectangular (k, ln z) grid and flattened arrays for evaluation.

    Args:
        n_k: Number of grid points for capital k.
        n_z: Number of grid points for log productivity ln z.
        m_minus: Lower multiple of deterministic steady-state k.
        m_plus: Upper multiple of deterministic steady-state k.
        z_std_range: Number of unconditional standard deviations of ln z
            to include on each side of its mean.

    Returns:
        A tuple (KK, LL, k_flat, k_min, k_max, lnz_min, lnz_max, z_flat) where:
            KK: 2D grid of k values with shape (n_z, n_k).
            LL: 2D grid of ln z values with shape (n_z, n_k).
            k_flat: Flattened k values, shape (n_z * n_k,).
            k_min: Minimum k in the grid.
            k_max: Maximum k in the grid.
            lnz_min: Minimum ln z in the grid.
            lnz_max: Maximum ln z in the grid.
            z_flat: Flattened z values (= exp(LL)), shape (n_z * n_k,).
    """
    k_star_det = steady_state_k(mp.theta, mp.delta, mp.r)
    k_min = m_minus * k_star_det
    k_max = m_plus * k_star_det

    # Unconditional distribution of ln z under AR(1) with lognormal shocks.
    sigma_ln_z = mp.sigma_eps / math.sqrt(1.0 - mp.rho * mp.rho)
    m_ln_z = -0.5 * (mp.sigma_eps**2) / (1.0 - mp.rho * mp.rho)
    lnz_min = m_ln_z - z_std_range * sigma_ln_z
    lnz_max = m_ln_z + z_std_range * sigma_ln_z

    k_grid = np.linspace(k_min, k_max, n_k, dtype=np.float32)
    lnz_grid = np.linspace(lnz_min, lnz_max, n_z, dtype=np.float32)
    KK, LL = np.meshgrid(k_grid, lnz_grid)

    # Flatten for batched evaluation, then exponentiate to recover z.
    k_flat = KK.ravel().astype(np.float32)
    z_flat = np.exp(LL.ravel().astype(np.float32)).astype(np.float32)

    return KK, LL, k_flat, k_min, k_max, lnz_min, lnz_max, z_flat


def plot_policy_heatmap(
    n_k: int = 100,
    n_z: int = 100,
    m_minus: float = 0.2,
    m_plus: float = 5.0,
    z_std_range: float = 2.5,
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a 2D heatmap of the policy function iota(k, z) = I/k.

    Args:
        n_k: Number of grid points for capital k.
        n_z: Number of grid points for log productivity ln z.
        m_minus: Lower multiple of deterministic steady-state k.
        m_plus: Upper multiple of deterministic steady-state k.
        z_std_range: Number of unconditional standard deviations of ln z
            on each side of its mean to cover.
        cmap: Matplotlib colormap name to use.
        save_path: Optional path to save the figure. If None, the figure
            is not saved.
        show: Whether to display the figure via `plt.show()`.

    Returns:
        None
    """
    KK, LL, k_flat, k_min, k_max, lnz_min, lnz_max, z_flat = _build_k_z_grid(
        n_k=n_k,
        n_z=n_z,
        m_minus=m_minus,
        m_plus=m_plus,
        z_std_range=z_std_range,
    )

    k_tf = tf.convert_to_tensor(k_flat, dtype=DTYPE)
    z_tf = tf.convert_to_tensor(z_flat, dtype=DTYPE)
    iota_flat = policy_iota(policy, k_tf, z_tf, training=False).numpy()
    iota_grid = iota_flat.reshape(KK.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        iota_grid,
        extent=[k_min, k_max, lnz_min, lnz_max],
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )
    ax.set_xlabel("Capital $k$")
    ax.set_ylabel(r"$\log z$")
    ax.set_title(r"Policy heatmap: $\iota(k, z) = I/k$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\iota = I / k$")

    # Draw horizontal line at ln z = 0 if the grid spans it.
    if lnz_min < 0.0 < lnz_max:
        ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_euler_residual_hist(
    n_k: int = 50,
    n_z: int = 50,
    m_minus: float = 0.5,
    m_plus: float = 1.5,
    z_std_range: float = 2.5,
    n_nodes: int = 20,
    bins: int = 60,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a histogram of Euler residuals g(k, z) on a rectangular grid.

    Args:
        n_k: Number of grid points for capital k.
        n_z: Number of grid points for log productivity ln z.
        m_minus: Lower multiple of deterministic steady-state k.
        m_plus: Upper multiple of deterministic steady-state k.
        z_std_range: Number of unconditional standard deviations of ln z
            on each side of its mean to cover.
        n_nodes: Number of Gauss–Hermite nodes used in residual evaluation.
        bins: Number of bins in the histogram.
        save_path: Optional path to save the figure. If None, the figure
            is not saved.
        show: Whether to display the figure via `plt.show()`.

    Returns:
        None
    """
    KK, LL, k_flat, _, _, _, _, z_flat = _build_k_z_grid(
        n_k=n_k,
        n_z=n_z,
        m_minus=m_minus,
        m_plus=m_plus,
        z_std_range=z_std_range,
    )

    # Only flattened values are used for evaluation.
    del KK, LL

    k_tf = tf.convert_to_tensor(k_flat, dtype=DTYPE)
    z_tf = tf.convert_to_tensor(z_flat, dtype=DTYPE)
    g_mean_tf, _, _ = euler_residuals_gh(k_tf, z_tf, n_nodes=n_nodes)
    g_np = g_mean_tf.numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(g_np, bins=bins, density=False, edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"Euler residual $g(k, z)$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Euler residuals on {n_k}×{n_z} grid (GH-{n_nodes})",
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0, label="0")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_ergodic_cloud_with_box(
    burn_in_steps: int | None = None,
    T: int | None = None,
    n_paths: int | None = None,
    q_low: float | None = None,
    q_high: float | None = None,
    expand_frac: float | None = None,
    M_coverage: int | None = None,
) -> None:
    """Plot ergodic on-policy cloud and coverage box in (ln k, ln z) space.

    Default arguments are taken from `fp` and `tp` if not provided.

    Args:
        burn_in_steps: Number of burn-in steps for the ergodic simulation.
        T: Total number of on-policy observations to simulate.
        n_paths: Number of parallel simulation paths.
        q_low: Lower quantile used to define the coverage box.
        q_high: Upper quantile used to define the coverage box.
        expand_frac: Fraction by which to widen quantile bounds.
        M_coverage: Number of coverage points used when constructing the box.

    Returns:
        None
    """
    if burn_in_steps is None:
        burn_in_steps = fp.burn_in_steps
    if T is None:
        T = fp.T_on_policy
    if n_paths is None:
        n_paths = tp.n_paths
    if q_low is None:
        q_low = fp.q_low
    if q_high is None:
        q_high = fp.q_high
    if expand_frac is None:
        expand_frac = fp.expand_frac
    if M_coverage is None:
        M_coverage = fp.M_coverage

    print(
        f"Simulating on-policy sample: burn-in={burn_in_steps}, "
        f"T={T}, paths={n_paths}",
    )
    k_ops, z_ops = simulate_on_policy_sample(
        burn_in_steps=burn_in_steps,
        T=T,
        n_paths=n_paths,
    )
    lnk_ops = np.log(k_ops.astype(np.float32))
    lnz_ops = np.log(z_ops.astype(np.float32))

    _, _, box = build_coverage_box_and_sample(
        lnk_ops,
        lnz_ops,
        q_low=q_low,
        q_high=q_high,
        expand=expand_frac,
        M=M_coverage,
    )

    print("Coverage box (log-space):")
    print(
        f"  ln k in [{box['lnk_min']:.3f}, {box['lnk_max']:.3f}], "
        f"ln z in [{box['lnz_min']:.3f}, {box['lnz_max']:.3f}]",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lnk_ops, lnz_ops, s=5, alpha=0.15, label="On-policy states")

    # Draw the box boundaries.
    ax.axvline(box["lnk_min"], color="red", linestyle="--")
    ax.axvline(box["lnk_max"], color="red", linestyle="--")
    ax.axhline(
        box["lnz_min"],
        color="red",
        linestyle="--",
        label="Coverage box",
    )
    ax.axhline(box["lnz_max"], color="red", linestyle="--")

    ax.set_xlabel(r"$\log k$")
    ax.set_ylabel(r"$\log z$")
    ax.set_title("Ergodic cloud and coverage box in $(\\log k, \\log z)$")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)