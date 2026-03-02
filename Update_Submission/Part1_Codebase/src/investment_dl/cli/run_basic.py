"""End-to-end CLI pipeline for DL-vs-VFI benchmarking on the basic model.

The script wires together model setup, DL training, VFI solution, multiple
diagnostic blocks, and plotting outputs used in project reports.
"""

from __future__ import annotations

import argparse

from investment_dl.config import BasicModelParams
from investment_dl.core.math_utils import tf_quantile_1d
from investment_dl.core.stochastic import tauchen_ln_z_grid
from investment_dl.core.tf_env import tf
from investment_dl.evaluation.euler_residuals import EulerResidualEvaluator
from investment_dl.evaluation.panels import compute_panel_moments
from investment_dl.evaluation.regimes import compute_regime_stats
from investment_dl.evaluation.values import PolicyValueEvaluator
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.plotting.distributions import plot_distribution
from investment_dl.plotting.moments import plot_moment_comparison
from investment_dl.plotting.policies import plot_policy_heatmaps, plot_policy_slice
from investment_dl.plotting.regimes import plot_regime_diagnostics
from investment_dl.plotting.training import plot_dl_convergence, plot_vfi_convergence
from investment_dl.plotting.values import plot_value_heatmaps
from investment_dl.simulation.simulators import PolicySimulator
from investment_dl.solvers.euler_dl import EulerEquationTrainer
from investment_dl.solvers.vfi import VFIInterpolatedPolicy, VFISolver


def run_experiment(
    n_k: int = 1001,
    n_z: int = 81,
    k_min_mul: float = 0.5,
    k_max_mul: float = 2.5,
) -> None:
    """Run the full DL-vs-VFI benchmark pipeline.

    The execution order is:
    1) initialize model and train DL policy,
    2) solve VFI benchmark and compare policies on-grid,
    3) evaluate Euler/Bellman diagnostics,
    4) compare panel and stationary-distribution moments,
    5) build regime-map diagnostics.
    """
    model = BasicInvestmentModel(BasicModelParams())

    print("Basic model parameters:")
    print(model.params)
    print(f"Steady-state capital (no adj. cost): k* = {model.k_star:.6f}")
    print(
        "Policy output bounds for iota=I/k: "
        f"[{model.iota_min:.4f}, {model.iota_max:.4f}]",
    )

    # Train the neural policy with Euler-equation residual minimization.
    trainer = EulerEquationTrainer(model)
    history = trainer.train(return_history=True)

    # Solve benchmark dynamic program on a dense discrete grid.
    vfi_solver = VFISolver(model)
    k_grid, z_grid, V_vfi, iota_vfi, vfi_hist = vfi_solver.solve(
        n_k=n_k,
        n_z=n_z,
        k_min_mul=k_min_mul,
        k_max_mul=k_max_mul,
        max_iter=1000,
        tol=1e-5,
        verbose=True,
        return_history=True,
    )

    # Evaluate learned policy on the same grid used by VFI.
    dl_policy_fn = trainer.make_policy_function()
    iota_dl = vfi_solver.evaluate_policy_on_grid(k_grid, z_grid, dl_policy_fn)

    # Policy comparison plots (surface and selected slices).
    plot_policy_heatmaps(k_grid, z_grid, iota_vfi, iota_dl, savepath="policy_comparison.png")
    plot_policy_slice(k_grid, z_grid, iota_vfi, iota_dl, z_index=0, savepath="policy_slice_lowz.png")
    plot_policy_slice(
        k_grid,
        z_grid,
        iota_vfi,
        iota_dl,
        z_index=z_grid.shape[0] // 2,
        savepath="policy_slice_midz.png",
    )
    plot_policy_slice(
        k_grid,
        z_grid,
        iota_vfi,
        iota_dl,
        z_index=z_grid.shape[0] - 1,
        savepath="policy_slice_highz.png",
    )

    # Convergence history plots for training diagnostics.
    if history is not None:
        plot_dl_convergence(history, savepath="dl_convergence.png")
    plot_vfi_convergence(vfi_hist, savepath="vfi_convergence.png")

    # Pointwise policy-gap summary on the VFI grid.
    policy_diff = iota_dl - iota_vfi
    rmse = float(tf.sqrt(tf.reduce_mean(tf.square(policy_diff))).numpy())
    mae = float(tf.reduce_mean(tf.abs(policy_diff)).numpy())
    max_abs = float(tf.reduce_max(tf.abs(policy_diff)).numpy())
    print(f"[Policy diff] RMSE={rmse:.4e}, MAE={mae:.4e}, max|diff|={max_abs:.4e}")

    # Euler-equation residual diagnostics on all (k,z) grid points.
    euler_eval = EulerResidualEvaluator(model)
    vfi_interp = VFIInterpolatedPolicy(k_grid, z_grid, iota_vfi)

    K, Z = tf.meshgrid(k_grid, z_grid, indexing="ij")
    k_flat = tf.reshape(K, [-1])
    z_flat = tf.reshape(Z, [-1])

    stats_dl = euler_eval.stats(dl_policy_fn, k_flat, z_flat, gh_nodes=7)
    stats_vfi = euler_eval.stats(vfi_interp, k_flat, z_flat, gh_nodes=7)
    print(
        "[Euler residuals DL]  "
        f"RMSE={stats_dl['rmse']:.4e}, MAE={stats_dl['mae']:.4e}, max={stats_dl['max_abs']:.4e}",
    )
    print(
        "[Euler residuals VFI] "
        f"RMSE={stats_vfi['rmse']:.4e}, MAE={stats_vfi['mae']:.4e}, max={stats_vfi['max_abs']:.4e}",
    )

    # Value/Bellman diagnostics use Tauchen transitions consistent with VFI.
    z_grid_eval, p_z = tauchen_ln_z_grid(model.params, n_z=z_grid.shape[0], m_std=3.0)
    z_diff = tf.reduce_max(tf.abs(z_grid_eval - z_grid))
    if float(z_diff.numpy()) > 1e-5:
        print("[Value eval] WARNING: z_grid mismatch vs Tauchen grid; using VFI grid.")

    value_eval = PolicyValueEvaluator(model)
    V_dl = value_eval.evaluate_policy_value_on_grid(
        k_grid,
        z_grid,
        iota_dl,
        p_z,
        max_iter=1000,
        tol=1e-5,
        init_V=V_vfi,
    )
    plot_value_heatmaps(k_grid, z_grid, V_vfi, V_dl, savepath="value_comparison.png")

    # Compare Bellman residuals for benchmark and learned policies.
    bell_vfi = value_eval.compute_bellman_residual_on_grid(
        k_grid,
        z_grid,
        iota_vfi,
        V_vfi,
        p_z,
    )
    bell_dl = value_eval.compute_bellman_residual_on_grid(
        k_grid,
        z_grid,
        iota_dl,
        V_dl,
        p_z,
    )

    stats_bell_vfi = value_eval.bellman_residual_stats(bell_vfi)
    stats_bell_dl = value_eval.bellman_residual_stats(bell_dl)
    print(
        "[Bellman residuals VFI] "
        f"RMSE={stats_bell_vfi['rmse']:.4e}, "
        f"MAE={stats_bell_vfi['mae']:.4e}, "
        f"max={stats_bell_vfi['max_abs']:.4e}",
    )
    print(
        "[Bellman residuals DL ] "
        f"RMSE={stats_bell_dl['rmse']:.4e}, "
        f"MAE={stats_bell_dl['mae']:.4e}, "
        f"max={stats_bell_dl['max_abs']:.4e}",
    )

    # Additional aggregate test for systematic Bellman underperformance.
    abs_vfi = tf.abs(tf.reshape(bell_vfi, [-1]))
    abs_dl = tf.abs(tf.reshape(bell_dl, [-1]))
    eps = 1e-12
    mean_ratio = stats_bell_dl["mean_abs"] / max(stats_bell_vfi["mean_abs"], eps)
    p90_ratio = stats_bell_dl["p90_abs"] / max(stats_bell_vfi["p90_abs"], eps)
    share_larger = float(tf.reduce_mean(tf.cast(abs_dl > abs_vfi, tf.float32)).numpy())

    systematic_tol = 0.10
    systematic_ok = (
        mean_ratio <= 1.0 + systematic_tol
        and p90_ratio <= 1.0 + systematic_tol
        and share_larger <= 0.5 + systematic_tol
    )
    print(
        "[Bellman residual test] "
        f"mean_ratio={mean_ratio:.3f}, "
        f"p90_ratio={p90_ratio:.3f}, "
        f"share(|DL|>|VFI|)={share_larger:.3f} "
        f"-> {'PASS' if systematic_ok else 'FAIL'}",
    )

    # Panel simulation block for moments and autocorrelation diagnostics.
    simulator = PolicySimulator(model)

    panel_burn_in = 5000
    panel_T = 20000
    panel_paths = 256
    panel_lags = [1, 4]

    k_dl_panel, z_dl_panel, iota_dl_panel = simulator.simulate_panel(
        dl_policy_fn,
        burn_in_steps=panel_burn_in,
        T=panel_T,
        n_paths=panel_paths,
        seed=123,
    )
    k_vfi_panel, z_vfi_panel, iota_vfi_panel = simulator.simulate_panel(
        vfi_interp,
        burn_in_steps=panel_burn_in,
        T=panel_T,
        n_paths=panel_paths,
        seed=123,
    )

    # Keep only aggregate investment/output/capital in the panel comparison plot.
    moments_dl = compute_panel_moments(k_dl_panel, z_dl_panel, iota_dl_panel, model, panel_lags)
    moments_vfi = compute_panel_moments(k_vfi_panel, z_vfi_panel, iota_vfi_panel, model, panel_lags)
    moments_dl.pop("iota", None)
    moments_vfi.pop("iota", None)
    plot_moment_comparison(moments_vfi, moments_dl, panel_lags, savepath="panel_moments.png")

    # Flattened ergodic draws for unconditional distribution comparisons.
    burn_in = 5000
    T_sim = 50000
    n_paths = 512

    k_dl, z_dl, iota_dl_sim = simulator.simulate_sample(
        dl_policy_fn,
        burn_in_steps=burn_in,
        T=T_sim,
        n_paths=n_paths,
        seed=123,
    )
    k_vfi, z_vfi, iota_vfi_sim = simulator.simulate_sample(
        vfi_interp,
        burn_in_steps=burn_in,
        T=T_sim,
        n_paths=n_paths,
        seed=123,
    )

    plot_distribution(
        k_dl,
        k_vfi,
        xlabel="k",
        title="Stationary Distribution of k",
        savepath="dist_k.png",
    )
    plot_distribution(
        iota_dl_sim,
        iota_vfi_sim,
        xlabel="iota",
        title="Stationary Distribution of iota",
        savepath="dist_iota.png",
    )

    # Build equally populated (quantile) bins shared by both methods.
    k_all = tf.concat([k_dl, k_vfi], axis=0)
    z_all = tf.concat([z_dl, z_vfi], axis=0)
    q = tf.constant([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=tf.float64)
    k_edges = tf_quantile_1d(k_all, q)
    z_edges = tf_quantile_1d(z_all, q)

    # Regime statistics summarize bin-level means, dispersion, and occupancy.
    mean_dl_map, std_dl_map, counts_dl, share_dl = compute_regime_stats(
        k_dl,
        z_dl,
        iota_dl_sim,
        k_edges,
        z_edges,
    )
    mean_vfi_map, std_vfi_map, counts_vfi, share_vfi = compute_regime_stats(
        k_vfi,
        z_vfi,
        iota_vfi_sim,
        k_edges,
        z_edges,
    )

    if bool(tf.reduce_any(counts_dl == 0).numpy()) or bool(tf.reduce_any(counts_vfi == 0).numpy()):
        print("[Regime map] WARNING: empty bins detected; consider fewer bins or more simulation.")

    low_k_high_z_dl = float(mean_dl_map[0, -1].numpy())
    low_k_high_z_vfi = float(mean_vfi_map[0, -1].numpy())
    print(
        "[Regime map] low-k/high-z mean iota: "
        f"DL={low_k_high_z_dl:.4e}, VFI={low_k_high_z_vfi:.4e}, "
        f"diff={(low_k_high_z_dl - low_k_high_z_vfi):.4e}",
    )

    plot_regime_diagnostics(
        mean_vfi_map,
        mean_dl_map,
        std_vfi_map,
        std_dl_map,
        share_vfi,
        share_dl,
        savepath="regime_map.png",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI options for VFI grid density and capital-grid coverage.

    These arguments mainly control benchmark resolution and therefore runtime
    and accuracy tradeoffs in the VFI reference solution.
    """
    parser = argparse.ArgumentParser(description="Run the basic DL-vs-VFI experiment.")
    parser.add_argument("--n-k", type=int, default=1001, help="Number of k-grid points for VFI.")
    parser.add_argument("--n-z", type=int, default=81, help="Number of z-grid points for VFI.")
    parser.add_argument("--k-min-mul", type=float, default=0.5, help="Lower bound multiple of k*.")
    parser.add_argument("--k-max-mul", type=float, default=2.5, help="Upper bound multiple of k*.")
    return parser.parse_args()


def main() -> None:
    """Program entry point for terminal execution.

    This wrapper keeps argument parsing separate from the implementation in
    :func:`run_experiment`, which also makes programmatic reuse easier.
    """
    args = parse_args()
    run_experiment(
        n_k=args.n_k,
        n_z=args.n_z,
        k_min_mul=args.k_min_mul,
        k_max_mul=args.k_max_mul,
    )


if __name__ == "__main__":
    main()
