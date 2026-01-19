"""
Entry point for training and diagnostics of Part 1 models.

Default behavior reproduces the basic-model pipeline:
- train the deep-learning Euler-equation solver
- run the final Gauss–Hermite-based test
- produce all diagnostic plots

Optionally, we can run the risky-debt model from the same entry point.
"""

from __future__ import annotations

import argparse


def run_basic_model() -> None:
    """Train and diagnose the basic investment model (default pipeline)."""
    from basic_model import fp, train_model
    from basic_model_diagnostic import (
        final_test,
        plot_deterministic_convergence,
        plot_euler_residual_hist,
        plot_ergodic_cloud_with_box,
        plot_policy_heatmap,
    )

    train_model()
    final_test(fp)

    # Optional plots mirroring notebook execution.
    plot_deterministic_convergence(
        k0_list=None,
        T=100,
        z_fixed=1.0,
        save_path=None,
        show=True,
    )
    plot_policy_heatmap(
        n_k=120,
        n_z=120,
        m_minus=0.2,
        m_plus=5.0,
        z_std_range=2.5,
        cmap="viridis",
        save_path=None,
        show=True,
    )
    plot_euler_residual_hist(
        n_k=50,
        n_z=50,
        m_minus=0.5,
        m_plus=1.5,
        z_std_range=2.5,
        n_nodes=20,
        bins=60,
        save_path=None,
        show=True,
    )
    plot_ergodic_cloud_with_box()


def run_risky_debt_model() -> None:
    """Train and diagnose the risky-debt model."""
    from risky_debt_model import run_training
    from config_part1 import RiskyDebtParams, RiskyDebtTrainingParams, RiskyDebtFinalTestParams

    run_training(RiskyDebtParams(), RiskyDebtTrainingParams(), RiskyDebtFinalTestParams())


def main(argv=None) -> None:
    """CLI dispatcher for Part 1."""
    parser = argparse.ArgumentParser(description="Run Part 1 models.")
    parser.add_argument(
        "--model",
        choices=("basic", "risky", "both"),
        default="risky",
        help="Which model pipeline to run (default: basic).",
    )
    args = parser.parse_args(argv)

    if args.model == "basic":
        run_basic_model()
    elif args.model == "risky":
        run_risky_debt_model()
    else:
        run_basic_model()
        run_risky_debt_model()


if __name__ == "__main__":
    main()