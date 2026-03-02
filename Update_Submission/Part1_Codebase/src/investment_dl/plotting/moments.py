"""Visualization of panel-moment comparisons between DL and VFI outputs.

The plotting layout is designed for quick side-by-side checks of levels,
dispersion, and persistence across selected state and flow variables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def plot_moment_comparison(
    moments_vfi: dict[str, dict],
    moments_dl: dict[str, dict],
    lags: list[int],
    savepath: str | None = None,
) -> None:
    """Plot grouped bars for means, variances, and selected autocorrelations.

    The first two panels always show mean/variance, and each requested lag
    adds one autocorrelation panel with the same variable ordering.
    """
    variables = list(moments_dl.keys())
    x = tf.range(len(variables), dtype=tf.float32).numpy()
    width = 0.35

    n_cols = 2 + len(lags)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Panel 1: levels.
    mean_dl = [moments_dl[v]["mean"] for v in variables]
    mean_vfi = [moments_vfi[v]["mean"] for v in variables]
    axes[0].bar(x - width / 2, mean_dl, width, label="DL")
    axes[0].bar(x + width / 2, mean_vfi, width, label="VFI")
    axes[0].set_title("Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variables)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    # Panel 2: dispersion.
    var_dl = [moments_dl[v]["var"] for v in variables]
    var_vfi = [moments_vfi[v]["var"] for v in variables]
    axes[1].bar(x - width / 2, var_dl, width, label="DL")
    axes[1].bar(x + width / 2, var_vfi, width, label="VFI")
    axes[1].set_title("Variance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variables)
    axes[1].grid(True, axis="y", alpha=0.3)

    for i, lag in enumerate(lags):
        # Remaining panels: persistence at each requested lag.
        ac_dl = [moments_dl[v]["ac"].get(lag, float("nan")) for v in variables]
        ac_vfi = [moments_vfi[v]["ac"].get(lag, float("nan")) for v in variables]
        ax = axes[2 + i]
        ax.bar(x - width / 2, ac_dl, width, label="DL")
        ax.bar(x + width / 2, ac_vfi, width, label="VFI")
        ax.set_title(f"Autocorr (lag {lag})")
        ax.set_xticks(x)
        ax.set_xticklabels(variables)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
