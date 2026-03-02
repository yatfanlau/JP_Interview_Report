"""Policy-visualization helpers for grid and slice comparisons.

These plots are used to inspect where learned policies align with, or deviate
from, the VFI benchmark across the state space.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def plot_policy_heatmaps(
    k_grid: tf.Tensor,
    z_grid: tf.Tensor,
    iota_vfi: tf.Tensor,
    iota_dl: tf.Tensor,
    savepath: str | None = None,
) -> None:
    """Plot VFI and DL policy heatmaps plus their difference map.

    The third panel is a direct subtraction (DL - VFI), which helps identify
    where state-contingent deviations are concentrated.
    """
    k_np = k_grid.numpy()
    z_np = z_grid.numpy()
    iota_vfi_t = tf.convert_to_tensor(iota_vfi)
    iota_dl_t = tf.convert_to_tensor(iota_dl)
    diff_t = iota_dl_t - iota_vfi_t

    # Shared axes keep spatial interpretation identical across panels.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        iota_vfi_t.numpy(),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
    )
    axes[0].set_title("VFI policy: iota(k,z)")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("k")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        iota_dl_t.numpy(),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
    )
    axes[1].set_title("DL policy: iota(k,z)")
    axes[1].set_xlabel("z")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        diff_t.numpy(),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
        cmap="bwr",
    )
    axes[2].set_title("Difference (DL - VFI)")
    axes[2].set_xlabel("z")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy_slice(
    k_grid: tf.Tensor,
    z_grid: tf.Tensor,
    iota_vfi: tf.Tensor,
    iota_dl: tf.Tensor,
    z_index: int | None = None,
    savepath: str | None = None,
) -> None:
    """Plot a one-dimensional ``k`` slice at a fixed productivity index.

    This view is useful for checking monotonicity and local slope differences
    that may be hard to spot in full heatmaps.
    """
    if z_index is None:
        z_index = z_grid.shape[0] // 2

    k_np = k_grid.numpy()
    iota_vfi_np = tf.convert_to_tensor(iota_vfi[:, z_index]).numpy()
    iota_dl_np = tf.convert_to_tensor(iota_dl[:, z_index]).numpy()
    z_val = float(z_grid[z_index].numpy())

    plt.figure(figsize=(6, 4))
    plt.plot(k_np, iota_vfi_np, label="VFI", lw=2)
    plt.plot(k_np, iota_dl_np, "--", label="DL", lw=2)
    plt.xlabel("k")
    plt.ylabel(f"iota(k, z={z_val:.3f})")
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
