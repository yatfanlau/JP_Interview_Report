"""Regime-map visualizations for bin-level policy diagnostics.

The figures here highlight local differences across discretized regions of
the ``(k,z)`` state space rather than only global averages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def _to_numpy(x):
    """Return NumPy-compatible data for Matplotlib rendering.

    TensorFlow tensors are converted eagerly; non-tensor inputs pass through.
    """
    return x.numpy() if tf.is_tensor(x) else x


def plot_regime_maps(
    map_vfi: tf.Tensor,
    map_dl: tf.Tensor,
    k_edges: tf.Tensor,
    z_edges: tf.Tensor,
    savepath: str | None = None,
) -> None:
    """Plot bin-level mean maps for VFI, DL, and their signed difference.

    The first two panels show levels; the third highlights where DL allocates
    relatively more or less investment than VFI within each regime cell.
    """
    del k_edges, z_edges
    map_vfi_np = _to_numpy(map_vfi)
    map_dl_np = _to_numpy(map_dl)
    diff = map_dl_np - map_vfi_np

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

    im0 = axes[0].imshow(map_vfi_np, origin="lower", aspect="auto")
    axes[0].set_title("Regime map (VFI): mean iota")
    axes[0].set_xlabel("z-bin")
    axes[0].set_ylabel("k-bin")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(map_dl_np, origin="lower", aspect="auto")
    axes[1].set_title("Regime map (DL): mean iota")
    axes[1].set_xlabel("z-bin")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="bwr")
    axes[2].set_title("Regime map diff (DL - VFI)")
    axes[2].set_xlabel("z-bin")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_regime_diagnostics(
    mean_vfi: tf.Tensor,
    mean_dl: tf.Tensor,
    std_vfi: tf.Tensor,
    std_dl: tf.Tensor,
    share_vfi: tf.Tensor,
    share_dl: tf.Tensor,
    savepath: str | None = None,
) -> None:
    """Plot mean, std, and occupancy-share diagnostics on a 3x3 grid.

    Rows correspond to statistics (mean/std/share), while columns compare VFI,
    DL, and the signed difference ``DL - VFI``.
    """
    mean_vfi_np = _to_numpy(mean_vfi)
    mean_dl_np = _to_numpy(mean_dl)
    std_vfi_np = _to_numpy(std_vfi)
    std_dl_np = _to_numpy(std_dl)
    share_vfi_np = _to_numpy(share_vfi)
    share_dl_np = _to_numpy(share_dl)

    # Difference panels use the same sign convention throughout.
    diff_mean = mean_dl_np - mean_vfi_np
    diff_std = std_dl_np - std_vfi_np
    diff_share = share_dl_np - share_vfi_np

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)

    im0 = axes[0, 0].imshow(mean_vfi_np, origin="lower", aspect="auto")
    axes[0, 0].set_title("Mean iota (VFI)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(mean_dl_np, origin="lower", aspect="auto")
    axes[0, 1].set_title("Mean iota (DL)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(diff_mean, origin="lower", aspect="auto", cmap="bwr")
    axes[0, 2].set_title("Mean diff (DL - VFI)")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(std_vfi_np, origin="lower", aspect="auto")
    axes[1, 0].set_title("Std iota (VFI)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(std_dl_np, origin="lower", aspect="auto")
    axes[1, 1].set_title("Std iota (DL)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(diff_std, origin="lower", aspect="auto", cmap="bwr")
    axes[1, 2].set_title("Std diff (DL - VFI)")
    plt.colorbar(im5, ax=axes[1, 2])

    im6 = axes[2, 0].imshow(share_vfi_np, origin="lower", aspect="auto")
    axes[2, 0].set_title("Regime share (VFI)")
    plt.colorbar(im6, ax=axes[2, 0])

    im7 = axes[2, 1].imshow(share_dl_np, origin="lower", aspect="auto")
    axes[2, 1].set_title("Regime share (DL)")
    plt.colorbar(im7, ax=axes[2, 1])

    im8 = axes[2, 2].imshow(diff_share, origin="lower", aspect="auto", cmap="bwr")
    axes[2, 2].set_title("Share diff (DL - VFI)")
    plt.colorbar(im8, ax=axes[2, 2])

    for ax in axes[-1, :]:
        ax.set_xlabel("z-bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("k-bin")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
