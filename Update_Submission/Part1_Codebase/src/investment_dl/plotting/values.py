"""Value-function plotting diagnostics for DL-vs-VFI comparisons.

These visuals compare levels and signed differences in policy-implied values
over the same state grid used for VFI benchmarking.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def _to_tensor(x, dtype=None) -> tf.Tensor:
    """Convert input data to a tensor, with optional dtype coercion.

    This helper keeps plotting functions tolerant to either NumPy arrays or
    TensorFlow tensors passed by callers.
    """
    t = tf.convert_to_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = tf.cast(t, dtype)
    return t


def _to_numpy(x):
    """Convert plotting inputs to NumPy arrays when needed.

    TensorFlow tensors are converted eagerly; all other array-like objects are
    returned unchanged.
    """
    return x.numpy() if tf.is_tensor(x) else x


def plot_value_heatmaps(
    k_grid: tf.Tensor,
    z_grid: tf.Tensor,
    V_vfi: tf.Tensor,
    V_dl: tf.Tensor,
    savepath: str | None = None,
) -> None:
    """Plot VFI and DL-policy values side-by-side with a difference panel.

    The two value panels share the same color limits so absolute levels remain
    directly comparable; the third panel uses a symmetric diverging scale.
    """
    k_np = _to_numpy(k_grid)
    z_np = _to_numpy(z_grid)
    V_vfi_t = _to_tensor(V_vfi, dtype=tf.float64)
    V_dl_t = _to_tensor(V_dl, dtype=tf.float64)
    diff_t = V_dl_t - V_vfi_t

    # Shared range for absolute-value panels.
    vmin = float(
        tf.reduce_min(tf.stack([tf.reduce_min(V_vfi_t), tf.reduce_min(V_dl_t)])).numpy(),
    )
    vmax = float(
        tf.reduce_max(tf.stack([tf.reduce_max(V_vfi_t), tf.reduce_max(V_dl_t)])).numpy(),
    )
    # Symmetric range around zero for signed difference panel.
    diff_max = float(tf.maximum(tf.reduce_max(tf.abs(diff_t)), 1e-12).numpy())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        _to_numpy(V_vfi_t),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("VFI value: V(k,z)")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("k")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        _to_numpy(V_dl_t),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("DL policy value: V^pi(k,z)")
    axes[1].set_xlabel("z")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        _to_numpy(diff_t),
        origin="lower",
        extent=(z_np[0], z_np[-1], k_np[0], k_np[-1]),
        aspect="auto",
        cmap="bwr",
        vmin=-diff_max,
        vmax=diff_max,
    )
    axes[2].set_title("Difference (DL - VFI)")
    axes[2].set_xlabel("z")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
