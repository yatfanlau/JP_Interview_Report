"""Distribution plotting helpers for one-dimensional simulated samples.

These utilities standardize histogram comparisons so DL and VFI distributions
can be contrasted with consistent visual settings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def _to_numpy(x):
    """Return a NumPy array view for plotting input ``x``.

    TensorFlow tensors are converted via ``.numpy()``, while existing NumPy
    arrays (or compatible array-like objects) are returned unchanged.
    """
    return x.numpy() if tf.is_tensor(x) else x


def plot_distribution(
    x_dl: tf.Tensor,
    x_vfi: tf.Tensor,
    xlabel: str,
    title: str,
    savepath: str | None = None,
) -> None:
    """Plot overlaid histograms for DL and VFI samples.

    The two histograms share the same binning/density settings so visual
    differences reflect distribution shifts rather than plotting choices.
    """
    plt.figure(figsize=(6.5, 4))
    plt.hist(_to_numpy(x_dl), bins=60, density=True, alpha=0.6, label="DL")
    plt.hist(_to_numpy(x_vfi), bins=60, density=True, alpha=0.6, label="VFI")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
