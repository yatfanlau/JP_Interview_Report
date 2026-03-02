"""Convergence plotting for DL training and VFI fixed-point iterations.

These plots summarize optimization trajectories so stability and termination
behavior can be compared across solution approaches.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from investment_dl.core.tf_env import tf


def _to_tensor(x, dtype=None) -> tf.Tensor:
    """Convert array-like input to a TensorFlow tensor.

    When ``dtype`` is provided, the output is cast so plotting code can apply
    TensorFlow ops without dtype mismatches.
    """
    t = tf.convert_to_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = tf.cast(t, dtype)
    return t


def _to_numpy(x):
    """Return a NumPy representation of ``x`` for Matplotlib routines.

    Tensor inputs are eagerly converted and non-tensor inputs are returned as-is.
    """
    return x.numpy() if tf.is_tensor(x) else x


def plot_dl_convergence(history: dict, savepath: str | None = None) -> None:
    """Plot pretraining and training AiO losses on a semilog scale.

    Pretraining and main training are displayed as separate segments so the
    transition in sampling strategy is visually explicit.
    """
    pre_steps = _to_tensor(history.get("pretrain_steps", []), dtype=tf.int32)
    pre_loss = _to_tensor(history.get("pretrain_loss", []), dtype=tf.float32)
    train_steps = _to_tensor(history.get("train_steps", []), dtype=tf.int32)
    train_loss = _to_tensor(history.get("train_loss", []), dtype=tf.float32)

    n_pre = int(tf.size(pre_steps).numpy())
    n_train = int(tf.size(train_steps).numpy())
    if n_pre == 0 and n_train == 0:
        return

    # Shift main-training step numbers after pretraining for one global axis.
    offset = int(tf.reduce_max(pre_steps).numpy()) if n_pre > 0 else 0
    train_steps_global = train_steps + offset

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if n_pre > 0:
        ax.semilogy(_to_numpy(pre_steps), _to_numpy(tf.abs(pre_loss) + 1e-12), label="Pretrain")
    if n_train > 0:
        ax.semilogy(
            _to_numpy(train_steps_global),
            _to_numpy(tf.abs(train_loss) + 1e-12),
            label="Train",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("|AiO loss|")
    ax.set_title("DL Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_vfi_convergence(vfi_hist: dict, savepath: str | None = None) -> None:
    """Plot VFI sup-norm differences across iterations.

    Exact zeros are handled separately from positive values to keep log-scale
    rendering stable and interpretable.
    """
    it = _to_tensor(vfi_hist.get("iter", []), dtype=tf.int32)
    diff = _to_tensor(vfi_hist.get("sup_norm", []), dtype=tf.float32)
    if int(tf.size(it).numpy()) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # Log-scale plotting must exclude exact zeros to avoid visual cliffs.
    positive_mask = diff > 0.0
    it_positive = tf.boolean_mask(it, positive_mask)
    diff_positive = tf.boolean_mask(diff, positive_mask)

    has_positive = int(tf.size(it_positive).numpy()) > 0
    has_zero = bool(tf.reduce_any(tf.logical_not(positive_mask)).numpy())

    if has_positive:
        ax.semilogy(
            _to_numpy(it_positive),
            _to_numpy(diff_positive),
            label="Sup-norm diff",
        )

    if has_zero:
        first_zero_idx = tf.where(tf.logical_not(positive_mask))[0, 0]
        first_zero_iter = float(tf.gather(it, first_zero_idx).numpy())
        ax.axvline(
            first_zero_iter,
            linestyle="--",
            alpha=0.7,
            color="tab:red",
            label="Numerical zero reached",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sup-norm diff")
    ax.set_title("VFI Convergence")
    ax.grid(True, alpha=0.3)
    if has_positive or has_zero:
        ax.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
