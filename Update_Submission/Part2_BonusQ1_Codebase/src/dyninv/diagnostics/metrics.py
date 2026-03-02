"""Diagnostic metrics and reporting helpers for estimator outputs.

The routines here provide lightweight summary statistics and interval-coverage
checks used by CLI scripts and notebook-style workflows.
"""

from __future__ import annotations

import pandas as pd

from dyninv.utils import DTYPE, tf, zcrit


def metric1_bias_sd_rmse(estimates, true_value: float) -> tuple[float, float, float, float]:
    """Compute mean, bias, sample SD, and RMSE for estimator draws.

    Args:
        estimates: 1D iterable/tensor of parameter estimates across replications.
        true_value: Ground-truth parameter value used for bias/RMSE.
    """
    est = tf.cast(tf.reshape(tf.convert_to_tensor(estimates), [-1]), tf.float64)
    true = tf.constant(float(true_value), dtype=tf.float64)
    mean_hat = tf.reduce_mean(est)
    bias = mean_hat - true

    n = tf.shape(est)[0]
    centered = est - mean_hat
    var_sample = tf.where(
        n > 1,
        tf.reduce_sum(centered * centered) / tf.cast(n - 1, tf.float64),
        tf.constant(float("nan"), tf.float64),
    )
    sd = tf.sqrt(var_sample)
    rmse = tf.sqrt(tf.reduce_mean((est - true) ** 2))
    return float(mean_hat.numpy()), float(bias.numpy()), float(sd.numpy()), float(rmse.numpy())


def run_metric_1(res_df: pd.DataFrame, theta_true: float, phi_true: float, label: str):
    """Print metric-1 summaries for ``theta`` and ``phi`` columns."""
    print(f"\n=== Metric 1 ({label}) ===")
    for name, true_val in [("theta", theta_true), ("phi", phi_true)]:
        est_col = f"{name}_hat"
        mean_hat, bias, sd, rmse = metric1_bias_sd_rmse(res_df[est_col].tolist(), true_val)
        print(
            f"{name}: mean={mean_hat:.8f}, true={true_val:.8f}, "
            f"bias={bias:.3e}, SD={sd:.3e}, RMSE={rmse:.3e}"
        )


def run_metric_2_coverage_hmc(res_df: pd.DataFrame, theta_true: float, phi_true: float, cfg=None):
    """Compute and print empirical credible-interval coverage for HMC output.

    Returns:
        A sorted copy of the results DataFrame with boolean coverage columns.
    """
    out = res_df.copy()
    out["cover_theta"] = (out["theta_lo"] <= theta_true) & (theta_true <= out["theta_hi"])
    out["cover_phi"] = (out["phi_lo"] <= phi_true) & (phi_true <= out["phi_hi"])

    cred = float(getattr(cfg, "cred_level", 0.95)) if cfg is not None else 0.95
    print("\n=== Metric 2 (HMC coverage) ===")
    print(f"Nominal level: {100 * cred:.1f}%")
    print(f"Theta coverage: {out['cover_theta'].mean():.3f}")
    print(f"Phi coverage:   {out['cover_phi'].mean():.3f}")
    return out.sort_values("rep").reset_index(drop=True)


def normal_ci(mean: tf.Tensor, se: tf.Tensor, alpha: float = 0.05) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute a two-sided normal-approximation confidence interval."""
    z = tf.constant(zcrit(alpha), dtype=DTYPE)
    return mean - z * se, mean + z * se
