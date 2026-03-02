"""Panel-moment utilities for simulated macro-finance time series.

These helpers summarize simulated panel outputs into moments that are easy to
compare across solution methods (DL policy vs VFI benchmark).
"""

from __future__ import annotations

from investment_dl.core.math_utils import tf_var
from investment_dl.core.tf_env import tf
from investment_dl.models.basic_investment import BasicInvestmentModel


def panel_autocorr(x_panel: tf.Tensor, lags: list[int]) -> dict[int, float]:
    """Compute pooled panel autocorrelations for requested lags.

    For each lag, all valid time-path pairs are flattened into two vectors and
    a correlation coefficient is computed from pooled covariance and variance.
    """
    out: dict[int, float] = {}
    x_panel = tf.convert_to_tensor(x_panel)
    T = int(x_panel.shape[0])
    for lag in lags:
        if lag <= 0 or lag >= T:
            out[lag] = float("nan")
            continue
        x0 = x_panel[lag:, :]
        x1 = x_panel[:-lag, :]
        # Flatten to pool across both time and cross-sectional dimensions.
        x0f = tf.reshape(x0, [-1])
        x1f = tf.reshape(x1, [-1])
        mean0 = tf.reduce_mean(x0f)
        mean1 = tf.reduce_mean(x1f)
        var0 = tf_var(x0f)
        var1 = tf_var(x1f)
        cov = tf.reduce_mean((x0f - mean0) * (x1f - mean1))
        denom = tf.sqrt(var0 * var1) + 1e-12
        out[lag] = float((cov / denom).numpy())
    return out


def compute_panel_moments(
    k_panel: tf.Tensor,
    z_panel: tf.Tensor,
    iota_panel: tf.Tensor,
    model: BasicInvestmentModel,
    lags: list[int],
) -> dict[str, dict]:
    """Return mean/variance/autocorrelation summaries for core model variables.

    The routine computes moments for ``k``, ``iota``, implied investment
    ``I = iota*k``, and output ``y = z*k^theta``.
    """
    k_panel_t = tf.convert_to_tensor(k_panel)
    z_panel_t = tf.convert_to_tensor(z_panel)
    iota_panel_t = tf.convert_to_tensor(iota_panel)

    y_panel = z_panel_t * tf.pow(k_panel_t, model.params.theta)
    I_panel = iota_panel_t * k_panel_t

    # Centralize variable construction so plotting/reporting use the same set.
    panels = {
        "k": k_panel_t,
        "iota": iota_panel_t,
        "I": I_panel,
        "y": y_panel,
    }

    out: dict[str, dict] = {}
    for name, panel in panels.items():
        out[name] = {
            "mean": float(tf.reduce_mean(panel).numpy()),
            "var": float(tf_var(panel).numpy()),
            "ac": panel_autocorr(panel, lags),
        }
    return out
