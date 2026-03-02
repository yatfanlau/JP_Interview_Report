"""Auxiliary panel-moment helpers for diagnostic comparisons.

These moments summarize broad panel behavior in a compact vector and are useful
for model-fit checks beyond the estimation objective itself.
"""

from __future__ import annotations

import pandas as pd

from dyninv.config import PanelColumnsPart2
from dyninv.utils import DTYPE, safe_corr_tf, tf


def aux_moments_tf(k_hist: tf.Tensor, z_hist: tf.Tensor) -> tf.Tensor:
    """Compute a compact vector of auxiliary panel moments.

    Args:
        k_hist: Capital history tensor shaped ``(firms, time)``.
        z_hist: Productivity history tensor shaped ``(firms, time)``.

    Returns:
        Tensor with moments ``[mean(z), std(log z), corr(log k, z), ar1(log k)]``.
    """
    k_hist = tf.cast(k_hist, DTYPE)
    z_hist = tf.cast(z_hist, DTYPE)

    k_flat = tf.reshape(k_hist, [-1])
    z_flat = tf.reshape(z_hist, [-1])

    logk = tf.math.log(k_flat + 1e-32)
    lnz = tf.math.log(z_flat + 1e-32)

    a1 = tf.reduce_mean(z_flat)
    a2 = tf.sqrt(tf.reduce_mean(tf.square(lnz - tf.reduce_mean(lnz))) + 1e-12)
    a3 = tf.cast(safe_corr_tf(logk, z_flat), DTYPE)

    logk_hist = tf.math.log(k_hist + 1e-32)
    a4 = tf.cast(safe_corr_tf(logk_hist[:, 1:], logk_hist[:, :-1]), DTYPE)
    return tf.stack([a1, a2, a3, a4], axis=0)


def aux_moments_from_df(df_rep: pd.DataFrame, cols: PanelColumnsPart2):
    """Compute auxiliary moments from one replication DataFrame.

    The input frame is sorted and reshaped into ``(firms, time)`` tensors before
    dispatching to ``aux_moments_tf``.
    """
    d = df_rep.sort_values([cols.firm, cols.time]).copy()
    n_firms = int(d[cols.firm].nunique())
    t_periods = int(d[cols.time].nunique())

    k = tf.constant(d[cols.k].tolist(), dtype=DTYPE)
    z = tf.constant(d[cols.z].tolist(), dtype=DTYPE)

    k_hist = tf.reshape(k, [n_firms, t_periods])
    z_hist = tf.reshape(z, [n_firms, t_periods])
    return aux_moments_tf(k_hist, z_hist).numpy()
