"""Unit tests for diagnostics moment and metric helper functions."""

from __future__ import annotations

import pandas as pd
import pytest

from dyninv.config import PanelColumnsPart2
from dyninv.diagnostics.aux_moments import aux_moments_from_df, aux_moments_tf
from dyninv.diagnostics.metrics import metric1_bias_sd_rmse, run_metric_2_coverage_hmc
from dyninv.utils import tf

pytestmark = pytest.mark.unit


def test_metric1_known_values():
    """Check metric-1 moments against a hand-computable baseline."""
    mean_hat, bias, sd, rmse = metric1_bias_sd_rmse([1.0, 2.0, 3.0], 2.0)
    assert abs(mean_hat - 2.0) < 1e-12
    assert abs(bias - 0.0) < 1e-12
    assert abs(sd - 1.0) < 1e-12
    assert abs(rmse - (2.0 / 3.0) ** 0.5) < 1e-12


def test_aux_moments_df_matches_tf():
    """Verify DataFrame and tensor pathways produce matching moments."""
    cols = PanelColumnsPart2()
    k_hist = tf.constant([[1.0, 1.1, 1.2], [2.0, 2.2, 2.4]], dtype=tf.float32)
    z_hist = tf.constant([[1.0, 0.9, 1.05], [1.1, 1.0, 0.95]], dtype=tf.float32)

    rows = []
    for firm in range(2):
        for t in range(3):
            rows.append(
                {
                    cols.rep: 0,
                    cols.firm: firm,
                    cols.time: t,
                    cols.k: float(k_hist[firm, t].numpy()),
                    cols.z: float(z_hist[firm, t].numpy()),
                }
            )
    df = pd.DataFrame(rows)
    m_df = aux_moments_from_df(df, cols)
    m_tf = aux_moments_tf(k_hist, z_hist).numpy()
    assert len(m_df) == 4
    assert m_tf.shape == (4,)
    assert max(abs(m_df - m_tf)) < 1e-5


def test_hmc_coverage_columns_added():
    """Ensure HMC coverage helper appends expected indicator columns."""
    df = pd.DataFrame(
        {
            "rep": [0, 1],
            "theta_lo": [0.6, 0.6],
            "theta_hi": [0.8, 0.7],
            "phi_lo": [1.0, 1.0],
            "phi_hi": [3.0, 1.5],
        }
    )
    out = run_metric_2_coverage_hmc(df, theta_true=0.65, phi_true=1.2)
    assert "cover_theta" in out.columns
    assert "cover_phi" in out.columns
