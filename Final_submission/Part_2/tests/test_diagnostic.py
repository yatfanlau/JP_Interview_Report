# tests/test_diagnostic.py
import numpy as np
import pandas as pd
import pytest
import utils_part2 as u

tf = u.tf

import diagnostic as diag
from config_part2 import PanelColumnsPart2


def test_metric1_bias_sd_rmse_known():
    est = [1.0, 2.0, 3.0]
    mean_hat, bias, sd, rmse = diag.metric1_bias_sd_rmse(est, true_value=2.0)

    assert mean_hat == pytest.approx(2.0)
    assert bias == pytest.approx(0.0)
    assert sd == pytest.approx(1.0)  # sample sd with ddof=1
    assert rmse == pytest.approx(np.sqrt(2.0 / 3.0))


def test_run_metric_2_coverage_hmc_adds_cover_columns():
    res_df = pd.DataFrame(
        {
            "rep": [0, 1],
            "theta_lo": [0.6, 0.6],
            "theta_hi": [0.8, 0.7],
            "phi_lo": [1.0, 1.0],
            "phi_hi": [3.0, 1.5],
        }
    )

    out = diag.run_metric_2_coverage_hmc(res_df, theta_true=0.65, phi_true=1.2)
    assert "cover_theta" in out.columns
    assert "cover_phi" in out.columns
    assert out["cover_theta"].tolist() == [True, True]
    assert out["cover_phi"].tolist() == [True, True]


def test_aux_moments_from_df_matches_tf():
    cols = PanelColumnsPart2()
    k_hist = np.array([[1.0, 1.1, 1.2], [2.0, 2.2, 2.4]], dtype=np.float32)
    z_hist = np.array([[1.0, 0.9, 1.05], [1.1, 1.0, 0.95]], dtype=np.float32)

    rows = []
    for firm in range(2):
        for t in range(3):
            rows.append(
                {
                    cols.rep: 0,
                    cols.firm: firm,
                    cols.time: t,
                    cols.k: float(k_hist[firm, t]),
                    cols.z: float(z_hist[firm, t]),
                }
            )
    df = pd.DataFrame(rows)

    mu_df = diag.aux_moments_from_df(df, cols)
    mu_tf = diag.aux_moments_tf(tf.constant(k_hist), tf.constant(z_hist)).numpy()

    assert mu_df.shape == (4,)
    assert mu_tf.shape == (4,)
    assert np.all(np.isfinite(mu_df))
    assert np.all(np.isfinite(mu_tf))
    assert np.allclose(mu_df, mu_tf, atol=1e-5, rtol=1e-5)