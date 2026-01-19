# tests/test_smm_estimation_unit.py
import numpy as np
import pandas as pd
import pytest
import utils_part2 as u

tf = u.tf

import SMM_estimation as smm


def test_compute_moments_from_df_matches_tf():
    cols = smm.COLS

    k_hist = np.array([[1.0, 1.1, 1.2, 1.1], [2.0, 2.1, 2.0, 2.2]], dtype=np.float32)
    z_hist = np.array([[1.0, 1.02, 0.99, 1.01], [0.98, 1.0, 1.03, 1.02]], dtype=np.float32)
    iota_hist = np.array([[0.0, 0.05, 0.02, 0.01], [0.01, 0.04, 0.03, 0.0]], dtype=np.float32)

    rows = []
    for firm in range(k_hist.shape[0]):
        for t in range(k_hist.shape[1]):
            rows.append(
                {
                    cols.rep: 0,
                    cols.firm: firm,
                    cols.time: t,
                    cols.k: float(k_hist[firm, t]),
                    cols.z: float(z_hist[firm, t]),
                    cols.iota: float(iota_hist[firm, t]),
                }
            )
    df = pd.DataFrame(rows)

    m_df = smm.compute_moments_from_df(df)
    m_tf = smm.compute_moments_tf(
        tf.constant(k_hist),
        tf.constant(z_hist),
        tf.constant(iota_hist),
    ).numpy()

    assert m_df.shape == (smm.N_TARGET_MOMENTS,)
    assert m_tf.shape == (smm.N_TARGET_MOMENTS,)
    assert np.all(np.isfinite(m_df))
    assert np.all(np.isfinite(m_tf))

    # float32 vs float64 + eps stabilizers => allow a bit of slack
    assert np.allclose(m_df, m_tf, atol=1e-3, rtol=1e-3)


def test_make_crn_draws_deterministic():
    eps1, lnz1 = smm.make_crn_draws(rep_id=7, base_seed=1234, n_sims=1)
    eps2, lnz2 = smm.make_crn_draws(rep_id=7, base_seed=1234, n_sims=1)

    assert np.allclose(eps1.numpy(), eps2.numpy())
    assert np.allclose(lnz1.numpy(), lnz2.numpy())

    eps3, lnz3 = smm.make_crn_draws(rep_id=7, base_seed=1234, n_sims=3)
    assert eps3.shape[0] == 3
    assert lnz3.shape[0] == 3


def test_bootstrap_moment_cov_by_firm_shape(synthetic_panel_csv):
    df = pd.read_csv(synthetic_panel_csv)
    df_rep = df[df[smm.COLS.rep] == 0].copy()

    Sigma = smm.bootstrap_moment_cov_by_firm(df_rep, n_boot=5, seed=0)
    q = smm.N_TARGET_MOMENTS
    assert Sigma.shape == (q, q)
    assert np.allclose(Sigma, Sigma.T, atol=1e-10)