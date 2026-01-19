# tests/test_gmm_estimation_unit.py
import numpy as np
import pandas as pd
import pytest
import utils_part2 as u

tf = u.tf

import GMM_estimation as gmm


def test_build_instruments_np_standardized():
    k = np.array([1.0, 2.0, 4.0], dtype=float)
    z = np.array([1.0, 1.1, 0.9], dtype=float)
    iota = np.array([0.0, 0.1, -0.05], dtype=float)

    Z = gmm.build_instruments_np(k, z, iota, standardize=True)
    assert Z.shape == (3, 7)
    assert np.allclose(Z[:, 0], 1.0)

    m = Z[:, 1:].mean(axis=0)
    assert np.all(np.abs(m) < 1e-5)
    assert np.all(np.isfinite(Z))


def test_prepare_gmm_data_from_df_shapes():
    cols = gmm.COLS
    rows = []
    for firm in [0, 1]:
        for t in [0, 1, 2]:
            rows.append(
                {
                    cols.rep: 0,
                    cols.firm: firm,
                    cols.time: t,
                    cols.k: 1.0 + firm + t,
                    cols.z: 1.0 + 0.1 * t,
                    cols.iota: 0.0,
                }
            )
    df = pd.DataFrame(rows)

    data = gmm.prepare_gmm_data_from_df(df, standardize_instr=True)
    assert data["k_t_tf"].shape[0] == 4  # 2 firms * (3-1)
    assert data["k_tp1_tf"].shape[0] == 4
    assert data["Z_tf"].shape[0] == 4
    assert data["Z_tf"].shape[1] == len(gmm.INSTRUMENT_NAMES)
    assert int(data["N_firms"]) == 2
    assert int(data["q"]) == len(gmm.INSTRUMENT_NAMES)


def test_euler_residual_tf_finite():
    theta = tf.constant(0.7, gmm.DTYPE)
    log_phi = tf.constant(np.log(2.0), gmm.DTYPE)

    k_t = tf.constant([1.0, 2.0], gmm.DTYPE)
    iota_t = tf.constant([0.0, 0.0], gmm.DTYPE)
    k_tp1 = tf.constant([0.9, 1.8], gmm.DTYPE)
    z_tp1 = tf.constant([1.0, 1.0], gmm.DTYPE)
    iota_tp1 = tf.constant([0.0, 0.0], gmm.DTYPE)

    u_res = gmm.euler_residual_tf(theta, log_phi, k_t, iota_t, k_tp1, z_tp1, iota_tp1).numpy()
    assert u_res.shape == (2,)
    assert np.all(np.isfinite(u_res))


def test_estimate_S_and_W_np_symmetry(synthetic_panel_csv):
    df = pd.read_csv(synthetic_panel_csv)
    df_rep = df[df[gmm.COLS.rep] == 0].copy()
    data = gmm.prepare_gmm_data_from_df(df_rep, standardize_instr=True)

    S, W = gmm.estimate_S_and_W_np(
        theta_hat=0.7,
        logphi_hat=np.log(2.0),
        data=data,
        ridge=1e-8,
        rcond=1e-10,
    )

    assert np.allclose(S, S.T, atol=1e-10)
    assert np.allclose(W, W.T, atol=1e-10)
    evals = np.linalg.eigvalsh(S)
    assert evals.min() > -1e-6


def test_run_gmm_optim_respects_bounds(synthetic_panel_csv):
    df = pd.read_csv(synthetic_panel_csv)
    df_rep = df[df[gmm.COLS.rep] == 0].copy()
    data = gmm.prepare_gmm_data_from_df(df_rep, standardize_instr=True)

    out = gmm._run_gmm_optim(
        data=data,
        theta_init=0.55,
        log_phi_init=np.log(1.1),
        W_np=None,
        learning_rate=0.01,
        train_steps=2,
        print_every=999999,
    )
    assert gmm.THETA_MIN_F <= out["theta_hat"] <= gmm.THETA_MAX_F
    assert gmm.PHI_MIN_F <= out["phi_hat"] <= gmm.PHI_MAX_F