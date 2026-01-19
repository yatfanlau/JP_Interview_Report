# tests/test_hmc_estimation_unit.py
import math
import numpy as np
import pandas as pd
import pytest
import utils_part2 as u

tf = u.tf

pytest.importorskip("tensorflow_probability")
import HMC_estimation_ as hmc  # noqa: E402

from config_part2 import (
    PanelColumnsPart2,
    ParamBoundsPart2,
    HMCConfigPart2,
    BasicModelParams,
)


def test_df_to_balanced_matrix_balanced():
    cols = PanelColumnsPart2()
    df = pd.DataFrame(
        {
            cols.firm: [0, 0, 0, 1, 1, 1],
            cols.time: [0, 1, 2, 0, 1, 2],
            "v": [10, 11, 12, 20, 21, 22],
        }
    )
    mat = hmc._df_to_balanced_matrix(df, "v", cols.firm, cols.time)
    assert mat.shape == (2, 3)
    assert np.allclose(mat[0], [10, 11, 12])
    assert np.allclose(mat[1], [20, 21, 22])


def test_df_to_balanced_matrix_unbalanced_raises():
    cols = PanelColumnsPart2()
    df = pd.DataFrame(
        {
            cols.firm: [0, 0, 1, 1],
            cols.time: [0, 1, 0, 1],
            "v": [10, 11, 20, 21],
        }
    ).iloc[:-1]  # drop one row -> unbalanced
    with pytest.raises(ValueError):
        _ = hmc._df_to_balanced_matrix(df, "v", cols.firm, cols.time)


def test_make_observations_from_truth_deterministic(synthetic_panel_csv):
    cols = PanelColumnsPart2()
    df = pd.read_csv(synthetic_panel_csv)
    df_rep = df[df[cols.rep] == 0].copy()

    cfg = HMCConfigPart2(sigma_y=0.05, sigma_logk_obs=0.02)
    y1, lk1 = hmc._make_observations_from_truth(df_rep, cols, theta_true=0.7, cfg=cfg, seed=123)
    y2, lk2 = hmc._make_observations_from_truth(df_rep, cols, theta_true=0.7, cfg=cfg, seed=123)

    assert np.allclose(y1, y2)
    assert np.allclose(lk1, lk2)


def test_build_hmc_context_Q_R_diag():
    mp = BasicModelParams(delta=0.1, rho=0.7, sigma_eps=0.15)
    cfg = HMCConfigPart2(sigma_y=0.05, sigma_logk_obs=0.02, sigma_logk_trans=0.01)
    ctx = hmc.build_hmc_context(mp, cfg)

    assert float(ctx["Q"][0, 0].numpy()) == pytest.approx(mp.sigma_eps**2)
    assert float(ctx["Q"][1, 1].numpy()) == pytest.approx(cfg.sigma_logk_trans**2)
    assert float(ctx["R"][0, 0].numpy()) == pytest.approx(cfg.sigma_y**2)
    assert float(ctx["R"][1, 1].numpy()) == pytest.approx(cfg.sigma_logk_obs**2)


def test_u_to_params_midpoint_and_logprior():
    bounds = ParamBoundsPart2(theta_min=0.5, theta_max=0.9, phi_min=0.5, phi_max=5.0)
    u0 = tf.constant([0.0, 0.0], dtype=hmc.DTYPE)

    theta, phi, lp = hmc._u_to_params(u0, bounds)
    assert float(theta.numpy()) == pytest.approx(0.7, abs=1e-7)
    assert float(phi.numpy()) == pytest.approx(2.75, abs=1e-7)
    assert float(lp.numpy()) == pytest.approx(2.0 * math.log(0.25), abs=1e-6)


def test_init_u_from_guess_roundtrip():
    bounds = ParamBoundsPart2()
    u_np = hmc._init_u_from_guess(theta_guess=0.65, phi_guess=2.0, bounds=bounds)
    theta, phi, _ = hmc._u_to_params(tf.constant(u_np, dtype=hmc.DTYPE), bounds)
    assert float(theta.numpy()) == pytest.approx(0.65, abs=1e-6)
    assert float(phi.numpy()) == pytest.approx(2.0, abs=1e-6)


def test_transition_fn_g_nonpositive_sets_logk_floor(dummy_policy_path_negative):
    policy = tf.keras.models.load_model(dummy_policy_path_negative, compile=False)

    mp = BasicModelParams(delta=0.1, rho=0.7, sigma_eps=0.15)
    cfg = HMCConfigPart2()
    ctx = hmc.build_hmc_context(mp, cfg)

    X = tf.constant([[[0.0, math.log(2.0)]]], dtype=hmc.DTYPE)  # [B=1, J=1, 2]
    out = hmc._transition_fn(
        X,
        theta=tf.constant(0.7, hmc.DTYPE),
        phi=tf.constant(2.0, hmc.DTYPE),
        policy=policy,
        ctx=ctx,
    )
    logk_next = float(out[0, 0, 1].numpy())
    assert logk_next == pytest.approx(float(ctx["logk_floor"].numpy()), abs=1e-12)


def test_mvn_logpdf_from_chol_identity_zero():
    obs = tf.zeros([1, 2], dtype=hmc.DTYPE)
    mean = tf.zeros([1, 2], dtype=hmc.DTYPE)
    chol = tf.eye(2, batch_shape=[1], dtype=hmc.DTYPE)

    val = float(hmc._mvn_logpdf_from_chol(obs, mean, chol).numpy()[0])
    assert val == pytest.approx(-math.log(2.0 * math.pi), abs=1e-6)


def test_make_ukf_loglik_fn_runs(synthetic_panel_csv, dummy_policy_path):
    cols = PanelColumnsPart2()
    df = pd.read_csv(synthetic_panel_csv)
    df_rep = df[df[cols.rep] == 0].copy()

    cfg = HMCConfigPart2(
        sigma_y=0.01,
        sigma_logk_obs=0.01,
        sigma_logk_trans=0.0,
        ukf_jitter=1e-6,
    )
    mp = BasicModelParams(delta=0.1, rho=0.7, sigma_eps=0.15)
    ctx = hmc.build_hmc_context(mp, cfg)

    policy = tf.keras.models.load_model(dummy_policy_path, compile=False)
    y_obs, logk_obs = hmc._make_observations_from_truth(df_rep, cols, theta_true=0.7, cfg=cfg, seed=1)

    loglik_fn = hmc.make_ukf_loglik_fn(
        tf.constant(y_obs, dtype=hmc.DTYPE),
        tf.constant(logk_obs, dtype=hmc.DTYPE),
        policy,
        ctx,
        cfg,
    )

    ll = float(loglik_fn(tf.constant(0.7, hmc.DTYPE), tf.constant(2.0, hmc.DTYPE)).numpy())
    assert np.isfinite(ll)