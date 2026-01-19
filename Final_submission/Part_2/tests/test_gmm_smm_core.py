# tests/test_gmm_smm_core.py
import math
import numpy as np
import pytest
import utils_part2 as u

tf = u.tf

from config_part2 import BasicModelParams
import gmm_smm_core as core


def test_build_basic_model_context_matches_formulas():
    mp = BasicModelParams(r=0.04, rho=0.7, sigma_eps=0.15, delta=0.1)
    ctx = core.build_basic_model_context(mp, dtype=tf.float64)

    beta_expected = 1.0 / (1.0 + mp.r)
    mu_expected = -0.5 * (mp.sigma_eps**2) / (1.0 + mp.rho)
    sigma_ln_z_expected = mp.sigma_eps / math.sqrt(1.0 - mp.rho**2)
    m_ln_z_expected = -0.5 * (mp.sigma_eps**2) / (1.0 - mp.rho**2)

    assert ctx["beta_f"] == pytest.approx(beta_expected)
    assert ctx["mu_ln_z_f"] == pytest.approx(mu_expected)
    assert ctx["sigma_ln_z_f"] == pytest.approx(sigma_ln_z_expected)
    assert ctx["m_ln_z_f"] == pytest.approx(m_ln_z_expected)
    assert ctx["delta_tf"].dtype == tf.float64


def test_iota_bounds():
    mp = BasicModelParams(delta=0.1, iota_upper=0.5, iota_lower_eps=0.99)
    iota_min, iota_max = core.iota_bounds(mp)
    assert iota_max == pytest.approx(mp.iota_upper)
    assert iota_min == pytest.approx(-(mp.iota_lower_eps) * (1.0 - mp.delta))


def test_select_rep_ids_first_n():
    import pandas as pd

    df = pd.DataFrame({"rep": list(range(10))})
    assert core.select_rep_ids(df, "rep", 3) == [0, 1, 2]


def test_policy_iota_tf_constant_policy(dummy_policy_path):
    policy = tf.keras.models.load_model(dummy_policy_path, compile=False)
    k = tf.constant([1.0, 2.0, 3.0], tf.float32)
    z = tf.constant([1.0, 1.1, 0.9], tf.float32)
    iota = core.policy_iota_tf(
        policy,
        k,
        z,
        theta=tf.constant(0.7, tf.float32),
        phi=tf.constant(2.0, tf.float32),
    )
    assert iota.shape == (3,)
    assert np.allclose(iota.numpy(), 0.0, atol=1e-7)


def test_simulate_panel_hist_tf_constant_policy_dynamics(dummy_policy_path):
    policy = tf.keras.models.load_model(dummy_policy_path, compile=False)

    mp = BasicModelParams(theta=0.6, delta=0.1, r=0.04, rho=0.7, sigma_eps=0.15, phi=2.0)
    ctx = core.build_basic_model_context(mp, dtype=tf.float32)

    N = 2
    burnin = 0
    sim_len = 3
    T_total = burnin + sim_len + 2

    eps = tf.zeros([N, T_total], tf.float32)
    lnz0 = tf.zeros([N], tf.float32)

    k_hist, z_hist, iota_hist = core.simulate_panel_hist_tf(
        policy,
        theta=tf.constant(mp.theta, tf.float32),
        log_phi=tf.constant(math.log(mp.phi), tf.float32),
        eps_all_tf=eps,
        lnz_init_tf=lnz0,
        burnin=burnin,
        sim_len=sim_len,
        ctx=ctx,
    )

    assert k_hist.shape == (N, sim_len)
    assert z_hist.shape == (N, sim_len)
    assert iota_hist.shape == (N, sim_len)
    assert np.allclose(iota_hist.numpy(), 0.0, atol=1e-7)

    k = k_hist.numpy()
    assert np.allclose(k[:, 1], k[:, 0] * (1.0 - mp.delta), rtol=1e-6, atol=1e-6)
    assert np.allclose(k[:, 2], k[:, 1] * (1.0 - mp.delta), rtol=1e-6, atol=1e-6)