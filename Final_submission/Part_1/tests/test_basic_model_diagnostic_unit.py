# tests/test_basic_model_diagnostic_unit.py
import importlib
import sys
import types
import numpy as np

import pytest

from config_part1 import BasicFinalTestParams
from model_core import tf, DTYPE, steady_state_k as ss_k


@pytest.fixture
def basic_diag(monkeypatch):
    """
    Import basic_model_diagnostic with a fake 'basic_model' module injected.
    """
    fake = types.ModuleType("basic_model")
    fake.tf = tf
    fake.DTYPE = DTYPE
    fake.K_SS_TF = tf.constant(1.0, dtype=DTYPE)

    # minimal params/namespaces the diagnostic module expects
    fake.mp = types.SimpleNamespace(theta=0.7, delta=0.1, r=0.04, rho=0.7, sigma_eps=0.15)
    fake.tp = types.SimpleNamespace(n_paths=4)
    fake.fp = BasicFinalTestParams(
        burn_in_steps=2,
        T_on_policy=20,
        M_coverage=10,
        q_low=0.01,
        q_high=0.99,
        expand_frac=0.05,
        batch_eval=1024,
        edge_points=5,
        tol_list=(1e-3, 1e-4),
        gh_nodes=10,
    )

    fake.policy = object()

    def policy_step(policy, k, z, training=False):
        # trivial stable dynamics
        return k, z, None

    def policy_iota(policy, k, z, training=False):
        return tf.zeros_like(k) + tf.constant(0.1, dtype=DTYPE)

    def eval_batched(k_np, z_np, n_nodes: int, batch: int):
        g = np.zeros_like(k_np, dtype=np.float32)
        denom = np.ones_like(k_np, dtype=np.float32)
        return g, denom

    fake.policy_step = policy_step
    fake.policy_iota = policy_iota
    fake.eval_batched = eval_batched
    fake.steady_state_k = ss_k
    fake.euler_residuals_gh = lambda k, z, n_nodes: (tf.zeros_like(k), None, None)

    monkeypatch.setitem(sys.modules, "basic_model", fake)

    # Ensure fresh import using the injected fake
    sys.modules.pop("basic_model_diagnostic", None)
    mod = importlib.import_module("basic_model_diagnostic")
    return mod


def test_build_coverage_box_and_sample_within_log_bounds(basic_diag):
    np.random.seed(0)
    lnk = np.linspace(-1.0, 1.0, 1000).astype(np.float32)
    lnz = np.linspace(-0.5, 0.5, 1000).astype(np.float32)

    k_cov, z_cov, box = basic_diag.build_coverage_box_and_sample(
        lnk, lnz, q_low=0.2, q_high=0.8, expand=0.0, M=500
    )

    assert k_cov.shape == (500,)
    assert z_cov.shape == (500,)
    assert k_cov.dtype == np.float32
    assert z_cov.dtype == np.float32

    lnk_u = np.log(k_cov)
    lnz_u = np.log(z_cov)
    assert np.all(lnk_u >= box["lnk_min"] - 1e-6)
    assert np.all(lnk_u <= box["lnk_max"] + 1e-6)
    assert np.all(lnz_u >= box["lnz_min"] - 1e-6)
    assert np.all(lnz_u <= box["lnz_max"] + 1e-6)


def test_edge_and_corner_points_count(basic_diag):
    box = dict(lnk_min=0.0, lnk_max=1.0, lnz_min=-1.0, lnz_max=-0.5)
    n_edge = 5
    k_edge, z_edge = basic_diag.edge_and_corner_points(box, n_edge=n_edge)

    # total points = 4*(n_edge-2) + 4 = 4*n_edge - 4
    assert len(k_edge) == 4 * n_edge - 4
    assert len(z_edge) == 4 * n_edge - 4
    assert np.all(k_edge > 0.0)
    assert np.all(z_edge > 0.0)


def test_gh_robustness_check_zero_when_eval_batched_constant(basic_diag, monkeypatch):
    # Force eval_batched to ignore n_nodes and return the same residuals
    def eval_const(k_np, z_np, n_nodes: int, batch: int):
        g = np.sin(k_np).astype(np.float32) * 0.0  # all zeros
        denom = np.ones_like(k_np, dtype=np.float32)
        return g, denom

    monkeypatch.setattr(basic_diag, "eval_batched", eval_const)

    k = np.linspace(0.5, 2.0, 200).astype(np.float32)
    z = np.linspace(0.8, 1.2, 200).astype(np.float32)

    out = basic_diag.gh_robustness_check(
        k_np=k,
        z_np=z,
        base_nodes=10,
        compare_nodes=(15, 20),
        n_sub=100,
        batch_eval=1024,
    )
    assert out["GH15_vs_GH10_RelChange_P50"] == 0.0
    assert out["GH15_vs_GH10_RelChange_P95"] == 0.0
    assert out["GH20_vs_GH10_RelChange_P50"] == 0.0
    assert out["GH20_vs_GH10_RelChange_P95"] == 0.0


def test_final_test_smoke_with_fakes(basic_diag, capsys):
    params = BasicFinalTestParams(
        burn_in_steps=2,
        T_on_policy=20,
        M_coverage=10,
        edge_points=5,
        batch_eval=1024,
        gh_nodes=10,
    )
    basic_diag.final_test(params)
    out = capsys.readouterr().out
    assert "Final Test: Begin" in out
    assert "Final Test: End" in out
    # With zero residuals from fake eval_batched, should print PASS
    assert "Coverage set: PASS" in out
    assert "On-policy set: PASS" in out