import math

import numpy as np
import pytest

from common import tf, DTYPE, risky_coverage_sampler, set_global_seed
from config import RiskyDebtParams, RiskyDebtTrainingParams
from risky_debt_model import RiskyDebtModel


@pytest.fixture
def small_risky_model():
    rp = RiskyDebtParams(seed=123)
    tp = RiskyDebtTrainingParams(
        batch_size=32,
        buffer_size=200_000,   # unused directly
        n_paths=2048,
        roll_steps=5,
        pretrain_steps=400,
        train_steps=10_000,
        coverage_final_share=0.15,
        lr=1e-3,
        log_every=50,
        eval_every=50,
        lambda_foc_warmup=0.05,
        lambda_foc_final=0.3,
    )
    set_global_seed(rp.seed)
    model = RiskyDebtModel(params=rp, train_params=tp)

    model.lambda_foc = tp.lambda_foc_warmup

    return model, rp, tp

def test_get_next_z_shape_and_positivity(small_risky_model):
    model, rp, tp = small_risky_model
    B = 10
    n_samples = 7
    z = tf.ones((B, 1), dtype=DTYPE) * 1.5

    z_next = model.get_next_z(z, n_samples)
    z_np = z_next.numpy()

    assert z_np.shape == (B, n_samples)
    assert (z_np > 0).all()


def test_production_and_adjustment_cost_zero_at_steady_investment(small_risky_model):
    model, rp, tp = small_risky_model
    k = tf.constant([[1.0], [2.0]], dtype=DTYPE)
    z = tf.constant([[1.0], [0.5]], dtype=DTYPE)

    pi = model.production(k, z)
    assert pi.shape == (2, 1)
    assert (pi.numpy() >= 0).all()

    k_prime = (1.0 - model.delta) * k
    cost = model.adjustment_cost(k, k_prime)
    assert np.allclose(cost.numpy(), 0.0, atol=1e-7)


def test_get_recovery_matches_formula(small_risky_model):
    model, rp, tp = small_risky_model
    k_prime = tf.constant([[1.0], [3.0]], dtype=DTYPE)
    z_prime = tf.constant([[1.0], [0.8]], dtype=DTYPE)

    rec = model.get_recovery(k_prime, z_prime).numpy()

    pi = model.production(k_prime, z_prime).numpy()
    val = (1.0 - model.tau) * pi + (1.0 - model.delta) * k_prime.numpy()
    expected = (1.0 - model.bankruptcy_cost) * val

    assert np.allclose(rec, expected, atol=1e-6)


def test_compute_equity_value_soft_limited_liability(small_risky_model):
    model, rp, tp = small_risky_model

    states = risky_coverage_sampler(64, rp)
    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]

    C, V = model.compute_equity_value(k, b, z, use_target=False)
    C_np = C.numpy()
    V_np = V.numpy()

    assert (V_np >= -1e-8).all()

    relu_C = np.maximum(C_np, 0.0)
    assert (V_np >= relu_C - 1e-6).all()


def test_default_probability_bounds(small_risky_model):
    model, rp, tp = small_risky_model
    states = risky_coverage_sampler(32, rp)

    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]

    pd = model.default_probability(k, b, z, n_draws=16, use_target=True)
    pd_np = pd.numpy()

    assert pd_np.shape == (32, 1)
    assert (pd_np >= 0.0 - 1e-8).all()
    assert (pd_np <= 1.0 + 1e-8).all()


def test_pricing_kernel_basic_properties(small_risky_model):
    model, rp, tp = small_risky_model

    B = 16
    k_prime = tf.ones((B, 1), dtype=DTYPE) * 1.0
    z_current = tf.ones((B, 1), dtype=DTYPE) * 1.0

    # Case 1: b' <= eps_b -> risk-free price
    b_prime_safe = tf.zeros((B, 1), dtype=DTYPE)
    q_safe = model.pricing_kernel(k_prime, b_prime_safe, z_current)
    q_rf = 1.0 / (1.0 + model.r)
    assert np.allclose(q_safe.numpy(), q_rf, atol=1e-6)

    # Case 2: positive debt -> price in (0.01, q_rf]
    b_prime_debt = tf.ones((B, 1), dtype=DTYPE) * 1.0
    q_debt = model.pricing_kernel(k_prime, b_prime_debt, z_current).numpy()
    assert (q_debt > 0.009).all()
    assert (q_debt <= q_rf + 1e-8).all()


def test_compute_rhs_shape_and_finite(small_risky_model):
    model, rp, tp = small_risky_model

    states = risky_coverage_sampler(20, rp)
    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]

    k_safe = tf.maximum(k, 1e-3)
    norm_k = tf.math.log(k_safe)
    norm_b = b / k_safe
    z_safe = tf.maximum(z, 1e-6)
    log_z = tf.math.log(z_safe)
    policy_input = tf.concat([norm_k, norm_b, log_z], axis=-1)
    k_prime, b_prime = model.policy_net(policy_input)

    rhs = model.compute_rhs(k, b, z, k_prime, b_prime)
    rhs_np = rhs.numpy()

    assert rhs_np.shape == (20, 1)
    assert np.isfinite(rhs_np).all()


def test_train_step_runs_and_gradients_applied(small_risky_model):
    model, rp, tp = small_risky_model

    states = risky_coverage_sampler(tp.batch_size, rp)

    before = [v.numpy().copy() for v in model.policy_net.trainable_variables]

    total_loss, loss_bell, loss_foc, kp, bp = model.train_step(states)

    assert math.isfinite(float(total_loss.numpy()))
    assert math.isfinite(float(loss_bell.numpy()))
    assert math.isfinite(float(loss_foc.numpy()))
    assert kp.shape == (tp.batch_size, 1)
    assert bp.shape == (tp.batch_size, 1)

    after = [v.numpy() for v in model.policy_net.trainable_variables]
    changed = any(not np.allclose(b, a) for b, a in zip(before, after))
    assert changed


def test_compute_residuals_structure_and_finite(small_risky_model):
    model, rp, tp = small_risky_model

    states = risky_coverage_sampler(64, rp)
    k = states[:, 0:1]
    b = states[:, 1:2]
    z = states[:, 2:3]

    res = model.compute_residuals(k, b, z, n_mc_rhs=5)

    expected_keys = {
        "bell_rel_mean",
        "bell_rel_max",
        "foc_mean_sq",
        "foc_max_abs",
        "ll_rel_mean",
        "ll_rel_max",
    }
    assert expected_keys.issubset(res.keys())
    for v in res.values():
        assert math.isfinite(v)
        assert v >= 0.0