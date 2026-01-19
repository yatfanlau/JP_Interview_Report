# tests/test_risky_debt_model_unit.py
import numpy as np

from model_core import tf, DTYPE


def _set_keras_model_all_zero(keras_model):
    weights = keras_model.get_weights()
    keras_model.set_weights([np.zeros_like(w) for w in weights])


def _set_value_net_constant(model, c: float):
    """
    Force value_net to output constant C=c for any input by zeroing all weights
    and setting final-layer bias to c.
    """
    weights = model.value_net.get_weights()
    new = [np.zeros_like(w) for w in weights]
    new[-1] = np.array([c], dtype=np.float32)  # last bias
    model.value_net.set_weights(new)


def test_coverage_sampler_bounds(risky_small):
    model, params, _ = risky_small
    states = model.coverage_sampler(500).numpy()
    assert states.shape == (500, 3)

    k = states[:, 0]
    b = states[:, 1]
    z = states[:, 2]

    assert np.all(k >= params.k_cov_min - 1e-6)
    assert np.all(k <= params.k_cov_max + 1e-6)
    assert np.all(b >= params.b_cov_min - 1e-6)
    assert np.all(b <= params.b_cov_max + 1e-6)
    assert np.all(z > 0.0)
    assert np.all(np.isfinite(states))


def test_state_features_safe_for_small_k_and_z(risky_small):
    model, _, _ = risky_small
    k = tf.constant([[0.0], [1e-9]], dtype=DTYPE)
    b = tf.constant([[1.0], [-1.0]], dtype=DTYPE)
    z = tf.constant([[0.0], [1e-12]], dtype=DTYPE)
    feats = model.state_features(k, b, z).numpy()
    assert feats.shape == (2, 3)
    assert np.all(np.isfinite(feats))


def test_get_next_z_shape_and_positive(risky_small):
    model, _, _ = risky_small
    z = tf.ones((4, 1), dtype=DTYPE)
    z_next = model.get_next_z(z, n_samples=7).numpy()
    assert z_next.shape == (4, 7)
    assert np.all(z_next > 0.0)
    assert np.all(np.isfinite(z_next))


def test_adjustment_cost_zero_at_replacement_investment(risky_small):
    model, _, _ = risky_small
    k = tf.constant([[2.0]], dtype=DTYPE)

    # Replacement investment implies k' = k (given the cost definition here)
    k_prime = tf.identity(k)
    cost = model.adjustment_cost(k, k_prime).numpy()
    assert cost.shape == (1, 1)
    assert abs(float(cost[0, 0])) < 1e-6


def test_pricing_kernel_riskfree_when_no_debt(risky_small):
    model, params, _ = risky_small
    B = 6
    k_prime = tf.ones((B, 1), dtype=DTYPE) * 1.1
    b_prime = tf.zeros((B, 1), dtype=DTYPE)  # no debt
    z = tf.ones((B, 1), dtype=DTYPE)

    q = model.pricing_kernel(k_prime, b_prime, z).numpy()
    q_rf = 1.0 / (1.0 + params.r)
    assert q.shape == (B, 1)
    assert np.allclose(q, q_rf, atol=1e-6)


def test_pricing_kernel_bounded_when_debt_positive(risky_small):
    model, params, _ = risky_small
    B = 6
    k_prime = tf.ones((B, 1), dtype=DTYPE) * 1.1
    b_prime = tf.ones((B, 1), dtype=DTYPE) * 0.2  # positive debt
    z = tf.ones((B, 1), dtype=DTYPE)

    q = model.pricing_kernel(k_prime, b_prime, z).numpy()
    q_rf = 1.0 / (1.0 + params.r)

    assert np.all(q >= 0.01 - 1e-7)
    assert np.all(q <= q_rf + 1e-7)


def test_compute_equity_value_softplus_limits(risky_small):
    model, _, _ = risky_small
    B = 4
    k = tf.ones((B, 1), dtype=DTYPE)
    b = tf.zeros((B, 1), dtype=DTYPE)
    z = tf.ones((B, 1), dtype=DTYPE)

    # Force constant positive C
    _set_value_net_constant(model, c=10.0)
    C_pos, V_pos = model.compute_equity_value(k, b, z, use_target=False)
    C_pos = C_pos.numpy()
    V_pos = V_pos.numpy()

    assert np.allclose(C_pos, 10.0, atol=1e-6)
    # For large positive C, softplus smoothing should be ~ identity
    assert np.allclose(V_pos, 10.0, atol=1e-3)

    # Force constant negative C
    _set_value_net_constant(model, c=-10.0)
    C_neg, V_neg = model.compute_equity_value(k, b, z, use_target=False)
    V_neg = V_neg.numpy()
    assert np.all(V_neg >= 0.0)
    assert np.max(V_neg) < 1e-4  # near zero


def test_default_probability_zero_when_policy_chooses_no_debt(risky_small):
    model, _, _ = risky_small

    # Force policy to pick b' = 0 always
    _set_keras_model_all_zero(model.policy_net)

    B = 5
    k = tf.ones((B, 1), dtype=DTYPE)
    b = tf.zeros((B, 1), dtype=DTYPE)
    z = tf.ones((B, 1), dtype=DTYPE)

    p = model.default_probability(k, b, z, n_draws=20, use_target=True).numpy()
    assert p.shape == (B, 1)
    assert np.allclose(p, 0.0, atol=1e-7)


def test_train_step_runs_and_increments_optimizer_iterations(risky_small):
    model, _, train_params = risky_small
    it0 = int(model.optimizer.iterations.numpy())

    states = model.coverage_sampler(train_params.batch_size)
    total_loss, loss_bell, loss_foc, kp, bp = model.train_step(states)

    assert np.isfinite(float(total_loss.numpy()))
    assert np.isfinite(float(loss_bell.numpy()))
    assert np.isfinite(float(loss_foc.numpy()))
    assert kp.shape == (train_params.batch_size, 1)
    assert bp.shape == (train_params.batch_size, 1)

    it1 = int(model.optimizer.iterations.numpy())
    assert it1 == it0 + 1