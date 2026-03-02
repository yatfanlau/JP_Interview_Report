"""Unit tests for model primitives and policy network behavior.

These tests validate shape/feasibility properties that many training and
evaluation routines implicitly rely on.
"""

from __future__ import annotations

from investment_dl.config import BasicModelParams
from investment_dl.core.tf_env import DTYPE, tf
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.models.policies import PolicyNetwork


def test_basic_investment_model_primitives_shapes() -> None:
    """Model primitive helpers should return correctly shaped tensors.

    The test also checks that next-period capital respects the positivity floor.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=7))

    k = tf.constant([model.k_star, 1.1 * model.k_star], dtype=DTYPE)
    z = tf.constant([1.0, 1.2], dtype=DTYPE)
    iota = tf.constant([0.0, 0.1], dtype=DTYPE)

    # Policy input should stack log-state features into two columns.
    x = model.policy_input(k, z)
    assert tuple(x.shape) == (2, 2)

    profit_k = model.profit_k(k, z)
    assert tuple(profit_k.shape) == (2,)

    # Transition equation should preserve vector shape and nonnegativity floor.
    next_k = model.next_capital(k, iota)
    assert tuple(next_k.shape) == (2,)
    assert float(tf.reduce_min(next_k).numpy()) >= float(model.k_floor.numpy())


def test_coverage_sampler_returns_positive_states() -> None:
    """Coverage sampler should return positive ``(k,z)`` states in bounds.

    This checks both positivity and support implied by ``m_minus``/``m_plus``.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=9))
    k, z = model.coverage_sampler(batch_size=256, m_minus=0.5, m_plus=1.5)

    assert tuple(k.shape) == (256,)
    assert tuple(z.shape) == (256,)
    assert float(tf.reduce_min(k).numpy()) > 0.0
    assert float(tf.reduce_min(z).numpy()) > 0.0

    # k-draws should stay within the configured multiplicative steady-state band.
    k_min = 0.5 * model.k_star
    k_max = 1.5 * model.k_star
    assert float(tf.reduce_min(k).numpy()) >= k_min * (1.0 - 1e-6)
    assert float(tf.reduce_max(k).numpy()) <= k_max * (1.0 + 1e-6)


def test_policy_network_respects_bounds() -> None:
    """Policy network output should always satisfy configured iota bounds.

    Because the architecture uses a bounded output layer, this property should
    hold even for arbitrary random feature inputs.
    """
    model = BasicInvestmentModel(BasicModelParams(seed=11))
    policy = PolicyNetwork(
        iota_min=model.iota_min,
        iota_max=model.iota_max,
        hidden_sizes=(8, 8),
        activation="tanh",
    )

    # Random features stress-test bound enforcement across a large batch.
    x = tf.random.normal((512, 2), dtype=DTYPE)
    iota = policy(x, training=False)[:, 0]

    assert tuple(iota.shape) == (512,)
    assert float(tf.reduce_min(iota).numpy()) >= model.iota_min - 1e-6
    assert float(tf.reduce_max(iota).numpy()) <= model.iota_max + 1e-6
