# tests/conftest.py
import os

# Must be set before importing matplotlib.pyplot in tests/modules
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import pytest


@pytest.fixture(autouse=True)
def _no_matplotlib_show(monkeypatch):
    """Avoid blocking/hanging tests due to plt.show()."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def risky_small():
    """
    Small RiskyDebtModel fixture for fast unit/integration tests.
    Returns (model, params, train_params).
    """
    from config_part1 import RiskyDebtParams, RiskyDebtTrainingParams
    from risky_debt_model import RiskyDebtModel
    from model_core import set_global_seed, tf, DTYPE

    set_global_seed(0)

    params = RiskyDebtParams(
        n_hidden=8,
        M_pricing=8,
        k_cov_min=0.5,
        k_cov_max=1.5,
        b_cov_min=-0.1,
        b_cov_max=0.5,
        lnz_cov_std=0.05,
        tau_V_init=0.5,
        tau_D_init=0.5,
        seed=0,
    )
    train_params = RiskyDebtTrainingParams(
        batch_size=8,
        rhs_n_z_draws_train=4,
        grad_clip_norm=1.0,
        polyak_tau=0.01,
        # others unused by RiskyDebtModel directly
    )

    model = RiskyDebtModel(params=params, train_params=train_params)
    model.lambda_foc = 0.1  # required by train_step

    # Build nets once so get_weights/set_weights works reliably
    dummy = tf.zeros((1, 3), dtype=DTYPE)
    _ = model.policy_net(dummy)
    _ = model.value_net(dummy)
    _ = model.target_value_net(dummy)

    return model, params, train_params