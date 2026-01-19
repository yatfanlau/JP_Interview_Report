# tests/conftest.py
import math
import numpy as np
import pandas as pd
import pytest

# IMPORTANT: import utils_part2 first so it sets TF env (CPU-only) before TF loads
import utils_part2 as u
tf = u.tf

from config_part2 import PathsPart2, PanelColumnsPart2, BasicModelParams


def _make_constant_policy(value: float) -> "tf.keras.Model":
    """Keras model: input (4,) -> output (1,) constant == value."""
    inp = tf.keras.Input(shape=(4,), dtype=tf.float32)
    out = tf.keras.layers.Dense(
        1,
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(value),
    )(inp)
    return tf.keras.Model(inp, out)


def _save_model_compat(model, base_path_no_suffix):
    """
    Save as .keras if possible; fall back to .h5.
    Returns the actual path string.
    """
    keras_path = str(base_path_no_suffix) + ".keras"
    h5_path = str(base_path_no_suffix) + ".h5"
    try:
        model.save(keras_path, overwrite=True)
        return keras_path
    except Exception:
        model.save(h5_path, overwrite=True)
        return h5_path


@pytest.fixture(scope="session")
def dummy_policy_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("policy")
    model = _make_constant_policy(0.0)
    return _save_model_compat(model, d / "dummy_policy")


@pytest.fixture(scope="session")
def dummy_policy_path_negative(tmp_path_factory):
    d = tmp_path_factory.mktemp("policy_neg")
    model = _make_constant_policy(-1.0)  # makes g = 1-delta+iota <= 0 for delta=0.1
    return _save_model_compat(model, d / "dummy_policy_neg")


@pytest.fixture(scope="session")
def synthetic_panel_csv(tmp_path_factory):
    cols = PanelColumnsPart2()

    # Keep true params within ParamBoundsPart2 defaults.
    theta_true = 0.7
    phi_true = 2.0

    mp = BasicModelParams(theta=theta_true, phi=phi_true)
    delta, rho, sigma_eps = mp.delta, mp.rho, mp.sigma_eps
    mu = -0.5 * (sigma_eps**2) / (1.0 + rho)

    n_reps, n_firms, T = 2, 3, 6

    rows = []
    for rep in range(n_reps):
        for firm in range(n_firms):
            k0 = 1.0 + 0.25 * firm + 0.1 * rep
            lnz0 = -0.02 + 0.05 * (firm - (n_firms - 1) / 2.0) - 0.02 * rep
            for t in range(T):
                rho_t = rho**t
                lnz_t = rho_t * lnz0 + mu * (1.0 - rho_t) / (1.0 - rho)
                z_t = float(math.exp(lnz_t))
                k_t = float(k0 * ((1.0 - delta) ** t))
                iota_t = 0.0

                rows.append(
                    {
                        cols.rep: rep,
                        cols.firm: firm,
                        cols.time: t,
                        cols.k: k_t,
                        cols.z: z_t,
                        cols.iota: iota_t,
                        "theta_true": theta_true,
                        "phi_true": phi_true,
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = tmp_path_factory.mktemp("data")
    csv_path = out_dir / "synthetic_panels.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def paths(synthetic_panel_csv, dummy_policy_path):
    return PathsPart2(data_csv=synthetic_panel_csv, policy_path=dummy_policy_path)