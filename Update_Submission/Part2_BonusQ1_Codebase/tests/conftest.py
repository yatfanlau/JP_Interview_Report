"""Shared fixtures for dyninv package tests."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from dyninv.config import BasicModelParams, PanelColumnsPart2
from dyninv.policy.network import ParamPolicyNet
from dyninv.utils import tf


@pytest.fixture(scope="session")
def dummy_policy_path(tmp_path_factory):
    """Create and persist a tiny deterministic policy model for tests."""
    d = tmp_path_factory.mktemp("policy")
    path = d / "dummy_policy.keras"

    model = ParamPolicyNet(hidden_sizes=(8, 8), activation="tanh")
    _ = model(tf.zeros([1, 4], dtype=tf.float32), training=False)

    # Force near-zero policy output by zeroing all trainable vars.
    for v in model.trainable_variables:
        v.assign(tf.zeros_like(v))

    model.save(str(path))
    return str(path)


@pytest.fixture(scope="session")
def synthetic_panel_csv(tmp_path_factory):
    """Build a small synthetic panel dataset and return its CSV path."""
    cols = PanelColumnsPart2()
    mp = BasicModelParams(theta=0.7, phi=2.0, delta=0.1, rho=0.7, sigma_eps=0.15)

    n_reps, n_firms, t_periods = 2, 5, 10
    mu = -0.5 * (mp.sigma_eps**2) / (1.0 + mp.rho)

    rows = []
    for rep in range(n_reps):
        for firm in range(n_firms):
            k0 = 1.0 + 0.1 * firm + 0.05 * rep
            lnz0 = -0.01 * rep + 0.02 * firm
            for t in range(t_periods):
                lnz_t = (mp.rho**t) * lnz0 + mu * (1.0 - mp.rho**t) / (1.0 - mp.rho)
                z_t = math.exp(lnz_t)
                k_t = k0 * ((1.0 - mp.delta) ** t)
                rows.append(
                    {
                        cols.rep: rep,
                        cols.firm: firm,
                        cols.time: t,
                        cols.k: float(k_t),
                        cols.z: float(z_t),
                        cols.iota: 0.0,
                        "theta_true": float(mp.theta),
                        "phi_true": float(mp.phi),
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = tmp_path_factory.mktemp("data")
    csv_path = out_dir / "synthetic_panels.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)
