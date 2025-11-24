import numpy as np
import pytest

from common import tf, DTYPE, risky_coverage_sampler, set_global_seed
from config import RiskyDebtParams, RiskyDebtTrainingParams, RiskyDebtFinalTestParams
from risky_debt_model import RiskyDebtModel


@pytest.mark.slow
def test_risky_debt_short_training_and_diagnostics():
    rp = RiskyDebtParams(seed=2024)
    tp = RiskyDebtTrainingParams(
        batch_size=32,
        buffer_size=200_000,
        n_paths=2048,
        roll_steps=5,
        pretrain_steps=400,
        train_steps=10_000,
        coverage_final_share=0.15,
        lr=1e-3,
        log_every=50,
        eval_every=50,
        lambda_foc_warmup=0.05,
        lambda_foc_final=0.30,
    )
    ftp = RiskyDebtFinalTestParams(
        n_coverage=200,
        n_onpolicy=200,
        gh_nodes=10,
        n_mc_rhs=20,
    )

    set_global_seed(rp.seed)
    model = RiskyDebtModel(params=rp, train_params=tp)

    # Initial states from coverage
    states = risky_coverage_sampler(tp.batch_size, rp)
    current_states = states

    # Short training loop
    for epoch in range(20):
        total_loss, loss_bell, loss_foc, kp, bp = model.train_step(current_states)

        z_curr = current_states[:, 2:3]
        z_next = model.get_next_z(z_curr, 1)
        next_states = tf.concat([kp, bp, z_next], axis=1)

        next_states = tf.clip_by_value(
            next_states,
            clip_value_min=tf.constant([0.1, -0.5, 0.1], dtype=DTYPE),
            clip_value_max=tf.constant([5.0, 5.0, 5.0], dtype=DTYPE),
        )
        current_states = next_states

        assert np.isfinite(float(total_loss.numpy()))
        assert np.isfinite(float(loss_bell.numpy()))
        assert np.isfinite(float(loss_foc.numpy()))

    # Coverage diagnostics
    cov_diag = model.test_coverage_residuals(
        n_samples=ftp.n_coverage,
        n_mc_rhs=ftp.n_mc_rhs,
    )
    assert "bell_rel_mean" in cov_diag
    assert np.isfinite(cov_diag["bell_rel_mean"])

    # Pseudo-ergodic panel from last states
    states_erg = current_states.numpy()
    reps = max(1, ftp.n_onpolicy // states_erg.shape[0])
    states_erg_tiled = np.tile(states_erg, (reps, 1))[: ftp.n_onpolicy]
    states_erg_tf = tf.convert_to_tensor(states_erg_tiled, dtype=DTYPE)

    onp_diag = model.test_onpolicy_residuals(
        states_erg_tf,
        n_samples=ftp.n_onpolicy,
        n_mc_rhs=ftp.n_mc_rhs,
    )
    assert "bell_rel_mean" in onp_diag
    assert np.isfinite(onp_diag["bell_rel_mean"])