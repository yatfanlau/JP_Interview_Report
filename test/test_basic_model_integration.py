import numpy as np
import pytest

from config import BasicModelParams, BasicTrainingParams, BasicFinalTestParams
from basic_model import BasicTrainer
from common import set_global_seed


@pytest.mark.slow
def test_basic_model_short_training_and_final_test():
    set_global_seed(123)

    mp = BasicModelParams(seed=123)

    tp = BasicTrainingParams(
        hidden_sizes=(16, 16),
        activation="tanh",
        buffer_size=2_000,
        n_paths=64,
        roll_steps=2,
        batch_size=64,
        pretrain_steps=5,
        train_steps=10,
        coverage_final_share=0.5,
        lr=5e-4,
        log_every=1000,
        eval_every=1000,
        test_size=512,
        k_cov_m_minus=0.2,
        k_cov_m_plus=2.0,
    )

    trainer = BasicTrainer(mp=mp, tp=tp)
    trainer.train()  # short run

    acc = trainer._eval_accuracy()
    assert "MAE" in acc and np.isfinite(acc["MAE"])
    assert acc["MAE"] >= 0.0

    fp = BasicFinalTestParams(
        burn_in_steps=5,
        T_on_policy=200,
        M_coverage=200,
        q_low=0.05,
        q_high=0.95,
        expand_frac=0.05,
        batch_eval=512,
        edge_points=10,
        tol_list=(1e-2, 1e-3),
    )

    trainer.final_test(fp)  # should run without errors