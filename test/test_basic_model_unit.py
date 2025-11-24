import math

import numpy as np
import pytest

from common import tf, DTYPE, steady_state_k, ReplayBuffer
from config import BasicModelParams, BasicTrainingParams
from basic_model import (
    PolicyNet,
    coverage_sampler,
    rollout_on_policy,
    BasicTrainer,
)


def test_policy_net_output_bounds():
    mp = BasicModelParams()
    net = PolicyNet(mp=mp)

    B = 64
    x = tf.random.uniform((B, 2), minval=-1.0, maxval=1.0, dtype=DTYPE)
    iota = net(x, training=False)  # [B, 1]

    iota_min = -(mp.iota_lower_eps) * (1.0 - mp.delta)
    iota_max = mp.iota_upper

    iota_np = iota.numpy()
    assert iota_np.shape == (B, 1)
    assert (iota_np >= iota_min - 1e-6).all()
    assert (iota_np <= iota_max + 1e-6).all()


def test_coverage_sampler_shape_and_ranges():
    mp = BasicModelParams()
    B = 512

    k, z = coverage_sampler(B, mp)
    k_np = k.numpy()
    z_np = z.numpy()

    assert k_np.shape == (B,)
    assert z_np.shape == (B,)
    assert (k_np > 0).all()
    assert (z_np > 0).all()

    k_star = steady_state_k(mp.theta, mp.delta, mp.r)
    m_minus = 0.2
    m_plus = 5.0
    assert (k_np >= m_minus * k_star - 1e-6).all()
    assert (k_np <= m_plus * k_star + 1e-6).all()


def test_rollout_on_policy_buffer_growth():
    mp = BasicModelParams()
    policy = PolicyNet(mp=mp)
    N_paths = 32
    steps = 3

    k0 = np.ones(N_paths, dtype=np.float32) * np.float32(
        steady_state_k(mp.theta, mp.delta, mp.r)
    )
    z0 = np.ones(N_paths, dtype=np.float32)

    buffer = ReplayBuffer(max_size=1000, state_dim=2, seed=0)
    k1, z1 = rollout_on_policy(
        policy,
        current_states=(k0, z0),
        mp=mp,
        n_steps=steps,
        buffer=buffer,
    )

    assert len(buffer) == N_paths * steps
    assert k1.shape == (N_paths,)
    assert z1.shape == (N_paths,)


def test_basic_trainer_euler_aio_loss_finite():
    mp = BasicModelParams(seed=123)
    tp = BasicTrainingParams(
        hidden_sizes=(16, 16),
        batch_size=64,
        buffer_size=1000,
        n_paths=64,
        roll_steps=2,
        pretrain_steps=10,
        train_steps=20,
        coverage_final_share=0.2,
        lr=1e-3,
        log_every=1000,
        eval_every=1000,
        test_size=256,
        k_cov_m_minus=0.2,
        k_cov_m_plus=2.0,
    )

    trainer = BasicTrainer(mp=mp, tp=tp)

    B = 32
    k_batch, z_batch = coverage_sampler(B, mp)
    loss = trainer._euler_aio_loss(k_batch, z_batch)

    loss_val = float(loss.numpy())
    assert math.isfinite(loss_val)


def test_eval_batched_consistency_with_different_batch_sizes():
    mp = BasicModelParams(seed=321)
    tp = BasicTrainingParams(
        hidden_sizes=(16, 16),
        batch_size=64,
        buffer_size=1000,
        n_paths=64,
        roll_steps=2,
        pretrain_steps=1,
        train_steps=1,
        coverage_final_share=0.5,
        lr=1e-3,
        log_every=1000,
        eval_every=1000,
        test_size=128,
        k_cov_m_minus=0.2,
        k_cov_m_plus=2.0,
    )
    trainer = BasicTrainer(mp=mp, tp=tp)

    N = 100
    k_np, z_np = trainer._build_fixed_eval_coverage(N)

    g1, d1 = trainer._eval_batched(k_np, z_np, n_nodes=5, batch=N)
    g2, d2 = trainer._eval_batched(k_np, z_np, n_nodes=5, batch=10)

    assert g1.shape == g2.shape == (N,)
    assert d1.shape == d2.shape == (N,)
    assert np.allclose(g1, g2)
    assert np.allclose(d1, d2)


def test_build_fixed_eval_coverage_reproducible():
    mp = BasicModelParams(seed=999)
    tp = BasicTrainingParams()
    trainer = BasicTrainer(mp=mp, tp=tp)

    k1, z1 = trainer._build_fixed_eval_coverage(200)
    k2, z2 = trainer._build_fixed_eval_coverage(200)

    assert np.allclose(k1, k2)
    assert np.allclose(z1, z2)