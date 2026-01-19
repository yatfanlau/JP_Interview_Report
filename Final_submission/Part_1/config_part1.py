"""
Configuration for Part 1 models.

This module centralizes all tunable parameters for:
- Basic stochastic investment model (Euler-equation solver)
- Risky-debt model (policy/value iteration with risky bond pricing)

"""

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------
# Basic Euler-equation model configuration
# ---------------------------------------------------------------------


@dataclass
class BasicModelParams:
    """Economic primitives for the basic stochastic investment model."""
    theta: float = 0.7        # production curvature in π(k,z)=z*k^theta
    delta: float = 0.10       # depreciation rate
    r: float = 0.04           # risk-free interest rate
    rho: float = 0.70         # AR(1) coefficient for ln z
    sigma_eps: float = 0.15   # std of ln z innovations
    phi: float = 2.0          # convex adjustment-cost curvature
    # bounds for iota = I/k
    iota_upper: float = 0.50        # upper bound for iota
    iota_lower_eps: float = 0.99    # iota_min = -iota_lower_eps*(1-delta)
    # random seed
    seed: int = 421


@dataclass
class BasicTrainingParams:
    """Training and evaluation hyperparameters for the basic model."""
    hidden_sizes: Tuple[int, int] = (64, 64)
    activation: str = "tanh"

    # Replay buffer and sampling
    buffer_size: int = 200_000
    n_paths: int = 2048
    roll_steps: int = 5

    # SGD / batch parameters
    batch_size: int = 4096
    pretrain_steps: int = 1000
    train_steps: int = 60_000
    coverage_final_share: float = 0.10

    # Optimizer
    lr: float = 1e-4

    # Logging and mid-training evaluation
    log_every: int = 200
    eval_every: int = 1000
    eval_gh_nodes: int = 10  # GH order used by mid-training evaluation

    # Fixed coverage evaluation set for mid-training diagnostics
    test_size: int = 20_000
    # kept for API compatibility; not used in GH evaluation
    test_mc: int = 1024

    # Coverage box for coverage_sampler
    k_cov_m_minus: float = 0.2
    k_cov_m_plus: float = 5.0


@dataclass
class BasicFinalTestParams:
    """Final GH-based test configuration for the basic model."""
    burn_in_steps: int = 10_000
    T_on_policy: int = 100_000
    M_coverage: int = 20_000
    q_low: float = 0.01
    q_high: float = 0.99
    expand_frac: float = 0.05
    batch_eval: int = 16_384
    edge_points: int = 50
    tol_list: Tuple[float, float] = (1e-3, 1e-4)
    gh_nodes: int = 10  # GH order used by the final test


# ---------------------------------------------------------------------
# Risky-debt model configuration
# ---------------------------------------------------------------------


@dataclass
class RiskyDebtParams:
    """Economic primitives for the risky-debt model."""
    # Technology and financing environment
    theta: float = 0.30           # capital elasticity in production
    delta: float = 0.10           # depreciation
    phi: float = 0.01             # adjustment cost param
    r: float = 0.04               # risk-free rate
    tau: float = 0.20             # corporate tax rate
    bankruptcy_cost: float = 0.15 # deadweight loss fraction in default

    # Productivity shock: AR(1) in log z
    rho_z: float = 0.90
    sigma_z: float = 0.10
    mu_ln_z: float = 0.0          # intercept in ln z

    # Costly external finance
    eta_0: float = 0.02
    eta_1: float = 0.05

    # Debt vs. savings threshold
    eps_b: float = 1e-3

    # Soft logic temperatures (initial)
    tau_V_init: float = 0.7
    tau_D_init: float = 0.7

    # Risky-debt pricing: inner Monte Carlo draws
    M_pricing: int = 128

    # Policy / value-network architecture
    n_hidden: int = 64

    # Coverage ranges for (k,b,z) for training / testing
    k_cov_min: float = 0.5
    k_cov_max: float = 3.0
    b_cov_min: float = -0.25
    b_cov_max: float = 2.0
    lnz_cov_mean: float = 0.0
    lnz_cov_std: float = 0.1

    # Random seed
    seed: int = 42


@dataclass
class RiskyDebtTrainingParams:
    """Training hyperparameters and schedules for the risky-debt model."""
    batch_size: int = 128
    lr: float = 1e-3

    # Total epochs and sampling schedule
    epochs: int = 2500
    warmup_epochs: int = 400
    onpolicy_ramp_epochs: int = 500
    max_onpolicy_frac: float = 0.85

    # FOC loss weight schedule
    lambda_foc_warmup: float = 0.05
    lambda_foc_final: float = 0.30

    # Bellman RHS sampling inside the double-sampling scheme
    rhs_n_z_draws_train: int = 128

    # Optimization stability
    grad_clip_norm: float = 1.0
    polyak_tau: float = 0.005

    # State evolution during training
    z_clip_min: float = 0.1
    z_clip_max: float = 5.0

    # Exploration / random resets
    reset_prob: float = 0.05
    reset_z_log_std: float = 0.1

    # Diagnostics / logging cadence
    diag_every: int = 50
    diag_n_samples: int = 512
    diag_n_mc_rhs: int = 50

    # Smoothing annealing (kept for completeness; default values preserve behavior)
    tau_min: float = 0.7
    tau_anneal_factor: float = 0.9


@dataclass
class RiskyDebtFinalTestParams:
    """Final out-of-sample diagnostics and plotting settings."""
    # Number of states for coverage and on-policy tests
    n_coverage: int = 2_000
    n_onpolicy: int = 2_000

    # Monte Carlo draws for MC-based diagnostics (Bellman/FOC/limited liability)
    n_mc_rhs: int = 100

    # Ergodic simulation settings
    ergodic_n_firms: int = 256
    ergodic_T: int = 400
    ergodic_burn_in: int = 100
    ergodic_k0_min: float = 0.5
    ergodic_k0_max: float = 2.0

    # Default-probability distribution evaluation
    stats_batch_size: int = 1024
    default_prob_draws: int = 1500

    # Histograms
    hist_bins: int = 40

    # (k,b) grid slices
    grid_z_fixed: float = 0.3
    grid_k_points: int = 100
    grid_b_points: int = 100
    grid_b_indices: Tuple[int, ...] = (0, 24, 49, 74, 99)

    # Default-region contours
    default_region_n_levels: int = 30
    default_region_contour_levels: Tuple[float, ...] = (0.01, 0.02)