# Configuration for both the basic model and the risky-debt model.

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
    seed: int = 42


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


# ---------------------------------------------------------------------
# Risky-debt model configuration
# ---------------------------------------------------------------------


@dataclass
class RiskyDebtParams:
    """Economic primitives for the risky-debt model."""
    # Technology and financing environment
    theta: float = 0.30           # capital elasticity in production
    delta: float = 0.10           # depreciation
    r: float = 0.04               # risk-free rate
    tau: float = 0.20             # corporate tax rate
    bankruptcy_cost: float = 0.15 # deadweight loss fraction in default

    # Productivity shock: AR(1) in log z
    rho_z: float = 0.90
    sigma_z: float = 0.10

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
    b_cov_min: float = 0.0
    b_cov_max: float = 2.0
    lnz_cov_mean: float = 0.0
    lnz_cov_std: float = 0.1

    # Random seed
    seed: int = 42


@dataclass
class RiskyDebtTrainingParams:
    """Training hyperparameters for the risky-debt model."""
    batch_size: int = 128

    # Replay buffer and sampling
    buffer_size: int = 200_000
    n_paths: int = 2048
    roll_steps: int = 5

    # Training schedule
    pretrain_steps: int = 400      # warm-up phase (coverage only)
    train_steps: int = 10000        # main training phase (coverage + on-policy)
    coverage_final_share: float = 0.15

    # Optimizer
    lr: float = 1e-3

    # Logging and mid-training diagnostics
    log_every: int = 50
    eval_every: int = 50

    # FOC AiO loss weights
    lambda_foc_warmup: float = 0.05
    lambda_foc_final: float = 0.30


@dataclass
class RiskyDebtFinalTestParams:
    """Final out-of-sample diagnostics for the risky-debt model."""
    # Number of states for coverage and on-policy tests
    n_coverage: int = 2_000
    n_onpolicy: int = 2_000

    # Gauss–Hermite nodes for outer expectation over shocks
    gh_nodes: int = 10

    # Monte Carlo draws for MC-based diagnostics (Bellman/FOC/limited liability)
    n_mc_rhs: int = 100