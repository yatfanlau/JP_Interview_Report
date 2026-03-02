"""Typed configuration objects for model, training, and final diagnostics.

The dataclasses in this module are intentionally lightweight containers.
They keep experiment settings explicit and serializable, while separating
economic assumptions from optimization/runtime choices.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BasicModelParams:
    """Economic primitives for the stochastic investment environment.

    These parameters define the technology, adjustment costs, and stochastic
    productivity process used consistently by VFI, DL training, and all
    downstream evaluation routines.
    """

    theta: float = 0.7  # Production elasticity with respect to capital.
    delta: float = 0.10  # Per-period depreciation rate.
    r: float = 0.04  # Interest rate determining the discount factor.
    rho: float = 0.70  # AR(1) persistence for log productivity.
    sigma_eps: float = 0.15  # Innovation volatility for log productivity.
    phi: float = 2.0  # Quadratic adjustment-cost curvature.
    iota_upper: float = 0.50  # Maximum feasible investment rate I/k.
    iota_lower_eps: float = 0.99  # Lower-bound factor for feasible disinvestment.
    seed: int = 42  # RNG seed for reproducibility.


@dataclass
class BasicTrainingParams:
    """Hyperparameters that govern policy-network training behavior.

    The fields here cover architecture, replay-buffer usage, sampling mixes,
    and optimizer cadence. They are grouped in one object so a run can be
    reproduced from a single parameter snapshot.
    """

    hidden_sizes: Tuple[int, int] = (64, 64)  # Two-layer MLP widths for policy net.
    activation: str = "tanh"  # Hidden-layer activation function.
    buffer_size: int = 200_000  # Replay-buffer capacity in number of states.
    n_paths: int = 2048  # Parallel rollout paths for on-policy state generation.
    roll_steps: int = 5  # Rollout refresh cadence during optimization.
    batch_size: int = 4096  # Number of states per SGD minibatch.
    pretrain_steps: int = 0  # Coverage-only warm-up optimization steps.
    train_steps: int = 45_000  # Main hybrid-training optimization steps.
    coverage_final_share: float = 0.10  # Final floor for coverage-sampled fraction.
    lr: float = 1e-4  # Adam learning rate.
    log_every: int = 200  # Logging cadence in optimizer steps.
    eval_every: int = 1000  # Reserved periodic-evaluation cadence.
    test_size: int = 20_000  # Reserved test-set size for compatibility.
    test_mc: int = 1024  # Reserved Monte Carlo sample count for compatibility.
    k_cov_m_minus: float = 0.2  # Coverage lower bound multiplier for k.
    k_cov_m_plus: float = 5.0  # Coverage upper bound multiplier for k.


@dataclass
class BasicFinalTestParams:
    """Settings for the final post-training diagnostic pass.

    These controls are primarily used when computing robust residual summaries
    on large on-policy samples and stress-test edge cases.
    """

    burn_in_steps: int = 10_000  # Burn-in transitions before on-policy collection.
    T_on_policy: int = 100_000  # Count of on-policy states for final tests.
    M_coverage: int = 20_000  # Count of coverage states for final tests.
    q_low: float = 0.01  # Lower quantile for robust state-range diagnostics.
    q_high: float = 0.99  # Upper quantile for robust state-range diagnostics.
    expand_frac: float = 0.05  # Quantile-range expansion fraction.
    batch_eval: int = 16_384  # Batch size used for final residual evaluation.
    edge_points: int = 50  # Number of explicit edge points in stress diagnostics.
    tol_list: Tuple[float, float] = (1e-3, 1e-4)  # Tolerance thresholds for reporting.
