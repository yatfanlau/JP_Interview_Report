# Configuration for both the basic model and the risky-debt model.

from dataclasses import dataclass
from typing import Tuple, Optional


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
    seed: int = 424


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
    train_steps: int = 40000
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
# Part 2 (SMM/GMM single) configuration
# ---------------------------------------------------------------------


@dataclass
class PathsPart2:
    data_csv: str = "synthetic_panels/synthetic_panels_all.csv"
    policy_path: str = "param_policy_theta_phi.keras"
    # optional: if we generate a dedicated HMC dataset (not required by hmc_estimation.py)
    hmc_data_csv: str = "synthetic_panels/synthetic_panels_hmc.csv"


@dataclass
class PanelColumnsPart2:
    rep: str = "rep"
    firm: str = "firm"
    time: str = "t"
    k: str = "k"
    z: str = "z"
    iota: str = "iota"


@dataclass
class ParamBoundsPart2:
    theta_min: float = 0.5
    theta_max: float = 0.9
    phi_min: float = 0.5
    phi_max: float = 5.0


@dataclass
class SMMConfigPart2:
    n_reps_eval: int = 100

    n_firms_sim: int = 200
    t_data: int = 80
    t_burnin: int = 200

    theta_init_guess: float = 0.55
    phi_init_guess: float = 1.1

    steps_1: int = 150
    steps_2: int = 100
    lr_1: float = 0.01
    lr_2: float = 0.005

    sims_per_obj_1: int = 1
    sims_per_obj_2: int = 1

    crn_base_seed: int = 1234

    boot_seed_base: int = 10000
    sim_seed_base: int = 50000

    w_n_boot: int = 30
    w_n_sims: int = 15
    w_ridge: float = 1e-8
    w_rcond: float = 1e-10
    include_sim_var_in_W: bool = True

    metric2_alpha: float = 0.05
    metric2_n_boot: int = 150
    metric2_n_sims: int = 80
    metric2_use_optimal_weight: bool = True
    metric2_ridge: float = 1e-8

    metric3_burnin: int = 400
    metric3_sim_len: int = 200
    metric3_n_firms_sim: int = 200
    metric3_seed_base: int = 777

    jtest_alpha: float = 0.05
    jtest_n_boot: int = 50
    jtest_n_sims: int = 30
    jtest_ridge: float = 1e-8

    diag_chi2_n_mc: int = 100000
    diag_chi2_seed: int = 12345


@dataclass
class GMMConfigPart2:
    n_reps_eval: int = 100

    theta_init_guess: float = 0.55
    phi_init_guess: float = 1.1

    steps_1: int = 250
    steps_2: int = 200
    lr_1: float = 0.02
    lr_2: float = 0.01

    w_ridge: float = 1e-8
    pinv_rcond: float = 1e-10

    standardize_instr: bool = True
    instrument_names: Tuple[str, ...] = (
        "1",
        "logk",
        "lnz",
        "iota",
        "logk^2",
        "lnz^2",
        "iota^2",
    )

    metric2_alpha: float = 0.05
    metric2_use_optimal_weight: bool = True
    metric2_ridge: float = 1e-8

    metric3_burnin: int = 400
    metric3_sim_len: int = 200
    metric3_n_firms_sim: int = 200
    metric3_seed_base: int = 777

    hansen_alpha: float = 0.05

    diag_chi2_n_mc: int = 100000
    diag_chi2_seed: int = 12345


# ---------------------------------------------------------------------
# Part 2 (Bayesian) configuration: UKF + HMC (theta, phi only)
# ---------------------------------------------------------------------


@dataclass
class HMCColumnsPart2:
    rep: str = "rep"
    firm: str = "firm"
    time: str = "t"
    y: str = "y"              # observed log profits
    logk_obs: str = "logk_obs"  # observed log capital


@dataclass
class HMCConfigPart2:
    # how many replications to estimate
    n_reps_eval: int = 50
    rep_ids: Tuple[int, ...] = ()  # if non-empty, overrides n_reps_eval selection

    # init guess (in (theta_min,theta_max) and (phi_min,phi_max))
    theta_init_guess: float = 0.65
    phi_init_guess: float = 1.8

    # observation construction (if we don't use a separate CSV)
    obs_seed_base: int = 90000
    sigma_y: float = 0.05          # measurement noise in y = ln profit
    sigma_logk_obs: float = 0.02   # measurement noise in ln k_obs

    # state transition noise (decision shock) in ln k_{t+1}
    sigma_logk_trans: float = 0.01

    # UKF parameters (scaled unscented transform)
    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_jitter: float = 1e-6

    # initial prior for latent state s0 = (x0, ln k0)
    init_x_mean: Optional[float] = None  # None -> stationary mean
    init_x_var: Optional[float] = None   # None -> stationary var
    init_logk_var: float = 0.25          # fairly diffuse prior var for ln k0 (obs at t=0 anchors it)

    # HMC sampling
    num_results: int = 300
    num_burnin: int = 300
    num_adaptation_steps: int = 250
    step_size: float = 0.03
    num_leapfrog_steps: int = 6
    target_accept: float = 0.75
    seed: int = 20260118

    # posterior summary
    cred_level: float = 0.95

    # Metric 3 simulation settings
    metric3_burnin: int = 400
    metric3_sim_len: int = 200
    metric3_n_firms_sim: int = 200
    metric3_seed_base: int = 777