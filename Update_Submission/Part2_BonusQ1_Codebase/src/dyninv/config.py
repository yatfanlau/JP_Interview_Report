"""Dataclass configuration schemas for the dyninv package.

The module groups default values for model primitives, training hyperparameters,
data paths, and estimator-specific settings so workflows can be configured in a
single, typed, discoverable place.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BasicModelParams:
    """Economic primitives for the baseline stochastic investment model.

    These parameters govern production, depreciation, shocks, adjustment costs,
    policy-action bounds, and reproducible random-number generation.
    """

    theta: float = 0.7  # Production curvature in pi(k, z) = z * k^theta.
    delta: float = 0.10  # Capital depreciation rate.
    r: float = 0.04  # Risk-free interest rate.
    rho: float = 0.70  # AR(1) persistence for log productivity.
    sigma_eps: float = 0.15  # Innovation standard deviation for log productivity.
    phi: float = 2.0  # Convex adjustment-cost curvature.
    iota_upper: float = 0.50  # Upper policy bound for iota = I / k.
    iota_lower_eps: float = 0.99  # Lower policy scaling; min iota = -eps * (1 - delta).
    seed: int = 424  # Global random seed for reproducibility.


@dataclass
class BasicTrainingParams:
    """Hyperparameters controlling policy training and diagnostics."""

    hidden_sizes: Tuple[int, int] = (64, 64)  # Hidden widths of the policy MLP.
    activation: str = "tanh"  # Activation function used in hidden layers.
    buffer_size: int = 200_000  # Replay buffer capacity (number of states).
    n_paths: int = 2048  # Number of on-policy paths used during rollout updates.
    roll_steps: int = 5  # Rollout frequency in SGD steps.
    batch_size: int = 4096  # Minibatch size for SGD updates.
    pretrain_steps: int = 1000  # Number of coverage-only warm-up updates.
    train_steps: int = 40000  # Number of hybrid replay+coverage updates.
    coverage_final_share: float = 0.10  # Floor share of coverage samples late in training.
    lr: float = 1e-4  # Adam learning rate.
    log_every: int = 200  # Logging interval (steps).
    eval_every: int = 1000  # Diagnostic evaluation interval (steps).
    test_size: int = 20_000  # Size of diagnostic evaluation sample.
    test_mc: int = 1024  # Number of Monte Carlo draws for diagnostics.
    k_cov_m_minus: float = 0.2  # Lower coverage multiplier for k / k*.
    k_cov_m_plus: float = 5.0  # Upper coverage multiplier for k / k*.


@dataclass
class BasicFinalTestParams:
    """Configuration for final grid/history-based policy diagnostics."""

    burn_in_steps: int = 10_000  # Burn-in transitions before final diagnostics.
    t_on_policy: int = 100_000  # On-policy horizon length for final evaluation.
    m_coverage: int = 20_000  # Number of coverage points in final grid checks.
    q_low: float = 0.01  # Lower quantile for evaluation support.
    q_high: float = 0.99  # Upper quantile for evaluation support.
    expand_frac: float = 0.05  # Fractional expansion of evaluation interval.
    batch_eval: int = 16_384  # Batch size for final vectorized evaluation.
    edge_points: int = 50  # Number of edge points used in boundary diagnostics.
    tol_list: Tuple[float, float] = (1e-3, 1e-4)  # Error tolerances for pass/fail checks.


@dataclass
class PathsPart2:
    """Filesystem paths for policy artifacts and generated datasets."""

    data_csv: str = "synthetic_panels/synthetic_panels_all.csv"  # Main synthetic panel CSV.
    policy_path: str = "param_policy_theta_phi.keras"  # Saved amortized policy model path.
    hmc_data_csv: str = "synthetic_panels/synthetic_panels_hmc.csv"  # Optional HMC-specific CSV.


@dataclass
class PanelColumnsPart2:
    """Canonical column names used by panel datasets."""

    rep: str = "rep"  # Replication identifier.
    firm: str = "firm"  # Firm identifier.
    time: str = "t"  # Time index.
    k: str = "k"  # Capital level.
    z: str = "z"  # Productivity level.
    iota: str = "iota"  # Investment rate I / k.


@dataclass
class ParamBoundsPart2:
    """Feasible bounds for structural parameters in estimation routines."""

    theta_min: float = 0.5  # Lower bound for theta.
    theta_max: float = 0.9  # Upper bound for theta.
    phi_min: float = 0.5  # Lower bound for phi.
    phi_max: float = 5.0  # Upper bound for phi.


@dataclass
class SMMConfigPart2:
    """Settings for two-step simulation method of moments estimation.

    Includes optimizer controls, Monte Carlo simulation sizes, covariance
    estimation options, and optional downstream metric diagnostics.
    """

    n_reps_eval: int = 100  # Number of replications to estimate.
    n_firms_sim: int = 200  # Number of simulated firms per objective evaluation.
    t_data: int = 80  # Retained simulation periods.
    t_burnin: int = 200  # Burn-in simulation periods.
    theta_init_guess: float = 0.55  # Initial theta value for step-1 optimization.
    phi_init_guess: float = 1.1  # Initial phi value for step-1 optimization.
    steps_1: int = 150  # Step-1 optimizer iterations.
    steps_2: int = 100  # Step-2 optimizer iterations.
    lr_1: float = 0.01  # Step-1 learning rate.
    lr_2: float = 0.005  # Step-2 learning rate.
    sims_per_obj_1: int = 1  # Simulations averaged per step-1 objective call.
    sims_per_obj_2: int = 1  # Simulations averaged per step-2 objective call.
    crn_base_seed: int = 1234  # Base seed for common-random-number draws.
    boot_seed_base: int = 10000  # Base seed for bootstrap covariance draws.
    sim_seed_base: int = 50000  # Base seed for simulation covariance draws.
    w_n_boot: int = 30  # Bootstrap draws for weighting-matrix estimation.
    w_n_sims: int = 15  # Simulation draws for weighting-matrix estimation.
    w_ridge: float = 1e-8  # Ridge regularization added to covariance estimates.
    w_rcond: float = 1e-10  # Reciprocal condition threshold for pseudo-inverse.
    include_sim_var_in_W: bool = True  # Include simulation variance in weighting matrix.
    metric2_alpha: float = 0.05  # Significance level for metric-2 intervals.
    metric2_n_boot: int = 150  # Bootstrap draws for metric-2 standard errors.
    metric2_n_sims: int = 80  # Simulation draws for metric-2 variance terms.
    metric2_use_optimal_weight: bool = True  # Use optimal weighting for metric-2 delta method.
    metric2_ridge: float = 1e-8  # Ridge added in metric-2 covariance inversion.
    metric3_burnin: int = 400  # Burn-in periods for metric-3 auxiliary simulations.
    metric3_sim_len: int = 200  # Retained periods for metric-3 auxiliary simulations.
    metric3_n_firms_sim: int = 200  # Number of firms in metric-3 auxiliary simulations.
    metric3_seed_base: int = 777  # Base seed for metric-3 simulations.
    jtest_alpha: float = 0.05  # Significance level for overidentification J-test.
    jtest_n_boot: int = 50  # Bootstrap draws for J-test uncertainty.
    jtest_n_sims: int = 30  # Simulation draws for J-test uncertainty.
    jtest_ridge: float = 1e-8  # Ridge regularization used in J-test covariance.
    diag_chi2_n_mc: int = 100000  # Monte Carlo draws for chi-square diagnostics.
    diag_chi2_seed: int = 12345  # Seed for chi-square diagnostic draws.


@dataclass
class GMMConfigPart2:
    """Settings for two-step generalized method of moments estimation."""

    n_reps_eval: int = 100  # Number of replications to estimate.
    theta_init_guess: float = 0.55  # Initial theta value for step-1 optimization.
    phi_init_guess: float = 1.1  # Initial phi value for step-1 optimization.
    steps_1: int = 250  # Step-1 optimizer iterations.
    steps_2: int = 200  # Step-2 optimizer iterations.
    lr_1: float = 0.02  # Step-1 learning rate.
    lr_2: float = 0.01  # Step-2 learning rate.
    w_ridge: float = 1e-8  # Ridge regularization added to covariance estimates.
    pinv_rcond: float = 1e-10  # Reciprocal condition threshold for pseudo-inverse.
    standardize_instr: bool = True  # Standardize non-constant instruments.
    instrument_names: Tuple[str, ...] = (
        "1",
        "logk",
        "lnz",
        "iota",
        "logk^2",
        "lnz^2",
        "iota^2",
    )  # Names of instrument columns used in reporting.
    metric2_alpha: float = 0.05  # Significance level for metric-2 intervals.
    metric2_use_optimal_weight: bool = True  # Use optimal-weight covariance simplification.
    metric2_ridge: float = 1e-8  # Ridge added in metric-2 covariance inversion.
    metric3_burnin: int = 400  # Burn-in periods for metric-3 auxiliary simulations.
    metric3_sim_len: int = 200  # Retained periods for metric-3 auxiliary simulations.
    metric3_n_firms_sim: int = 200  # Number of firms in metric-3 auxiliary simulations.
    metric3_seed_base: int = 777  # Base seed for metric-3 simulations.
    hansen_alpha: float = 0.05  # Significance level for Hansen J-test.
    diag_chi2_n_mc: int = 100000  # Monte Carlo draws for chi-square diagnostics.
    diag_chi2_seed: int = 12345  # Seed for chi-square diagnostic draws.


@dataclass
class HMCColumnsPart2:
    """Column names expected by Bayesian observation datasets."""

    rep: str = "rep"  # Replication identifier.
    firm: str = "firm"  # Firm identifier.
    time: str = "t"  # Time index.
    y: str = "y"  # Observed output/profit measurement.
    logk_obs: str = "logk_obs"  # Observed log capital measurement.


@dataclass
class HMCConfigPart2:
    """Settings for Bayesian HMC estimation and posterior summaries."""

    n_reps_eval: int = 50  # Number of replications to estimate.
    rep_ids: Tuple[int, ...] = ()  # Optional explicit list of replication IDs.
    theta_init_guess: float = 0.65  # Initial theta value for HMC initialization.
    phi_init_guess: float = 1.8  # Initial phi value for HMC initialization.
    obs_seed_base: int = 90000  # Base seed for synthetic observation construction.
    sigma_y: float = 0.05  # Observation noise standard deviation for y.
    sigma_logk_obs: float = 0.02  # Observation noise standard deviation for log k.
    sigma_logk_trans: float = 0.01  # Transition noise standard deviation for log k.
    ukf_alpha: float = 1.0  # UKF spread parameter alpha.
    ukf_beta: float = 2.0  # UKF distribution parameter beta.
    ukf_kappa: float = 0.0  # UKF secondary spread parameter kappa.
    ukf_jitter: float = 1e-6  # UKF covariance jitter for numerical stability.
    init_x_mean: Optional[float] = None  # Optional initial mean for latent log productivity.
    init_x_var: Optional[float] = None  # Optional initial variance for latent log productivity.
    init_logk_var: float = 0.25  # Initial variance for latent log capital.
    num_results: int = 300  # Number of retained posterior draws.
    num_burnin: int = 300  # Number of warm-up draws discarded before retention.
    num_adaptation_steps: int = 250  # Number of step-size adaptation draws.
    step_size: float = 0.03  # Initial HMC leapfrog step size.
    num_leapfrog_steps: int = 6  # Number of leapfrog steps per HMC proposal.
    target_accept: float = 0.75  # Target acceptance probability for adaptation.
    seed: int = 20260118  # Base random seed for HMC sampling.
    cred_level: float = 0.95  # Posterior credible interval level.
    metric3_burnin: int = 400  # Burn-in periods for metric-3 auxiliary simulations.
    metric3_sim_len: int = 200  # Retained periods for metric-3 auxiliary simulations.
    metric3_n_firms_sim: int = 200  # Number of firms in metric-3 auxiliary simulations.
    metric3_seed_base: int = 777  # Base seed for metric-3 simulations.
