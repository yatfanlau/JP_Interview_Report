"""Bayesian HMC estimator for structural parameters ``(theta, phi)``.

The implementation samples in unconstrained coordinates and applies explicit
bounds transformations with Jacobian correction to target the posterior over
economically meaningful parameter ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from dyninv.config import BasicModelParams, HMCColumnsPart2, HMCConfigPart2, PanelColumnsPart2, ParamBoundsPart2, PathsPart2
from dyninv.estimation.base import BaseEstimator, EstimationResult
from dyninv.model.context import build_basic_model_context
from dyninv.utils import DTYPE, tf

try:
    import tensorflow_probability as tfp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "tensorflow_probability is required for HMCEstimator. Install with `pip install tensorflow-probability`."
    ) from e


tfd = tfp.distributions


@dataclass(eq=False)
class HMCEstimator(BaseEstimator):
    """Bayesian estimator using TensorFlow Probability HMC."""

    paths: PathsPart2 = field(default_factory=PathsPart2)
    cols: PanelColumnsPart2 = field(default_factory=PanelColumnsPart2)
    hmc_cols: HMCColumnsPart2 = field(default_factory=HMCColumnsPart2)
    bounds: ParamBoundsPart2 = field(default_factory=ParamBoundsPart2)
    cfg: HMCConfigPart2 = field(default_factory=HMCConfigPart2)
    mp: BasicModelParams = field(default_factory=BasicModelParams)

    def __post_init__(self):
        """Cache model constants and transformed parameter bounds."""
        self.ctx = build_basic_model_context(self.mp, dtype=DTYPE)
        self.theta_min = tf.constant(float(self.bounds.theta_min), dtype=DTYPE)
        self.theta_max = tf.constant(float(self.bounds.theta_max), dtype=DTYPE)
        self.log_phi_min = tf.constant(math.log(float(self.bounds.phi_min)), dtype=DTYPE)
        self.log_phi_max = tf.constant(math.log(float(self.bounds.phi_max)), dtype=DTYPE)

    def _load_data(self):
        """Load panel CSV and extract true parameter labels."""
        df = pd.read_csv(self.paths.data_csv)
        theta_true = float(df["theta_true"].iloc[0])
        phi_true = float(df["phi_true"].iloc[0])
        return df, theta_true, phi_true

    def _select_rep_ids(self, df_all: pd.DataFrame) -> list[int]:
        """Select replication identifiers used in the HMC run."""
        all_rep_ids = sorted(df_all[self.cols.rep].unique().tolist())
        if self.cfg.rep_ids:
            keep = [int(r) for r in self.cfg.rep_ids if int(r) in all_rep_ids]
            return keep
        if self.cfg.n_reps_eval >= len(all_rep_ids):
            return all_rep_ids
        return all_rep_ids[: int(self.cfg.n_reps_eval)]

    def _prepare_data(self, df_rep: pd.DataFrame) -> dict:
        """Prepare one replication into aligned tensors for likelihood terms."""
        d = df_rep.sort_values([self.cols.firm, self.cols.time]).copy()
        d["k_tp1"] = d.groupby(self.cols.firm)[self.cols.k].shift(-1)
        d["z_tp1"] = d.groupby(self.cols.firm)[self.cols.z].shift(-1)
        d["iota_tp1"] = d.groupby(self.cols.firm)[self.cols.iota].shift(-1)
        d = d.dropna(subset=["k_tp1", "z_tp1", "iota_tp1"]).copy()

        return {
            "k_t": tf.constant(d[self.cols.k].tolist(), dtype=DTYPE),
            "iota_t": tf.constant(d[self.cols.iota].tolist(), dtype=DTYPE),
            "k_tp1": tf.constant(d["k_tp1"].tolist(), dtype=DTYPE),
            "z_tp1": tf.constant(d["z_tp1"].tolist(), dtype=DTYPE),
            "iota_tp1": tf.constant(d["iota_tp1"].tolist(), dtype=DTYPE),
        }

    @tf.function
    def _euler_residual(self, theta, log_phi, data: dict):
        """Compute Euler residual vector under current parameter values."""
        one = self.ctx["one_tf"]
        delta = self.ctx["delta_tf"]
        beta = self.ctx["beta_tf"]
        one_minus_delta = self.ctx["one_minus_delta_tf"]
        k_floor = self.ctx["k_floor_tf"]

        k_tp1 = tf.maximum(data["k_tp1"], k_floor)
        phi = tf.exp(log_phi)

        psi_i_t = phi * (data["iota_t"] - delta)
        psi_i_tp1 = phi * (data["iota_tp1"] - delta)
        psi_k_tp1 = 0.5 * phi * (delta * delta - data["iota_tp1"] * data["iota_tp1"])
        pi_k_tp1 = theta * data["z_tp1"] * tf.pow(k_tp1, theta - one)
        rhs = pi_k_tp1 - psi_k_tp1 + one_minus_delta * (one + psi_i_tp1)
        return (one + psi_i_t) - beta * rhs

    @tf.function
    def _u_to_params(self, u: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map unconstrained HMC coordinates to bounded parameters.

        Returns transformed ``theta``, ``phi``, and the log absolute Jacobian
        term needed for correct density transformation.
        """
        # Sigmoid maps unconstrained HMC coordinates into (0,1) intervals.
        t_s = tf.sigmoid(u[0])
        p_s = tf.sigmoid(u[1])
        # Affine map to structural parameter bounds.
        theta = self.theta_min + (self.theta_max - self.theta_min) * t_s
        log_phi = self.log_phi_min + (self.log_phi_max - self.log_phi_min) * p_s
        phi = tf.exp(log_phi)

        # Jacobian term for the transformed density:
        # u -> sigmoid -> bounded interval, plus log_phi -> phi exponential map.
        log_jac = (
            tf.math.log(self.theta_max - self.theta_min)
            + tf.math.log(t_s)
            + tf.math.log(1.0 - t_s)
            + tf.math.log(self.log_phi_max - self.log_phi_min)
            + tf.math.log(p_s)
            + tf.math.log(1.0 - p_s)
            + log_phi
        )
        return theta, phi, log_jac

    def _init_u_from_guess(self, theta_guess: float, phi_guess: float) -> tf.Tensor:
        """Map bounded initialization guesses into unconstrained coordinates."""
        t_span = float(self.bounds.theta_max - self.bounds.theta_min)
        p_span = float(math.log(self.bounds.phi_max) - math.log(self.bounds.phi_min))

        t_p = (theta_guess - float(self.bounds.theta_min)) / max(t_span, 1e-8)
        p_p = (math.log(phi_guess) - math.log(self.bounds.phi_min)) / max(p_span, 1e-8)
        # Clip away from hard boundaries so inverse-logit is finite.
        t_p = min(max(t_p, 1e-6), 1.0 - 1e-6)
        p_p = min(max(p_p, 1e-6), 1.0 - 1e-6)
        return tf.constant([math.log(t_p) - math.log(1.0 - t_p), math.log(p_p) - math.log(1.0 - p_p)], dtype=DTYPE)

    def _target_log_prob_fn(self, data: dict):
        """Build transformed log-posterior callable for HMC sampling."""
        sigma = tf.constant(float(self.cfg.sigma_y), dtype=DTYPE)
        # Constant Gaussian normalization term reused for every residual.
        log_sigma_term = tf.math.log(2.0 * math.pi * sigma * sigma)

        @tf.function
        def fn(u):
            """Evaluate transformed log posterior at one unconstrained state."""
            theta, phi, log_jac = self._u_to_params(u)
            log_phi = tf.math.log(phi)
            resid = self._euler_residual(theta, log_phi, data)
            # Gaussian pseudo-likelihood over Euler residuals plus Jacobian correction.
            ll = -0.5 * tf.reduce_sum(tf.square(resid / sigma) + log_sigma_term)
            return ll + log_jac

        return fn

    def _run_chain(self, data: dict, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Run one adapted HMC chain and return draws plus acceptance flags."""
        target = self._target_log_prob_fn(data)
        init = self._init_u_from_guess(float(self.cfg.theta_init_guess), float(self.cfg.phi_init_guess))

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target,
            step_size=float(self.cfg.step_size),
            num_leapfrog_steps=int(self.cfg.num_leapfrog_steps),
        )
        # Adaptive wrapper tunes step size during warm-up toward target acceptance.
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            num_adaptation_steps=int(self.cfg.num_adaptation_steps),
            target_accept_prob=float(self.cfg.target_accept),
        )

        @tf.function
        def run_chain_tf():
            """Sample the chain and trace Metropolis acceptance decisions."""
            return tfp.mcmc.sample_chain(
                num_results=int(self.cfg.num_results),
                num_burnin_steps=int(self.cfg.num_burnin),
                current_state=init,
                kernel=kernel,
                # Stateless seed vector keeps per-rep chains reproducible.
                seed=tf.constant([int(seed), int(seed) + 1], dtype=tf.int32),
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )

        return run_chain_tf()

    @tf.function
    def _transform_chain(self, samples_u: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Transform unconstrained posterior draws into ``(theta, phi)`` draws."""
        # Apply same bounded transforms as the target density definition.
        t_s = tf.sigmoid(samples_u[:, 0])
        p_s = tf.sigmoid(samples_u[:, 1])
        theta = self.theta_min + (self.theta_max - self.theta_min) * t_s
        log_phi = self.log_phi_min + (self.log_phi_max - self.log_phi_min) * p_s
        phi = tf.exp(log_phi)
        return theta, phi

    def run(self) -> EstimationResult:
        """Run Bayesian HMC estimation across selected replications.

        For each replication, this method samples posterior draws, reports
        posterior means, and computes equal-tail credible intervals.
        """
        df_all, theta_true, phi_true = self._load_data()
        rep_ids = self._select_rep_ids(df_all)

        rows: list[dict] = []
        alpha = float(1.0 - self.cfg.cred_level)
        q_lo = 100.0 * (alpha / 2.0)
        q_hi = 100.0 * (1.0 - alpha / 2.0)

        for rep_id in rep_ids:
            df_rep = df_all[df_all[self.cols.rep] == rep_id].copy()
            data = self._prepare_data(df_rep)
            samples_u, accepted = self._run_chain(data, seed=int(self.cfg.seed + rep_id))
            theta_s, phi_s = self._transform_chain(samples_u)

            theta_hat = float(tf.reduce_mean(theta_s).numpy())
            phi_hat = float(tf.reduce_mean(phi_s).numpy())
            # Equal-tail credible intervals from empirical posterior quantiles.
            theta_lo = float(tfp.stats.percentile(theta_s, q_lo).numpy())
            theta_hi = float(tfp.stats.percentile(theta_s, q_hi).numpy())
            phi_lo = float(tfp.stats.percentile(phi_s, q_lo).numpy())
            phi_hi = float(tfp.stats.percentile(phi_s, q_hi).numpy())
            # Average Metropolis acceptance indicator for chain diagnostics.
            acc_rate = float(tf.reduce_mean(tf.cast(accepted, DTYPE)).numpy())

            rows.append(
                {
                    "rep": int(rep_id),
                    "theta_hat": theta_hat,
                    "phi_hat": phi_hat,
                    "theta_lo": theta_lo,
                    "theta_hi": theta_hi,
                    "phi_lo": phi_lo,
                    "phi_hi": phi_hi,
                    "accept_rate": acc_rate,
                }
            )

        res_df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
        return EstimationResult(
            res_df=res_df,
            theta_true=theta_true,
            phi_true=phi_true,
            extra={"rep_ids": rep_ids},
        )
