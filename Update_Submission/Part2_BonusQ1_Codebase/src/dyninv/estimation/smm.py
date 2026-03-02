"""Two-step SMM estimator with TensorFlow-based panel simulation.

The estimator matches data moments to simulated moments from the amortized
policy model, first with identity weighting and then with a data-driven
second-step weighting matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd  # DataFrame I/O and firm-level bootstrap reshaping.

from dyninv.config import BasicModelParams, PanelColumnsPart2, ParamBoundsPart2, PathsPart2, SMMConfigPart2
from dyninv.estimation.base import BaseEstimator, EstimationResult
from dyninv.model.context import build_basic_model_context
from dyninv.policy.network import policy_iota
from dyninv.utils import DTYPE, pinv_psd_tf, safe_corr_tf, symmetrize_tf, tf


@dataclass(eq=False)
class SMMEstimator(BaseEstimator):
    """Two-step simulation method of moments estimator."""

    paths: PathsPart2 = field(default_factory=PathsPart2)
    cols: PanelColumnsPart2 = field(default_factory=PanelColumnsPart2)
    bounds: ParamBoundsPart2 = field(default_factory=ParamBoundsPart2)
    cfg: SMMConfigPart2 = field(default_factory=SMMConfigPart2)
    mp: BasicModelParams = field(default_factory=BasicModelParams)

    def __post_init__(self):
        """Cache model context and transformed parameter bounds."""
        self.ctx = build_basic_model_context(self.mp, dtype=DTYPE)
        self.theta_min = tf.constant(float(self.bounds.theta_min), dtype=DTYPE)
        self.theta_max = tf.constant(float(self.bounds.theta_max), dtype=DTYPE)
        self.log_phi_min = tf.constant(math.log(float(self.bounds.phi_min)), dtype=DTYPE)
        self.log_phi_max = tf.constant(math.log(float(self.bounds.phi_max)), dtype=DTYPE)

    def _load_policy(self):
        """Load the trained policy model used in simulated moments."""
        return tf.keras.models.load_model(self.paths.policy_path, compile=False)

    def _load_data(self):
        """Load panel CSV and true parameter labels."""
        df = pd.read_csv(self.paths.data_csv)
        theta_true = float(df["theta_true"].iloc[0])
        phi_true = float(df["phi_true"].iloc[0])
        return df, theta_true, phi_true

    def _select_rep_ids(self, df_all: pd.DataFrame) -> list[int]:
        """Select replication IDs included in the run."""
        rep_ids = sorted(df_all[self.cols.rep].unique().tolist())
        if self.cfg.n_reps_eval >= len(rep_ids):
            return rep_ids
        return rep_ids[: int(self.cfg.n_reps_eval)]

    def _panel_matrices(self, df_rep: pd.DataFrame) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Reshape one replication into ``(firms, periods)`` tensors."""
        df_sorted = df_rep.sort_values([self.cols.firm, self.cols.time])
        n_firms = int(df_sorted[self.cols.firm].nunique())
        t_periods = int(df_sorted[self.cols.time].nunique())

        k = tf.constant(df_sorted[self.cols.k].tolist(), dtype=DTYPE)
        z = tf.constant(df_sorted[self.cols.z].tolist(), dtype=DTYPE)
        iota = tf.constant(df_sorted[self.cols.iota].tolist(), dtype=DTYPE)

        k_hist = tf.reshape(k, [n_firms, t_periods])
        z_hist = tf.reshape(z, [n_firms, t_periods])
        iota_hist = tf.reshape(iota, [n_firms, t_periods])
        return k_hist, z_hist, iota_hist

    @tf.function
    def _compute_moments_tf(self, k_hist: tf.Tensor, z_hist: tf.Tensor, iota_hist: tf.Tensor) -> tf.Tensor:
        """Compute the 13-dimensional SMM moment vector from panel histories.

        Moments combine unconditional levels/variances, serial dependence, and
        simple predictive relationships designed to identify ``theta`` and ``phi``.
        """
        # Use a tiny floor before logs so degenerate states do not create -inf values.
        logk = tf.math.log(k_hist + 1e-32)
        lnz = tf.math.log(z_hist + 1e-32)

        # Flatten `(firms, time)` panels into one vector when moments are unconditional.
        k_flat = tf.reshape(k_hist, [-1])
        logk_flat = tf.reshape(logk, [-1])
        lnz_flat = tf.reshape(lnz, [-1])
        iota_flat = tf.reshape(iota_hist, [-1])

        # m1: average log capital level.
        m1 = tf.reduce_mean(logk_flat)
        # m2: contemporaneous corr(iota_t, ln z_t).
        m2 = tf.cast(safe_corr_tf(iota_flat, lnz_flat), DTYPE)

        iota_t = iota_hist[:, 1:]
        iota_lag = iota_hist[:, :-1]
        # m3: persistence of the investment-rate process.
        m3 = tf.cast(safe_corr_tf(iota_t, iota_lag), DTYPE)

        delta = self.ctx["delta_tf"]
        # m4: second moment around depreciation, tied to adjustment-cost curvature.
        m4 = tf.reduce_mean(tf.square(iota_flat - delta))

        dlogk = logk[:, 1:] - logk[:, :-1]
        dlogk_flat = tf.reshape(dlogk, [-1])
        # m5: variance of capital growth.
        m5 = tf.reduce_mean(tf.square(dlogk_flat - tf.reduce_mean(dlogk_flat)))

        iota_lead = iota_hist[:, 1:]
        lnz_t = lnz[:, :-1]
        # m6: predictive corr(iota_{t+1}, ln z_t).
        m6 = tf.cast(safe_corr_tf(iota_lead, lnz_t), DTYPE)

        # m7-m9: mean/std and difference variance of iota.
        m7 = tf.reduce_mean(iota_flat)
        m8 = tf.sqrt(tf.reduce_mean(tf.square(iota_flat - m7)) + 1e-12)

        diota = iota_hist[:, 1:] - iota_hist[:, :-1]
        diota_flat = tf.reshape(diota, [-1])
        m9 = tf.reduce_mean(tf.square(diota_flat - tf.reduce_mean(diota_flat)))

        # Innovation in ln z implied by the AR(1) law with model intercept.
        innov = lnz[:, 1:] - (self.ctx["mu_ln_z_tf"] + self.ctx["rho_tf"] * lnz[:, :-1])
        # m10-m11: corr(iota, current innovation) and corr(iota, lagged innovation).
        m10 = tf.cast(safe_corr_tf(iota_hist[:, 1:], innov), DTYPE)

        m11 = tf.cast(safe_corr_tf(iota_hist[:, 2:], innov[:, :-1]), DTYPE)

        # m12-m13: OLS slopes from iota_{t+1} on [iota_t, innovation_t] using moments only.
        y = tf.reshape(iota_hist[:, 1:], [-1])
        x1 = tf.reshape(iota_hist[:, :-1], [-1])
        x2 = tf.reshape(innov, [-1])
        my = tf.reduce_mean(y)
        mx1 = tf.reduce_mean(x1)
        mx2 = tf.reduce_mean(x2)
        dy = y - my
        dx1 = x1 - mx1
        dx2 = x2 - mx2
        s11 = tf.reduce_mean(dx1 * dx1)
        s22 = tf.reduce_mean(dx2 * dx2)
        s12 = tf.reduce_mean(dx1 * dx2)
        c1y = tf.reduce_mean(dx1 * dy)
        c2y = tf.reduce_mean(dx2 * dy)
        # Determinant of regressor covariance matrix; near-zero means unstable inversion.
        den = s11 * s22 - s12 * s12
        den_ok = tf.abs(den) > 1e-12
        # Closed-form 2x2 normal-equation solution with a safe fallback at singular points.
        m12 = tf.where(den_ok, (s22 * c1y - s12 * c2y) / den, tf.zeros([], dtype=DTYPE))
        m13 = tf.where(den_ok, (-s12 * c1y + s11 * c2y) / den, tf.zeros([], dtype=DTYPE))

        return tf.stack([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13], axis=0)

    def _compute_data_moments(self, df_rep: pd.DataFrame) -> tf.Tensor:
        """Compute target moment vector from observed panel data."""
        k_hist, z_hist, iota_hist = self._panel_matrices(df_rep)
        return self._compute_moments_tf(k_hist, z_hist, iota_hist)

    def _make_crn_draws(self, rep_id: int, n_sims: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate common-random-number shocks for one replication.

        Using CRN keeps objective comparisons less noisy across parameter values.
        """
        # CRN means every parameter evaluation sees the same underlying shocks,
        # which reduces optimizer noise when comparing objective values.
        seed = int(self.cfg.crn_base_seed + rep_id)
        eps = tf.random.stateless_normal(
            shape=[n_sims, int(self.cfg.n_firms_sim), int(self.cfg.t_burnin + self.cfg.t_data)],
            seed=tf.constant([seed, seed + 1], dtype=tf.int32),
            stddev=self.ctx["sigma_eps_f"],
            dtype=DTYPE,
        )
        lnz_init = tf.random.stateless_normal(
            shape=[n_sims, int(self.cfg.n_firms_sim)],
            seed=tf.constant([seed + 2, seed + 3], dtype=tf.int32),
            mean=self.ctx["m_ln_z_f"],
            stddev=self.ctx["sigma_ln_z_f"],
            dtype=DTYPE,
        )
        return eps, lnz_init

    @tf.function
    def _simulate_one(self, policy, theta: tf.Tensor, phi: tf.Tensor, eps_path: tf.Tensor, lnz_init: tf.Tensor) -> tf.Tensor:
        """Simulate one retained panel path and return its moment vector."""
        n = tf.shape(eps_path)[0]
        t_total = tf.shape(eps_path)[1]

        # Initialize capital at the frictionless steady state implied by `(theta, r, delta)`.
        k_ss = tf.exp(tf.math.log(theta / (self.ctx["r_tf"] + self.ctx["delta_tf"])) / (1.0 - theta))
        k = tf.fill([n], tf.squeeze(k_ss))
        lnz = lnz_init

        burnin = int(self.cfg.t_burnin)
        sim_len = int(self.cfg.t_data)

        # TensorArray is used because graph-mode loops cannot append to Python lists.
        k_arr = tf.TensorArray(dtype=DTYPE, size=sim_len)
        z_arr = tf.TensorArray(dtype=DTYPE, size=sim_len)
        i_arr = tf.TensorArray(dtype=DTYPE, size=sim_len)

        write_idx = tf.constant(0, dtype=tf.int32)
        t = tf.constant(0, dtype=tf.int32)

        def cond(t, write_idx, *_):
            """Continue until all periods are processed or retained array is full."""
            return tf.logical_and(t < t_total, write_idx < sim_len)

        def body(t, write_idx, k, lnz, k_arr, z_arr, i_arr):
            """Advance one period and write retained observations after burn-in."""
            lnz = self.ctx["mu_ln_z_tf"] + self.ctx["rho_tf"] * lnz + eps_path[:, t]
            z = tf.exp(lnz)
            theta_vec = tf.fill([n], theta)
            phi_vec = tf.fill([n], phi)
            iota = policy_iota(policy, k, z, theta_vec, phi_vec, training=False)
            # Capital law of motion with floor avoids exploding negatives in logs.
            k_next = tf.maximum(self.ctx["k_floor_tf"], (self.ctx["one_minus_delta_tf"] + iota) * k)

            def write_vals(k_arr, z_arr, i_arr, write_idx):
                """Write current-period states into retained-history arrays."""
                k_arr = k_arr.write(write_idx, k)
                z_arr = z_arr.write(write_idx, z)
                i_arr = i_arr.write(write_idx, iota)
                return k_arr, z_arr, i_arr, write_idx + 1

            def skip_vals(k_arr, z_arr, i_arr, write_idx):
                """Skip writes while burn-in periods are still being processed."""
                return k_arr, z_arr, i_arr, write_idx

            k_arr, z_arr, i_arr, write_idx = tf.cond(
                t >= burnin,
                lambda: write_vals(k_arr, z_arr, i_arr, write_idx),
                lambda: skip_vals(k_arr, z_arr, i_arr, write_idx),
            )
            return t + 1, write_idx, k_next, lnz, k_arr, z_arr, i_arr

        _, _, _, _, k_arr, z_arr, i_arr = tf.while_loop(
            cond,
            body,
            loop_vars=[t, write_idx, k, lnz, k_arr, z_arr, i_arr],
            # Keep loop sequential to preserve deterministic write order.
            parallel_iterations=1,
        )

        # Arrays are stacked as `(time, firms)`; transpose to `(firms, time)` for moments.
        k_hist = tf.transpose(k_arr.stack(), [1, 0])
        z_hist = tf.transpose(z_arr.stack(), [1, 0])
        iota_hist = tf.transpose(i_arr.stack(), [1, 0])
        return self._compute_moments_tf(k_hist, z_hist, iota_hist)

    def _simulate_moments_avg(self, policy, theta: tf.Tensor, phi: tf.Tensor, rep_id: int, n_sims: int) -> tf.Tensor:
        """Average simulated moments over multiple CRN simulation paths."""
        eps, lnz = self._make_crn_draws(rep_id=rep_id, n_sims=n_sims)
        moms = []
        for s in range(n_sims):
            moms.append(self._simulate_one(policy, theta, phi, eps[s], lnz[s]))
        return tf.reduce_mean(tf.stack(moms, axis=0), axis=0)

    @tf.function
    def _params_from_u(self, u_theta: tf.Tensor, u_phi: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Map unconstrained optimizer variables into bounded parameters."""
        # Sigmoid map enforces bounds exactly while allowing unconstrained Adam updates.
        theta = self.theta_min + (self.theta_max - self.theta_min) * tf.sigmoid(u_theta)
        log_phi = self.log_phi_min + (self.log_phi_max - self.log_phi_min) * tf.sigmoid(u_phi)
        return theta, tf.exp(log_phi)

    def _u_from_init(self, theta_init: float, phi_init: float) -> tuple[tf.Tensor, tf.Tensor]:
        """Map bounded initial guesses into unconstrained logit coordinates."""
        t_span = float(self.bounds.theta_max - self.bounds.theta_min)
        p_span = float(math.log(self.bounds.phi_max) - math.log(self.bounds.phi_min))

        t_p = (theta_init - float(self.bounds.theta_min)) / max(t_span, 1e-8)
        p_p = (math.log(phi_init) - math.log(self.bounds.phi_min)) / max(p_span, 1e-8)
        # Clip probabilities away from {0,1} so inverse-logit stays finite.
        t_p = min(max(t_p, 1e-6), 1.0 - 1e-6)
        p_p = min(max(p_p, 1e-6), 1.0 - 1e-6)
        return (
            tf.constant(math.log(t_p) - math.log(1.0 - t_p), dtype=DTYPE),
            tf.constant(math.log(p_p) - math.log(1.0 - p_p), dtype=DTYPE),
        )

    def _bootstrap_cov(self, df_rep: pd.DataFrame, n_boot: int, seed: int) -> tf.Tensor:
        """Estimate covariance of data moments via firm-level bootstrap."""
        firms = sorted(df_rep[self.cols.firm].unique().tolist())
        m_list = []
        for b in range(int(n_boot)):
            draw_firms = (
                df_rep[[self.cols.firm]]
                .drop_duplicates()
                .sample(n=len(firms), replace=True, random_state=int(seed + b))[self.cols.firm]
                .tolist()
            )
            d = df_rep[df_rep[self.cols.firm].isin(draw_firms)].copy()
            # Re-index sampled firms to dense IDs required by panel reshaping logic.
            d[self.cols.firm] = d[self.cols.firm].astype("category").cat.codes
            m_list.append(self._compute_data_moments(d))
        m = tf.stack(m_list, axis=0)
        # Standard sample covariance over bootstrap draws.
        m_center = m - tf.reduce_mean(m, axis=0, keepdims=True)
        denom = tf.cast(tf.shape(m_center)[0] - 1, DTYPE)
        return tf.matmul(m_center, m_center, transpose_a=True) / tf.maximum(denom, 1.0)

    def _simulate_cov(self, policy, theta_hat: float, phi_hat: float, n_sims: int, seed: int) -> tf.Tensor:
        """Estimate covariance of simulated moments via repeated simulations."""
        moms = []
        theta_t = tf.constant(theta_hat, dtype=DTYPE)
        phi_t = tf.constant(phi_hat, dtype=DTYPE)
        for s in range(int(n_sims)):
            # Use distinct seeds per draw so covariance reflects simulation noise.
            rep_seed = int(seed + s + 1)
            eps, lnz = self._make_crn_draws(rep_id=rep_seed, n_sims=1)
            moms.append(self._simulate_one(policy, theta_t, phi_t, eps[0], lnz[0]))
        m = tf.stack(moms, axis=0)
        m_center = m - tf.reduce_mean(m, axis=0, keepdims=True)
        denom = tf.cast(tf.shape(m_center)[0] - 1, DTYPE)
        return tf.matmul(m_center, m_center, transpose_a=True) / tf.maximum(denom, 1.0)

    def _run_optim(self, policy, rep_id: int, m_data: tf.Tensor, theta_init: float, phi_init: float, w: tf.Tensor, lr: float, steps: int, sims_per_obj: int):
        """Run Adam optimization for a fixed SMM weighting matrix.

        Args:
            policy: Trained policy model used for simulation.
            rep_id: Replication ID used for deterministic CRN seeds.
            m_data: Target moment vector from observed data.
            theta_init: Initial theta guess.
            phi_init: Initial phi guess.
            w: Current weighting matrix.
            lr: Adam learning rate.
            steps: Number of optimization updates.
            sims_per_obj: Number of simulated panels averaged per objective call.
        """
        u_theta0, u_phi0 = self._u_from_init(theta_init, phi_init)
        u_theta = tf.Variable(u_theta0, trainable=True)
        u_phi = tf.Variable(u_phi0, trainable=True)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                theta, phi = self._params_from_u(u_theta, u_phi)
                m_sim = self._simulate_moments_avg(policy, theta, phi, rep_id=rep_id, n_sims=sims_per_obj)
                g = m_data - m_sim
                gv = g[:, None]
                # Quadratic form g' W g with column-vector moments.
                obj = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(gv), w), gv))
            grads = tape.gradient(obj, [u_theta, u_phi])
            opt.apply_gradients(zip(grads, [u_theta, u_phi]))

        theta_hat, phi_hat = self._params_from_u(u_theta, u_phi)
        m_sim = self._simulate_moments_avg(policy, theta_hat, phi_hat, rep_id=rep_id, n_sims=sims_per_obj)
        g = m_data - m_sim
        gv = g[:, None]
        obj = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(gv), w), gv))
        return {
            "theta_hat": float(theta_hat.numpy()),
            "phi_hat": float(phi_hat.numpy()),
            "obj": float(obj.numpy()),
        }

    def run(self) -> EstimationResult:
        """Run two-step SMM across selected replications.

        Each replication performs step-1 identity-weight estimation, covariance
        estimation for weighting, and step-2 re-estimation under optimal weights.
        """
        policy = self._load_policy()
        df_all, theta_true, phi_true = self._load_data()
        rep_ids = self._select_rep_ids(df_all)

        rows: list[dict] = []
        q = 13
        # Step-1 uses identity weighting (classic first-stage SMM).
        i_q = tf.eye(q, dtype=DTYPE)

        for rep_id in rep_ids:
            df_rep = df_all[df_all[self.cols.rep] == rep_id].copy()
            m_data = self._compute_data_moments(df_rep)

            step1 = self._run_optim(
                policy=policy,
                rep_id=rep_id,
                m_data=m_data,
                theta_init=float(self.cfg.theta_init_guess),
                phi_init=float(self.cfg.phi_init_guess),
                w=i_q,
                lr=float(self.cfg.lr_1),
                steps=int(self.cfg.steps_1),
                sims_per_obj=int(self.cfg.sims_per_obj_1),
            )

            sigma_data = self._bootstrap_cov(
                df_rep,
                n_boot=int(self.cfg.w_n_boot),
                seed=int(self.cfg.boot_seed_base + rep_id),
            )
            if self.cfg.include_sim_var_in_W:
                sigma_sim = self._simulate_cov(
                    policy,
                    theta_hat=step1["theta_hat"],
                    phi_hat=step1["phi_hat"],
                    n_sims=int(self.cfg.w_n_sims),
                    seed=int(self.cfg.sim_seed_base + rep_id),
                )
                # Two-step weighting can include both data and simulation noise.
                sigma = sigma_data + sigma_sim / max(int(self.cfg.sims_per_obj_2), 1)
            else:
                sigma = sigma_data

            # Symmetrize and ridge-stabilize before pseudo-inversion.
            sigma = symmetrize_tf(tf.cast(sigma, tf.float64)) + tf.cast(self.cfg.w_ridge, tf.float64) * tf.eye(
                q, dtype=tf.float64
            )
            w2 = tf.cast(pinv_psd_tf(sigma, rcond=self.cfg.w_rcond), DTYPE)

            step2 = self._run_optim(
                policy=policy,
                rep_id=rep_id,
                m_data=m_data,
                theta_init=float(step1["theta_hat"]),
                phi_init=float(step1["phi_hat"]),
                w=w2,
                lr=float(self.cfg.lr_2),
                steps=int(self.cfg.steps_2),
                sims_per_obj=int(self.cfg.sims_per_obj_2),
            )

            rows.append(
                {
                    "rep": int(rep_id),
                    "theta_hat": float(step2["theta_hat"]),
                    "phi_hat": float(step2["phi_hat"]),
                    "step1_obj": float(step1["obj"]),
                    "step2_obj": float(step2["obj"]),
                }
            )

        res_df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
        return EstimationResult(
            res_df=res_df,
            theta_true=theta_true,
            phi_true=phi_true,
            extra={"policy": policy, "rep_ids": rep_ids},
        )
