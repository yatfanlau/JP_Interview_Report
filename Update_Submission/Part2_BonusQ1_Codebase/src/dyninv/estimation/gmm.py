"""Two-step GMM estimator built around TensorFlow moment evaluation.

The implementation follows a standard two-step workflow:
1) estimate with identity weighting;
2) estimate an optimal weighting matrix from first-step residual moments;
3) re-optimize under the second-step weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from dyninv.config import GMMConfigPart2, PanelColumnsPart2, ParamBoundsPart2, PathsPart2
from dyninv.estimation.base import BaseEstimator, EstimationResult
from dyninv.model.context import build_basic_model_context
from dyninv.utils import DTYPE, chi2_sf, pinv_psd_tf, symmetrize_tf, tf


@dataclass(eq=False)
class GMMEstimator(BaseEstimator):
    """Two-step GMM estimator with bounded-parameter optimization."""

    paths: PathsPart2 = field(default_factory=PathsPart2)
    cols: PanelColumnsPart2 = field(default_factory=PanelColumnsPart2)
    bounds: ParamBoundsPart2 = field(default_factory=ParamBoundsPart2)
    cfg: GMMConfigPart2 = field(default_factory=GMMConfigPart2)

    def __post_init__(self):
        """Cache model constants and transformed parameter bounds.

        Bounds are stored in both level and log space so optimization can run
        on unconstrained coordinates while preserving feasible parameters.
        """
        self.ctx = build_basic_model_context(dtype=DTYPE)
        self.theta_min = tf.constant(float(self.bounds.theta_min), dtype=DTYPE)
        self.theta_max = tf.constant(float(self.bounds.theta_max), dtype=DTYPE)
        self.log_phi_min = tf.constant(math.log(float(self.bounds.phi_min)), dtype=DTYPE)
        self.log_phi_max = tf.constant(math.log(float(self.bounds.phi_max)), dtype=DTYPE)

    def _load_data(self):
        """Load the panel dataset and pull true parameter labels."""
        df = pd.read_csv(self.paths.data_csv)
        theta_true = float(df["theta_true"].iloc[0])
        phi_true = float(df["phi_true"].iloc[0])
        return df, theta_true, phi_true

    def _select_rep_ids(self, df_all: pd.DataFrame) -> list[int]:
        """Choose replication IDs included in the estimation run."""
        rep_ids = sorted(df_all[self.cols.rep].unique().tolist())
        if self.cfg.n_reps_eval >= len(rep_ids):
            return rep_ids
        return rep_ids[: int(self.cfg.n_reps_eval)]

    @tf.function
    def _build_instruments(self, k_t: tf.Tensor, z_t: tf.Tensor, iota_t: tf.Tensor) -> tf.Tensor:
        """Construct instrument matrix from current-period observables.

        Instrument columns include levels and squares of transformed state/action
        variables, with optional standardization of non-constant terms.
        """
        # Build instruments in log space where panel distributions are better behaved.
        logk = tf.math.log(k_t + 1e-32)
        lnz = tf.math.log(z_t + 1e-32)
        z_mat = tf.stack(
            [
                tf.ones_like(logk),
                logk,
                lnz,
                iota_t,
                tf.square(logk),
                tf.square(lnz),
                tf.square(iota_t),
            ],
            axis=1,
        )
        if self.cfg.standardize_instr:
            # Leave the constant instrument untouched and standardize the rest.
            x = z_mat[:, 1:]
            m = tf.reduce_mean(x, axis=0)
            s = tf.sqrt(tf.reduce_mean(tf.square(x - m), axis=0) + 1e-12)
            s = tf.where(s > 1e-12, s, tf.ones_like(s))
            z_mat = tf.concat([z_mat[:, :1], (x - m) / s], axis=1)
        # Replace invalid values so a single bad row does not break moment evaluation.
        z_mat = tf.where(tf.math.is_finite(z_mat), z_mat, tf.zeros_like(z_mat))
        return z_mat

    def _prepare_data(self, df_rep: pd.DataFrame) -> dict:
        """Prepare one replication for tensor-based GMM calculations.

        The output dictionary contains aligned current/next-period tensors,
        instrument matrix, and firm indexing metadata for cluster aggregation.
        """
        d = df_rep.sort_values([self.cols.firm, self.cols.time]).copy()
        d["k_tp1"] = d.groupby(self.cols.firm)[self.cols.k].shift(-1)
        d["z_tp1"] = d.groupby(self.cols.firm)[self.cols.z].shift(-1)
        d["iota_tp1"] = d.groupby(self.cols.firm)[self.cols.iota].shift(-1)
        d = d.dropna(subset=["k_tp1", "z_tp1", "iota_tp1"]).copy()

        # Build dense firm indices for `unsorted_segment_mean` in graph mode.
        uniq_firms = sorted(d[self.cols.firm].unique().tolist())
        firm_map = {f: i for i, f in enumerate(uniq_firms)}
        d["firm_idx"] = d[self.cols.firm].map(firm_map).astype(int)
        # Stable sorting keeps temporal order inside each firm block.
        d = d.sort_values(["firm_idx", self.cols.time], kind="mergesort")

        k_t = tf.constant(d[self.cols.k].tolist(), dtype=DTYPE)
        z_t = tf.constant(d[self.cols.z].tolist(), dtype=DTYPE)
        iota_t = tf.constant(d[self.cols.iota].tolist(), dtype=DTYPE)
        k_tp1 = tf.constant(d["k_tp1"].tolist(), dtype=DTYPE)
        z_tp1 = tf.constant(d["z_tp1"].tolist(), dtype=DTYPE)
        iota_tp1 = tf.constant(d["iota_tp1"].tolist(), dtype=DTYPE)
        firm_idx = tf.constant(d["firm_idx"].tolist(), dtype=tf.int32)

        z_instr = self._build_instruments(k_t, z_t, iota_t)
        return {
            "k_t": k_t,
            "iota_t": iota_t,
            "k_tp1": k_tp1,
            "z_tp1": z_tp1,
            "iota_tp1": iota_tp1,
            "z_instr": z_instr,
            "firm_idx": firm_idx,
            "n_firms": len(uniq_firms),
            "q": int(z_instr.shape[1]),
            "n_obs": len(d),
        }

    @tf.function
    def _euler_residual(
        self,
        theta: tf.Tensor,
        log_phi: tf.Tensor,
        k_t: tf.Tensor,
        iota_t: tf.Tensor,
        k_tp1: tf.Tensor,
        z_tp1: tf.Tensor,
        iota_tp1: tf.Tensor,
    ) -> tf.Tensor:
        """Compute Euler residuals for one candidate ``(theta, log_phi)``."""
        phi = tf.exp(log_phi)
        one = self.ctx["one_tf"]
        delta = self.ctx["delta_tf"]
        beta = self.ctx["beta_tf"]
        one_minus_delta = self.ctx["one_minus_delta_tf"]
        k_floor = self.ctx["k_floor_tf"]

        # Enforce strictly positive capital to keep power/log operations well-defined.
        k_tp1 = tf.maximum(k_tp1, k_floor)
        psi_i_t = phi * (iota_t - delta)
        psi_i_tp1 = phi * (iota_tp1 - delta)
        psi_k_tp1 = 0.5 * phi * (delta * delta - iota_tp1 * iota_tp1)
        pi_k_tp1 = theta * z_tp1 * tf.pow(k_tp1, theta - one)

        rhs = pi_k_tp1 - psi_k_tp1 + one_minus_delta * (one + psi_i_tp1)
        return (one + psi_i_t) - beta * rhs

    @tf.function
    def _moments(self, theta: tf.Tensor, log_phi: tf.Tensor, data: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute sample mean moments and firm-level aggregated moments."""
        u = self._euler_residual(
            theta,
            log_phi,
            data["k_t"],
            data["iota_t"],
            data["k_tp1"],
            data["z_tp1"],
            data["iota_tp1"],
        )
        # Observation-level moment vectors: u_t * z_t.
        m_t = u[:, None] * data["z_instr"]
        # Sample mean moments used directly in objective.
        g = tf.reduce_mean(m_t, axis=0)
        # Firm-level averages enable cluster-robust covariance estimation.
        mbar = tf.math.unsorted_segment_mean(m_t, data["firm_idx"], num_segments=data["n_firms"])
        return g, mbar

    @tf.function
    def _objective(self, theta: tf.Tensor, log_phi: tf.Tensor, data: dict, w: tf.Tensor) -> tf.Tensor:
        """Evaluate the quadratic objective ``g(theta)' W g(theta)``."""
        g, _ = self._moments(theta, log_phi, data)
        gv = g[:, None]
        # Explicit column-vector quadratic form g' W g.
        return tf.squeeze(tf.matmul(tf.matmul(tf.transpose(gv), w), gv))

    @tf.function
    def _params_from_u(self, u_theta: tf.Tensor, u_phi: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map unconstrained optimizer variables to bounded model parameters."""
        # Sigmoid transforms map R -> (0,1), then affine map enforces parameter bounds.
        theta = self.theta_min + (self.theta_max - self.theta_min) * tf.sigmoid(u_theta)
        log_phi = self.log_phi_min + (self.log_phi_max - self.log_phi_min) * tf.sigmoid(u_phi)
        return theta, tf.exp(log_phi), log_phi

    def _u_from_init(self, theta_init: float, phi_init: float) -> tuple[tf.Tensor, tf.Tensor]:
        """Map bounded initial guesses into unconstrained logit coordinates."""
        t_span = float(self.bounds.theta_max - self.bounds.theta_min)
        p_span = float(math.log(self.bounds.phi_max) - math.log(self.bounds.phi_min))

        t_p = (theta_init - float(self.bounds.theta_min)) / max(t_span, 1e-8)
        p_p = (math.log(phi_init) - math.log(self.bounds.phi_min)) / max(p_span, 1e-8)
        # Keep away from boundaries so logit is finite.
        t_p = min(max(t_p, 1e-6), 1.0 - 1e-6)
        p_p = min(max(p_p, 1e-6), 1.0 - 1e-6)

        u_theta = tf.constant(math.log(t_p) - math.log(1.0 - t_p), dtype=DTYPE)
        u_phi = tf.constant(math.log(p_p) - math.log(1.0 - p_p), dtype=DTYPE)
        return u_theta, u_phi

    def _run_optim(self, data: dict, theta_init: float, phi_init: float, w: tf.Tensor | None, lr: float, steps: int):
        """Run Adam optimization under a fixed weighting matrix.

        Returns point estimates, objective value, and cached moment objects used
        by later weighting-matrix or reporting calculations.
        """
        # First stage defaults to identity weighting when `w` is not supplied.
        w_mat = w if w is not None else tf.eye(data["q"], dtype=DTYPE)

        u_theta0, u_phi0 = self._u_from_init(theta_init, phi_init)
        u_theta = tf.Variable(u_theta0, trainable=True)
        u_phi = tf.Variable(u_phi0, trainable=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                theta, phi, log_phi = self._params_from_u(u_theta, u_phi)
                obj = self._objective(theta, log_phi, data, w_mat)
            grads = tape.gradient(obj, [u_theta, u_phi])
            opt.apply_gradients(zip(grads, [u_theta, u_phi]))

        theta_hat, phi_hat, log_phi_hat = self._params_from_u(u_theta, u_phi)
        obj_hat = self._objective(theta_hat, log_phi_hat, data, w_mat)
        g_hat, mbar = self._moments(theta_hat, log_phi_hat, data)
        return {
            "theta_hat": float(theta_hat.numpy()),
            "phi_hat": float(phi_hat.numpy()),
            "logphi_hat": float(log_phi_hat.numpy()),
            "obj": float(obj_hat.numpy()),
            "g": g_hat,
            "mbar": mbar,
            "w": w_mat,
        }

    def _estimate_w(self, theta_hat: float, logphi_hat: float, data: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Estimate robust moment covariance and its pseudo-inverse weight."""
        g, mbar = self._moments(tf.constant(theta_hat, DTYPE), tf.constant(logphi_hat, DTYPE), data)
        _ = g
        n_firms = tf.constant(float(data["n_firms"]), dtype=DTYPE)
        # Cluster-robust covariance proxy from firm-level moment means.
        s = tf.matmul(mbar, mbar, transpose_a=True) / n_firms
        # Numerical cleanup before inversion: symmetrize + ridge.
        s = symmetrize_tf(tf.cast(s, tf.float64)) + tf.cast(self.cfg.w_ridge, tf.float64) * tf.eye(
            data["q"], dtype=tf.float64
        )
        w = pinv_psd_tf(s, rcond=self.cfg.pinv_rcond)
        return tf.cast(s, DTYPE), tf.cast(w, DTYPE)

    def run(self) -> EstimationResult:
        """Execute two-step GMM for each selected replication and collect results.

        For each replication, the method runs step-1 and step-2 optimization,
        computes Hansen's over-identification statistic, and returns a combined
        results table.
        """
        df_all, theta_true, phi_true = self._load_data()
        rep_ids = self._select_rep_ids(df_all)

        rows: list[dict] = []
        for rep_id in rep_ids:
            df_rep = df_all[df_all[self.cols.rep] == rep_id].copy()
            data = self._prepare_data(df_rep)

            step1 = self._run_optim(
                data=data,
                theta_init=float(self.cfg.theta_init_guess),
                phi_init=float(self.cfg.phi_init_guess),
                w=None,
                lr=float(self.cfg.lr_1),
                steps=int(self.cfg.steps_1),
            )
            _, w2 = self._estimate_w(step1["theta_hat"], step1["logphi_hat"], data)
            step2 = self._run_optim(
                data=data,
                theta_init=float(step1["theta_hat"]),
                phi_init=float(step1["phi_hat"]),
                w=w2,
                lr=float(self.cfg.lr_2),
                steps=int(self.cfg.steps_2),
            )

            n_obs = float(data["n_obs"])
            # Standard over-identification statistic J = N * g'Wg, df = moments - parameters.
            j_stat = n_obs * float(step2["obj"])
            df_j = max(int(data["q"] - 2), 1)
            p_val = chi2_sf(j_stat, df=df_j)

            rows.append(
                {
                    "rep": int(rep_id),
                    "theta_hat": float(step2["theta_hat"]),
                    "phi_hat": float(step2["phi_hat"]),
                    "step1_obj": float(step1["obj"]),
                    "step2_obj": float(step2["obj"]),
                    "J": float(j_stat),
                    "df": int(df_j),
                    "p_value": float(p_val),
                }
            )

        res_df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
        return EstimationResult(
            res_df=res_df,
            theta_true=theta_true,
            phi_true=phi_true,
            extra={"rep_ids": rep_ids},
        )
