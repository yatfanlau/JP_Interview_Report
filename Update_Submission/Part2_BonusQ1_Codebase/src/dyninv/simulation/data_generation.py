"""Facade for end-to-end synthetic panel generation.

This module glues together policy loading, repeated simulation replications,
and CSV writing into a single callable interface used by CLI entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dyninv.config import BasicModelParams
from dyninv.simulation.panel_simulator import PanelSimulator
from dyninv.simulation.panels import PanelData
from dyninv.utils import tf


@dataclass
class DataGenerationConfig:
    """Configuration bundle for repeated synthetic panel generation."""

    n_firms: int = 100
    t_periods: int = 40
    burn_in: int = 100
    n_reps: int = 200
    theta_true: float = 0.7
    phi_true: float = 2.0
    policy_path: str = "param_policy_theta_phi.keras"
    output_csv: str = "synthetic_panels/synthetic_panels_all.csv"


class DataGenerator:
    """Generate replicated panels from a saved policy and persist to CSV."""

    def __init__(self, mp: BasicModelParams | None = None):
        """Initialize generator with model primitives and simulator backend."""
        self.mp = mp or BasicModelParams()
        self.simulator = PanelSimulator(self.mp)

    def load_policy(self, policy_path: str):
        """Load a previously trained Keras policy model from disk."""
        return tf.keras.models.load_model(policy_path, compile=False)

    def generate(self, cfg: DataGenerationConfig) -> str:
        """Generate ``n_reps`` synthetic panels and append them into one CSV.

        Args:
            cfg: Data-generation settings including panel dimensions, true
                parameters, policy path, and output location.

        Returns:
            Output CSV path as a string.
        """
        policy = self.load_policy(cfg.policy_path)
        out_path = Path(cfg.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Replications use deterministic seed offsets for reproducible Monte Carlo.
        for rep in range(cfg.n_reps):
            panel = self.simulator.simulate_panel(
                policy=policy,
                theta_true=cfg.theta_true,
                phi_true=cfg.phi_true,
                n_firms=cfg.n_firms,
                t_periods=cfg.t_periods,
                burn_in=cfg.burn_in,
                base_seed=self.mp.seed + rep,
            )
            panel_data = PanelData.from_simulation(panel, rep, cfg.theta_true, cfg.phi_true)
            panel_data.save_csv(str(out_path), append=(rep > 0))
            print(f"[rep {rep + 1}/{cfg.n_reps}] wrote {int(tf.shape(panel_data.k)[0].numpy())} rows")

        return str(out_path)
