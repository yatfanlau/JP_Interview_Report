"""CLI entry point for amortized policy training.

This script instantiates default model/training configs, fits the policy
network, and prints a concise training summary.
"""

from __future__ import annotations

from dyninv.config import BasicModelParams, BasicTrainingParams
from dyninv.policy.training import PolicyTrainer


def main() -> None:
    """Train the policy model with default settings and save it to disk."""
    trainer = PolicyTrainer(mp=BasicModelParams(), tp=BasicTrainingParams())
    out = trainer.train(save_path="param_policy_theta_phi.keras")
    print(f"Final diagnostic loss: {out.final_loss:.6e}")
    print(f"Training seconds: {out.seconds:.2f}")
    print(f"Saved policy: {out.save_path}")


if __name__ == "__main__":
    main()
