"""CLI entry point for synthetic panel data generation.

It loads the trained policy and writes replicated synthetic panels to CSV using
default generation settings.
"""

from __future__ import annotations

from dyninv.config import BasicModelParams
from dyninv.simulation.data_generation import DataGenerationConfig, DataGenerator


def main() -> None:
    """Generate synthetic panel data with the default configuration."""
    generator = DataGenerator(mp=BasicModelParams())
    out = generator.generate(DataGenerationConfig())
    print(f"Wrote synthetic data to: {out}")


if __name__ == "__main__":
    main()
