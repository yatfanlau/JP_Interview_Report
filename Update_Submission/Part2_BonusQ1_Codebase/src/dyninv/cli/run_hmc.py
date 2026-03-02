"""CLI entry point for Bayesian HMC estimation.

Instantiates default HMC configuration and prints a preview of posterior
summary statistics by replication.
"""

from __future__ import annotations

from dyninv.config import BasicModelParams, HMCColumnsPart2, HMCConfigPart2, PanelColumnsPart2, ParamBoundsPart2, PathsPart2
from dyninv.estimation.bayesian_hmc import HMCEstimator


def main() -> None:
    """Run Bayesian HMC estimation with package-default settings."""
    est = HMCEstimator(
        paths=PathsPart2(),
        cols=PanelColumnsPart2(),
        hmc_cols=HMCColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=HMCConfigPart2(),
        mp=BasicModelParams(),
    )
    out = est.run()
    print(out.res_df.head())


if __name__ == "__main__":
    main()
