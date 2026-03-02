"""CLI entry point for two-step SMM estimation.

Loads default simulation/optimization settings and prints a preview of
replication-level parameter estimates.
"""

from __future__ import annotations

from dyninv.config import BasicModelParams, PanelColumnsPart2, ParamBoundsPart2, PathsPart2, SMMConfigPart2
from dyninv.estimation.smm import SMMEstimator


def main() -> None:
    """Run SMM estimation with package-default configuration objects."""
    est = SMMEstimator(
        paths=PathsPart2(),
        cols=PanelColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=SMMConfigPart2(),
        mp=BasicModelParams(),
    )
    out = est.run()
    print(out.res_df.head())


if __name__ == "__main__":
    main()
