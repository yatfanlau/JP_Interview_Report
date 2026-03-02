"""CLI entry point for two-step GMM estimation.

Uses package-default paths, bounds, and optimizer settings, then prints a
preview of replication-level estimates.
"""

from __future__ import annotations

from dyninv.config import GMMConfigPart2, PanelColumnsPart2, ParamBoundsPart2, PathsPart2
from dyninv.estimation.gmm import GMMEstimator


def main() -> None:
    """Run GMM estimation with default package configuration objects."""
    est = GMMEstimator(
        paths=PathsPart2(),
        cols=PanelColumnsPart2(),
        bounds=ParamBoundsPart2(),
        cfg=GMMConfigPart2(),
    )
    out = est.run()
    print(out.res_df.head())


if __name__ == "__main__":
    main()
