"""CLI entry for the full train/simulate/estimate pipeline.

This orchestrates policy training, synthetic data generation, and all three
estimators (SMM, GMM, HMC) with default configurations.
"""

from __future__ import annotations

from dyninv.cli.generate_data import main as generate_data_main
from dyninv.cli.run_gmm import main as run_gmm_main
from dyninv.cli.run_hmc import main as run_hmc_main
from dyninv.cli.run_smm import main as run_smm_main
from dyninv.cli.train_policy import main as train_policy_main
from dyninv.config import (
    BasicModelParams,
    GMMConfigPart2,
    HMCConfigPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    PathsPart2,
    SMMConfigPart2,
)
from dyninv.diagnostics.metrics import run_metric_1
from dyninv.estimation.gmm import GMMEstimator
from dyninv.estimation.smm import SMMEstimator


def main() -> None:
    """Run the full policy-training, simulation, and estimation pipeline."""
    train_policy_main()
    generate_data_main()

    paths = PathsPart2()
    cols = PanelColumnsPart2()
    bounds = ParamBoundsPart2()

    smm = SMMEstimator(paths=paths, cols=cols, bounds=bounds, cfg=SMMConfigPart2(), mp=BasicModelParams()).run()
    gmm = GMMEstimator(paths=paths, cols=cols, bounds=bounds, cfg=GMMConfigPart2()).run()

    run_metric_1(smm.res_df, smm.theta_true, smm.phi_true, label="SMM")
    run_metric_1(gmm.res_df, gmm.theta_true, gmm.phi_true, label="GMM")

    run_smm_main()
    run_gmm_main()
    run_hmc_main()


if __name__ == "__main__":
    main()
