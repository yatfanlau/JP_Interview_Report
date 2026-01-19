import pandas as pd

from config_part2 import (
    PathsPart2,
    PanelColumnsPart2,
    ParamBoundsPart2,
    SMMConfigPart2,
    GMMConfigPart2,
    HMCConfigPart2,
    HMCColumnsPart2,
    BasicModelParams,
)

import SMM_estimation as smm
import GMM_estimation as gmm
import HMC_estimation_ as hmc
import diagnostic as diag


def main():
    paths = PathsPart2()
    cols = PanelColumnsPart2()
    hmc_cols = HMCColumnsPart2()
    bounds = ParamBoundsPart2()
    smm_cfg = SMMConfigPart2()
    gmm_cfg = GMMConfigPart2()
    hmc_cfg = HMCConfigPart2()
    mp = BasicModelParams()

    # --- SMM
    smm_ctx = smm.run_smm_estimation(paths=paths, cols=cols, bounds=bounds, cfg=smm_cfg, mp=mp)
    diag.run_metric_1(smm_ctx["res_df"], smm_ctx["theta_true"], smm_ctx["phi_true"], label="SMM")

    df_all = pd.read_csv(paths.data_csv)
    metric2_df = diag.run_metric_2_coverage_smm(
        smm_ctx["res_df"],
        df_all,
        theta_true=smm_ctx["theta_true"],
        phi_true=smm_ctx["phi_true"],
        policy=smm_ctx["policy"],
        paths=paths,
        cols=cols,
        bounds=bounds,
        cfg=smm_cfg,
    )
    print(metric2_df.head())

    metric3_df = diag.run_metric_3_aux_fit_smm(
        smm_ctx["res_df"],
        df_all,
        policy=smm_ctx["policy"],
        cols=cols,
        cfg=smm_cfg,
        mp=mp,
    )
    print(metric3_df.head())

    jtest_df = smm.run_j_test_overid(
        smm_ctx["res_df"],
        df_all,
        policy=smm_ctx["policy"],
        alpha=smm_cfg.jtest_alpha,
        n_boot=smm_cfg.jtest_n_boot,
        n_sims=smm_cfg.jtest_n_sims,
        ridge=smm_cfg.jtest_ridge,
    )
    print(jtest_df.head())

    smm.diagnostic_step2_loss_vs_chi2(
        smm_ctx["res_df"],
        df_chi=smm.N_TARGET_MOMENTS - 2,
        n_mc=smm_cfg.diag_chi2_n_mc,
        random_seed=smm_cfg.diag_chi2_seed,
    )

    # --- GMM
    gmm_ctx = gmm.run_gmm_estimation(paths=paths, cols=cols, bounds=bounds, cfg=gmm_cfg)
    diag.run_metric_1(gmm_ctx["res_df"], gmm_ctx["theta_true"], gmm_ctx["phi_true"], label="GMM")

    df_all = pd.read_csv(paths.data_csv)
    metric2_df = diag.run_metric_2_coverage_gmm(
        gmm_ctx["res_df"],
        df_all,
        theta_true=gmm_ctx["theta_true"],
        phi_true=gmm_ctx["phi_true"],
        cols=cols,
        bounds=bounds,
        cfg=gmm_cfg,
    )
    print(metric2_df.head())

    metric3_df = diag.run_metric_3_aux_fit_gmm(
        gmm_ctx["res_df"],
        df_all,
        paths=paths,
        cols=cols,
        cfg=gmm_cfg,
        mp=BasicModelParams(),
    )
    print(metric3_df.head())

    jtest_df = gmm.run_j_test_overid_gmm(gmm_ctx["res_df"], alpha=gmm_cfg.hansen_alpha)
    print(jtest_df.head())

    df_chi = int(gmm_ctx["res_df"]["df"].iloc[0]) if len(gmm_ctx["res_df"]) else 0
    if df_chi > 0:
        gmm.diagnostic_Jstat_vs_chi2(
            gmm_ctx["res_df"],
            df_chi=df_chi,
            n_mc=gmm_cfg.diag_chi2_n_mc,
            random_seed=gmm_cfg.diag_chi2_seed,
        )

    # --- HMC + UKF (theta, phi only)
    hmc_ctx = hmc.run_hmc_estimation(paths=paths, cols=cols, hmc_cols=hmc_cols, bounds=bounds, cfg=hmc_cfg, mp=mp)
    diag.run_metric_1(hmc_ctx["res_df"], hmc_ctx["theta_true"], hmc_ctx["phi_true"], label="HMC-UKF")

    metric2_hmc_df = diag.run_metric_2_coverage_hmc(
        hmc_ctx["res_df"],
        theta_true=hmc_ctx["theta_true"],
        phi_true=hmc_ctx["phi_true"],
        cfg=hmc_cfg,
    )
    print(metric2_hmc_df.head())

    df_all = pd.read_csv(paths.data_csv)
    metric3_hmc_df = diag.run_metric_3_aux_fit_hmc(
        hmc_ctx["res_df"],
        df_all,
        policy=hmc_ctx["policy"],
        cols=cols,
        cfg=hmc_cfg,
        mp=mp,
    )
    print(metric3_hmc_df.head())


if __name__ == "__main__":
    main()