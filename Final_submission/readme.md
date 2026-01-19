## Repository Layout

```text
Final_submission/
├── Part_1/
│   ├── tests/     
│   ├── basic_model.py                       Functions for solving basic model
│   ├── basic_model_diagnostic.py            Functions for basic model diagnostics
│   ├── config_part1.py                      All params are put here
│   ├── main_part1.py                        main file to run the solver
│   ├── model_core.py                        Common functions for both basic model and risky-debt model
│   ├── readme.md
│   ├── risky_debt_diagnostic.py             Functions for risky-debt model diagnostics
│   ├── risky_debt_model.py                  Functions for solving risky-debt model
│   └── utils_part1.py                       Some useful utils
└── Part_2/
    ├── tests/
    ├── GMM_estimation.py                    Functions for GMM estimation
    ├── HMC_estimation_.py                   Functions for HMC estimation
    ├── SMM_estimation.py                    Functions for SMM estimation
    ├── amortized_policy.py                  A parametrized policy solver of basic model
    ├── config_part2.py                      All params are put here
    ├── data_generation.py                   To generate the synthetic data needed
    ├── diagnostic.py                        Fucntions for diagnostic
    ├── gmm_smm_core.py                      Functions common to both SMM/GMM
    ├── main_part2.py                        main file to run estimation methods
    ├── param_policy_theta_phi.keras         trained amortized policy
    ├── readme.md
    └── utils_part2.py                       Some useful utils
