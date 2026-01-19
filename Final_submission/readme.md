## A breif summary of changes for intermediate and final submission for Part 1:
- For basic model solver, we mainly replace some plots with more informative one. Not much difference in the solver.
- We also reorganize the code to improve readability and maintainability. 
- For risky-debt model, we
  - correct the risky-debt pricing formula both in report and codes as we follow Strebulaev's paper before but their paper has inconsistency in Eq. 3.27 about the definition of $b'$
  - redefine the default condition in our code such that only $b'>0$, there is meaning to talk about default or not. But not be simply judged by whether Continuation value $C<0$ or not.
  - modify the minor definition of adjustment cost such that it is aligned with our report.
  - allow explicit default and firm restart as described in Strebulaev
  - Use line-plots over heatmap for more straightforward illustration
  - Some other minor changes


## Tests
- Run `pytest -q` and it can be shown that all tests in the two tests files would be passed for both part 1 and part 2 cdoes.

## Repository Layout
- Run `python main_part1.py` and `python main_part2.py` for solving and estimating the model. Note that for part 2 one must run `amortized_policy.py` first to get the parametrized policy and run `data_generation.py` to get synthetic data needed before running the estimation.
- All files outside the folder `Final_submission` are from intermediate submission and has not been touched after submission on 24 Nov.

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



