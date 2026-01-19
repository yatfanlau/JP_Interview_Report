## Note that Part 1, Part 2 and Bonus Q1 are attmpted and the HMC method is included in Part for easier comparison.

## Summary of Changes for Intermediate and Final Submission (Part 1)

For both model, we reorganized the codebase to enhance readability and maintainability.

### Basic Model Solver

- Replaced some plots with more informative visualizations; the core solver remains largely unchanged.


### Risky-Debt Model

- Corrected the risky-debt pricing formula in both the report and code. Previously, we followed Strebulaev’s paper, which contains an inconsistency in Equation 3.27 regarding the definition of $b'$.
- Redefined the default condition in the code: default is now only considered when $b' > 0$, ensuring it's meaningful to assess default. This replaces the earlier criterion based solely on whether the continuation value $C<0$.
- Aligned the definition of adjustment costs in the code with the description in the report.
- Introduced explicit default and firm restart mechanisms, as described in Strebulaev.
- Replaced heatmaps with line plots for clearer and more intuitive illustrations.
- Incorporated various minor improvements and refinements. 


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



