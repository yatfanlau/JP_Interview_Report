# investment-dl

`investment-dl` is a Python package for solving and benchmarking a basic stochastic
investment model with:

- a deep-learning Euler-equation solver,
- a value-function-iteration (VFI) benchmark,
- simulation, evaluation, and plotting utilities.

## Package Layout

```text
pyproject.toml
README.md

src/
  investment_dl/
    __init__.py
    config.py                # BasicModelParams, BasicTrainingParams, BasicFinalTestParams

    core/
      __init__.py
      tf_env.py              # TF setup, logging, CPU-only, DTYPE, set_global_seed
      math_utils.py          # steady_state_k and related helpers
      stochastic.py          # ar1_step_ln_z, tauchen_ln_z_grid, get_gh_nodes
      replay_buffer.py       # ReplayBuffer class

    models/
      __init__.py
      basic_investment.py    # BasicInvestmentModel (economic primitives, constants)
      policies.py            # BoundedTanh, PolicyNetwork

    solvers/
      __init__.py
      euler_dl.py            # EulerEquationTrainer (DL Euler-equation solver)
      vfi.py                 # VFISolver, VFIInterpolatedPolicy

    simulation/
      __init__.py
      simulators.py          # PolicySimulator (simulate_sample, simulate_panel, etc.)

    evaluation/
      __init__.py
      euler_residuals.py     # EulerResidualEvaluator (GH residuals + stats)
      values.py              # PolicyValueEvaluator (value eval, Bellman residuals + stats)
      regimes.py             # compute_regime_map, compute_regime_stats
      panels.py              # panel_autocorr, compute_panel_moments

    plotting/
      __init__.py
      training.py            # DL/VFI convergence plots
      policies.py            # policy heatmaps & slices
      values.py              # value-function heatmaps
      distributions.py       # histograms for k, iota, etc.
      regimes.py             # regime maps & diagnostics
      moments.py             # panel-moment bar charts

    cli/
      __init__.py
      run_basic.py           # high-level experiment script / CLI

tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Usage

### Run full experiment (CLI)

```bash
python -m investment_dl.cli.run_basic
```

### Run with custom VFI grid bounds/size

```bash
python -m investment_dl.cli.run_basic --n-k 801 --n-z 41 --k-min-mul 0.5 --k-max-mul 2.5
```

### Use as a library

```python
from investment_dl.models.basic_investment import BasicInvestmentModel
from investment_dl.solvers.euler_dl import EulerEquationTrainer
from investment_dl.solvers.vfi import VFISolver

model = BasicInvestmentModel()
trainer = EulerEquationTrainer(model)
trainer.train(return_history=False)

solver = VFISolver(model)
k_grid, z_grid, V_vfi, iota_vfi = solver.solve(return_history=False)
```

## Outputs

The CLI generates benchmark plots in the working directory, including:

- `policy_comparison.png`
- `value_comparison.png`
- `dl_convergence.png`
- `vfi_convergence.png`
- `panel_moments.png`
- `dist_k.png`
- `dist_iota.png`
- `regime_map.png`

## Notes

- If you do not install with `pip install -e .`, run with:

```bash
PYTHONPATH=src python -m investment_dl.cli.run_basic
```

## Running Tests

Install test dependency:

```bash
python -m pip install pytest
```

Run all unit + integration tests:

```bash
python -m pytest -q
```

Output:

```text
...........                                                              [100%]
11 passed in 4.52s
```
