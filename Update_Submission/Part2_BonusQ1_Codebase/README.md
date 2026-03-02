# dyninv

`dyninv` is a TensorFlow-based Python package for dynamic investment policy training,
panel simulation, and structural estimation (SMM, GMM, UKF-HMC).

## Package layout

```text
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ src/
│  └─ dyninv/
│     ├─ __init__.py
│     ├─ config.py                # Dataclass configs (model, training, SMM, GMM, HMC, paths, columns)
│     │
│     ├─ utils/
│     │  ├─ __init__.py
│     │  ├─ tf_setup.py           # TF/DTYPE setup and CPU forcing
│     │  ├─ random.py             # Global seed helpers
│     │  ├─ replay_buffer.py      # ReplayBuffer class
│     │  ├─ numerics.py           # Symmetrization, PSD pseudo-inverse, chi2/z helpers
│     │  └─ stats.py              # Correlation, OLS, logit/log-sigmoid helpers
│     │
│     ├─ model/
│     │  ├─ __init__.py
│     │  ├─ primitives.py         # profit_k, psi_i, psi_k, euler_term
│     │  ├─ processes.py          # AR(1) in ln z and steady-state helpers
│     │  ├─ context.py            # build_basic_model_context, iota_bounds
│     │  └─ environment.py        # EconomicEnvironment class
│     │
│     ├─ policy/
│     │  ├─ __init__.py
│     │  ├─ network.py            # ParamPolicyNet and custom feature layer
│     │  ├─ sampler.py            # Coverage samplers for (k, z, theta, phi)
│     │  ├─ rollout.py            # PolicyRollout class using ReplayBuffer
│     │  └─ training.py           # PolicyTrainer class (AiO Euler loss, train, save)
│     │
│     ├─ simulation/
│     │  ├─ __init__.py
│     │  ├─ panel_simulator.py    # PanelSimulator class (TF simulation kernels)
│     │  ├─ data_generation.py    # DataGenerator facade for replicated panel CSV output
│     │  └─ panels.py             # PanelData class with CSV load/save
│     │
│     ├─ estimation/
│     │  ├─ __init__.py
│     │  ├─ base.py               # BaseEstimator abstraction and EstimationResult dataclass
│     │  ├─ smm.py                # SMMEstimator
│     │  ├─ gmm.py                # GMMEstimator
│     │  └─ bayesian_hmc.py       # HMCEstimator (TFP HMC)
│     │
│     ├─ diagnostics/
│     │  ├─ __init__.py
│     │  ├─ aux_moments.py        # aux_moments_from_df / aux_moments_tf
│     │  └─ metrics.py            # Metric summaries and coverage diagnostics
│     │
│     └─ cli/
│        ├─ __init__.py
│        ├─ train_policy.py       # main(): train amortized policy and save model
│        ├─ generate_data.py      # main(): load policy, simulate panels, write CSV
│        ├─ run_smm.py            # main(): run SMM over reps
│        ├─ run_gmm.py            # main(): run GMM over reps
│        ├─ run_hmc.py            # main(): run Bayesian HMC over reps
│        └─ run_all.py            # main(): orchestrate full pipeline
│
└─ tests/
   ├─ conftest.py
   ├─ unit/
   │  ├─ test_policy_training.py
   │  ├─ test_simulation.py
   │  ├─ test_smm.py
   │  ├─ test_gmm.py
   │  ├─ test_hmc.py
   │  ├─ test_diagnostics.py
   │  └─ test_utils.py
   └─ integration/
      └─ test_estimators_smoke.py
```

The codebase is now primarily organized in an OOP style: core workflows such as training, rollout, simulation, and estimation are implemented through classes (for example, `PolicyTrainer`, `PolicyRollout`, `PanelSimulator`, `DataGenerator`, `SMMEstimator`, `GMMEstimator`, and `HMCEstimator`) within the `dyninv` package hierarchy. The remaining functional-style pieces are mostly stateless utility and math helpers (for example, tensor numerics, statistical helpers, and low-level model primitives), which are kept as functions because they are deterministic transformations without internal state.

## Quick start

```bash
pip install -e .
dyninv-train-policy
dyninv-generate-data
dyninv-run-all
```

## Testing

Run tests with `pytest`:

```bash
python -m pytest -q -m unit
python -m pytest -q -m integration
python -m pytest -q
```

- `-m unit`: fast unit tests.
- `-m integration`: integration tests covering estimator pipeline interactions.
- no marker: full test suite.

Example run results:

```text
$ python -m pytest -q -m unit
............                                                             [100%]
12 passed, 3 deselected

$ python -m pytest -q -m integration
...                                                                      [100%]
3 passed, 12 deselected

$ python -m pytest -q
...............                                                          [100%]
15 passed
```
