"""Public re-exports for shared numerical and runtime utilities.

Importing from ``investment_dl.core`` provides a stable, compact entry point
for common helpers used across training, simulation, and evaluation modules.
"""

from investment_dl.core.math_utils import steady_state_k, tf_quantile_1d, tf_var
from investment_dl.core.replay_buffer import ReplayBuffer
from investment_dl.core.stochastic import ar1_step_ln_z, get_gh_nodes, tauchen_ln_z_grid
from investment_dl.core.tf_env import DTYPE, set_global_seed, tf

__all__ = [
    "tf",
    "DTYPE",
    "set_global_seed",
    "steady_state_k",
    "tf_quantile_1d",
    "tf_var",
    "ReplayBuffer",
    "ar1_step_ln_z",
    "get_gh_nodes",
    "tauchen_ln_z_grid",
]
