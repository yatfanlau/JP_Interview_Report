"""Public exports for TensorFlow setup, numerics, and random utilities."""

from dyninv.utils.tf_setup import DTYPE, as_tf_constant, tf
from dyninv.utils.random import set_global_seed
from dyninv.utils.replay_buffer import ReplayBuffer
from dyninv.utils.numerics import chi2_sf, pinv_psd_np, pinv_psd_tf, symmetrize_np, symmetrize_tf, zcrit
from dyninv.utils.stats import (
    log1m_sigmoid_tf,
    log_sigmoid_prime_tf,
    log_sigmoid_tf,
    logit_np,
    logit_tf,
    ols_slopes_2reg_with_intercept_np,
    ols_slopes_2reg_with_intercept_tf,
    safe_corr_np,
    safe_corr_tf,
)

__all__ = [
    "tf",
    "DTYPE",
    "as_tf_constant",
    "set_global_seed",
    "ReplayBuffer",
    "symmetrize_tf",
    "pinv_psd_tf",
    "symmetrize_np",
    "pinv_psd_np",
    "zcrit",
    "chi2_sf",
    "safe_corr_tf",
    "safe_corr_np",
    "ols_slopes_2reg_with_intercept_tf",
    "ols_slopes_2reg_with_intercept_np",
    "logit_tf",
    "logit_np",
    "log_sigmoid_tf",
    "log1m_sigmoid_tf",
    "log_sigmoid_prime_tf",
]
