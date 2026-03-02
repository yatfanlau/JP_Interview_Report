"""Convenience exports for evaluation and diagnostic functionality.

These symbols cover residual checks, value diagnostics, regime aggregation,
and panel-moment statistics used in benchmark scripts.
"""

from investment_dl.evaluation.euler_residuals import EulerResidualEvaluator
from investment_dl.evaluation.panels import compute_panel_moments, panel_autocorr
from investment_dl.evaluation.regimes import compute_regime_map, compute_regime_stats
from investment_dl.evaluation.values import PolicyValueEvaluator

__all__ = [
    "EulerResidualEvaluator",
    "PolicyValueEvaluator",
    "compute_regime_map",
    "compute_regime_stats",
    "panel_autocorr",
    "compute_panel_moments",
]
