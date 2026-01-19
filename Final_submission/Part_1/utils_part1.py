"""
General-purpose utilities for Part 1.

These helpers are intentionally model-agnostic and are shared by both
training and diagnostic modules.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def compute_stats(
    g: np.ndarray,
    denom: np.ndarray,
    tol_list: Tuple[float, ...] = (1e-3, 1e-4),
) -> Dict[str, float]:
    """
    Compute absolute and relative residual statistics.

    Args:
        g: Residual array, typically E[g] evaluated at many states.
        denom: Scale array used for relative residuals.
        tol_list: Tolerances for reporting the share of points within tolerance.

    Returns:
        Dictionary of summary statistics (means, quantiles, maxima, and shares).
    """
    abs_g = np.abs(g)
    mae = abs_g.mean()
    rmse = np.sqrt((g**2).mean())
    med = np.quantile(abs_g, 0.5)
    p95 = np.quantile(abs_g, 0.95)
    mx = abs_g.max()

    # Log-scale diagnostics are useful when residuals span many magnitudes.
    abs_clip = np.maximum(abs_g, 1e-30)
    log10_abs = np.log10(abs_clip)
    log10_med = float(np.quantile(log10_abs, 0.5))
    log10_p95 = float(np.quantile(log10_abs, 0.95))

    # Relative residual normalization to reduce scale dependence.
    rel = abs_g / np.maximum(denom, 1e-12)
    rel_mean = float(rel.mean())
    rel_med = float(np.quantile(rel, 0.5))
    rel_p95 = float(np.quantile(rel, 0.95))

    out = dict(
        N=len(g),
        Abs_MAE=float(mae),
        Abs_RMSE=float(rmse),
        Abs_Median=float(med),
        Abs_P95=float(p95),
        Abs_Max=float(mx),
        Log10Abs_Median=log10_med,
        Log10Abs_P95=log10_p95,
        Rel_Mean=rel_mean,
        Rel_Median=rel_med,
        Rel_P95=rel_p95,
    )
    for t in tol_list:
        out[f"Share(|E[g]|<= {t:.0e})"] = float((abs_g <= t).mean())
    return out


def print_stats(
    title: str,
    stats: Dict[str, float],
    tol_list: Tuple[float, ...] = (1e-3, 1e-4),
) -> None:
    """
    Pretty-print residual statistics.

    Args:
        title: Section title shown in the header.
        stats: Statistics dictionary produced by `compute_stats`.
        tol_list: Tolerances that were included in `stats`.
    """
    print(f"\n[{title}]")
    print(f"- N = {stats['N']}")
    print(
        f"- Absolute residual |E[g]|: MAE={stats['Abs_MAE']:.3e}, "
        f"RMSE={stats['Abs_RMSE']:.3e}, Median={stats['Abs_Median']:.3e}, "
        f"P95={stats['Abs_P95']:.3e}, Max={stats['Abs_Max']:.3e}",
    )
    print(
        f"- log10(|E[g]|): Median={stats['Log10Abs_Median']:.3f}, "
        f"P95={stats['Log10Abs_P95']:.3f}",
    )
    print(
        "- Relative residual |E[g]|/(|1+psi_I|+|beta*E[term]|): "
        f"Mean={stats['Rel_Mean']:.3e}, Median={stats['Rel_Median']:.3e}, "
        f"P95={stats['Rel_P95']:.3e}",
    )
    for t in tol_list:
        key = f"Share(|E[g]|<= {t:.0e})"
        print(f"- {key}: {stats[key]:.3f}")