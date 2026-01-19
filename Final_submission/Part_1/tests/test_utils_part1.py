# tests/test_utils_part1.py
import numpy as np

from utils_part1 import compute_stats, print_stats


def test_compute_stats_known_values_matches_numpy():
    g = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    denom = np.ones_like(g)

    stats = compute_stats(g, denom, tol_list=(1e-3,))

    abs_g = np.abs(g)
    assert stats["N"] == 4
    assert np.isclose(stats["Abs_MAE"], abs_g.mean())
    assert np.isclose(stats["Abs_RMSE"], np.sqrt((g**2).mean()))
    assert np.isclose(stats["Abs_Median"], np.quantile(abs_g, 0.5))
    assert np.isclose(stats["Abs_P95"], np.quantile(abs_g, 0.95))
    assert np.isclose(stats["Abs_Max"], abs_g.max())

    # Share within tolerance
    assert np.isclose(stats["Share(|E[g]|<= 1e-03)"], float((abs_g <= 1e-3).mean()))


def test_compute_stats_handles_zero_denominator():
    g = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    denom = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    stats = compute_stats(g, denom)
    assert np.isfinite(stats["Rel_Mean"])
    assert np.isfinite(stats["Rel_P95"])


def test_print_stats_smoke(capsys):
    g = np.array([0.0, 1.0], dtype=np.float32)
    denom = np.ones_like(g)
    stats = compute_stats(g, denom, tol_list=(1e-3,))
    print_stats("MyTitle", stats, tol_list=(1e-3,))
    out = capsys.readouterr().out
    assert "MyTitle" in out
    assert "Absolute residual" in out
    assert "Relative residual" in out