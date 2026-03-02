"""Common estimator abstractions and result containers.

Concrete estimators share this lightweight interface so CLI and diagnostics code
can consume outputs consistently regardless of estimation method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EstimationResult:
    """Standardized estimator output payload.

    Attributes:
        res_df: Per-replication result table (typically a pandas DataFrame).
        theta_true: Ground-truth ``theta`` used in simulation.
        phi_true: Ground-truth ``phi`` used in simulation.
        extra: Method-specific metadata (replication IDs, diagnostics, etc.).
    """

    res_df: Any
    theta_true: float
    phi_true: float
    extra: dict


class BaseEstimator(ABC):
    """Abstract base class implemented by each concrete estimator."""

    @abstractmethod
    def run(self) -> EstimationResult:
        """Execute estimation and return standardized structured outputs."""
        raise NotImplementedError
