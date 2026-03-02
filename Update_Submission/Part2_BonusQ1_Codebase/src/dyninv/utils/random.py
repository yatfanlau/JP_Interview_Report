"""Randomness utilities for deterministic runs.

Centralizing seed setup keeps reproducibility behavior consistent across
training, simulation, and tests.
"""

from __future__ import annotations

import random

import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Set seeds for Python and TensorFlow random-number generators."""
    random.seed(seed)
    tf.random.set_seed(seed)
