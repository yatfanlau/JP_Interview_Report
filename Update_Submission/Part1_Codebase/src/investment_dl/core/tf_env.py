"""TensorFlow runtime defaults shared by the whole package.

The package consistently imports TensorFlow from this module so runtime
configuration (CPU selection, dtype defaults, and seeding conventions)
is centralized in one place.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


def set_global_seed(seed: int) -> None:
    """Set Python and TensorFlow RNG seeds for reproducibility.

    This does not make every low-level op fully deterministic on every
    platform, but it aligns the main random sources used in this project.
    """
    random.seed(seed)
    tf.random.set_seed(seed)
