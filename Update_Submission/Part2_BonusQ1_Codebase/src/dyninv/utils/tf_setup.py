"""TensorFlow runtime setup and shared dtype utilities.

The module configures TensorFlow for CPU-first execution and exposes a single
package-wide dtype constant used across numerical code.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


def as_tf_constant(value: float | int, dtype: tf.dtypes.DType = DTYPE) -> tf.Tensor:
    """Create a scalar TensorFlow constant using the package default dtype."""
    return tf.constant(value, dtype=dtype)
