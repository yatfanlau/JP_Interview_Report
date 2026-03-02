"""TensorFlow-backed replay buffer for training state samples.

The buffer implements ring-buffer semantics with fixed capacity and uniform
sampling over stored rows.
"""

from __future__ import annotations

import tensorflow as tf

from dyninv.utils.tf_setup import DTYPE


class ReplayBuffer:
    """Fixed-capacity ring buffer with TensorFlow-native storage."""

    def __init__(self, max_size: int, state_dim: int, seed: int = 42):
        """Allocate fixed-size storage and initialize ring-buffer pointers."""
        self.max_size = int(max_size)
        self.state_dim = int(state_dim)
        self._buffer = tf.Variable(
            initial_value=tf.zeros([self.max_size, self.state_dim], dtype=DTYPE),
            trainable=False,
            name="replay_buffer_storage",
        )
        self._size = tf.Variable(0, dtype=tf.int32, trainable=False, name="replay_buffer_size")
        self._ptr = tf.Variable(0, dtype=tf.int32, trainable=False, name="replay_buffer_ptr")
        self._seed = int(seed)

    @property
    def buffer(self):
        """Expose raw storage as NumPy for legacy callers/tests."""
        return self._buffer.numpy()

    def push_batch_tf(self, states: tf.Tensor) -> None:
        """Insert a batch of states into the ring buffer in TensorFlow space."""
        states = tf.cast(states, DTYPE)
        n = tf.shape(states)[0]
        # Circular write indices implement fixed-capacity overwrite semantics.
        idx = tf.math.mod(tf.range(self._ptr, self._ptr + n), self.max_size)
        updated = tf.tensor_scatter_nd_update(self._buffer, idx[:, None], states)
        self._buffer.assign(updated)
        self._ptr.assign(tf.math.mod(self._ptr + n, self.max_size))
        self._size.assign(tf.minimum(self.max_size, self._size + n))

    def push_batch(self, states) -> None:
        """Validate shape and insert a batch of states."""
        states_tf = tf.convert_to_tensor(states, dtype=DTYPE)
        if states_tf.shape.rank != 2 or states_tf.shape[1] != self.state_dim:
            raise AssertionError(
                f"Expected states with shape [B,{self.state_dim}], got {tuple(states_tf.shape)}"
            )
        self.push_batch_tf(states_tf)

    def sample_tf(self, batch_size: int) -> tf.Tensor:
        """Uniformly sample a random batch from currently stored states."""
        if int(self._size.numpy()) <= 0:
            raise RuntimeError("ReplayBuffer is empty.")
        # Sample uniformly from currently filled prefix `[0, size)`.
        idx = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=int(self._size.numpy()),
            dtype=tf.int32,
        )
        return tf.gather(self._buffer, idx)

    def sample(self, batch_size: int):
        """Sample a random batch and return NumPy output."""
        return self.sample_tf(int(batch_size)).numpy()

    def __len__(self) -> int:
        """Return number of valid rows currently stored in the buffer."""
        return int(self._size.numpy())
