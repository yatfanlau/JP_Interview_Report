"""Replay-buffer implementation for state sampling.

The buffer stores 2D state rows and supports ring overwrite semantics so memory
usage remains fixed over long training runs.
"""

from __future__ import annotations

from investment_dl.core.tf_env import DTYPE, tf


class ReplayBuffer:
    """Fixed-capacity ring buffer for vector-valued simulation states.

    The implementation is optimized for repeated batch insertion from policy
    rollouts and uniform random sampling during stochastic training updates.
    """

    def __init__(self, max_size: int, state_dim: int, seed: int = 42):
        """Create a fixed-size ring buffer.

        Parameters
        ----------
        max_size:
            Maximum number of state rows retained.
        state_dim:
            Width of each stored state vector.
        seed:
            Seed used to initialize TensorFlow RNG state.
        """
        self.max_size = int(max_size)
        self.state_dim = int(state_dim)
        self.max_size_tf = tf.constant(self.max_size, dtype=tf.int32)
        self.state_dim_tf = tf.constant(self.state_dim, dtype=tf.int32)
        self.buffer = tf.Variable(
            tf.zeros((self.max_size, self.state_dim), dtype=DTYPE),
            trainable=False,
        )
        self.size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.ptr = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.rng = tf.random.Generator.from_seed(seed)

    @tf.function
    def push_batch(self, states: tf.Tensor) -> None:
        """Insert a batch of states and overwrite oldest entries if full.

        The write indices are computed from the current pointer and wrapped
        modulo ``max_size`` so this operation behaves like a circular queue.
        """
        states = tf.convert_to_tensor(states, dtype=DTYPE)
        tf.debugging.assert_rank(states, 2)
        tf.debugging.assert_equal(
            tf.shape(states)[1],
            self.state_dim_tf,
            message="ReplayBuffer: wrong state dimension.",
        )
        n = tf.shape(states)[0]
        # Consecutive indices from ptr, wrapped around buffer capacity.
        idx = tf.range(n, dtype=tf.int32) + self.ptr
        idx = tf.math.floormod(idx, self.max_size_tf)
        self.buffer.scatter_nd_update(tf.expand_dims(idx, 1), states)
        # Update pointer and effective size after write.
        self.ptr.assign(tf.math.floormod(self.ptr + n, self.max_size_tf))
        self.size.assign(tf.minimum(self.size + n, self.max_size_tf))

    def sample(self, batch_size: int) -> tf.Tensor:
        """Uniformly sample a batch of states from populated rows only."""
        size_int = int(self.size.numpy())
        if size_int == 0:
            raise RuntimeError("ReplayBuffer is empty.")
        idx = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=size_int,
            dtype=tf.int32,
        )
        return tf.gather(self.buffer, idx)

    def __len__(self) -> int:
        """Return the number of currently stored samples."""
        return int(self.size.numpy())
