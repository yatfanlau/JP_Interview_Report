"""Regime-map statistics over discretized state-space bins.

The functions here aggregate simulated observations into ``(k,z)`` cells so
local policy behavior can be compared across methods and regions of the state
space rather than only via global summary moments.
"""

from __future__ import annotations

from investment_dl.core.tf_env import tf


def compute_regime_map(
    k: tf.Tensor,
    z: tf.Tensor,
    iota: tf.Tensor,
    k_edges: tf.Tensor,
    z_edges: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return per-bin mean ``iota`` values and observation counts.

    Bins are constructed from user-supplied edges in ``k`` and ``z``. Each
    observation is mapped to a segment index, then aggregated with unsorted
    segment reductions for speed.
    """
    k = tf.convert_to_tensor(k)
    z = tf.convert_to_tensor(z)
    iota = tf.convert_to_tensor(iota)
    k_edges = tf.cast(k_edges, k.dtype)
    z_edges = tf.cast(z_edges, z.dtype)

    n_k_bins = tf.shape(k_edges)[0] - 1
    n_z_bins = tf.shape(z_edges)[0] - 1
    n_segments = n_k_bins * n_z_bins

    # Convert 2D bin coordinates into a flat segment id.
    k_bin = tf.searchsorted(k_edges[1:-1], k, side="right", out_type=tf.int32)
    z_bin = tf.searchsorted(z_edges[1:-1], z, side="right", out_type=tf.int32)
    seg = k_bin * n_z_bins + z_bin

    # Aggregate counts and sums by segment id.
    ones = tf.ones_like(iota, dtype=tf.int32)
    counts = tf.math.unsorted_segment_sum(ones, seg, n_segments)
    sum_iota = tf.math.unsorted_segment_sum(tf.cast(iota, tf.float32), seg, n_segments)

    counts_f = tf.cast(counts, tf.float32)
    mean = tf.where(
        counts_f > 0,
        sum_iota / counts_f,
        tf.fill([n_segments], tf.constant(float("nan"), dtype=tf.float32)),
    )

    out = tf.reshape(mean, [n_k_bins, n_z_bins])
    counts = tf.reshape(counts, [n_k_bins, n_z_bins])
    return out, counts


def compute_regime_stats(
    k: tf.Tensor,
    z: tf.Tensor,
    iota: tf.Tensor,
    k_edges: tf.Tensor,
    z_edges: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return bin-wise mean/std/count/share diagnostics for ``iota``.

    Shares are defined relative to the full sample size and therefore sum
    to one (up to numerical precision) across all bins.
    """
    k = tf.convert_to_tensor(k)
    z = tf.convert_to_tensor(z)
    iota = tf.convert_to_tensor(iota)
    k_edges = tf.cast(k_edges, k.dtype)
    z_edges = tf.cast(z_edges, z.dtype)

    n_k_bins = tf.shape(k_edges)[0] - 1
    n_z_bins = tf.shape(z_edges)[0] - 1
    n_segments = n_k_bins * n_z_bins

    # Segment index for each observation in the flattened bin lattice.
    k_bin = tf.searchsorted(k_edges[1:-1], k, side="right", out_type=tf.int32)
    z_bin = tf.searchsorted(z_edges[1:-1], z, side="right", out_type=tf.int32)
    seg = k_bin * n_z_bins + z_bin

    ones = tf.ones_like(iota, dtype=tf.int32)
    counts = tf.math.unsorted_segment_sum(ones, seg, n_segments)
    counts_f = tf.cast(counts, tf.float32)

    # Compute E[iota] and E[iota^2] for variance reconstruction.
    iota_f = tf.cast(iota, tf.float32)
    sum_iota = tf.math.unsorted_segment_sum(iota_f, seg, n_segments)
    sum_sq = tf.math.unsorted_segment_sum(tf.square(iota_f), seg, n_segments)
    mean = tf.where(
        counts_f > 0,
        sum_iota / counts_f,
        tf.fill([n_segments], tf.constant(float("nan"), dtype=tf.float32)),
    )
    var = tf.where(
        counts_f > 0,
        tf.maximum(sum_sq / counts_f - tf.square(mean), 0.0),
        tf.fill([n_segments], tf.constant(float("nan"), dtype=tf.float32)),
    )
    std = tf.sqrt(var)

    total = tf.reduce_sum(counts_f)
    share = counts_f / tf.maximum(total, 1.0)

    mean = tf.reshape(mean, [n_k_bins, n_z_bins])
    std = tf.reshape(std, [n_k_bins, n_z_bins])
    counts = tf.reshape(counts, [n_k_bins, n_z_bins])
    share = tf.reshape(share, [n_k_bins, n_z_bins])
    return mean, std, counts, share
