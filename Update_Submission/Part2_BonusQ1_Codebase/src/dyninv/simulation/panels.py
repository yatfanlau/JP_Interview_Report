"""Panel-tensor container with CSV serialization helpers.

The estimators expect long-format panel rows with fixed column names, while
simulators naturally produce matrix-shaped tensors. This module bridges those
representations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from dyninv.utils import DTYPE, tf


@dataclass
class PanelData:
    """Wrap panel tensors and provide CSV conversion utilities.

    Fields store flattened long-format vectors (one row per firm-time pair).
    """

    rep: tf.Tensor
    firm: tf.Tensor
    t: tf.Tensor
    k: tf.Tensor
    i: tf.Tensor
    iota: tf.Tensor
    z: tf.Tensor
    k_next: tf.Tensor
    z_next: tf.Tensor
    theta_true: tf.Tensor
    phi_true: tf.Tensor

    @classmethod
    def from_simulation(cls, panel, rep_index: int, theta_true: float, phi_true: float) -> "PanelData":
        """Convert matrix-shaped simulation output into flattened panel vectors.

        Args:
            panel: ``SimulationPanel`` with shape ``(firms, time)`` fields.
            rep_index: Replication identifier to attach to every row.
            theta_true: True replication-level ``theta`` value.
            phi_true: True replication-level ``phi`` value.
        """
        n_firms = tf.shape(panel.k)[0]
        t_periods = tf.shape(panel.k)[1]

        # Construct identifier grids aligned with `(firms, time)` panel tensors.
        firm_idx = tf.tile(tf.range(n_firms, dtype=tf.int32)[:, None], [1, t_periods])
        t_idx = tf.tile(tf.range(t_periods, dtype=tf.int32)[None, :], [n_firms, 1])
        rep_idx = tf.fill([n_firms, t_periods], tf.constant(rep_index, dtype=tf.int32))

        return cls(
            rep=tf.reshape(rep_idx, [-1]),
            firm=tf.reshape(firm_idx, [-1]),
            t=tf.reshape(t_idx, [-1]),
            k=tf.reshape(panel.k, [-1]),
            i=tf.reshape(panel.i, [-1]),
            iota=tf.reshape(panel.iota, [-1]),
            z=tf.reshape(panel.z, [-1]),
            k_next=tf.reshape(panel.k_next, [-1]),
            z_next=tf.reshape(panel.z_next, [-1]),
            theta_true=tf.fill([n_firms * t_periods], tf.constant(theta_true, dtype=DTYPE)),
            phi_true=tf.fill([n_firms * t_periods], tf.constant(phi_true, dtype=DTYPE)),
        )

    def to_rows(self) -> list[dict]:
        """Convert tensor fields into Python dictionaries for CSV writing.

        Returns:
            A list of row dictionaries in the canonical schema expected by
            downstream estimators.
        """
        # Convert once to Python lists so the row loop stays pure Python.
        n = int(tf.shape(self.k)[0].numpy())
        rep = self.rep.numpy().tolist()
        firm = self.firm.numpy().tolist()
        t = self.t.numpy().tolist()
        k = self.k.numpy().tolist()
        i = self.i.numpy().tolist()
        iota = self.iota.numpy().tolist()
        z = self.z.numpy().tolist()
        k_next = self.k_next.numpy().tolist()
        z_next = self.z_next.numpy().tolist()
        theta_true = self.theta_true.numpy().tolist()
        phi_true = self.phi_true.numpy().tolist()

        rows = []
        for j in range(n):
            # Keep canonical column names used by estimators and diagnostics.
            rows.append(
                {
                    "rep": rep[j],
                    "firm": firm[j],
                    "t": t[j],
                    "k": k[j],
                    "I": i[j],
                    "iota": iota[j],
                    "z": z[j],
                    "k_next": k_next[j],
                    "z_next": z_next[j],
                    "theta_true": theta_true[j],
                    "phi_true": phi_true[j],
                }
            )
        return rows

    def save_csv(self, path: str, append: bool = False) -> None:
        """Write panel rows to CSV using the package's canonical column order.

        Args:
            path: Destination CSV path.
            append: If ``True``, append rows and write header only when needed.
        """
        headers = [
            "rep",
            "firm",
            "t",
            "k",
            "I",
            "iota",
            "z",
            "k_next",
            "z_next",
            "theta_true",
            "phi_true",
        ]
        mode = "a" if append else "w"
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Write header on first write only; appends should keep existing schema.
        write_header = (not append) or (not path_obj.exists())
        with path_obj.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerows(self.to_rows())

    @classmethod
    def load_csv(cls, path: str) -> "PanelData":
        """Load canonical panel CSV rows into tensor-backed fields.

        Numeric values are parsed as floats and cast to integer dtypes for
        identifier columns.
        """
        rows: list[dict] = []
        with Path(path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

        def col(name: str, dtype):
            """Read one named CSV column into a TensorFlow tensor."""
            values = [float(r[name]) for r in rows]
            return tf.constant(values, dtype=dtype)

        return cls(
            rep=tf.cast(col("rep", tf.float32), tf.int32),
            firm=tf.cast(col("firm", tf.float32), tf.int32),
            t=tf.cast(col("t", tf.float32), tf.int32),
            k=col("k", DTYPE),
            i=col("I", DTYPE),
            iota=col("iota", DTYPE),
            z=col("z", DTYPE),
            k_next=col("k_next", DTYPE),
            z_next=col("z_next", DTYPE),
            theta_true=col("theta_true", DTYPE),
            phi_true=col("phi_true", DTYPE),
        )
