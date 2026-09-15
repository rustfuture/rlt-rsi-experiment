"""Leakage-safe deterministic data generation for the sequence-parity task."""

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    tokens: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    seed: int
    min_length: int
    max_length: int

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for value in (self.tokens, self.labels, self.lengths):
            h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": int(len(self.labels)),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "seed": self.seed,
            "fingerprint": self.fingerprint(),
        }


def make_split(
    *, n: int, min_length: int, max_length: int, seed: int
) -> DatasetSplit:
    """Generate independent examples; no example is reused across splits.

    Token 0/1 is a bit. The label is XOR/parity over the *valid* prefix. The
    arrays are padded with token 0, while ``lengths`` tells the model which
    prefix is valid. For the fixed-length smoke runs, every row has one length.
    """
    if n <= 0 or min_length <= 0 or max_length < min_length:
        raise ValueError("n > 0 and 0 < min_length <= max_length are required")
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_length, max_length + 1, size=n, dtype=np.int64)
    tokens = np.zeros((n, max_length), dtype=np.int64)
    for i, length in enumerate(lengths):
        tokens[i, :length] = rng.integers(0, 2, size=int(length), dtype=np.int64)
    labels = np.asarray(
        [int(np.bitwise_xor.reduce(row[: int(length)])) for row, length in zip(tokens, lengths)],
        dtype=np.int64,
    )
    return DatasetSplit(tokens, labels, lengths, seed, min_length, max_length)


def split_manifest(splits: Dict[str, DatasetSplit]) -> str:
    """Stable JSON manifest used in result files and reproducibility checks."""
    return json.dumps({name: split.to_dict() for name, split in splits.items()}, sort_keys=True)
