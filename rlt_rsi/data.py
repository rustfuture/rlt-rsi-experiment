"""Leakage-safe deterministic data generation for the sequence-parity task.

Splits are made *example-disjoint by construction*: a split refuses to draw any
``(length, token sequence)`` pair that already appeared in this split or in the
splits passed via ``exclude_keys``. Sharing only distinct PRNG seeds is not
sufficient on this task because the example space for short lengths is tiny
(e.g. only ``2**4 = 16`` distinct length-4 sequences), so independent PRNG draws
collide often. The helpers below make that overlap measurable.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

#: An exact example identity: ``(length, (token_0, ..., token_{length-1}))``.
ExampleKey = Tuple[int, Tuple[int, ...]]


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
            "length_counts": {
                str(int(length)): int(count)
                for length, count in zip(*np.unique(self.lengths, return_counts=True))
            },
        }


def example_keys(split: DatasetSplit) -> Set[ExampleKey]:
    """Return the exact set of examples in ``split``.

    Two rows are the same example when they have the same valid length and the
    same token sequence, regardless of any padding beyond ``length``.
    """
    return {
        (int(length), tuple(int(token) for token in row[: int(length)]))
        for row, length in zip(split.tokens, split.lengths)
    }


def count_overlap(a: DatasetSplit, b: DatasetSplit) -> int:
    """Number of exact examples shared by ``a`` and ``b`` (same length, same tokens)."""
    return len(example_keys(a) & example_keys(b))


def overlap_report(splits: Dict[str, DatasetSplit]) -> Dict[str, int]:
    """Pairwise exact-example overlap counts for a named split bundle."""
    names = list(splits)
    out: Dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            out[f"{left}_{right}"] = count_overlap(splits[left], splits[right])
    return out


def make_split(
    *,
    n: int,
    min_length: int,
    max_length: int,
    seed: int,
    exclude_keys: Optional[Iterable[ExampleKey]] = None,
    unique_examples: bool = True,
) -> DatasetSplit:
    """Generate examples, rejecting any that repeat or that appear in ``exclude_keys``.

    Token 0/1 is a bit. The label is XOR/parity over the *valid* prefix. The
    arrays are padded with token 0, while ``lengths`` tells the model which
    prefix is valid.

    With ``unique_examples=True`` (the default) no ``(length, tokens)`` pair is
    drawn twice, and none is drawn from ``exclude_keys``. This can only fail
    loudly: if ``n`` exceeds the available example space the loop raises
    ``ValueError`` instead of silently returning duplicates.
    """
    if n <= 0 or min_length <= 0 or max_length < min_length:
        raise ValueError("n > 0 and 0 < min_length <= max_length are required")
    rng = np.random.default_rng(seed)
    excluded: Set[ExampleKey] = set(exclude_keys) if exclude_keys else set()
    if unique_examples and max_length <= 20:
        capacity = sum(1 << length for length in range(min_length, max_length + 1))
        capacity -= sum(1 for (length, _) in excluded if min_length <= length <= max_length)
        if n > capacity:
            raise ValueError(
                f"example space is too small: requested n={n} distinct examples but only {capacity} "
                f"exist for lengths [{min_length}, {max_length}]"
            )
    seen: Set[ExampleKey] = set()
    lengths_list = []
    token_rows = []
    attempts = 0
    max_attempts = 200 * n + 10_000
    while len(lengths_list) < n:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"example space is too small: could not draw {n} distinct examples for lengths "
                f"[{min_length}, {max_length}] after {attempts} rejection-sampling attempts "
                "(and/or the excluded keys cover too much of it)"
            )
        length = int(rng.integers(min_length, max_length + 1))
        row = tuple(int(bit) for bit in rng.integers(0, 2, size=length))
        key = (length, row)
        if key in excluded:
            continue
        if unique_examples:
            if key in seen:
                continue
            seen.add(key)
        lengths_list.append(length)
        token_rows.append(row)

    lengths = np.asarray(lengths_list, dtype=np.int64)
    tokens = np.zeros((n, max_length), dtype=np.int64)
    for i, (length, row) in enumerate(zip(lengths_list, token_rows)):
        tokens[i, :length] = row
    labels = np.asarray(
        [int(np.bitwise_xor.reduce(row)) for row in token_rows],
        dtype=np.int64,
    )
    return DatasetSplit(tokens, labels, lengths, seed, min_length, max_length)


def select_rows(split: DatasetSplit, indices: np.ndarray, *, seed: int) -> DatasetSplit:
    """Return a new split containing ``indices`` or ``split`` (order preserved)."""
    idx = np.asarray(indices, dtype=np.int64)
    return DatasetSplit(
        tokens=np.ascontiguousarray(split.tokens[idx]),
        labels=np.ascontiguousarray(split.labels[idx]),
        lengths=np.ascontiguousarray(split.lengths[idx]),
        seed=seed,
        min_length=split.min_length,
        max_length=split.max_length,
    )


def make_splits(
    *,
    train_n: int,
    dev_n: int,
    heldout_n: int,
    seed: int,
    train_lengths: Tuple[int, int] = (4, 8),
    heldout_lengths: Tuple[int, int] = (12, 16),
) -> Dict[str, DatasetSplit]:
    """Build train/dev/held-out splits that are pairwise example-disjoint.

    Train and dev are drawn as **one joint pool** of unique short-length examples
    with a single PRNG seed, then randomly partitioned (``seed + 404``). Drawing
    train first and then rejecting train examples from dev would let train
    exhaust the tiny short-length example spaces entirely (e.g. only ``2**4 = 16``
    length-4 sequences exist), leaving dev without any length-4 examples; the
    joint pool keeps the length distribution usable for both splits. Held-out is
    drawn from disjoint lengths and additionally excludes anything in train/dev.
    """
    pool = make_split(
        n=train_n + dev_n,
        min_length=train_lengths[0],
        max_length=train_lengths[1],
        seed=seed + 101,
    )
    rng = np.random.default_rng(seed + 404)
    order = rng.permutation(train_n + dev_n)
    train = select_rows(pool, order[:train_n], seed=seed + 101)
    dev = select_rows(pool, order[train_n:], seed=seed + 202)
    heldout = make_split(
        n=heldout_n,
        min_length=heldout_lengths[0],
        max_length=heldout_lengths[1],
        seed=seed + 303,
        exclude_keys=example_keys(train) | example_keys(dev),
    )
    return {"train": train, "dev": dev, "heldout": heldout}


def split_manifest(splits: Dict[str, DatasetSplit]) -> str:
    """Stable JSON manifest used in result files and reproducibility checks."""
    return json.dumps({name: split.to_dict() for name, split in splits.items()}, sort_keys=True)
