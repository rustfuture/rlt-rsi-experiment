"""Tiny transformer models.

The NumPy implementation freezes the transformer and trains only a binary
readout, so it is a real, dependency-light smoke path—not a proxy claim for
the full PyTorch experiment. With torch installed, ``Torch*`` trains all
parameters and is the intended research run.

The two backends are **not** the same architecture and their parameter counts
differ by construction (never claim they are equal):

* NumPy: manual scaled-dot-product attention, RMS-style normalization, frozen
  backbone, and a trainable binary readout + bias (``d_model + 1`` trainable
  parameters, everything else frozen).
* PyTorch: ``nn.MultiheadAttention`` + ``nn.LayerNorm`` and full end-to-end
  training of every parameter.

Only the *width/head/feed-forward shape* (``d_model``, ``n_heads``, ``d_ff``)
and the loop-count ablation are shared between backends.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass
class NumpyConfig:
    d_model: int = 24
    n_heads: int = 4
    d_ff: int = 48
    seed: int = 7


class NumpyTransformerBlock:
    """Pre-LN self-attention + MLP block with frozen, seeded weights."""

    def __init__(self, cfg: NumpyConfig, rng: np.random.Generator):
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        scale = 1.0 / math.sqrt(cfg.d_model)
        self.d_model, self.n_heads = cfg.d_model, cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.wq = rng.normal(0, scale, (cfg.d_model, cfg.d_model))
        self.wk = rng.normal(0, scale, (cfg.d_model, cfg.d_model))
        self.wv = rng.normal(0, scale, (cfg.d_model, cfg.d_model))
        self.wo = rng.normal(0, scale, (cfg.d_model, cfg.d_model))
        self.w1 = rng.normal(0, scale, (cfg.d_model, cfg.d_ff))
        self.w2 = rng.normal(0, scale, (cfg.d_ff, cfg.d_model))

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # x: [batch, time, d_model]
        b, t, d = x.shape
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = np.where(mask[:, None, None, :], scores, -1e9)
        attn = _softmax(scores, axis=-1)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, d) @ self.wo
        x = x + y
        # RMS-style normalization keeps this tiny implementation numerically stable.
        norm = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        h = x / norm
        x = x + np.maximum(h @ self.w1, 0) @ self.w2
        return x


class NumpyTransformerClassifier:
    """Conventional one-pass transformer with a trainable binary readout."""

    def __init__(self, cfg: NumpyConfig):
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.embedding = rng.normal(0, 0.15, (2, cfg.d_model))
        self.block = NumpyTransformerBlock(cfg, rng)
        self.readout = np.zeros(cfg.d_model, dtype=np.float64)
        self.bias = 0.0

    def features(self, tokens: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        _, t = tokens.shape
        pos = np.arange(t)[:, None]
        dims = np.arange(self.cfg.d_model)[None, :]
        pe = np.sin(pos / np.power(10000.0, (2 * (dims // 2)) / self.cfg.d_model))
        x = self.embedding[tokens] + 0.05 * pe[None, :, :]
        valid = np.arange(t)[None, :] < lengths[:, None]
        x = self.block(x, valid)
        return x[np.arange(len(tokens)), np.maximum(lengths - 1, 0)]

    def logits(self, features: np.ndarray) -> np.ndarray:
        return features @ self.readout + self.bias

    @property
    def loop_count(self) -> int:
        return 1

    @property
    def estimated_block_calls(self) -> int:
        """Sequential applications of the shared block (a structural counter).

        This is **not** a FLOP count and **not** a latency measurement; wall-clock
        time is reported separately as ``seconds``. Name retained for backward
        compatibility with earlier result files.
        """
        return 1

    def architecture_parameters(self) -> int:
        """Frozen backbone subtotal: embedding + transformer block weights only.

        This deliberately excludes the trainable readout (``d_model``) and bias
        (``1``), so it must never be reported as the total parameter count. Use
        :meth:`total_parameters`, :meth:`trainable_parameters`, or
        :meth:`frozen_parameters` for honest accounting. Kept for backward
        compatibility with earlier result files that used this scope.
        """
        backbone = int(self.embedding.size)
        backbone += sum(
            int(w.size)
            for w in (self.block.wq, self.block.wk, self.block.wv, self.block.wo, self.block.w1, self.block.w2)
        )
        return backbone

    def trainable_parameters(self) -> int:
        """Parameters updated by the NumPy trainer: readout vector + scalar bias."""
        return int(self.readout.size + 1)

    def frozen_parameters(self) -> int:
        """Parameters created but never updated by the NumPy trainer."""
        return self.architecture_parameters()

    def total_parameters(self) -> int:
        """All created parameters, including the trainable readout and bias."""
        return self.frozen_parameters() + self.trainable_parameters()

    def parameter_counts(self) -> Dict[str, int]:
        total = self.total_parameters()
        trainable = self.trainable_parameters()
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class NumpyLoopedTransformerClassifier(NumpyTransformerClassifier):
    """Shared-weight block applied ``loop_count`` times."""

    def __init__(self, cfg: NumpyConfig, loop_count: int):
        super().__init__(cfg)
        if loop_count < 1:
            raise ValueError("loop_count must be >= 1")
        self._loop_count = loop_count

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def estimated_block_calls(self) -> int:
        """Sequential shared-block applications; a structural counter, not FLOPs or latency."""
        return self._loop_count

    def features(self, tokens: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        _, t = tokens.shape
        pos = np.arange(t)[:, None]
        dims = np.arange(self.cfg.d_model)[None, :]
        pe = np.sin(pos / np.power(10000.0, (2 * (dims // 2)) / self.cfg.d_model))
        x = self.embedding[tokens] + 0.05 * pe[None, :, :]
        valid = np.arange(t)[None, :] < lengths[:, None]
        for _ in range(self.loop_count):
            x = self.block(x, valid)
        return x[np.arange(len(tokens)), np.maximum(lengths - 1, 0)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
