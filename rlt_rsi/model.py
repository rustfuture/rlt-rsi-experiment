"""Tiny transformer models.

The NumPy implementation freezes the transformer and trains only a binary
readout, so it is a real, dependency-light smoke path—not a proxy claim for
the full PyTorch experiment. With torch installed, ``Torch*`` trains all
parameters and is the intended research run.
"""

from dataclasses import dataclass
import math
from typing import Optional

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
        return 1

    def architecture_parameters(self) -> int:
        c = self.cfg
        return int(2 * c.d_model + 4 * c.d_model * c.d_model + 2 * c.d_model * c.d_ff)


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
