"""CLI runner for the parity experiment (NumPy smoke or full PyTorch)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import DatasetSplit, make_split, split_manifest
from .model import (
    NumpyConfig,
    NumpyLoopedTransformerClassifier,
    NumpyTransformerClassifier,
    sigmoid,
)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((logits >= 0).astype(np.int64) == labels))


def evaluate_numpy(model, split: DatasetSplit, features: np.ndarray | None = None) -> Dict[str, object]:
    x = model.features(split.tokens, split.lengths) if features is None else features
    logits = model.logits(x)
    probs = sigmoid(logits)
    eps = 1e-8
    out: Dict[str, object] = {
        "accuracy": accuracy(logits, split.labels),
        "bce": float(-np.mean(split.labels * np.log(probs + eps) + (1 - split.labels) * np.log(1 - probs + eps))),
        "n": int(len(split.labels)),
    }
    by_length = {}
    for length in sorted(set(split.lengths.tolist())):
        mask = split.lengths == length
        by_length[str(length)] = accuracy(logits[mask], split.labels[mask])
    out["accuracy_by_length"] = by_length
    return out


def fit_numpy(model, train: DatasetSplit, *, epochs: int, learning_rate: float, l2: float) -> Dict[str, object]:
    features = model.features(train.tokens, train.lengths)
    y = train.labels.astype(np.float64)
    losses: List[float] = []
    for _ in range(epochs):
        logits = model.logits(features)
        p = sigmoid(logits)
        losses.append(float(-np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))))
        grad = p - y
        model.readout -= learning_rate * (features.T @ grad / len(y) + l2 * model.readout)
        model.bias -= learning_rate * float(np.mean(grad))
    return {"epochs": epochs, "final_train_bce": losses[-1], "features": features}


def _torch_components():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch backend requested but torch is unavailable. Install with "
            '`pip install -e ".[torch]"` (or use `--backend numpy`).'
        ) from exc
    return torch, nn


def run_torch(model_kind: str, loop_count: int, splits: Dict[str, DatasetSplit], cfg: NumpyConfig,
              *, epochs: int, learning_rate: float, seed: int) -> Dict[str, object]:
    torch, nn = _torch_components()
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(cfg.d_model)
            self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

        def forward(self, x, padding_mask):
            h = self.norm1(x)
            a, _ = self.attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)
            x = x + a
            return x + self.ff(self.norm2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(2, cfg.d_model)
            self.block = Block()
            self.readout = nn.Linear(cfg.d_model, 1)

        def forward(self, tokens, lengths):
            t = tokens.shape[1]
            pos = torch.arange(t, device=tokens.device).float()[:, None]
            dim = torch.arange(cfg.d_model, device=tokens.device).float()[None, :]
            pe = torch.sin(pos / (10000 ** (2 * torch.floor(dim / 2) / cfg.d_model)))
            x = self.embedding(tokens) + 0.05 * pe[None, :, :]
            mask = torch.arange(t, device=tokens.device)[None, :] >= lengths[:, None]
            n = 1 if model_kind == "baseline" else loop_count
            for _ in range(n):
                x = self.block(x, mask)
            return self.readout(x[torch.arange(len(tokens), device=tokens.device), lengths - 1]).squeeze(-1)

    model = Model()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    tx = torch.from_numpy(splits["train"].tokens)
    tl = torch.from_numpy(splits["train"].lengths)
    ty = torch.from_numpy(splits["train"].labels).float()
    start = time.perf_counter()
    model.train()
    losses = []
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(tx, tl), ty)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    model.eval()
    result = {"epochs": epochs, "final_train_bce": losses[-1], "seconds": time.perf_counter() - start}
    with torch.no_grad():
        for name, split in splits.items():
            logits = model(torch.from_numpy(split.tokens), torch.from_numpy(split.lengths)).numpy()
            result[name] = {
                "accuracy": accuracy(logits, split.labels),
                "bce": float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(split.labels).float())),
                "n": int(len(split.labels)),
            }
    result["architecture_parameters"] = sum(p.numel() for p in model.parameters())
    result["trainable_parameters"] = result["architecture_parameters"]
    return result


def run_numpy(model_kind: str, loop_count: int, splits: Dict[str, DatasetSplit], cfg: NumpyConfig,
              *, epochs: int, learning_rate: float, l2: float, seed: int = 0) -> Dict[str, object]:
    model = (NumpyTransformerClassifier(cfg) if model_kind == "baseline"
             else NumpyLoopedTransformerClassifier(cfg, loop_count))
    start = time.perf_counter()
    fit = fit_numpy(model, splits["train"], epochs=epochs, learning_rate=learning_rate, l2=l2)
    result = {
        "epochs": epochs,
        "final_train_bce": fit["final_train_bce"],
        "seconds": time.perf_counter() - start,
        "architecture_parameters": model.architecture_parameters(),
        "trainable_parameters": cfg.d_model + 1,
        "backend_note": "NumPy fallback freezes transformer weights and trains only the binary readout; use --backend torch for the full trainable experiment.",
    }
    for name, split in splits.items():
        result[name] = evaluate_numpy(model, split)
    return result


def write_report(path: Path, payload: Dict[str, object]) -> None:
    rows = payload["results"]
    lines = ["# Smoke experiment report", "", f"Backend: **{payload['backend']}**", "", "Accuracy (higher is better):", "", "| model | loops | train | dev | held-out length |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        r = row["metrics"]
        lines.append(f"| {row['model']} | {row['loop_count']} | {r['train']['accuracy']:.3f} | {r['dev']['accuracy']:.3f} | {r['heldout']['accuracy']:.3f} |")
    lines += ["", "The held-out split uses lengths outside the train/dev range. This is an illustrative smoke run, not evidence that recurrence improves reasoning.", "", "See DESIGN.md for hypotheses, controls, and limitations."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--dev-size", type=int, default=128)
    ap.add_argument("--heldout-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    backend = args.backend
    if backend == "auto":
        try:
            _torch_components()
            backend = "torch"
        except RuntimeError:
            backend = "numpy"
    epochs = args.epochs if args.epochs is not None else (40 if backend == "numpy" else 80)
    splits = {
        "train": make_split(n=args.train_size, min_length=4, max_length=8, seed=args.seed + 101),
        "dev": make_split(n=args.dev_size, min_length=4, max_length=8, seed=args.seed + 202),
        "heldout": make_split(n=args.heldout_size, min_length=12, max_length=16, seed=args.seed + 303),
    }
    cfg = NumpyConfig(seed=args.seed)
    results = []
    for model, loops in (("baseline", 1), ("looped", 2), ("looped", 4)):
        metrics = (run_numpy if backend == "numpy" else run_torch)(
            model, loops, splits, cfg, epochs=epochs,
            learning_rate=0.08 if backend == "numpy" else 3e-3,
            **({"l2": 1e-4} if backend == "numpy" else {}), seed=args.seed,
        )
        results.append({"model": model, "loop_count": loops, "metrics": metrics})
    payload = {
        "task": "binary sequence parity",
        "backend": backend,
        "seed": args.seed,
        "config": vars(cfg),
        "split_manifest": json.loads(split_manifest(splits)),
        "results": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = args.report or args.output.with_suffix(".md")
    write_report(report, payload)
    print(json.dumps({"output": str(args.output), "report": str(report), "backend": backend, "results": [{"model": x["model"], "loops": x["loop_count"], "heldout_accuracy": x["metrics"]["heldout"]["accuracy"]} for x in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
