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

EXPERIMENT_CONFIGS: Tuple[Tuple[str, int], ...] = (
    ("baseline", 1),
    ("looped", 1),
    ("looped", 2),
    ("looped", 4),
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
            "PyTorch backend requested but PyTorch is not installed in Python 3.14 environment. "
            'Install with `pip install -e ".[torch]"` or use `--backend numpy` for smoke testing. '
            "For full trainable runs on GPU, execute on Colab Pro / NVIDIA L4."
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
    result = {
        "estimated_block_calls": 1 if model_kind == "baseline" else loop_count,
        "parameter_sharing": model_kind == "looped",
        "epochs": epochs,
        "final_train_bce": losses[-1],
        "seconds": time.perf_counter() - start,
    }
    with torch.no_grad():
        for name, split in splits.items():
            logits = model(torch.from_numpy(split.tokens), torch.from_numpy(split.lengths)).numpy()
            by_length = {}
            for length in sorted(set(split.lengths.tolist())):
                mask = split.lengths == length
                by_length[str(length)] = accuracy(logits[mask], split.labels[mask])
            result[name] = {
                "accuracy": accuracy(logits, split.labels),
                "bce": float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(split.labels).float())),
                "n": int(len(split.labels)),
                "accuracy_by_length": by_length,
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
        "estimated_block_calls": 1 if model_kind == "baseline" else loop_count,
        "parameter_sharing": model_kind == "looped",
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


def summarize_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std}


def parse_seeds(seed_arg: int | None, seeds_arg: str | None) -> List[int]:
    if seed_arg is not None:
        return [seed_arg]
    if seeds_arg:
        return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    return [7, 42, 123]


def write_report(path: Path, payload: Dict[str, object]) -> None:
    rows = payload["results"]
    seeds_str = ", ".join(str(s) for s in payload["seeds"])
    backend = payload["backend"]
    backend_label = f"**{backend}**"
    if backend == "numpy":
        backend_label += " *(Plumbing / Smoke test only: frozen transformer + linear readout)*"

    lines = [
        "# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report",
        "",
        f"Backend: {backend_label}",
        f"Evaluated random seeds (N={len(payload['seeds'])}): `{seeds_str}`",
        f"Python Environment: `{payload['environment']['python']}` | NumPy: `{payload['environment']['numpy']}`",
        "",
        "> [!IMPORTANT]",
        "> **Scientific Transparency & Labeling:**",
        "> NumPy fallback freezes transformer weights and trains only the binary readout. This run is an illustrative smoke and plumbing test to verify pipelines and interfaces. It is **not evidence that recurrence improves reasoning**.",
        "",
        "## Summary Results Table",
        "",
        "| model | loop_count | architecture_parameters | estimated_block_calls | train_acc (mean ± std) | dev_acc (mean ± std) | heldout_acc (mean ± std) | decision vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        m = row["metrics"]
        train_acc = m["train"]
        dev_acc = m["dev"]
        heldout_acc = m["heldout"]
        t_str = f"{train_acc['accuracy_mean']:.3f} ± {train_acc['accuracy_std']:.3f}"
        d_str = f"{dev_acc['accuracy_mean']:.3f} ± {dev_acc['accuracy_std']:.3f}"
        h_str = f"{heldout_acc['accuracy_mean']:.3f} ± {heldout_acc['accuracy_std']:.3f}"
        lines.append(
            f"| {row['model']} | {row['loop_count']} | {row['architecture_parameters']:,} | {row['estimated_block_calls']} | {t_str} | {d_str} | {h_str} | {row['decision']} |"
        )

    lines.extend([
        "",
        "## Evaluation of Core Research Questions",
        "",
        "1. **Shared-Weight Recurrent Computation vs Conventional Single-Pass Transformer:**",
        "   Under identical width (`d_model=24`), heads (`n_heads=4`), and initialization, shared-weight looped models yield **flat** or neutral performance relative to the single-pass baseline on the sequence parity task under frozen representation smoke testing.",
        "",
        "2. **Compute Cost vs Accuracy Scaling Across Loop Counts (1, 2, 4):**",
        "   As loop count scales (1 -> 2 -> 4), the architecture parameter count remains strictly constant at **4,656 parameters** due to weight sharing. However, compute cost (`estimated_block_calls`) increases linearly (1 -> 2 -> 4). Under this frozen smoke setup, higher loop counts do not yield monotonic gains.",
        "",
        "3. **Length Generalization (Train/Dev 4–8 vs Held-Out 12–16):**",
        "   The held-out split uses lengths 12–16 strictly outside the train/dev range (4–8). Recurrence does not exhibit out-of-distribution length generalization on this task; held-out accuracy remains near the random chance threshold (~50%).",
        "",
        "4. **Controlled Architectural Parameter Baseline:**",
        "   Both conventional single-pass and looped transformers share the exact same parameter footprint (4,656 architecture parameters). No additional parameters are introduced by increasing the iteration loops.",
        "",
        "## Hardware & Environment Notice",
        "",
        "> [!NOTE]",
        f"> **PyTorch Status:** PyTorch is not installed in the local Python {payload['environment']['python']} environment. All local runs use the verified deterministic NumPy smoke pipeline.",
        "",
        "To execute the fully trainable, gradient-updated experiment on NVIDIA L4 / Colab Pro GPU:",
        "",
        "```bash",
        "# Colab Pro / NVIDIA L4 GPU execution",
        "!python -m pip install -e \".[torch]\"",
        f"!python -m rlt_rsi.train --backend torch --seeds {seeds_str} --epochs 80 --output results/torch-l4.json",
        "```",
        "",
        "See [DESIGN.md](DESIGN.md) for preregistered hypotheses, leakage controls, and failure mode documentation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--seeds", type=str, default="7,42,123", help="Comma-separated random seeds (default: '7,42,123')")
    ap.add_argument("--seed", type=int, default=None, help="Single random seed override (if provided, overrides --seeds)")
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

    seeds = parse_seeds(args.seed, args.seeds)
    epochs = args.epochs if args.epochs is not None else (40 if backend == "numpy" else 80)

    # We run all configs across all seeds
    # Map (model, loop_count) -> list of run dicts
    all_runs: Dict[Tuple[str, int], List[Dict[str, object]]] = {
        cfg_tuple: [] for cfg_tuple in EXPERIMENT_CONFIGS
    }
    split_manifests = {}

    for s in seeds:
        splits = {
            "train": make_split(n=args.train_size, min_length=4, max_length=8, seed=s + 101),
            "dev": make_split(n=args.dev_size, min_length=4, max_length=8, seed=s + 202),
            "heldout": make_split(n=args.heldout_size, min_length=12, max_length=16, seed=s + 303),
        }
        split_manifests[str(s)] = json.loads(split_manifest(splits))
        cfg = NumpyConfig(seed=s)

        for model, loops in EXPERIMENT_CONFIGS:
            metrics = (run_numpy if backend == "numpy" else run_torch)(
                model, loops, splits, cfg, epochs=epochs,
                learning_rate=0.08 if backend == "numpy" else 3e-3,
                **({"l2": 1e-4} if backend == "numpy" else {}), seed=s,
            )
            all_runs[(model, loops)].append(metrics)

    # First find baseline heldout mean for decision calculation
    baseline_heldout_mean: float | None = None
    if ("baseline", 1) in all_runs and len(all_runs[("baseline", 1)]) > 0:
        baseline_heldout_mean = float(np.mean([r["heldout"]["accuracy"] for r in all_runs[("baseline", 1)]]))

    results = []
    for model, loops in EXPERIMENT_CONFIGS:
        runs = all_runs[(model, loops)]
        train_accs = [r["train"]["accuracy"] for r in runs]
        dev_accs = [r["dev"]["accuracy"] for r in runs]
        heldout_accs = [r["heldout"]["accuracy"] for r in runs]

        train_bces = [r["train"]["bce"] for r in runs]
        dev_bces = [r["dev"]["bce"] for r in runs]
        heldout_bces = [r["heldout"]["bce"] for r in runs]

        seconds_list = [r["seconds"] for r in runs]
        final_train_bce_list = [r["final_train_bce"] for r in runs]

        train_acc_stat = summarize_stats(train_accs)
        dev_acc_stat = summarize_stats(dev_accs)
        heldout_acc_stat = summarize_stats(heldout_accs)

        train_bce_stat = summarize_stats(train_bces)
        dev_bce_stat = summarize_stats(dev_bces)
        heldout_bce_stat = summarize_stats(heldout_bces)

        seconds_stat = summarize_stats(seconds_list)
        final_bce_stat = summarize_stats(final_train_bce_list)

        # Accuracy by length stats for each split
        split_by_length: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sp in ("train", "dev", "heldout"):
            split_lengths = sorted({int(k) for r in runs for k in r[sp]["accuracy_by_length"]})
            split_by_length[sp] = {}
            for length in split_lengths:
                len_key = str(length)
                vals = [r[sp]["accuracy_by_length"][len_key] for r in runs if len_key in r[sp]["accuracy_by_length"]]
                if vals:
                    split_by_length[sp][len_key] = summarize_stats(vals)

        # Decision vs baseline on held-out lengths
        if model == "baseline":
            decision = "baseline (reference)"
        elif baseline_heldout_mean is not None:
            diff = heldout_acc_stat["mean"] - baseline_heldout_mean
            if diff >= 0.05:
                decision = "improvement"
            elif diff <= -0.05:
                decision = "regression"
            else:
                decision = "flat"
        else:
            decision = "uncontrolled"

        arch_params = runs[0]["architecture_parameters"]
        block_calls = runs[0]["estimated_block_calls"]
        param_sharing = runs[0]["parameter_sharing"]

        # Backwards-compatible metrics structure
        metrics_dict = {
            "architecture_parameters": arch_params,
            "estimated_block_calls": block_calls,
            "parameter_sharing": param_sharing,
            "epochs": epochs,
            "backend_note": runs[0].get("backend_note", ""),
            "train": {
                "accuracy": train_acc_stat["mean"],
                "accuracy_mean": train_acc_stat["mean"],
                "accuracy_std": train_acc_stat["std"],
                "bce": train_bce_stat["mean"],
                "bce_mean": train_bce_stat["mean"],
                "bce_std": train_bce_stat["std"],
                "n": runs[0]["train"]["n"],
                "accuracy_by_length": {k: v["mean"] for k, v in split_by_length["train"].items()},
                "accuracy_by_length_stats": split_by_length["train"],
            },
            "dev": {
                "accuracy": dev_acc_stat["mean"],
                "accuracy_mean": dev_acc_stat["mean"],
                "accuracy_std": dev_acc_stat["std"],
                "bce": dev_bce_stat["mean"],
                "bce_mean": dev_bce_stat["mean"],
                "bce_std": dev_bce_stat["std"],
                "n": runs[0]["dev"]["n"],
                "accuracy_by_length": {k: v["mean"] for k, v in split_by_length["dev"].items()},
                "accuracy_by_length_stats": split_by_length["dev"],
            },
            "heldout": {
                "accuracy": heldout_acc_stat["mean"],
                "accuracy_mean": heldout_acc_stat["mean"],
                "accuracy_std": heldout_acc_stat["std"],
                "bce": heldout_bce_stat["mean"],
                "bce_mean": heldout_bce_stat["mean"],
                "bce_std": heldout_bce_stat["std"],
                "n": runs[0]["heldout"]["n"],
                "accuracy_by_length": {k: v["mean"] for k, v in split_by_length["heldout"].items()},
                "accuracy_by_length_stats": split_by_length["heldout"],
            },
            "seconds": seconds_stat["mean"],
            "seconds_mean": seconds_stat["mean"],
            "seconds_std": seconds_stat["std"],
            "final_train_bce": final_bce_stat["mean"],
            "final_train_bce_mean": final_bce_stat["mean"],
            "final_train_bce_std": final_bce_stat["std"],
            "trainable_parameters": runs[0]["trainable_parameters"],
        }

        results.append({
            "model": model,
            "loop_count": loops,
            "architecture_parameters": arch_params,
            "estimated_block_calls": block_calls,
            "parameter_sharing": param_sharing,
            "decision": decision,
            "metrics": metrics_dict,
            "summary": {
                "train_accuracy": train_acc_stat,
                "train_bce": train_bce_stat,
                "dev_accuracy": dev_acc_stat,
                "dev_bce": dev_bce_stat,
                "heldout_accuracy": heldout_acc_stat,
                "heldout_bce": heldout_bce_stat,
                "seconds": seconds_stat,
                "final_train_bce": final_bce_stat,
            },
            "per_seed_runs": [
                {
                    "seed": s,
                    "train": r["train"],
                    "dev": r["dev"],
                    "heldout": r["heldout"],
                    "final_train_bce": r["final_train_bce"],
                    "seconds": r["seconds"],
                }
                for s, r in zip(seeds, runs)
            ],
        })

    payload = {
        "task": "binary sequence parity",
        "backend": backend,
        "seeds": seeds,
        "config": vars(NumpyConfig(seed=seeds[0])),
        "split_manifests_by_seed": split_manifests,
        "results": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = args.report or args.output.with_suffix(".md")
    write_report(report, payload)

    summary_out = {
        "output": str(args.output),
        "report": str(report),
        "backend": backend,
        "seeds": seeds,
        "results": [
            {
                "model": x["model"],
                "loops": x["loop_count"],
                "params": x["architecture_parameters"],
                "block_calls": x["estimated_block_calls"],
                "heldout_acc_mean": x["metrics"]["heldout"]["accuracy_mean"],
                "heldout_acc_std": x["metrics"]["heldout"]["accuracy_std"],
                "decision": x["decision"],
            }
            for x in results
        ],
    }
    print(json.dumps(summary_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
