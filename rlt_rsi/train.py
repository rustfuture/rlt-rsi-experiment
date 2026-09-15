"""CLI runner for the parity experiment (NumPy smoke or full PyTorch).

Honesty notes that the report and payload repeat verbatim:

* ``estimated_block_calls`` (name kept for backward compatibility) counts
  sequential applications of the shared transformer block. It is a *structural
  counter*: not a FLOP count and not a latency measurement. Measured wall-clock
  time is reported separately as ``seconds``.
* Equal ``epochs`` across configurations is **not** an equal-compute budget: a
  looped model with ``loop_count=k`` performs ``k`` block applications per
  optimisation step.
* ``requires_grad``/trainable accounting is derived from the real parameters of
  the object that ran; the NumPy and PyTorch architectures are different and
  their parameter counts are not comparable across backends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .data import (
    DatasetSplit,
    make_splits,
    overlap_report,
    split_manifest,
)
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

#: Operational decision rule threshold (percentage points as a fraction). This is
#: NOT a statistical significance level and NOT an equivalence bound.
DECISION_THRESHOLD = 0.05

PAIRED_METRICS = (
    "heldout_accuracy",
    "heldout_bce",
    "in_distribution_accuracy",
    "length_generalization_drop",
    "train_accuracy",
)

BLOCK_CALL_NOTE = (
    "`estimated_block_calls` counts sequential applications of the shared transformer "
    "block. It is a structural counter, not a FLOP count and not a latency measurement; "
    "see `seconds` for measured wall-clock time."
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


def _numpy_parameter_fingerprint(model) -> str:
    """SHA-256 over the created NumPy parameters (used to prove identical init)."""
    h = hashlib.sha256()
    named = [("embedding", model.embedding)]
    for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
        named.append((f"block.{name}", getattr(model.block, name)))
    named.append(("readout", np.asarray(model.readout, dtype=np.float64)))
    named.append(("bias", np.asarray(model.bias, dtype=np.float64)))
    for name, value in named:
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return h.hexdigest()


def _torch_components():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch backend requested but PyTorch is not importable in this Python "
            f"{platform.python_version()} environment ({exc}). Install it into the active "
            'virtualenv with `pip install -e ".[torch]"` (or `pip install torch>=2.2`) and '
            "re-run, or use `--backend numpy` for the frozen-feature smoke path."
        ) from exc
    return torch, nn


def _import_torch_version() -> Optional[str]:
    """Version string if torch is importable, else ``None`` (never claims absence)."""
    try:
        import torch
    except ImportError:
        return None
    return str(torch.__version__)


def resolve_device(requested: str):
    """Resolve ``auto|cpu|mps|cuda`` to a ``torch.device``.

    ``cuda``/``mps`` raise ``RuntimeError`` when unavailable; there is no silent
    fallback. ``auto`` prefers cuda, then mps, then cpu.
    """
    torch, _ = _torch_components()
    requested = str(requested).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is False. "
                "No silent fallback is performed; use --device auto, --device mps, or --device cpu."
            )
        return torch.device("cuda")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "--device mps requested but torch.backends.mps.is_available() is False. "
                "No silent fallback is performed; use --device auto or --device cpu."
            )
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device request {requested!r}; expected auto|cpu|mps|cuda")


def _build_torch_model(torch, nn, cfg: NumpyConfig, model_kind: str, loop_count: int):
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

    return Model()


def build_torch_model(cfg: NumpyConfig, model_kind: str, loop_count: int, *, seed: Optional[int] = None):
    """Build the torch model, seeding first when ``seed`` is given.

    Seeding before construction is what makes the baseline and every looped
    configuration for a seed start from identical shared-module parameters.
    """
    torch, nn = _torch_components()
    if seed is not None:
        torch.manual_seed(seed)
    return _build_torch_model(torch, nn, cfg, model_kind, loop_count)


def _torch_parameter_counts(model) -> Dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _torch_state_fingerprint(torch, model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        # Move to CPU *before* upcasting: MPS does not support float64, and some torch
        # builds silently return zeros for ``mps_tensor.to("cpu", torch.float64)``.
        h.update(tensor.detach().to("cpu").to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def _torch_batch(torch, split: DatasetSplit, device):
    return (
        torch.from_numpy(split.tokens).to(device),
        torch.from_numpy(split.lengths).to(device),
        torch.from_numpy(split.labels).float().to(device),
    )


def run_torch(
    model_kind: str,
    loop_count: int,
    splits: Dict[str, DatasetSplit],
    cfg: NumpyConfig,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device=None,
    device_requested: str = "auto",
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, object]:
    torch, nn = _torch_components()
    device = torch.device(device) if device is not None else resolve_device(device_requested)
    model = build_torch_model(cfg, model_kind, loop_count, seed=seed).to(device)
    counts = _torch_parameter_counts(model)
    init_fingerprint = _torch_state_fingerprint(torch, model)
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tx, tl, ty = _torch_batch(torch, splits["train"], device)
    start = time.perf_counter()
    model.train()
    losses = []
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(tx, tl), ty)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start

    result: Dict[str, object] = {
        "estimated_block_calls": 1 if model_kind == "baseline" else loop_count,
        "sequential_block_applications": 1 if model_kind == "baseline" else loop_count,
        "parameter_sharing": model_kind == "looped",
        "epochs": epochs,
        "seed": seed,
        "final_train_bce": losses[-1],
        "seconds": elapsed,
        "device": str(device),
        "device_requested": device_requested,
        "dtype": dtype,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "parameter_counts": counts,
        "initialization_fingerprint": init_fingerprint,
        "torch_manifest": {
            "total_parameters": counts["total"],
            "trainable_parameters": counts["trainable"],
            "frozen_parameters": counts["frozen"],
        },
    }
    with torch.no_grad():
        for name, split in splits.items():
            tokens, lengths, labels = _torch_batch(torch, split, device)
            logits = model(tokens, lengths).detach().cpu().numpy()
            labels_np = split.labels
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            eps = 1e-8
            by_length = {}
            for length in sorted(set(split.lengths.tolist())):
                mask = split.lengths == length
                by_length[str(length)] = accuracy(logits[mask], labels_np[mask])
            result[name] = {
                "accuracy": accuracy(logits, labels_np),
                "bce": float(-np.mean(labels_np * np.log(probs + eps) + (1 - labels_np) * np.log(1 - probs + eps))),
                "n": int(len(labels_np)),
                "accuracy_by_length": by_length,
            }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": {
                    "seed": seed,
                    "model": model_kind,
                    "loop_count": loop_count,
                    "cfg": vars(cfg),
                    "device": str(device),
                    "dtype": dtype,
                    "epochs": epochs,
                    "optimizer": "AdamW",
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "torch_version": str(torch.__version__),
                },
            },
            checkpoint_path,
        )
        result["checkpoint_path"] = str(checkpoint_path)
    return result


def run_numpy(
    model_kind: str,
    loop_count: int,
    splits: Dict[str, DatasetSplit],
    cfg: NumpyConfig,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int = 0,
) -> Dict[str, object]:
    model = (NumpyTransformerClassifier(cfg) if model_kind == "baseline"
             else NumpyLoopedTransformerClassifier(cfg, loop_count))
    counts = model.parameter_counts()
    init_fingerprint = _numpy_parameter_fingerprint(model)  # captured before any update
    start = time.perf_counter()
    fit = fit_numpy(model, splits["train"], epochs=epochs, learning_rate=learning_rate, l2=l2)
    result = {
        "estimated_block_calls": 1 if model_kind == "baseline" else loop_count,
        "sequential_block_applications": 1 if model_kind == "baseline" else loop_count,
        "parameter_sharing": model_kind == "looped",
        "epochs": epochs,
        "seed": seed,
        "final_train_bce": fit["final_train_bce"],
        "seconds": time.perf_counter() - start,
        "device": "cpu",
        "device_requested": "cpu",
        "dtype": "float64",
        "optimizer": "numpy full-batch gradient descent on readout+bias only",
        "learning_rate": learning_rate,
        "weight_decay": l2,
        "torch_version": None,
        "python_version": platform.python_version(),
        "parameter_counts": counts,
        "initialization_fingerprint": init_fingerprint,
        "trainable_scope": "readout + bias only; embedding and transformer block are frozen",
        "backend_note": (
            "NumPy smoke backend: transformer block and embeddings are frozen and only the "
            "linear readout + bias are trained. This is a plumbing check, not evidence about "
            "trained recurrence."
        ),
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


def _pooled_accuracy(train_metrics: Dict[str, object], dev_metrics: Dict[str, object]) -> float:
    n = int(train_metrics["n"]) + int(dev_metrics["n"])
    return (
        float(train_metrics["accuracy"]) * int(train_metrics["n"])
        + float(dev_metrics["accuracy"]) * int(dev_metrics["n"])
    ) / n


def _seed_metrics(run: Dict[str, object]) -> Dict[str, float]:
    """Per-seed scalar metrics, including the H2 length-generalization drop."""
    # Dev examples are disjoint from training; fitting accuracy is not a
    # generalization baseline. This statistic remains exploratory: a smaller
    # gap can also result from worse short-sequence accuracy.
    in_distribution = float(run["dev"]["accuracy"])
    return {
        "train_accuracy": float(run["train"]["accuracy"]),
        "dev_accuracy": float(run["dev"]["accuracy"]),
        "heldout_accuracy": float(run["heldout"]["accuracy"]),
        "in_distribution_accuracy": in_distribution,
        "length_generalization_drop": in_distribution - float(run["heldout"]["accuracy"]),
        "heldout_bce": float(run["heldout"]["bce"]),
        "train_bce": float(run["train"]["bce"]),
        "final_train_bce": float(run["final_train_bce"]),
        "seconds": float(run["seconds"]),
    }


def _decision_word(value: float, threshold: float = DECISION_THRESHOLD) -> str:
    if value >= threshold:
        return "improvement"
    if value <= -threshold:
        return "regression"
    return "flat"


def _h2_word(value: float, threshold: float = DECISION_THRESHOLD) -> str:
    if value >= threshold:
        return "supported"
    if value <= -threshold:
        return "not supported (opposite direction beyond the +/-0.05 rule)"
    return "not supported (within the +/-0.05 rule)"


def determinism_notes(backend: str, device: str) -> List[str]:
    notes = [
        "torch.manual_seed set per run before model construction; MPS/CUDA kernels and some "
        "attention paths are not guaranteed bit-exact across runs/devices.",
        "num_workers=0 / single process; training is full-batch over all train examples with no DataLoader.",
        "accuracy metrics are computed from a single training run per seed (no repeated trials per seed).",
        "equal epoch counts across configurations are NOT equal compute budgets: a loop_count=k "
        "configuration performs k sequential block applications per optimisation step.",
        BLOCK_CALL_NOTE,
    ]
    if backend == "numpy":
        notes.append(
            "NumPy backend uses float64 on CPU and trains only the readout+bias via deterministic "
            "full-batch gradient descent; the frozen features make it a plumbing check."
        )
    elif device == "mps":
        notes.append("MPS was used for this run; MPS results are not guaranteed bit-identical to CPU runs.")
    elif device == "cuda":
        notes.append("CUDA was used for this run; CUDA results are not guaranteed bit-identical to CPU runs.")
    return notes


def _git_state(repo_root: Path) -> Dict[str, object]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            proc = subprocess.run(args, cwd=str(repo_root), capture_output=True, text=True, check=False)
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    # Include untracked implementation files too; exclude generated results,
    # virtual environments and bytecode. Hash contents, not only dirty names.
    source_files = sorted({*repo_root.glob('rlt_rsi/*.py'),
                           *repo_root.glob('tests/*.py'),
                           *repo_root.glob('.github/workflows/*.yml'),
                           *repo_root.glob('*.toml')})
    hashes = {str(p.relative_to(repo_root)): _sha256_file(p) for p in source_files}
    tree_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "git_commit": commit,
        "source_files_sha256": hashes,
        "source_tree_sha256": tree_hash,
        "git_status_porcelain": porcelain.splitlines() if porcelain else [],
        "git_dirty": bool(porcelain),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _reload_and_reevaluate(
    checkpoint_path: Path,
    record: Dict[str, object],
    splits: Dict[str, DatasetSplit],
    cfg: NumpyConfig,
    device,
    tolerance: float = 1e-4,
) -> Dict[str, object]:
    """Load a checkpoint, re-run evaluation, and compare with the recorded metrics."""
    torch, nn = _torch_components()
    device = torch.device(device)
    model = build_torch_model(cfg, str(record["model"]), int(record["loop_count"]), seed=int(record["seed"]))
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # older torch without weights_only
        payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    model.eval()
    recorded = record["recorded_metrics"]
    diffs: Dict[str, float] = {}
    with torch.no_grad():
        for name, split in splits.items():
            tokens, lengths, _ = _torch_batch(torch, split, device)
            logits = model(tokens, lengths).detach().cpu().numpy()
            acc = accuracy(logits, split.labels)
            diffs[f"{name}_accuracy"] = abs(acc - float(recorded[name]["accuracy"]))
    max_diff = max(diffs.values()) if diffs else 0.0
    return {
        "checkpoint": checkpoint_path.name,
        "seed": int(record["seed"]),
        "model": record["model"],
        "loop_count": int(record["loop_count"]),
        "matches_recorded": bool(max_diff <= tolerance),
        "max_abs_accuracy_diff": float(max_diff),
        "tolerance": tolerance,
        "per_split_abs_accuracy_diff": diffs,
    }


def write_artifacts(
    artifacts_dir: Path,
    *,
    checkpoint_records: List[Dict[str, object]],
    reload_checks: List[Dict[str, object]],
    payload: Dict[str, object],
    rerun_command: str,
    repo_root: Path,
) -> Path:
    """Write ``checkpoint_manifest.json`` (the only committed artifact record)."""
    sizes = [int(record["bytes"]) for record in checkpoint_records]
    size_note = (
        f"{min(sizes)}-{max(sizes)} bytes per file" if sizes else "no checkpoint files were written"
    )
    manifest = {
        "note": (
            "Checkpoint files under checkpoints/ are gitignored local artifacts "
            f"({size_note}; see `bytes`/`sha256` per file below). Only this JSON manifest is tracked. "
            "The checkpoints are NOT portable: another machine or a different torch build may be unable "
            "to load them, and they must be regenerated with the documented command."
        ),
        "rerun_command": rerun_command,
        "source": _git_state(repo_root),
        "environment": payload.get("environment", {}),
        "training": payload.get("training", {}),
        "data": payload.get("data", {}),
        "paired_summary": [
            {
                "model": row["model"],
                "loop_count": row["loop_count"],
                "decision": row["decision"],
                "parameter_counts": row["parameter_counts"],
                "sequential_block_applications": row["sequential_block_applications"],
                "paired_delta_vs_baseline": {
                    metric: {
                        "mean": values["mean"],
                        "std": values["std"],
                        "stderr": values["stderr"],
                        "per_seed": values["per_seed"],
                        "n_seeds": values["n_seeds"],
                    }
                    for metric, values in row["paired_delta_vs_baseline"].items()
                },
            }
            for row in payload.get("results", [])
        ],
        "files": checkpoint_records,
        "reload_and_reevaluate": reload_checks,
    }
    path = artifacts_dir / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _stat(stat: Dict[str, float]) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _fmt_signed(value: float, digits: int = 4) -> str:
    return f"{0.0 if value == 0 else value:+.{digits}f}"


def _env_torch_sentence(payload: Dict[str, object]) -> str:
    env = payload["environment"]
    backend = payload["backend"]
    device = payload["training"]["device"]
    if env.get("torch_available"):
        if backend == "torch":
            return (
                f"PyTorch {env['torch']} is available in this environment and was used for this run "
                f"on device `{device}`."
            )
        return (
            f"PyTorch {env['torch']} is available in this environment, but this run used the NumPy "
            "backend on CPU; no torch device was used."
        )
    return (
        "PyTorch is not installed in this environment, so no torch device was used and this run "
        "used the NumPy backend."
    )


def write_report(path: Path, payload: Dict[str, object]) -> None:
    """Render the markdown report **entirely from the payload** (no hardcoded claims)."""
    results = payload["results"]
    seeds = payload["seeds"]
    seeds_str = ", ".join(str(s) for s in seeds)
    backend = payload["backend"]
    training = payload["training"]
    env = payload["environment"]
    data = payload["data"]
    threshold = DECISION_THRESHOLD

    scope = payload["scope_statement"]
    lines: List[str] = [
        "# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report",
        "",
        f"- **Backend:** `{backend}`",
        f"- **Seeds (N={len(seeds)}):** `{seeds_str}`",
        f"- **Epochs per configuration:** {training['epochs']}",
        f"- **Optimizer:** {training['optimizer']} | **learning rate:** {training['learning_rate']} "
        f"| **weight decay:** {training['weight_decay']}",
        f"- **Device:** requested `{training['device_requested']}` -> resolved `{training['device']}` "
        f"| **dtype:** `{training['dtype']}`",
        f"- **Python:** `{env['python']}` | **NumPy:** `{env['numpy']}` | **PyTorch:** "
        f"{env['torch'] if env.get('torch_available') else 'not importable in this environment'}",
        f"- **Train/dev size:** {data['train_size']}/{data['dev_size']} (lengths {data['train_lengths'][0]}-{data['train_lengths'][1]}) "
        f"| **Held-out size:** {data['heldout_size']} (lengths {data['heldout_lengths'][0]}-{data['heldout_lengths'][1]})",
        "",
        "> [!IMPORTANT]",
        f"> **Scope of this run.** {scope}",
        "",
        "## 1. Setup actually used",
        "",
        f"- Backend `{backend}`; device requested `{training['device_requested']}`, resolved `{training['device']}`.",
        f"- dtype `{training['dtype']}`; torch version "
        f"`{env['torch'] if env.get('torch_available') else 'not installed'}`; python `{env['python']}`; numpy `{env['numpy']}`.",
        f"- Epochs {training['epochs']}; optimizer {training['optimizer']}; learning rate {training['learning_rate']}; "
        f"weight decay {training['weight_decay']}.",
        f"- Seeds `{seeds_str}`; gradient/training scope: {training['trainable_scope']}.",
        f"- Batch: {training['batch']}.",
        "",
    ]

    # Parameter accounting, derived per configuration from the payload.
    lines.extend([
        "### Parameter accounting (counted from the real model object)",
        "",
        "| configuration | total parameters | trainable | frozen |",
        "|---|---:|---:|---:|",
    ])
    for row in results:
        pc = row["parameter_counts"]
        lines.append(
            f"| {row['model']} (loops={row['loop_count']}) | {pc['total']:,} | {pc['trainable']:,} | {pc['frozen']:,} |"
        )
    lines.append("")
    if backend == "numpy":
        first = results[0]["parameter_counts"]
        d_model = payload["config"]["d_model"]
        lines.append(
            "The NumPy model freezes the embedding and transformer block and trains **only the linear "
            "readout + bias**. For the configurations above: total "
            f"{first['total']:,} = frozen {first['frozen']:,} + trainable {first['trainable']:,} "
            f"(trainable = d_model + 1 with d_model={d_model}). Earlier reports quoted the frozen backbone "
            "subtotal (embedding + block, excluding the readout and bias) as if it were the total; that was "
            "wrong and is corrected here."
        )
    else:
        all_frozen_zero = all(row["parameter_counts"]["frozen"] == 0 for row in results)
        trainable_is_total = all(
            row["parameter_counts"]["trainable"] == row["parameter_counts"]["total"] for row in results
        )
        if all_frozen_zero and trainable_is_total:
            lines.append(
                "The PyTorch run reported here has no frozen parameters (`frozen = 0` and "
                "`trainable = total` for every configuration), so all parameters receive gradients. Counts "
                "are derived from `model.parameters()` and `requires_grad`."
            )
        else:
            lines.append(
                "The PyTorch run reported here has some frozen parameters; see the per-configuration table "
                "above. Counts are derived from `model.parameters()` and `requires_grad`."
            )
    lines.append("")

    lines.extend([
        "The NumPy and PyTorch models are **different architectures** (NumPy: manual attention + RMS-style "
        "normalization + frozen backbone with a trained readout; torch: `nn.MultiheadAttention` + `LayerNorm` "
        "with full training). Their parameter counts are not expected to match and must never be compared "
        "as if they were the same model.",
        "",
        "## 2. Summary results",
        "",
        "| model | loops | total params | trainable | frozen | sequential block applications | train acc | dev acc | in-distribution acc (4-8) | held-out acc (12-16) | seconds (mean) | decision vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in results:
        m = row["metrics"]
        pc = row["parameter_counts"]
        lines.append(
            f"| {row['model']} | {row['loop_count']} | {pc['total']:,} | {pc['trainable']:,} | {pc['frozen']:,} | "
            f"{row['sequential_block_applications']} | {_stat(m['train_accuracy'])} | {_stat(m['dev_accuracy'])} | "
            f"{_stat(m['in_distribution_accuracy'])} | {_stat(m['heldout_accuracy'])} | {m['seconds']['mean']:.4f} | "
            f"{row['decision']} |"
        )
    lines.extend([
        "",
        "`sequential block applications` is a structural counter (see metric notes at the end): it is not "
        "FLOPs and not a latency measurement. `seconds` is measured wall-clock training time for "
        f"{training['epochs']} epochs on this machine.",
        "",
        "## 3. Paired per-seed differences vs the baseline",
        "",
        "Differences are computed **per seed** against the baseline run with the same seed (paired), then "
        "averaged. `n_seeds` is small; the standard error is reported so it is not mistaken for a "
        "significance test.",
        "",
        "| model | loops | metric | per-seed (looped - baseline) | mean | std | stderr |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in results:
        if row["model"] == "baseline":
            continue
        for metric in PAIRED_METRICS:
            pd = row["paired_delta_vs_baseline"][metric]
            per_seed = ", ".join(_fmt_signed(v) for v in pd["per_seed"])
            lines.append(
                f"| {row['model']} | {row['loop_count']} | {metric} | {per_seed} | {pd['mean']:+.4f} | "
                f"{pd['std']:.4f} | {pd['stderr']:.4f} |"
            )
    lines.append("")

    # Decision rule + honesty about what it is.
    lines.extend([
        "### Decision rule (operational, not statistical)",
        "",
        f"The decision rule is: difference >= +{threshold:.2f} -> `improvement`; difference <= -{threshold:.2f} "
        f"-> `regression`; otherwise `flat`. This is an **operational decision rule only**. It is not a "
        "statistical significance test, not a confidence interval, and not an equivalence test; with "
        f"{len(seeds)} seed(s) and held-out n={data['heldout_size']} it cannot establish either superiority or "
        "equivalence.",
        "",
        "## 4. Evaluation of Core Research Questions",
        "",
    ])

    by_config = {(row["model"], row["loop_count"]): row for row in results}
    baseline_row = by_config[("baseline", 1)]

    # RQ1
    looped_rows = [row for row in results if row["model"] != "baseline"]
    outcomes = ", ".join(
        f"looped-{row['loop_count']}: `{row['decision']}` (held-out delta {row['paired_delta_vs_baseline']['heldout_accuracy']['mean']:+.4f}"
        f" ± {row['paired_delta_vs_baseline']['heldout_accuracy']['std']:.4f})"
        for row in looped_rows
    )
    lines.extend([
        "1. **Shared-weight recurrence vs conventional single-pass transformer.** "
        f"Operational decision-rule outcomes on held-out accuracy (baseline mean {_stat(baseline_row['metrics']['heldout_accuracy'])}): "
        f"{outcomes}. Scoped to this run, this task (binary sequence parity), these seeds ({seeds_str}), "
        f"these sample sizes (train {data['train_size']}, dev {data['dev_size']}, held-out {data['heldout_size']}), "
        f"this training budget ({training['epochs']} epochs, {training['optimizer']}), and backend `{backend}` "
        f"on device `{training['device']}`. "
        + (
            "Because this is the frozen-feature NumPy smoke path, it does not measure whether trained "
            "recurrence helps; only the readout+bias are trained."
            if backend == "numpy"
            else "Both baselines and looped configurations were trained end-to-end under this budget."
        ),
        "",
    ])

    # RQ2
    params_totals = ", ".join(f"{value:,}" for value in sorted({row["parameter_counts"]["total"] for row in results}))
    block_scaling = ", ".join(
        f"{row['model']}(loops={row['loop_count']}) -> {row['sequential_block_applications']} block application(s)"
        for row in results
    )
    seconds_by_loops = ", ".join(
        f"{row['model']}(loops={row['loop_count']}): {row['metrics']['seconds']['mean']:.4f}s" for row in results
    )
    heldout_by_loops = ", ".join(
        f"{row['model']}(loops={row['loop_count']}): {row['metrics']['heldout_accuracy']['mean']:.3f}" for row in results
    )
    lines.extend([
        "2. **Loop scaling and compute trade-off (1, 2, 4).** "
        f"Total parameter counts across the reported configurations: {params_totals}. "
        f"Sequential block applications scale as {block_scaling}. "
        f"Measured wall-clock seconds: {seconds_by_loops}. Held-out accuracy by loop count: {heldout_by_loops}. "
        "Equal epoch counts are **not** equal compute budgets: each additional loop applies the shared "
        "block once more per optimisation step, so the higher-loop configurations perform more work per "
        "step; the measured `seconds` column is the only latency evidence in this report.",
        "",
    ])

    # RQ3 / H2
    h2_entries = [h for h in payload["hypotheses"] if h["id"] == "H2"]
    h2_detail = ""
    if h2_entries:
        per_config = ", ".join(
            f"looped-{c['loop_count']}: {_fmt_signed(c['statistic'])} -> {c['result']}" for c in h2_entries[0]["per_config"]
        )
        h2_detail = (
            " Length-generalization drop is defined per seed as "
            "`in_distribution_accuracy(dev only, lengths 4-8) - heldout_accuracy(lengths 12-16)`; "
            f"H2 statistic = drop(baseline) - drop(looped), evaluated against the same +/-0.05 operational "
            f"rule. Results: {per_config}."
        )
    lines.extend([
        "3. **Length generalization (train/dev 4-8 vs held-out 12-16).** "
        "These length ranges are disjoint and split generation is example-disjoint by construction."
        + h2_detail,
        "",
        f"   Exact-example overlap measured per seed (train_dev / train_heldout / dev_heldout): "
        + "; ".join(
            f"seed {s}: " + "/".join(str(v) for v in ov.values()) for s, ov in payload["data"]["example_overlap_by_seed"].items()
        )
        + ".",
        "",
    ])

    # RQ4
    rows_txt = ", ".join(
        f"{row['model']}(loops={row['loop_count']}): total {row['parameter_counts']['total']:,}, "
        f"trainable {row['parameter_counts']['trainable']:,}, frozen {row['parameter_counts']['frozen']:,}"
        for row in results
    )
    lines.extend([
        "4. **Controlled parameter comparison (structural).** "
        f"Within backend `{backend}`: {rows_txt}. Trainable parameter counts are constant across loop "
        "counts, so increasing loops adds sequential block applications but no new parameters. This is a "
        "structural check of the implementation, not a claim that the two backends have equal counts.",
        "",
        "## 5. Hypothesis evaluation status",
        "",
        "| id | statement | status | statistic | threshold | result |",
        "|---|---|---|---:|---:|---|",
    ])
    for hyp in payload["hypotheses"]:
        if hyp.get("per_config"):
            statistic = "; ".join(
                f"loops={c['loop_count']}: {_fmt_signed(c['statistic'])}" for c in hyp["per_config"]
            )
        else:
            statistic = hyp.get("statistic", "n/a")
        lines.append(
            f"| {hyp['id']} | {hyp['statement']} | {hyp['status']} | {statistic} | "
            f"{hyp.get('threshold') if hyp.get('threshold') is not None else 'n/a'} | {hyp['result']} |"
        )
    lines.extend([
        "",
        "Each hypothesis above is marked `evaluated` or `not evaluated`. The +/-0.05 rule is an operational "
        "decision rule (see section 3) and is applied identically to H1 and H2.",
        "",
        "### Not evaluated by this experiment",
        "",
    ])
    for item in payload["not_evaluated"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 6. Determinism and reproducibility notes",
        "",
    ])
    for note in payload["determinism_notes"]:
        lines.append(f"- {note}")
    lines.extend([
        "",
        "Identical initialization check (per seed, over the shared modules): the recorded initialization "
        "fingerprints are identical across the baseline and all looped configurations.",
        "| seed | identical shared init | fingerprint (first 16 hex chars) |",
        "|---|---|---|",
    ])
    for seed_str, info in payload["initialization"]["by_seed"].items():
        lines.append(
            f"| {seed_str} | {info['identical_shared_init']} | `{info['fingerprint'][:16]}` |"
        )
    lines.extend([
        "",
        "## 7. Environment and PyTorch status (derived from this run)",
        "",
        _env_torch_sentence(payload),
        "",
    ])
    if payload.get("diagnostic"):
        diag = payload["diagnostic"]
        lines.extend([
            "### Optimization diagnostic (separate small dataset, not the held-out split)",
            "",
            f"{diag['purpose']}",
            f"- Dataset: {diag['dataset']['n']} examples, lengths {diag['dataset']['min_length']}-{diag['dataset']['max_length']}, "
            f"seed {diag['dataset']['seed']} (a separate diagnostic draw, never the held-out split).",
            f"- Device `{diag['device']}`, {diag['epochs']} epochs, learning rate {diag['learning_rate']}, "
            f"optimizer {diag['optimizer']}, measured {diag['seconds']:.2f}s.",
            f"- Train accuracy: {diag['initial_train_accuracy']:.3f} -> {diag['final_train_accuracy']:.3f}; "
            f"train loss (BCE): {diag['initial_loss']:.4f} -> {diag['final_loss']:.4f}.",
            f"- This is a diagnostic of optimization only; it says nothing about held-out or length generalization.",
            "",
        ])
    lines.extend([
        "## 8. Artifacts",
        "",
    ])
    if payload.get("artifacts"):
        art = payload["artifacts"]
        lines.extend([
            f"- Directory: `{art['artifacts_dir']}` (run.json, run.md, checkpoint_manifest.json, checkpoints/).",
            f"- Exact rerun command: `{art['rerun_command']}`",
            f"- Checkpoint manifest SHA-256: `{art['checkpoint_manifest_sha256']}`",
            f"- Checkpoints verified by reload-and-re-evaluate: "
            f"{sum(1 for c in art['reload_checks'] if c['matches_recorded'])}/{len(art['reload_checks'])} "
            f"matched the recorded metrics (tolerance {art['reload_tolerance']}).",
            f"- Portability limit: {art['portability_note']}",
        ])
    else:
        lines.append("- No checkpoint artifacts were requested for this run.")
    lines.extend([
        "",
        "### Metric notes",
        "",
    ])
    for key, value in payload["metric_notes"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "Historical note: `results/smoke.json` and `results/smoke.md` predate this generator and are kept "
        "as historical evidence; they are not regenerated by this report writer.",
        "",
        "See [DESIGN.md](DESIGN.md) for the specified hypotheses and the fixed decision rule. The commit "
        "history does **not** establish preregistration: DESIGN.md and the first results were committed in "
        "the same commit (161752a), so these are specified hypotheses and a fixed decision rule, not a "
        "preregistration claim.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="Torch device request. `cuda`/`mps` raise if unavailable (no silent fallback).")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=None,
                    help="Write run.json, run.md, checkpoints/ and checkpoint_manifest.json here.")
    ap.add_argument("--seeds", type=str, default="7,42,123", help="Comma-separated random seeds (default: '7,42,123')")
    ap.add_argument("--seed", type=int, default=None, help="Single random seed override (if provided, overrides --seeds)")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--dev-size", type=int, default=128)
    ap.add_argument("--heldout-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--diagnostic-epochs", type=int, default=600,
                    help="Epochs for the separate 'can it learn at all' overfit diagnostic (torch only).")
    ap.add_argument("--diagnostic-learning-rate", type=float, default=1e-3,
                    help="Learning rate for the optimization diagnostic (separate from the experiment LR).")
    ap.add_argument("--skip-diagnostic", action="store_true")
    args = ap.parse_args(raw_argv)

    backend = args.backend
    if backend == "auto":
        backend = "torch" if _import_torch_version() is not None else "numpy"

    if backend == "numpy" and args.device not in ("auto", "cpu"):
        raise SystemExit(
            f"--device {args.device} is only meaningful for --backend torch; the NumPy backend runs on CPU. "
            "Use --device cpu or --device auto with --backend numpy."
        )

    seeds = parse_seeds(args.seed, args.seeds)
    epochs = args.epochs if args.epochs is not None else (40 if backend == "numpy" else 80)
    learning_rate = 0.08 if backend == "numpy" else 3e-3
    weight_decay = args.weight_decay if backend == "torch" else 1e-4
    repo_root = Path(__file__).resolve().parent.parent

    artifacts_dir: Optional[Path] = Path(args.artifacts_dir) if args.artifacts_dir else None
    output: Path = args.output or (artifacts_dir / "run.json" if artifacts_dir else Path("results/smoke.json"))
    report_path: Path = args.report or (artifacts_dir / "run.md" if artifacts_dir else output.with_suffix(".md"))

    torch_device = None
    if backend == "torch":
        torch_device = resolve_device(args.device)
    device_str = str(torch_device) if torch_device is not None else "cpu"

    all_runs: Dict[Tuple[str, int], List[Dict[str, object]]] = {
        cfg_tuple: [] for cfg_tuple in EXPERIMENT_CONFIGS
    }
    split_manifests: Dict[str, object] = {}
    overlap_by_seed: Dict[str, Dict[str, int]] = {}
    init_by_seed: Dict[str, Dict[str, object]] = {}
    checkpoint_records: List[Dict[str, object]] = []
    reload_checks: List[Dict[str, object]] = []

    checkpoint_dir = artifacts_dir / "checkpoints" if artifacts_dir else None

    for s in seeds:
        splits = make_splits(
            train_n=args.train_size,
            dev_n=args.dev_size,
            heldout_n=args.heldout_size,
            seed=s,
        )
        overlap = overlap_report(splits)
        if any(value != 0 for value in overlap.values()):
            raise RuntimeError(
                f"split generation produced overlapping exact examples for seed {s}: {overlap}. "
                "This must never happen; the splits are documented as example-disjoint."
            )
        overlap_by_seed[str(s)] = overlap
        split_manifests[str(s)] = json.loads(split_manifest(splits))
        cfg = NumpyConfig(seed=s)

        seed_fingerprints: Dict[str, str] = {}
        for model, loops in EXPERIMENT_CONFIGS:
            ckpt_path = (
                checkpoint_dir / f"seed{s}-{model}-loop{loops}.pt" if checkpoint_dir is not None else None
            )
            if backend == "numpy":
                metrics = run_numpy(
                    model, loops, splits, cfg, epochs=epochs, learning_rate=learning_rate, l2=1e-4, seed=s
                )
            else:
                metrics = run_torch(
                    model, loops, splits, cfg, epochs=epochs, learning_rate=learning_rate,
                    weight_decay=weight_decay, seed=s, device=torch_device,
                    device_requested=args.device, checkpoint_path=ckpt_path,
                )
            all_runs[(model, loops)].append(metrics)
            seed_fingerprints[f"{model}-loop{loops}"] = str(metrics["initialization_fingerprint"])
            if ckpt_path is not None:
                rel = ckpt_path.relative_to(artifacts_dir) if artifacts_dir else Path(ckpt_path.name)
                checkpoint_records.append({
                    "path": str(rel),
                    "sha256": _sha256_file(ckpt_path),
                    "bytes": ckpt_path.stat().st_size,
                    "seed": s,
                    "model": model,
                    "loop_count": loops,
                    "parameter_counts": metrics["parameter_counts"],
                    "recorded_metrics": {
                        name: metrics[name] for name in ("train", "dev", "heldout")
                    },
                })

        unique_fps = set(seed_fingerprints.values())
        if len(unique_fps) != 1:
            raise RuntimeError(
                f"configurations for seed {s} did not start from identical shared parameters: {seed_fingerprints}"
            )
        init_by_seed[str(s)] = {
            "by_config": seed_fingerprints,
            "identical_shared_init": True,
            "fingerprint": next(iter(unique_fps)),
        }

        if backend == "torch" and ckpt_path is not None:
            for record in [r for r in checkpoint_records if r["seed"] == s]:
                ckpt = artifacts_dir / str(record["path"])
                reload_checks.append(
                    _reload_and_reevaluate(ckpt, record, splits, cfg, torch_device)
                )

    baseline_runs = all_runs[("baseline", 1)]
    baseline_seed_metrics = [_seed_metrics(r) for r in baseline_runs]

    results = []
    for model, loops in EXPERIMENT_CONFIGS:
        runs = all_runs[(model, loops)]
        seed_metrics = [_seed_metrics(r) for r in runs]
        paired: Dict[str, Dict[str, object]] = {}
        for metric in PAIRED_METRICS:
            deltas = [m[metric] - b[metric] for m, b in zip(seed_metrics, baseline_seed_metrics)]
            st = summarize_stats(deltas)
            paired[metric] = {
                "per_seed": deltas,
                "mean": st["mean"],
                "std": st["std"],
                "stderr": st["std"] / float(np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
                "n_seeds": len(deltas),
            }

        accuracy_stats = {
            key: summarize_stats([m[key] for m in seed_metrics])
            for key in (
                "train_accuracy", "dev_accuracy", "heldout_accuracy", "in_distribution_accuracy",
                "length_generalization_drop", "train_bce", "heldout_bce", "final_train_bce", "seconds",
            )
        }

        if model == "baseline":
            decision = "baseline (reference)"
        else:
            decision = _decision_word(float(paired["heldout_accuracy"]["mean"]))

        split_by_length: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sp in ("train", "dev", "heldout"):
            split_lengths = sorted({int(k) for r in runs for k in r[sp]["accuracy_by_length"]})
            split_by_length[sp] = {}
            for length in split_lengths:
                len_key = str(length)
                vals = [r[sp]["accuracy_by_length"][len_key] for r in runs if len_key in r[sp]["accuracy_by_length"]]
                if vals:
                    split_by_length[sp][len_key] = summarize_stats(vals)

        results.append({
            "model": model,
            "loop_count": loops,
            "parameter_counts": runs[0]["parameter_counts"],
            "sequential_block_applications": runs[0]["sequential_block_applications"],
            "estimated_block_calls": runs[0]["estimated_block_calls"],
            "parameter_sharing": runs[0]["parameter_sharing"],
            "decision": decision,
            "metrics": {
                "epochs": epochs,
                "backend_note": runs[0].get("backend_note", ""),
                "train_accuracy": accuracy_stats["train_accuracy"],
                "dev_accuracy": accuracy_stats["dev_accuracy"],
                "heldout_accuracy": accuracy_stats["heldout_accuracy"],
                "in_distribution_accuracy": accuracy_stats["in_distribution_accuracy"],
                "length_generalization_drop": accuracy_stats["length_generalization_drop"],
                "train_bce": accuracy_stats["train_bce"],
                "heldout_bce": accuracy_stats["heldout_bce"],
                "final_train_bce": accuracy_stats["final_train_bce"],
                "seconds": accuracy_stats["seconds"],
                "accuracy_by_length": {
                    sp: {k: v["mean"] for k, v in split_by_length[sp].items()} for sp in split_by_length
                },
                "accuracy_by_length_stats": split_by_length,
            },
            "paired_delta_vs_baseline": paired,
            "per_seed_runs": [
                {
                    "seed": s,
                    "train": r["train"],
                    "dev": r["dev"],
                    "heldout": r["heldout"],
                    "in_distribution_accuracy": m["in_distribution_accuracy"],
                    "length_generalization_drop": m["length_generalization_drop"],
                    "final_train_bce": r["final_train_bce"],
                    "seconds": r["seconds"],
                    "sequential_block_applications": r["sequential_block_applications"],
                    "paired_delta_vs_baseline": {
                        metric: m[metric] - b[metric] for metric in PAIRED_METRICS
                    },
                    "run_metadata": {
                        "device": r.get("device"),
                        "device_requested": r.get("device_requested"),
                        "dtype": r.get("dtype"),
                        "torch_version": r.get("torch_version"),
                        "python_version": r.get("python_version"),
                        "optimizer": r.get("optimizer"),
                        "learning_rate": r.get("learning_rate"),
                        "weight_decay": r.get("weight_decay"),
                        "epochs": r.get("epochs"),
                        "parameter_counts": r.get("parameter_counts"),
                        "initialization_fingerprint": r.get("initialization_fingerprint"),
                    },
                }
                for s, r, m, b in zip(seeds, runs, seed_metrics, baseline_seed_metrics)
            ],
        })

    # Hypotheses: H1, H2 and the H0 null are computed from the measured values.
    non_baseline = [row for row in results if row["model"] != "baseline"]
    h1_per_config = [
        {
            "model": row["model"],
            "loop_count": row["loop_count"],
            "statistic": row["paired_delta_vs_baseline"]["heldout_accuracy"]["mean"],
            "std": row["paired_delta_vs_baseline"]["heldout_accuracy"]["std"],
            "stderr": row["paired_delta_vs_baseline"]["heldout_accuracy"]["stderr"],
            "result": row["decision"],
        }
        for row in non_baseline
    ]
    h2_per_config = [
        {
            "model": row["model"],
            "loop_count": row["loop_count"],
            "statistic": -row["paired_delta_vs_baseline"]["length_generalization_drop"]["mean"],
            "std": row["paired_delta_vs_baseline"]["length_generalization_drop"]["std"],
            "stderr": row["paired_delta_vs_baseline"]["length_generalization_drop"]["stderr"],
            "result": _h2_word(-row["paired_delta_vs_baseline"]["length_generalization_drop"]["mean"]),
        }
        for row in non_baseline
    ]
    frozen_scope = backend == "numpy"
    scope_note = (
        "Evaluated on the frozen-feature NumPy smoke path only (readout+bias trained); the decisions do "
        "not measure trained recurrence."
        if frozen_scope
        else "Evaluated on a full end-to-end trained run under this budget."
    )
    hypotheses = [
        {
            "id": "H1",
            "statement": "Looped configurations (2 or 4 loops) exceed the single-pass baseline on held-out "
                         "parity accuracy by >= 0.05.",
            "status": "evaluated",
            "threshold": DECISION_THRESHOLD,
            "per_config": h1_per_config,
            "result": (
                "supported for at least one loop count"
                if any(c["result"] == "improvement" for c in h1_per_config)
                else "not supported in this run"
            ),
            "scope": scope_note,
            "notes": f"N={len(seeds)} seeds, held-out n={args.heldout_size}; operational rule only.",
        },
        {
            "id": "H2",
            "statement": "The accuracy drop from in-distribution lengths (4-8) to held-out lengths (12-16) "
                         "is at least 0.05 smaller for a looped configuration than for the baseline.",
            "status": "evaluated",
            "threshold": DECISION_THRESHOLD,
            "per_config": h2_per_config,
            "result": (
                "supported for at least one loop count"
                if any(c["result"] == "supported" for c in h2_per_config)
                else "not supported in this run"
            ),
            "scope": scope_note,
            "notes": "Exploratory: drop uses dev only, excluding training. A smaller gap can reflect worse dev accuracy, not improved held-out accuracy. statistic = drop(baseline) - drop(looped), paired per seed.",
        },
        {
            "id": "H0",
            "statement": "No looped configuration meets the +/-0.05 improvement threshold (flat or worse "
                         "held-out accuracy than the baseline).",
            "status": "evaluated",
            "threshold": DECISION_THRESHOLD,
            "result": (
                "rejected (at least one configuration met the improvement threshold)"
                if any(row["decision"] == "improvement" for row in non_baseline)
                else "not rejected by this run's data (all configurations flat or regressing)"
            ),
            "scope": scope_note,
            "notes": "H0 is the complement of the H1 rule and is reported for completeness.",
        },
    ]

    torch_version = _import_torch_version()
    diagnostic = None
    if backend == "torch" and not args.skip_diagnostic:
        from .diagnostics import overfit_diagnostic

        diagnostic = overfit_diagnostic(
            seed=seeds[0], epochs=args.diagnostic_epochs, device=torch_device,
            learning_rate=args.diagnostic_learning_rate, weight_decay=weight_decay,
        )

    payload: Dict[str, object] = {
        "task": "binary sequence parity",
        "backend": backend,
        "seeds": seeds,
        "config": vars(NumpyConfig(seed=seeds[0])),
        "scope_statement": (
            "This run used the NumPy frozen-feature smoke path: the transformer block and embeddings are "
            "frozen and only the linear readout + bias are trained. Frozen-feature smoke results do not "
            "measure whether trained recurrence helps, and nothing here is evidence about trained "
            "recurrence or general reasoning."
            if backend == "numpy"
            else
            f"This run trained all parameters end-to-end with {training_optimizer(backend)} for {epochs} "
            f"epochs on device `{device_str}` for seeds {', '.join(str(s) for s in seeds)}. Conclusions are scoped to this run, this "
            "task (binary sequence parity), these seeds, these sample sizes and this training budget; they "
            "are not evidence about general reasoning or about recurrence at other scales."
        ),
        "data": {
            "train_size": args.train_size,
            "dev_size": args.dev_size,
            "heldout_size": args.heldout_size,
            "train_lengths": [4, 8],
            "heldout_lengths": [12, 16],
            "split_manifests_by_seed": split_manifests,
            "example_overlap_by_seed": overlap_by_seed,
            "generator": (
                "rejection-sampled, example-disjoint by construction (rlt_rsi/data.py); overlap is "
                "measured and recorded per seed and asserted to be zero"
            ),
        },
        "training": {
            "epochs": epochs,
            "optimizer": "AdamW" if backend == "torch" else "numpy full-batch gradient descent on readout+bias only",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch": "full-batch (all train examples per step)",
            "device_requested": args.device,
            "device": device_str,
            "dtype": "float32" if backend == "torch" else "float64",
            "trainable_scope": (
                "all parameters (end-to-end)" if backend == "torch"
                else "readout + bias only; embedding and transformer block are frozen"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch_version,
            "torch_available": torch_version is not None,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "determinism_notes": determinism_notes(backend, device_str),
        "metric_notes": {
            "estimated_block_calls": BLOCK_CALL_NOTE,
            "seconds": "Measured wall-clock training time (per configuration, averaged over seeds).",
            "in_distribution_accuracy": "Accuracy on disjoint dev examples only (lengths 4-8); training examples excluded. H2 is exploratory: a smaller gap may reflect worse dev accuracy, not improved held-out performance.",
            "length_generalization_drop": "in_distribution_accuracy - heldout_accuracy (per seed).",
            "paired_delta_vs_baseline": "Per-seed difference against the same-seed baseline run, then mean/std.",
            "decision": (
                "Operational rule only: >= +0.05 improvement, <= -0.05 regression, else flat. Not a "
                "significance test and not an equivalence test."
            ),
            "parameter_counts": "total/trainable/frozen counted from the real model object (requires_grad for torch).",
        },
        "initialization": {
            "by_seed": init_by_seed,
            "note": "Baseline and all looped configurations for a seed share identical initialization of the "
                    "shared modules; the fingerprint is SHA-256 over the constructed parameters before training.",
        },
        "results": results,
        "hypotheses": hypotheses,
        "not_evaluated": [
            "General reasoning ability: binary sequence parity is an isolated algorithmic check.",
            "Statistical significance or equivalence: the +/-0.05 rule is operational, and the seed count is small.",
            "Equal-compute comparisons: equal epochs are not equal compute; only measured seconds and block applications are reported.",
            "Trained recurrence at other scales/architectures, or on other tasks: not tested here.",
        ],
        "diagnostic": diagnostic,
        "artifacts": None,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = None
    if artifacts_dir is not None:
        rerun_command = "python -m rlt_rsi.train " + " ".join(shlex.quote(a) for a in raw_argv)
        manifest_path = write_artifacts(
            artifacts_dir,
            checkpoint_records=checkpoint_records,
            reload_checks=reload_checks,
            payload=payload,
            rerun_command=rerun_command,
            repo_root=repo_root,
        )
        payload["artifacts"] = {
            "artifacts_dir": str(artifacts_dir),
            "run_json": str(output),
            "run_md": str(report_path),
            "checkpoint_manifest": str(manifest_path),
            "checkpoint_manifest_sha256": _sha256_file(manifest_path),
            "rerun_command": rerun_command,
            "reload_checks": reload_checks,
            "reload_tolerance": 1e-4,
            "portability_note": (
                f"Checkpoint .pt files are gitignored local artifacts ({len(checkpoint_records)} files, "
                f"{min([int(r['bytes']) for r in checkpoint_records])}-{max([int(r['bytes']) for r in checkpoint_records])} "
                "bytes each) and are NOT portable; only checkpoint_manifest.json (with SHA-256 hashes) is "
                "tracked. Regenerate checkpoints with the rerun command on a matching torch build."
                if checkpoint_records else
                "No checkpoints were written for this run."
            ),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_report(report_path, payload)

    summary_out = {
        "output": str(output),
        "report": str(report_path),
        "backend": backend,
        "device": device_str,
        "seeds": seeds,
        "results": [
            {
                "model": x["model"],
                "loops": x["loop_count"],
                "total_params": x["parameter_counts"]["total"],
                "trainable_params": x["parameter_counts"]["trainable"],
                "frozen_params": x["parameter_counts"]["frozen"],
                "block_applications": x["sequential_block_applications"],
                "heldout_acc_mean": x["metrics"]["heldout_accuracy"]["mean"],
                "heldout_acc_std": x["metrics"]["heldout_accuracy"]["std"],
                "seconds_mean": x["metrics"]["seconds"]["mean"],
                "decision": x["decision"],
            }
            for x in results
        ],
        "hypotheses": [
            {"id": h["id"], "status": h["status"], "result": h["result"]} for h in hypotheses
        ],
    }
    print(json.dumps(summary_out, indent=2))
    return 0


def training_optimizer(backend: str) -> str:
    return "AdamW" if backend == "torch" else "numpy full-batch gradient descent"


if __name__ == "__main__":
    raise SystemExit(main())
