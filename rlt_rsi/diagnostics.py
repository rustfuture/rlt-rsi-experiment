"""Optimization diagnostics.

These checks deliberately use a **separate small diagnostic dataset** (never the
held-out split). They answer one narrow question -- "can the optimizer reduce
loss on a handful of examples at all?" -- and are not evidence about held-out
accuracy, length generalization, or recurrence benefits.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

DIAGNOSTIC_PURPOSE = (
    "Overfit a tiny separate diagnostic batch of parity examples (lengths 4-8) to check that the "
    "optimizer actually reduces loss. This is a diagnostic of optimization plumbing, not a held-out "
    "claim; the diagnostic draw is never used for evaluation."
)


def overfit_diagnostic(
    *,
    seed: int = 7,
    epochs: int = 600,
    device=None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n: int = 32,
    min_length: int = 4,
    max_length: int = 8,
    model_kind: str = "looped",
    loop_count: int = 2,
) -> Dict[str, object]:
    """Overfit ``n`` diagnostic examples and record the train trajectory.

    Uses the primary research variant (shared-weight looped block) so that the
    diagnostic exercises the same code path as the looped experiment.
    """
    from .data import make_split
    from .model import NumpyConfig
    from .train import _torch_batch, _torch_components, build_torch_model

    torch, nn = _torch_components()
    device = torch.device(device or "cpu")
    diagnostic_seed = seed + 9001  # deliberately distinct from the experiment splits
    split = make_split(n=n, min_length=min_length, max_length=max_length, seed=diagnostic_seed)
    cfg = NumpyConfig(seed=seed)
    model = build_torch_model(cfg, model_kind, loop_count, seed=seed).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tx, tl, ty = _torch_batch(torch, split, device)
    labels_np = split.labels

    losses: List[float] = []
    accuracies: List[float] = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        logits = model(tx, tl)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ty)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = (model(tx, tl).detach().cpu().numpy() >= 0).astype(np.int64)
            accuracies.append(float(np.mean(pred == labels_np)))
    seconds = time.perf_counter() - start
    return {
        "purpose": DIAGNOSTIC_PURPOSE,
        "not_heldout": True,
        "model": model_kind,
        "loop_count": loop_count,
        "dataset": {
            "n": n,
            "min_length": min_length,
            "max_length": max_length,
            "seed": diagnostic_seed,
            "fingerprint": split.fingerprint(),
        },
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seconds": seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial_train_accuracy": accuracies[0],
        "final_train_accuracy": accuracies[-1],
        "best_train_accuracy": max(accuracies),
        "loss_trajectory": losses,
        "accuracy_trajectory": accuracies,
    }
