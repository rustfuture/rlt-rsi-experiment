import copy
import time
from typing import Dict, List, Optional
import numpy as np

from .data import DatasetSplit
from .model import NumpyConfig
from .train import build_torch_model, _torch_components, _torch_batch, accuracy, _torch_parameter_counts, _torch_state_fingerprint, _numpy_parameter_fingerprint

def run_rsi_torch(
    splits: Dict[str, DatasetSplit],
    cfg: NumpyConfig,
    *,
    generations: int,
    epochs_per_gen: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device,
) -> Dict[str, object]:
    """
    Runs the RSI experiment loop.
    Returns tracking info.
    """
    torch, nn = _torch_components()
    
    # Initialize state
    current_loop_count = 1
    # Build initial model just to get the starting weights
    initial_model = build_torch_model(cfg, "looped", current_loop_count, seed=seed).to(device)
    current_state_dict = copy.deepcopy(initial_model.state_dict())
    init_fingerprint = _torch_state_fingerprint(torch, initial_model)
    
    lineage = []
    
    total_seconds = 0.0
    total_block_applications = 0
    total_training_steps = 0
    
    tx, tl, ty = _torch_batch(torch, splits["train"], device)
    vx, vl, vy_np = _torch_batch(torch, splits["dev"], device)
    vy_np = splits["dev"].labels
    
    for gen in range(1, generations + 1):
        # Proposals: current, current + 1, current - 1 (min 1, max 8)
        proposals = [current_loop_count]
        if current_loop_count < 8:
            proposals.append(current_loop_count + 1)
        if current_loop_count > 1:
            proposals.append(current_loop_count - 1)
            
        candidate_results = []
        
        for p_loop in proposals:
            model = build_torch_model(cfg, "looped", p_loop, seed=seed).to(device)
            model.load_state_dict(copy.deepcopy(current_state_dict))
            model.train()
            
            optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            start = time.perf_counter()
            for _ in range(epochs_per_gen):
                optim.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(tx, tl), ty)
                loss.backward()
                optim.step()
                total_block_applications += p_loop
                total_training_steps += 1
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            
            # Eval on dev and train
            model.eval()
            with torch.no_grad():
                logits = model(vx, vl).detach().cpu().numpy()
                acc = accuracy(logits, vy_np)
                
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
                eps = 1e-8
                bce = float(-np.mean(vy_np * np.log(probs + eps) + (1 - vy_np) * np.log(1 - probs + eps)))
                
                # Compute train BCE
                t_logits = model(tx, tl).detach().cpu().numpy()
                t_probs = 1.0 / (1.0 + np.exp(-np.clip(t_logits, -40, 40)))
                ty_np = splits["train"].labels
                t_bce = float(-np.mean(ty_np * np.log(t_probs + eps) + (1 - ty_np) * np.log(1 - t_probs + eps)))
                
            candidate_results.append({
                "loop_count": p_loop,
                "dev_accuracy": acc,
                "dev_bce": bce,
                "train_bce": t_bce,
                "state_dict": copy.deepcopy(model.state_dict())
            })
            
        # Select best candidate (highest dev_accuracy, lowest dev_bce)
        candidate_results.sort(key=lambda x: (x["dev_accuracy"], -x["dev_bce"]), reverse=True)
        best = candidate_results[0]
        
        current_loop_count = best["loop_count"]
        current_state_dict = best["state_dict"]
        
        lineage.append({
            "generation": gen,
            "proposals": [c["loop_count"] for c in candidate_results],
            "accepted_loop_count": current_loop_count,
            "dev_accuracy": best["dev_accuracy"],
            "dev_bce": best["dev_bce"],
            "train_bce": best["train_bce"],
        })
        
    # Final eval on all splits
    final_model = build_torch_model(cfg, "looped", current_loop_count, seed=seed).to(device)
    final_model.load_state_dict(current_state_dict)
    final_model.eval()
    
    final_metrics = {}
    with torch.no_grad():
        for name, split in splits.items():
            x, l, y_t = _torch_batch(torch, split, device)
            logits = final_model(x, l).detach().cpu().numpy()
            y_np = split.labels
            acc = accuracy(logits, y_np)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            eps = 1e-8
            bce = float(-np.mean(y_np * np.log(probs + eps) + (1 - y_np) * np.log(1 - probs + eps)))
            
            by_length = {}
            for length in sorted(set(split.lengths.tolist())):
                mask = split.lengths == length
                by_length[str(length)] = accuracy(logits[mask], y_np[mask])
                
            final_metrics[name] = {
                "accuracy": acc,
                "bce": bce,
                "n": int(len(y_np)),
                "accuracy_by_length": by_length,
            }
            
    return {
        "model": "rsi_looped",
        "final_loop_count": current_loop_count,
        "generations": generations,
        "epochs_per_gen": epochs_per_gen,
        "lineage": lineage,
        "total_seconds": total_seconds,
        "total_block_applications": total_block_applications,
        "total_training_steps": total_training_steps,
        **final_metrics,
        "parameter_counts": _torch_parameter_counts(final_model),
        "initialization_fingerprint": init_fingerprint,
        "estimated_block_calls": current_loop_count,
        "sequential_block_applications": current_loop_count,
        "parameter_sharing": True,
        "epochs": generations * epochs_per_gen,
        "seed": seed,
        "final_train_bce": lineage[-1]["train_bce"] if lineage else 0.0,
        "seconds": total_seconds,
        "device": str(device),
        "device_requested": str(device),
        "dtype": "float32",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "torch_version": torch.__version__,
        "python_version": "3",
        "torch_manifest": {
            "total_parameters": _torch_parameter_counts(final_model)["total"],
            "trainable_parameters": _torch_parameter_counts(final_model)["trainable"],
            "frozen_parameters": _torch_parameter_counts(final_model)["frozen"],
        }
    }

from .train import NumpyTransformerClassifier, NumpyLoopedTransformerClassifier, fit_numpy, evaluate_numpy

def run_rsi_numpy(
    splits: Dict[str, DatasetSplit],
    cfg: NumpyConfig,
    *,
    generations: int,
    epochs_per_gen: int,
    learning_rate: float,
    weight_decay: float,
    seed: int = 0,
) -> Dict[str, object]:
    current_loop_count = 1
    
    # Init initial model parameters
    initial_model = NumpyLoopedTransformerClassifier(cfg, current_loop_count)
    init_fingerprint = _numpy_parameter_fingerprint(initial_model)
    
    current_readout = initial_model.readout.copy()
    current_bias = float(initial_model.bias)
    
    lineage = []
    
    total_seconds = 0.0
    total_block_applications = 0
    total_training_steps = 0
    
    for gen in range(1, generations + 1):
        proposals = [current_loop_count]
        if current_loop_count < 8:
            proposals.append(current_loop_count + 1)
        if current_loop_count > 1:
            proposals.append(current_loop_count - 1)
            
        candidate_results = []
        
        for p_loop in proposals:
            model = NumpyLoopedTransformerClassifier(cfg, p_loop)
            model.readout = current_readout.copy()
            model.bias = current_bias
            
            start = time.perf_counter()
            fit = fit_numpy(model, splits["train"], epochs=epochs_per_gen, learning_rate=learning_rate, l2=weight_decay)
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            total_block_applications += p_loop * epochs_per_gen
            total_training_steps += epochs_per_gen
            
            eval_res = evaluate_numpy(model, splits["dev"])
            
            candidate_results.append({
                "loop_count": p_loop,
                "dev_accuracy": eval_res["accuracy"],
                "dev_bce": eval_res["bce"],
                "train_bce": fit["final_train_bce"],
                "readout": model.readout.copy(),
                "bias": model.bias
            })
            
        candidate_results.sort(key=lambda x: (x["dev_accuracy"], -x["dev_bce"]), reverse=True)
        best = candidate_results[0]
        
        current_loop_count = best["loop_count"]
        current_readout = best["readout"]
        current_bias = best["bias"]
        
        lineage.append({
            "generation": gen,
            "proposals": [c["loop_count"] for c in candidate_results],
            "accepted_loop_count": current_loop_count,
            "dev_accuracy": best["dev_accuracy"],
            "dev_bce": best["dev_bce"],
            "train_bce": best["train_bce"],
        })
        
    final_model = NumpyLoopedTransformerClassifier(cfg, current_loop_count)
    final_model.readout = current_readout
    final_model.bias = current_bias
    
    final_metrics = {}
    for name, split in splits.items():
        final_metrics[name] = evaluate_numpy(final_model, split)
        
    return {
        "model": "rsi_looped",
        "final_loop_count": current_loop_count,
        "generations": generations,
        "epochs_per_gen": epochs_per_gen,
        "lineage": lineage,
        "total_seconds": total_seconds,
        "total_block_applications": total_block_applications,
        "total_training_steps": total_training_steps,
        **final_metrics,
        "parameter_counts": final_model.parameter_counts(),
        "initialization_fingerprint": init_fingerprint,
        "estimated_block_calls": current_loop_count,
        "sequential_block_applications": current_loop_count,
        "parameter_sharing": True,
        "epochs": generations * epochs_per_gen,
        "seed": seed,
        "final_train_bce": lineage[-1]["train_bce"] if lineage else 0.0,
        "seconds": total_seconds,
        "device": "cpu",
        "device_requested": "cpu",
        "dtype": "float64",
        "optimizer": "numpy",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "torch_version": None,
        "python_version": "3",
        "trainable_scope": "readout + bias only; embedding and transformer block are frozen",
        "backend_note": "NumPy RSI mode." 
    }
