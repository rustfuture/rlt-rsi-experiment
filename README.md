# rlt-rsi-experiment

Reproducible minimal experiment comparing a conventional transformer with a
shared-weight recurrent/looped transformer on binary sequence parity. The
experiment tests iterative computation and length generalization; it does not
test or claim general reasoning ability.

## Reproduce the CPU smoke run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m rlt_rsi.train --backend numpy --output results/smoke.json
pytest -q
```

The command writes machine-readable `results/smoke.json` and a Markdown summary
at `results/smoke.md`. `--seed`, split sizes, and `--epochs` are configurable.
The default smoke path is deterministic and needs only NumPy. Run the full
trainable experiment in Colab/L4 with:

```bash
python -m pip install -e '.[torch]'
python -m rlt_rsi.train --backend torch --seed 7 --epochs 80 --output results/torch-seed7.json
```

Use `--backend torch` to fail fast with the exact install instruction if torch
is unavailable; `--backend auto` selects torch when installed and otherwise
selects the labeled NumPy fallback.

## Interpreting results

The committed smoke output is illustrative. Compare baseline and looped rows,
including the held-out 12–16 length split and `accuracy_by_length`. A flat or
regressed looped score is expected to remain a valid outcome. Do not infer that
recurrence improves reasoning from this task.

See [DESIGN.md](DESIGN.md) for hypotheses, leakage controls, matched controls,
failure cases, and scientific limitations.
