# Design and preregistered questions

## Research question

On a small algorithmic task, does applying one transformer block repeatedly with
shared weights change accuracy or length generalization compared with a
same-width conventional one-pass transformer under a matched initialization?

This is deliberately narrower than “reasoning”: parity is a controlled probe of
iterative computation, not a reasoning benchmark.

## Falsifiable hypotheses

* **H1 (fit):** at a fixed width and training budget, the 2- or 4-loop model has
  higher held-out parity accuracy than the one-pass baseline by at least 5
  percentage points.
* **H2 (length generalization):** the recurrent model's accuracy drop from
  train-range lengths 4–8 to held-out lengths 12–16 is smaller than the
  baseline's by at least 5 points.
* **H0:** neither threshold is met; flat or worse performance is a valid result.

The thresholds are fixed before inspecting the output. A future L4 run should
repeat seeds (7, 17, 27), report mean and standard deviation, and use identical
optimizer steps and examples for all variants.

## Task and leakage controls

Each example is an independently generated binary sequence. The target is XOR
of its valid prefix. Train/dev lengths are sampled from 4–8; held-out lengths
are sampled only from 12–16. Split RNG seeds are `seed+101`, `seed+202`, and
`seed+303`; no generated rows are shared. Padding is masked in attention and the
final valid token is classified.

## Models and controls

The conventional model has one pre-normalized multi-head self-attention + MLP
block. The looped model has the exact same block and applies it 2 or 4 times,
sharing every block parameter. Width, heads, embeddings, positional encoding,
readout, data, and optimizer are otherwise held fixed. The report includes
architecture parameter count and loop count (a transparent compute proxy).

The `torch` backend trains every parameter. Because this checkout does not have
PyTorch, the committed smoke result uses the explicit NumPy fallback: the
transformer weights are seeded and frozen, and only the binary readout is fit.
It is useful for plumbing/reproducibility and is not a substitute for the full
trainable experiment.

## Failure cases and limitations

Parity is intentionally tiny and may be too hard for a random frozen feature
map; a 50% result is not surprising. One smoke seed is not statistical
evidence, and loop count changes sequential compute even when parameter count
is shared. The fallback does not support end-to-end gradient training. Results
must not be generalized to reasoning, language modeling, or capability claims.
