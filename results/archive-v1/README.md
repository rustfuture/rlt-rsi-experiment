# Archived v1 NumPy smoke artifacts (superseded)

These files are the NumPy smoke output produced by the previous split generator
and the previous report writer (base commit `f83bfb5`). They are preserved
verbatim for traceability and must not be mixed with the current
`results/smoke.json` / `results/smoke.md`.

Why they were superseded:

- The old split generator drew train and dev with independent PRNG seeds and did
  **not** enforce exact-example disjointness. Measured on this repo's own data,
  train/dev overlapped by 42, 55 and 50 exact `(length, token sequence)` examples
  for seeds 7, 42 and 123.
- The old report writer emitted hardcoded claims that were false in this
  environment, most notably an unconditional "PyTorch is not installed in the
  local Python 3.14.5 environment" (torch 2.14.0 is installed and was used for
  the real training run) and a hardcoded "held-out accuracy remains near the
  random chance threshold (~50%)".
- The old parameter accounting presented the frozen backbone subtotal (4,656) in
  a way that read as the total; the NumPy total is 4,681 (4,656 frozen + 25
  trainable readout/bias).

Current artifacts: `results/smoke.json` / `results/smoke.md` (NumPy,
example-disjoint splits) and `results/torch-mps-2026-09-15/` (real end-to-end
torch training on MPS). Do not pool v1 smoke numbers with the current ones.
