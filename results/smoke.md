# Smoke experiment report

Backend: **numpy**

Accuracy (higher is better):

| model | loops | train | dev | held-out length |
|---|---:|---:|---:|---:|
| baseline | 1 | 0.547 | 0.523 | 0.484 |
| looped | 2 | 0.547 | 0.523 | 0.484 |
| looped | 4 | 0.492 | 0.469 | 0.469 |

The held-out split uses lengths outside the train/dev range. This is an illustrative smoke run, not evidence that recurrence improves reasoning.

See DESIGN.md for hypotheses, controls, and limitations.
