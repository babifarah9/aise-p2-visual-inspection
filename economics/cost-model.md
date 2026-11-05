# P2 Cost Model

Implemented in `p2_inspection/economics.py`.

| Symbol | Value | Meaning |
| --- | --- | --- |
| V | 1,000,000 | Parts per year |
| p | 0.005 | Defect prevalence |
| C_FN | $400 | Cost of a missed defect |
| C_FP | $8 | Cost of a false alarm |
| K | $250,000 | Project cost cap |
| r0, f0 | 0.90, 0.04 | Baseline recall and FPR |

Required recall: `r*(f) = r0 + (K / V − (1 − p)(f0 − f) C_FP) / (p C_FN)`.
Slope: each point of FPR costs (1 − p) C_FP / (p C_FN) ≈ 3.98 points of recall.

Monte Carlo ranges (uniform): V 0.8–1.2 M, p 0.3–0.8%, C_FN $200–700, K $200–400 k, adoption 60–100%.
