# P2: Visual Quality Inspection

Companion code for the running project **P2** in *AI Systems Engineering* (300-page edition). P2 supplies most of the book's worked numbers.

A camera images each part on a production line, and the system decides **pass, review, or reject**. Defects are rare, a missed defect costs far more than a false alarm, and inference runs at the edge.

| | |
| --- | --- |
| Decision supported | Pass, reject, or send a part to a human |
| Baseline today | Manual sampling, plus a rule-based vision check |
| Costliest error | A defective part that ships (missed defect) |
| Stresses | Rare events, edge deployment, error costs |

## The business case in code (Chapter 4)

```
Net = V [ p (r − r0) C_FN + (1 − p) (f0 − f) C_FP ] − K
```

With V = 1,000,000, p = 0.005, C_FN = $400, C_FP = $8, K = $250,000, r0 = 0.90 and f0 = 0.04:

```bash
python -m p2_inspection frontier   # required recall at each FPR; 0.9653 at FPR 0.025
python -m p2_inspection case       # Monte Carlo: P(Net > 0) over 1, 2 and 3 years, and the drivers
```

## The system

- `images.py`: synthetic 16 × 16 part images with lighting gradients, sensor noise, scratches and pits. Swap in a public defect-image dataset by yielding the same `(image, label)` pairs.
- `features.py`: plane detrending (removes lighting), then line and blob detectors.
- `model.py`: logistic regression (pure Python), and the rule-based baseline it must beat.
- `decision.py`: cost-minimising operating point, Wilson intervals, and the three-way pass/review/reject policy.
- `pipeline.py`: train → choose thresholds on validation → test once → gate.

```bash
python -m p2_inspection test    # report
python -m p2_inspection gate    # exit code 1 if any knock-out fails
python -m unittest discover -s tests -v
```

### The gate (knock-outs, Chapter 4 and Chapter 5)

| Check | Rule |
| --- | --- |
| Frontier | Lower 95% bound of test recall ≥ required recall at the upper 95% bound of FPR |
| Business case | Net benefit at (recall low, FPR high) ≥ 0 |
| Edge latency | p99 inference ≤ 20 ms on the CI machine |
| Auto-reject band | Precision of automatic rejects ≥ 0.97 |

Test sets are **enriched** with defects (5%) so recall has a usable confidence interval with 400 defects; costs are computed at the plant's real prevalence of 0.5%, which is valid because recall and FPR do not depend on prevalence.

Standard library only, Python 3.10+. All data is synthetic and every figure is illustrative.

## Layout

`method/` brief and gates · `architecture/decisions/` ADRs · `economics/` cost model · `eval/` test protocol · `ops/` runbook and SLOs · `data/` data contract.

## License

MIT
