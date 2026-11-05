# Test Protocol

| Set | Size | Defect share | Seed | Use |
| --- | --- | --- | --- | --- |
| Train | 2,000 | 10% | 2026 | Fit the model |
| Validation | 3,000 | 5% | 2027 | Choose review and reject thresholds |
| Test | 8,000 | 5% | 2028 | Used once; recall and FPR with Wilson 95% intervals |

Leakage check: the three sets come from independent seeds; no thresholds are set on test.
