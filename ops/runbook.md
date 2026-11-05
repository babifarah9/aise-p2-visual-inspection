# Runbook: P2 Inspection Cell

| Symptom | Check | Action |
| --- | --- | --- |
| Review share rises above staffing | Score distribution against last week | Check lighting and lens (see ticket 1042: condensation); recalibrate before retuning thresholds |
| Customer returns rise | Weekly audit sample of passed parts | Estimate field recall; if its lower bound is below the frontier, raise sampling and open a retraining ticket |
| Edge latency p99 above 20 ms | Edge server load and disk | Restart inference service; fail safe = route every part to review |
| Model or camera change | | Re-run `python -m p2_inspection gate` on fresh labelled data before release |
