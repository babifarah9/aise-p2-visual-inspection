# P2 Brief: Visual Quality Inspection

| Item | Value |
| --- | --- |
| Decision | Pass, reject, or send a part to human review |
| Decision owner | Quality manager, production line 2 |
| Baseline | Manual sampling plus a rule-based vision check: recall 0.90 at FPR 0.04 (book figure) |
| Baseline cost | $518,400 a year in error cost (1,000,000 parts, p = 0.005) |
| Costliest error | A missed defect: $400 each, against $8 for a false alarm |
| Best non-AI alternative | A tuned rule-based vision check; carried into Selection and knocked out there on the frontier |
| Data | Synthetic images in this repository; a public defect-image dataset for the book's exercises |
| Cost cap | K = $250,000 over the period |

The required operating point is recall 0.965 at FPR 0.025 (`python -m p2_inspection frontier`).
