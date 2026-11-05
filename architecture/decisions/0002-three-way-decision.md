# ADR 0002: Pass / review / reject, not pass / reject

**Status:** accepted

**Decision.** Scores between the review and auto-reject thresholds go to a human. The auto-reject threshold is set where the Wilson lower bound of precision on validation is ≥ 0.97.

**Consequences.** Recall is measured on everything flagged (review or reject). Human review load is reported as `review_share` and must fit the line's staffing.
