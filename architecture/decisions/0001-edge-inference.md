# ADR 0001: Run inference at the edge

**Status:** accepted

**Context.** The line moves one part every ~300 ms; a round trip to the cloud adds jitter and a network dependency on the line.

**Decision.** The model runs on the inspection cell's edge server. Features are hand-built and the model is a logistic regression, so inference is well under 1 ms in pure Python.

**Consequences.** Model updates ship as signed artifacts to the edge; monitoring data (scores, decisions) is sent upstream asynchronously.
