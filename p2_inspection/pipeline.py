"""Train, select the operating point, test against the frontier, and gate."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .decision import (ThreeWayPolicy, auto_reject_threshold, min_cost_threshold, rates_at,
                       threshold_for_fpr, wilson)
from .economics import CaseInputs, annual_error_cost, net_benefit, required_recall
from .features import extract
from .images import make_dataset
from .model import LogisticModel, RuleBaseline


@dataclass
class TestReport:
    recall: float
    fpr: float
    recall_ci: tuple[float, float]
    fpr_ci: tuple[float, float]
    required_recall_at_worst_fpr: float
    net_benefit_conservative: float
    annual_error_cost: float
    baseline_recall_at_baseline_fpr: float
    review_share: float
    auto_reject_precision: float
    latency_p99_ms: float
    n_defects: int
    n_good: int
    policy: ThreeWayPolicy

    def gate(self, latency_budget_ms: float = 20.0) -> list[str]:
        problems = []
        lb = self.recall_ci[0]
        if lb < self.required_recall_at_worst_fpr:
            problems.append(f"recall lower bound {lb:.4f} < frontier {self.required_recall_at_worst_fpr:.4f}")
        if self.net_benefit_conservative < 0:
            problems.append(f"conservative net benefit {self.net_benefit_conservative:,.0f} < 0")
        if self.latency_p99_ms > latency_budget_ms:
            problems.append(f"p99 latency {self.latency_p99_ms:.1f} ms > {latency_budget_ms} ms")
        if self.auto_reject_precision < 0.97:
            problems.append(f"auto-reject precision {self.auto_reject_precision:.3f} < 0.97")
        return problems


def run(seed: int = 2026, case: CaseInputs = CaseInputs(), n_train: int = 2000, n_val: int = 3000, n_test: int = 8000) -> TestReport:
    train = make_dataset(n_train, 0.10, seed)
    val = make_dataset(n_val, 0.05, seed + 1)
    test = make_dataset(n_test, 0.05, seed + 2)

    model = LogisticModel().fit([extract(p.image) for p in train], [int(p.defective) for p in train])

    # Operating point and reject band are chosen on validation only.
    v_scores = [model.score(p.image) for p in val]
    v_labels = [p.defective for p in val]
    review_at = min_cost_threshold(v_scores, v_labels, case)
    reject_at = max(review_at, auto_reject_threshold(v_scores, v_labels))
    policy = ThreeWayPolicy(review_at, reject_at)

    # Test once, timing each inference as the edge device would run it.
    t_scores, lat = [], []
    for p in test:
        t0 = time.perf_counter()
        t_scores.append(model.score(p.image))
        lat.append((time.perf_counter() - t0) * 1000)
    t_labels = [p.defective for p in test]
    r = rates_at(t_scores, t_labels, review_at)
    r_ci = wilson(r.tp, r.n_pos)
    f_ci = wilson(r.fp, r.n_neg)

    decisions = [policy.decide(s) for s in t_scores]
    flagged = sum(d != "pass" for d in decisions)
    rejected = [(d, l) for d, l in zip(decisions, t_labels) if d == "reject"]
    rej_prec = sum(l for _, l in rejected) / len(rejected) if rejected else 1.0

    # Baseline measured the same way, at the baseline's stated FPR.
    rule = RuleBaseline()
    b_scores = [rule.score(p.image) for p in test]
    b_t = threshold_for_fpr([rule.score(p.image) for p in val], v_labels, case.baseline_fpr)
    b = rates_at(b_scores, t_labels, b_t)

    lat.sort()
    return TestReport(
        recall=r.recall, fpr=r.fpr, recall_ci=r_ci, fpr_ci=f_ci,
        required_recall_at_worst_fpr=required_recall(f_ci[1], case),
        net_benefit_conservative=net_benefit(r_ci[0], f_ci[1], case),
        annual_error_cost=annual_error_cost(r.recall, r.fpr, case),
        baseline_recall_at_baseline_fpr=b.recall,
        review_share=(flagged - len(rejected)) / len(test),
        auto_reject_precision=rej_prec,
        latency_p99_ms=lat[int(0.99 * (len(lat) - 1))],
        n_defects=r.n_pos, n_good=r.n_neg, policy=policy,
    )
