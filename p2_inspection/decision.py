"""Operating points and the three-way pass / review / reject policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .economics import CaseInputs, annual_error_cost


@dataclass(frozen=True)
class Rates:
    recall: float
    fpr: float
    n_pos: int
    n_neg: int
    tp: int
    fp: int


def rates_at(scores: list[float], labels: list[bool], threshold: float) -> Rates:
    tp = sum(s >= threshold and l for s, l in zip(scores, labels))
    fp = sum(s >= threshold and not l for s, l in zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return Rates(tp / n_pos, fp / n_neg, n_pos, n_neg, tp, fp)


def threshold_for_fpr(scores: list[float], labels: list[bool], target_fpr: float) -> float:
    neg = sorted((s for s, l in zip(scores, labels) if not l), reverse=True)
    k = int(math.floor(target_fpr * len(neg)))
    return neg[k] + 1e-12 if k < len(neg) else min(neg)


def min_cost_threshold(scores: list[float], labels: list[bool], case: CaseInputs = CaseInputs()) -> float:
    """Threshold that minimises annual error cost at the plant's real prevalence."""
    best_t, best_c = None, float("inf")
    for t in sorted(set(scores)):
        r = rates_at(scores, labels, t)
        c = annual_error_cost(r.recall, r.fpr, case)
        if c < best_c:
            best_t, best_c = t, c
    return best_t


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


@dataclass(frozen=True)
class ThreeWayPolicy:
    """score < review_at -> pass; review_at <= score < reject_at -> human review; score >= reject_at -> reject."""

    review_at: float
    reject_at: float

    def decide(self, score: float) -> str:
        if score >= self.reject_at:
            return "reject"
        if score >= self.review_at:
            return "review"
        return "pass"


def auto_reject_threshold(scores: list[float], labels: list[bool], min_precision: float = 0.97, floor: float = 0.5) -> float:
    """Lowest threshold at which the Wilson lower bound on auto-reject precision stays above `min_precision`.

    Using the lower bound, not the point estimate, keeps the band honest on a
    small validation set: a threshold that looks 99% precise on 150 parts may
    not be on the next 150.
    """
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    best = 1.0 + 1e-9
    for s, l in pairs:
        tp += l
        fp += not l
        if s >= floor and wilson(tp, tp + fp)[0] >= min_precision:
            best = s
    return best
