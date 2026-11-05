"""The P2 business case (Chapter 4, Section 4.2).

    Net = V [ p (r - r0) C_FN + (1 - p) (f0 - f) C_FP ] - K

Setting Net = 0 and solving for r gives the required performance frontier:
the lowest recall the system can have at each false positive rate and still
break even.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CaseInputs:
    volume: float = 1_000_000      # V, parts inspected per year
    prevalence: float = 0.005      # p, share of parts that are defective
    cost_fn: float = 400.0         # C_FN, cost of a missed defect ($)
    cost_fp: float = 8.0           # C_FP, cost of a false alarm ($)
    project_cost: float = 250_000  # K, cost cap over the period ($)
    baseline_recall: float = 0.90  # r0
    baseline_fpr: float = 0.04     # f0
    adoption: float = 1.0          # share of volume that actually runs through the system
    years: float = 1.0             # years of benefit counted against K


def net_benefit(r: float, f: float, c: CaseInputs = CaseInputs()) -> float:
    per_part = c.prevalence * (r - c.baseline_recall) * c.cost_fn + (1 - c.prevalence) * (c.baseline_fpr - f) * c.cost_fp
    return c.years * c.adoption * c.volume * per_part - c.project_cost


def required_recall(f: float, c: CaseInputs = CaseInputs()) -> float:
    """Recall at which Net = 0 for a given false positive rate."""
    scale = c.years * c.adoption * c.volume
    fp_gain = (1 - c.prevalence) * (c.baseline_fpr - f) * c.cost_fp
    return c.baseline_recall + (c.project_cost / scale - fp_gain) / (c.prevalence * c.cost_fn)


def frontier(fprs=(0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040), c: CaseInputs = CaseInputs()) -> list[tuple[float, float]]:
    return [(f, required_recall(f, c)) for f in fprs]


def annual_error_cost(r: float, f: float, c: CaseInputs = CaseInputs()) -> float:
    return c.volume * (c.prevalence * (1 - r) * c.cost_fn + (1 - c.prevalence) * f * c.cost_fp)


@dataclass(frozen=True)
class Ranges:
    volume: tuple[float, float] = (800_000, 1_200_000)
    prevalence: tuple[float, float] = (0.003, 0.008)
    cost_fn: tuple[float, float] = (200.0, 700.0)
    project_cost: tuple[float, float] = (200_000, 400_000)
    adoption: tuple[float, float] = (0.6, 1.0)


@dataclass
class MonteCarloResult:
    p_positive: float
    median: float
    p05: float
    p95: float
    drivers: dict[str, float]  # Spearman rank correlation of each input with Net


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def _spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def monte_carlo(r: float, f: float, base: CaseInputs = CaseInputs(), ranges: Ranges = Ranges(),
                n: int = 100_000, seed: int = 7, years: float = 1.0) -> MonteCarloResult:
    """Draw each uncertain input uniformly from its range and evaluate Net."""
    rng = random.Random(seed)
    samples = {k: [] for k in ("volume", "prevalence", "cost_fn", "project_cost", "adoption")}
    nets = []
    for _ in range(n):
        draw = {k: rng.uniform(*getattr(ranges, k)) for k in samples}
        for k, v in draw.items():
            samples[k].append(v)
        nets.append(net_benefit(r, f, replace(base, years=years, **draw)))
    q = statistics.quantiles(nets, n=20)
    return MonteCarloResult(
        p_positive=sum(x > 0 for x in nets) / n,
        median=statistics.median(nets),
        p05=q[0],
        p95=q[-1],
        drivers={k: _spearman(v, nets) for k, v in samples.items()},
    )
