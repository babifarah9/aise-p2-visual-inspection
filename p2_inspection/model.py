"""Logistic regression trained by gradient descent, plus the rule-based baseline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .features import FEATURE_NAMES, extract, raw_darkest


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticModel:
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)

    def _standardise(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.means, self.stds)]

    def fit(self, X: list[list[float]], y: list[int], epochs: int = 300, lr: float = 0.5, l2: float = 1e-3, seed: int = 0) -> "LogisticModel":
        d = len(X[0])
        self.means = [sum(r[j] for r in X) / len(X) for j in range(d)]
        self.stds = [max(1e-9, (sum((r[j] - self.means[j]) ** 2 for r in X) / len(X)) ** 0.5) for j in range(d)]
        Z = [self._standardise(r) for r in X]
        # Balance the classes: defects are rare even in an enriched training set.
        pos = sum(y)
        w_pos, w_neg = len(y) / (2 * pos), len(y) / (2 * (len(y) - pos))
        rng = random.Random(seed)
        self.weights = [rng.gauss(0, 0.01) for _ in range(d)]
        self.bias = 0.0
        n = len(Z)
        for _ in range(epochs):
            gw, gb = [0.0] * d, 0.0
            for z, t in zip(Z, y):
                p = _sigmoid(sum(w * v for w, v in zip(self.weights, z)) + self.bias)
                g = (p - t) * (w_pos if t else w_neg)
                for j in range(d):
                    gw[j] += g * z[j]
                gb += g
            self.weights = [w - lr * (g / n + l2 * w) for w, g in zip(self.weights, gw)]
            self.bias -= lr * gb / n
        return self

    def score_features(self, x: list[float]) -> float:
        z = self._standardise(x)
        return _sigmoid(sum(w * v for w, v in zip(self.weights, z)) + self.bias)

    def score(self, image) -> float:
        return self.score_features(extract(image))

    def describe(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.weights))


class RuleBaseline:
    """Today's rule-based vision check: flag a part whose darkest pixel is far below the mean."""

    def score(self, image) -> float:
        return raw_darkest(image)
