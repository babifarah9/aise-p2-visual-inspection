import random
import unittest

from p2_inspection.decision import ThreeWayPolicy, rates_at, threshold_for_fpr, wilson
from p2_inspection.features import detrend, extract
from p2_inspection.images import SIZE, make_dataset, make_part
from p2_inspection.model import LogisticModel


class FeatureTest(unittest.TestCase):
    def test_detrend_removes_a_plane(self):
        img = [[0.3 + 0.01 * x - 0.02 * y for x in range(SIZE)] for y in range(SIZE)]
        r = detrend(img)
        self.assertLess(max(abs(v) for row in r for v in row), 1e-9)

    def test_defects_score_darker_than_good_parts(self):
        rng = random.Random(0)
        good = [extract(make_part(rng, False).image)[3] for _ in range(50)]
        bad = [extract(make_part(rng, True).image)[3] for _ in range(50)]
        self.assertGreater(sum(bad) / 50, sum(good) / 50)


class DecisionTest(unittest.TestCase):
    def test_rates(self):
        r = rates_at([0.9, 0.8, 0.2, 0.1], [True, False, True, False], 0.5)
        self.assertEqual((r.recall, r.fpr), (0.5, 0.5))

    def test_threshold_for_fpr(self):
        scores = [i / 100 for i in range(100)]
        labels = [False] * 100
        t = threshold_for_fpr(scores, labels, 0.05)
        self.assertLessEqual(rates_at(scores + [1.0], labels + [True], t).fpr, 0.05)

    def test_wilson_contains_point(self):
        lo, hi = wilson(390, 400)
        self.assertLess(lo, 390 / 400)
        self.assertGreater(hi, 390 / 400)

    def test_three_way(self):
        p = ThreeWayPolicy(0.4, 0.8)
        self.assertEqual([p.decide(s) for s in (0.1, 0.5, 0.9)], ["pass", "review", "reject"])


class ModelTest(unittest.TestCase):
    def test_model_beats_chance(self):
        train = make_dataset(400, 0.2, 1)
        test = make_dataset(400, 0.2, 2)
        m = LogisticModel().fit([extract(p.image) for p in train], [int(p.defective) for p in train], epochs=100)
        scores = [m.score(p.image) for p in test]
        pos = [s for s, p in zip(scores, test) if p.defective]
        neg = [s for s, p in zip(scores, test) if not p.defective]
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
        self.assertGreater(auc, 0.9)


if __name__ == "__main__":
    unittest.main()
