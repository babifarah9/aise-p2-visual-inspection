import unittest

from p2_inspection.economics import CaseInputs, annual_error_cost, monte_carlo, net_benefit, required_recall


class FrontierTest(unittest.TestCase):
    def test_book_operating_point(self):
        # Chapter 4: at FPR 0.025 the system must reach recall 0.965.
        self.assertAlmostEqual(required_recall(0.025), 0.9653, places=4)

    def test_frontier_is_break_even(self):
        for f in (0.01, 0.02, 0.03):
            self.assertAlmostEqual(net_benefit(required_recall(f), f), 0.0, places=6)

    def test_frontier_slope(self):
        # Slope = (1 - p) C_FP / (p C_FN) = 3.98: each point of FPR costs about
        # four points of recall (two points per half point of FPR).
        slope = (required_recall(0.03) - required_recall(0.02)) / 0.01
        self.assertAlmostEqual(slope, 3.98, places=2)

    def test_baseline_error_cost(self):
        self.assertAlmostEqual(annual_error_cost(0.90, 0.04), 518_400, places=0)

    def test_longer_horizon_lowers_the_bar(self):
        self.assertLess(required_recall(0.025, CaseInputs(years=3)), required_recall(0.025))


class MonteCarloTest(unittest.TestCase):
    def test_one_year_case_is_unlikely_to_pay(self):
        mc = monte_carlo(0.95, 0.025, n=20_000)
        self.assertLess(mc.p_positive, 0.15)
        self.assertLess(mc.median, 0)
        # Project cost is the largest driver: cost control is the first lever.
        top = max(mc.drivers, key=lambda k: abs(mc.drivers[k]))
        self.assertEqual(top, "project_cost")

    def test_three_years_usually_pays(self):
        self.assertGreater(monte_carlo(0.95, 0.025, n=20_000, years=3).p_positive, 0.9)

    def test_reproducible(self):
        a = monte_carlo(0.95, 0.025, n=2_000, seed=1)
        b = monte_carlo(0.95, 0.025, n=2_000, seed=1)
        self.assertEqual(a.median, b.median)


if __name__ == "__main__":
    unittest.main()
