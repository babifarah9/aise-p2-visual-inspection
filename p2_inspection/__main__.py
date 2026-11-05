"""Command line: `python -m p2_inspection frontier|case|test|gate`."""

from __future__ import annotations

import argparse
import sys

from .economics import CaseInputs, annual_error_cost, frontier, monte_carlo, net_benefit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="p2_inspection", description="P2 visual quality inspection")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frontier", help="print the required performance frontier")
    c = sub.add_parser("case", help="Monte Carlo business case")
    c.add_argument("--recall", type=float, default=0.95)
    c.add_argument("--fpr", type=float, default=0.025)
    c.add_argument("--n", type=int, default=100_000)
    for name in ("test", "gate"):
        s = sub.add_parser(name, help="train, test against the frontier" + (" and fail if the gate is not met" if name == "gate" else ""))
        s.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    case = CaseInputs()

    if args.cmd == "frontier":
        print("FPR     required recall")
        for f, r in frontier(c=case):
            print(f"{f:.3f}   {r:.4f}")
        print(f"\nBaseline annual error cost (r0={case.baseline_recall}, f0={case.baseline_fpr}): "
              f"${annual_error_cost(case.baseline_recall, case.baseline_fpr, case):,.0f}")
        return 0

    if args.cmd == "case":
        print(f"Point estimate at r={args.recall}, f={args.fpr}: Net = ${net_benefit(args.recall, args.fpr, case):,.0f}")
        for years in (1, 2, 3):
            mc = monte_carlo(args.recall, args.fpr, case, n=args.n, years=years)
            print(f"{years} year(s): P(Net > 0) = {mc.p_positive:.1%}, median ${mc.median:,.0f}, "
                  f"90% interval ${mc.p05:,.0f} to ${mc.p95:,.0f}")
            if years == 1:
                drivers = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(mc.drivers.items(), key=lambda kv: -abs(kv[1])))
                print(f"  drivers (rank correlation with Net): {drivers}")
        return 0

    from .pipeline import run
    rep = run(seed=args.seed, case=case)
    print(f"Test set: {rep.n_defects} defective, {rep.n_good} good parts (defect-enriched; costs use p={case.prevalence})")
    print(f"Recall {rep.recall:.4f} (95% CI {rep.recall_ci[0]:.4f} to {rep.recall_ci[1]:.4f})")
    print(f"FPR    {rep.fpr:.4f} (95% CI {rep.fpr_ci[0]:.4f} to {rep.fpr_ci[1]:.4f})")
    print(f"Frontier at worst-case FPR: recall >= {rep.required_recall_at_worst_fpr:.4f}")
    print(f"Conservative net benefit (recall low, FPR high): ${rep.net_benefit_conservative:,.0f}")
    print(f"Annual error cost at point estimate: ${rep.annual_error_cost:,.0f}")
    print(f"Rule baseline recall at FPR {case.baseline_fpr}: {rep.baseline_recall_at_baseline_fpr:.4f}")
    print(f"Policy: review at score >= {rep.policy.review_at:.3f}, auto-reject at >= {rep.policy.reject_at:.3f}; "
          f"{rep.review_share:.1%} of parts go to a human; auto-reject precision {rep.auto_reject_precision:.3f}")
    print(f"p99 inference latency: {rep.latency_p99_ms:.2f} ms")
    if args.cmd == "gate":
        problems = rep.gate()
        if problems:
            print("GATE FAILED: " + "; ".join(problems))
            return 1
        print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
