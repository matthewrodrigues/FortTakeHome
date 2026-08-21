"""``python -m wristset.eval`` — print every phase gate's metric in one table."""

from __future__ import annotations

import argparse
import sys

from wristset.eval.metrics import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wristset.eval",
        description="Cross-phase metric harness: every gate number, synthetic-validated.",
    )
    parser.add_argument("--quick", action="store_true",
                        help="use one corpus seed for the model heads instead of three")
    args = parser.parse_args(argv)

    report = run_evaluation(quick=args.quick)

    # ASCII-only: the harness runs on stock Windows terminals (cp1252).
    print("wristset evaluation - all figures are SYNTHETIC-VALIDATED")
    print("(measured against the generator's own ground truth, not real lifting)\n")
    print(report.to_table())
    print()
    print(f"{'ALL GATES PASS' if report.all_passed else 'SOME GATES FAILED'} "
          f"({report.elapsed_s:.0f}s)")
    return 0 if report.all_passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m wristset.eval`
    sys.exit(main())
