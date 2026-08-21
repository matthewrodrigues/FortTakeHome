"""Phase 9 — cross-phase metric harness.

``python -m wristset.eval`` re-derives every phase gate's headline number against the
generator's ground truth and prints them together. Shippable code rather than test-only
helpers, so the same figures can be quoted, tracked, or regenerated after a change.
"""

from wristset.eval.metrics import EvalReport, Metric, run_evaluation

__all__ = ["Metric", "EvalReport", "run_evaluation"]
