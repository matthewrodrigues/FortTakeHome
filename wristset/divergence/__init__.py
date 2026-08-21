"""Layer 6 — reported-vs-mechanical divergence (§11).

Compares what the lifter reported against what the movement implied, expressed as a
percentile within the RIR head's predictive distribution rather than a raw difference so
model uncertainty is built into the threshold (§11.3).
"""

from wristset.divergence.signal import (
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    Divergence,
    divergence_from_rir,
)

__all__ = ["Divergence", "divergence_from_rir", "LOW_PERCENTILE", "HIGH_PERCENTILE"]
