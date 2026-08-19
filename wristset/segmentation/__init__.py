"""Layer 3 — segmentation (§6.1).

Turns a ConditionedSet into rep boundaries. Milestone 3 is the binding gate: nothing
downstream is meaningful unless segmentation is accurate (§13). The primary metric is
exact-match rate of the detected completed-rep count against ``reported_reps`` — free
weak supervision on every set (§6.1).
"""

from wristset.segmentation.dtw import (
    build_template,
    dtw_distance,
    template_distance,
)
from wristset.segmentation.reps import (
    RepBoundary,
    SegmentationResult,
    segment_reps,
)

__all__ = [
    "RepBoundary",
    "SegmentationResult",
    "segment_reps",
    "dtw_distance",
    "build_template",
    "template_distance",
]
