"""Layer 2 — signal conditioning (§5).

An ordered pipeline of pure functions:

    raw (irregular, device frame)
      -> resample to uniform 100 Hz grid (+ gap flags)      §5.1
      -> rotate into gravity-aligned world frame            §5.2-5.3
      -> dual-branch filtering (kinematic + tremor)         §5.4
      -> ZUPT velocity & displacement                       §5.5
      -> quality gates                                      §5.6
    = ConditionedSet

Each stage is independently unit-testable; :func:`condition_set` wires them together.
"""

from wristset.conditioning.pipeline import ConditionedSet, condition_set

__all__ = ["ConditionedSet", "condition_set"]
