"""RPE head (Layer 5c, §10): ordinal, hierarchical, proportional-odds.

* ``dataset`` — set-level design incl. whole-set retrospective features (§10.4).
* ``ordinal`` — CORAL / proportional-odds fit by scipy; ridge-penalized per-user b_u (§10.3).
* ``eval``    — ordinal accuracy, b_u-vs-bias recovery, by-user split (§10.5 verification).

The head models the user's SUBJECTIVE mapping, not the objective state — ``b_u`` is the
object of interest (§10.5), which is what keeps it non-redundant with the RIR head.
"""

from wristset.models.rpe.dataset import (
    LEVELS,
    RpeDataset,
    RpeMeta,
    build_rpe_dataset,
    feature_row,
    prepare_rpe_sets,
)
from wristset.models.rpe.eval import (
    bias_recovery,
    by_user_split,
    ordinal_accuracy_within,
)
from wristset.models.rpe.ordinal import OrdinalRpeModel

__all__ = [
    "LEVELS",
    "RpeDataset",
    "RpeMeta",
    "build_rpe_dataset",
    "feature_row",
    "prepare_rpe_sets",
    "OrdinalRpeModel",
    "by_user_split",
    "ordinal_accuracy_within",
    "bias_recovery",
]
