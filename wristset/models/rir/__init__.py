"""RIR head (Layer 5b, §9): discrete-time survival hazard -> reps-in-reserve distribution.

* ``dataset`` — person-period expansion of causal features (§9.3).
* ``hazard``  — scipy-fit logistic hazard with a ridge-penalized per-user intercept (§9.2).
* ``rir``     — hazard -> P(RIR=k) via linear forward-projection, K capped at 8 (§9.4).
* ``eval``    — C-index, calibration, by-user split (§9.6).
"""

from wristset.models.rir.dataset import (
    PersonPeriod,
    RirMeta,
    build_person_period,
    feature_row,
    prepare_sets,
)
from wristset.models.rir.eval import (
    EXPECTED_SIGNS,
    Calibration,
    Collinearity,
    by_user_split,
    c_index,
    calibration,
    collinearity_report,
    completed_rep_c_index,
)
from wristset.models.rir.hazard import HazardModel
from wristset.models.rir.readiness import (
    MIN_FAILURE_SETS,
    TARGET_FAILURE_SETS,
    RirReadiness,
    assess_rir_readiness,
)
from wristset.models.rir.rir import (
    CONFIDENT_HORIZON,
    RIRPrediction,
    rir_distribution,
)

__all__ = [
    "PersonPeriod",
    "RirMeta",
    "build_person_period",
    "feature_row",
    "prepare_sets",
    "HazardModel",
    "RirReadiness",
    "assess_rir_readiness",
    "MIN_FAILURE_SETS",
    "TARGET_FAILURE_SETS",
    "RIRPrediction",
    "rir_distribution",
    "CONFIDENT_HORIZON",
    "by_user_split",
    "c_index",
    "completed_rep_c_index",
    "calibration",
    "Calibration",
    "Collinearity",
    "collinearity_report",
    "EXPECTED_SIGNS",
]
