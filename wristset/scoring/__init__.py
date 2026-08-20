"""Layer 5a — set score and insights (§8).

Phase 4 implements the deterministic FORM half only (§8.3-8.4): four equal-weight
subscores and their composite, each decomposing into >=1 structured pointer. The effort
half (RIR/RPE proximity), the composite weighting, the divergence flag, and narrative
assembly (§8.1-8.2, 8.5-8.6) arrive in Phase 8.
"""

from wristset.scoring.form import (
    MIN_REPS_BEYOND_REFERENCE,
    NormRef,
    FormSubscore,
    FormSubscores,
    Pointer,
    SUBSCORE_NAMES,
    score_form,
    squash,
)

__all__ = [
    "squash",
    "MIN_REPS_BEYOND_REFERENCE",
    "score_form",
    "NormRef",
    "Pointer",
    "FormSubscore",
    "FormSubscores",
    "SUBSCORE_NAMES",
]
