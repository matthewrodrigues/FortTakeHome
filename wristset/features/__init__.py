"""Layer 4 — feature extraction (§6.2-6.3) and Layer 7 — baselines (§7).

    ConditionedSet + RepBoundary[]
      -> per-rep features            §6.2   rep_features.py
      -> baseline resolution         §7     baseline.py
      -> path_dtw_baseline filled    §6.2
      -> set-level summaries         §6.3   trajectory.py
    = SetFeatures

:func:`extract_set_features` wires these in the required order — the baseline must resolve
before ``path_dtw_baseline`` exists, and that channel is one of the five §6.3 summarises.
"""

from wristset.features.baseline import (
    Baseline,
    BaselineSource,
    RepSource,
    attach_path_dtw_baseline,
    resolve_baseline,
)
from wristset.features.pipeline import SetFeatures, extract_set_features
from wristset.features.rep_features import (
    FEATURE_NAMES,
    OPTIONAL_FEATURE_NAMES,
    RepFeatures,
    extract_rep_features,
)
from wristset.features.trajectory import (
    KEY_CHANNELS,
    SUMMARY_NAMES,
    causal_summaries,
    retrospective_summaries,
    summarize_channel,
)

__all__ = [
    "Baseline",
    "BaselineSource",
    "RepSource",
    "resolve_baseline",
    "attach_path_dtw_baseline",
    "RepFeatures",
    "extract_rep_features",
    "FEATURE_NAMES",
    "OPTIONAL_FEATURE_NAMES",
    "KEY_CHANNELS",
    "SUMMARY_NAMES",
    "summarize_channel",
    "retrospective_summaries",
    "causal_summaries",
    "SetFeatures",
    "extract_set_features",
]
