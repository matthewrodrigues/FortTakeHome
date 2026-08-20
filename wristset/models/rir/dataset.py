"""Person-period expansion for the discrete-time RIR hazard (§9.3) — Layer 5b.

§9.3's implementation shortcut: a discrete-time hazard model is fit as ordinary logistic
regression on a "person-period" table — one row per attempted rep, binary outcome "was this
the failure rep". A failure set contributes ``R-1`` zero rows and one positive row; a
censored set contributes all zeros. Summing the row-wise Bernoulli log-likelihood over that
table *is* the §9.3 censored likelihood (the product form written additively).

**Only causal features may enter (§9.2).** Every column here comes from either the per-rep
``RepFeatures`` (a rep's own values) or the ``SetFeatures.causal[r]`` summaries (reps 1..r).
No whole-set / ``retro_`` summary is ever read — a test enforces this. Feeding a
retrospective feature here would leak the future into a prediction about the future and
silently inflate the C-index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wristset.conditioning import condition_set
from wristset.features import SetFeatures, extract_set_features
from wristset.segmentation import segment_reps
from wristset.synth import GeneratedSet

__all__ = ["RirMeta", "PersonPeriod", "feature_row", "build_person_period", "prepare_sets"]

#: Per-rep causal driver columns (§9.2). Order is fixed so design matrices align.
NUMERIC_COLUMNS: tuple[str, ...] = (
    "rep_index",
    "load_kg",
    "conc_mean_vel",           # conc_mean_vel(r)
    "path_dtw_baseline",       # path_dtw_baseline(r)
    "vel_loss_to_date",        # causal_conc_mean_vel_total_change @ r
    "ecc_change_to_date",      # causal_ecc_duration_total_change @ r
    "tremor_change_to_date",   # causal_tremor_power_8_12_total_change @ r
)


@dataclass
class RirMeta:
    """The labels/grouping a person-period row needs beyond its features."""

    set_id: str
    user_id: str
    exercise: str
    reached_failure: bool
    load_kg: float


@dataclass
class PersonPeriod:
    """A fitted-ready person-period table (§9.3)."""

    X: np.ndarray                 # (n_rows, n_numeric) raw causal features, NUMERIC_COLUMNS order
    y: np.ndarray                 # (n_rows,) 0/1 failure outcome
    user_idx: np.ndarray          # (n_rows,) index into `users`
    exercise: np.ndarray          # (n_rows,) exercise label per row
    set_id: np.ndarray            # (n_rows,) grouping id per row (for eval)
    rep_index: np.ndarray         # (n_rows,) 1-based rep index per row
    columns: tuple[str, ...] = NUMERIC_COLUMNS
    users: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]


def feature_row(sf: SetFeatures, rep) -> dict[str, float]:
    """The causal §9.2 feature vector for one rep of one set.

    ``None`` "to-date change" summaries (rep 1-2, before a trend is defined) map to 0.0 —
    "no change yet" by construction, not missing data. ``path_dtw_baseline`` is 0.0 under
    cold start (no template), matching the Phase-3 "None = no reference" handling.
    """
    causal = sf.causal.get(rep.rep_index, {})

    def c(key: str) -> float:
        v = causal.get(key)
        return float(v) if v is not None else 0.0

    return {
        "rep_index": float(rep.rep_index),
        "load_kg": float(rep.load_kg),
        "conc_mean_vel": float(rep.conc_mean_vel),
        "path_dtw_baseline": float(rep.path_dtw_baseline or 0.0),
        "vel_loss_to_date": c("causal_conc_mean_vel_total_change"),
        "ecc_change_to_date": c("causal_ecc_duration_total_change"),
        "tremor_change_to_date": c("causal_tremor_power_8_12_total_change"),
    }


def build_person_period(labeled: list[tuple[SetFeatures, RirMeta]]) -> PersonPeriod:
    """Expand scored sets into the §9.3 person-period table.

    Outcome rule: ``y=1`` on the failed attempt — the LAST detected rep of a set whose
    ``reached_failure`` is true — and 0 everywhere else. Censored sets are all zeros. Using
    the last *detected* rep (rather than the ground-truth index) keeps labels aligned with
    the same segmentation the features are drawn from.
    """
    rows: list[list[float]] = []
    y: list[int] = []
    user_of_row: list[str] = []
    exercise_of_row: list[str] = []
    set_of_row: list[str] = []
    rep_of_row: list[int] = []

    for sf, meta in labeled:
        if not sf.reps:
            continue
        last_idx = max(r.rep_index for r in sf.reps)
        for rep in sf.reps:
            fr = feature_row(sf, rep)
            rows.append([fr[c] for c in NUMERIC_COLUMNS])
            y.append(1 if (meta.reached_failure and rep.rep_index == last_idx) else 0)
            user_of_row.append(meta.user_id)
            exercise_of_row.append(meta.exercise)
            set_of_row.append(meta.set_id)
            rep_of_row.append(rep.rep_index)

    users = sorted(set(user_of_row))
    user_to_i = {u: i for i, u in enumerate(users)}
    return PersonPeriod(
        X=np.asarray(rows, dtype=np.float64).reshape(-1, len(NUMERIC_COLUMNS)),
        y=np.asarray(y, dtype=np.float64),
        user_idx=np.asarray([user_to_i[u] for u in user_of_row], dtype=np.int64),
        exercise=np.asarray(exercise_of_row, dtype=object),
        set_id=np.asarray(set_of_row, dtype=object),
        rep_index=np.asarray(rep_of_row, dtype=np.int64),
        users=users,
    )


def prepare_sets(
    generated: list[GeneratedSet],
    *,
    history=None,
    warmups=None,
) -> list[tuple[SetFeatures, RirMeta]]:
    """Run the Layer 2-4 pipeline (causal on) over raw synthetic sets and pair each with
    its ``RirMeta``. This is the corpus-prep step the dataset builder and eval consume."""
    out: list[tuple[SetFeatures, RirMeta]] = []
    for g in generated:
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        result = segment_reps(cs, exercise=g.exercise, reached_failure=g.reached_failure)
        sf = extract_set_features(
            cs, result.reps, exercise=g.exercise, load_kg=g.load_kg,
            set_type=g.set_type, history=history, warmups=warmups, causal=True,
        )
        meta = RirMeta(
            set_id=g.set_id, user_id=g.user_id, exercise=g.exercise,
            reached_failure=g.reached_failure, load_kg=g.load_kg,
        )
        out.append((sf, meta))
    return out
