"""Set-level design for the ordinal RPE head (§10.4) — Layer 5c.

The RPE head is a **set-level** model — one reported RPE per set — so its features are
whole-set quantities. Crucially, whole-set **retrospective** summaries are legitimate here
(§10.4): the user reports RPE *after* the set, so the model may use the full set. This is
the deliberate mirror image of the RIR head's causal-only rule (§9.2) — a test asserts the
retrospective features ARE present.

The head's target is the reported RPE on the {6, 6.5, ..., 10} half-point grid; the
generator's per-user ``rpe_bias`` is carried in ``RpeMeta`` for evaluation only (§10.5's
b_u-recovery check) and is never a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wristset.features import SetFeatures
from wristset.models.rir.dataset import prepare_sets
from wristset.synth import GeneratedSet

__all__ = ["LEVELS", "RpeMeta", "RpeDataset", "feature_row", "build_rpe_dataset",
           "prepare_rpe_sets"]

#: The RPE scale (§10.2): 9 ordinal levels, 8 thresholds.
LEVELS: np.ndarray = np.arange(6.0, 10.0 + 0.5, 0.5)

#: Whole-set retrospective channels used as effort features (§10.4). ``total_change`` of
#: each is the degradation magnitude across the set — the strongest set-level effort signal.
_RETRO_CHANNELS: tuple[str, ...] = (
    "conc_mean_vel", "rom_vertical", "ecc_duration", "tremor_power_8_12",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "reported_reps",
    "load_kg",
    *(f"retro_{ch}_total_change" for ch in _RETRO_CHANNELS),
)


@dataclass
class RpeMeta:
    """Labels and grouping for one set. ``true_rpe_bias`` is eval-only (§10.5), never a
    feature."""

    set_id: str
    user_id: str
    exercise: str
    load_kg: float
    reported_reps: int
    reported_rpe: float
    true_rpe_bias: float


@dataclass
class RpeDataset:
    """A fit-ready set-level ordinal design (§10)."""

    X: np.ndarray                 # (n_sets, n_numeric) raw features, NUMERIC_COLUMNS order
    y: np.ndarray                 # (n_sets,) ordinal level index into LEVELS
    user_idx: np.ndarray          # (n_sets,) index into `users`
    exercise: np.ndarray          # (n_sets,) exercise label
    set_id: np.ndarray            # (n_sets,) grouping id
    columns: tuple[str, ...] = NUMERIC_COLUMNS
    levels: np.ndarray = field(default_factory=lambda: LEVELS.copy())
    users: list[str] = field(default_factory=list)

    @property
    def n_sets(self) -> int:
        return self.X.shape[0]


def feature_row(sf: SetFeatures, meta: RpeMeta) -> dict[str, float]:
    """The §10.4 set-level feature vector for one set. ``None`` retrospective summaries
    (a set too short to have a trend) map to 0.0 — no measured degradation."""
    retro = sf.retrospective

    def r(ch: str) -> float:
        v = retro.get(f"retro_{ch}_total_change")
        return float(v) if v is not None else 0.0

    row = {"reported_reps": float(meta.reported_reps), "load_kg": float(meta.load_kg)}
    for ch in _RETRO_CHANNELS:
        row[f"retro_{ch}_total_change"] = r(ch)
    return row


def _level_index(rpe: float) -> int:
    return int(np.argmin(np.abs(LEVELS - rpe)))


def build_rpe_dataset(labeled: list[tuple[SetFeatures, RpeMeta]]) -> RpeDataset:
    """Assemble the set-level ordinal design from scored sets + their RPE labels."""
    rows, y, users_of, ex_of, set_of = [], [], [], [], []
    for sf, meta in labeled:
        fr = feature_row(sf, meta)
        rows.append([fr[c] for c in NUMERIC_COLUMNS])
        y.append(_level_index(meta.reported_rpe))
        users_of.append(meta.user_id)
        ex_of.append(meta.exercise)
        set_of.append(meta.set_id)

    users = sorted(set(users_of))
    to_i = {u: i for i, u in enumerate(users)}
    return RpeDataset(
        X=np.asarray(rows, dtype=np.float64).reshape(-1, len(NUMERIC_COLUMNS)),
        y=np.asarray(y, dtype=np.int64),
        user_idx=np.asarray([to_i[u] for u in users_of], dtype=np.int64),
        exercise=np.asarray(ex_of, dtype=object),
        set_id=np.asarray(set_of, dtype=object),
        users=users,
    )


def prepare_rpe_sets(generated: list[GeneratedSet]) -> list[tuple[SetFeatures, RpeMeta]]:
    """Run the Layer 2-4 pipeline and pair each set with its ``RpeMeta``.

    Reuses ``rir.prepare_sets`` for the raw->features wiring (one pipeline definition), then
    reads the RPE label and the injected bias off each set's ``ground_truth``.
    """
    prepared = prepare_sets(generated)  # [(SetFeatures, RirMeta)], same order as `generated`
    out: list[tuple[SetFeatures, RpeMeta]] = []
    for g, (sf, _) in zip(generated, prepared):
        gt = g.ground_truth
        out.append((sf, RpeMeta(
            set_id=g.set_id, user_id=g.user_id, exercise=g.exercise, load_kg=g.load_kg,
            reported_reps=g.reported_reps, reported_rpe=gt.reported_rpe,
            true_rpe_bias=gt.rpe_bias,
        )))
    return out
