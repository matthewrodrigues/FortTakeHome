"""Raw-timeseries and relational-metadata schemas + controlled vocabularies.

Raw channels (§3.2). One row per delivered IMU sample, one Parquet file per set:

    t_ns        int64   device-monotonic timestamp (nanoseconds) — NOT wall clock
    lin_acc_*   float64 linear (gravity-removed) acceleration, device frame, m/s^2
    rot_rate_*  float64 gyroscope rotation rate, device frame, rad/s
    quat_*      float64 attitude quaternion (w,x,y,z), device->world
    grav_*      float64 gravity vector, device frame, m/s^2

Note the channels are stored in the **device** frame exactly as CoreMotion
``CMDeviceMotion`` delivers them; world-frame rotation is a pipeline step (§5.3),
not a storage concern. Storing device-frame + quaternion (rather than pre-rotated
signals) keeps the raw layer immutable and lets the rotation be re-derived/debugged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

# --- controlled vocabularies (§1.3, §3.5) ---------------------------------------

#: Prototype exercise vocabulary. Only bench + squat are active (§1.3); the rest are
#: declared so the enum is stable when they are enabled later.
EXERCISES: tuple[str, ...] = (
    "bench_press",
    "back_squat",
    "deadlift",
    "overhead_press",
    "barbell_row",
    "dumbbell_curl",
)

#: Active subset for the prototype.
ACTIVE_EXERCISES: tuple[str, ...] = ("bench_press", "back_squat")

SET_TYPES: tuple[str, ...] = ("warmup", "working", "backoff")

WRISTS: tuple[str, ...] = ("left", "right")

#: Nominal capture rate (§3.3). Storage does not assume uniform spacing — t_ns is the
#: source of truth — but this is the grid the conditioning layer resamples onto (§5.1).
NOMINAL_HZ: int = 100

# --- raw timeseries schema (§3.2) -----------------------------------------------

RAW_SCHEMA: dict[str, pl.DataType] = {
    "t_ns": pl.Int64,
    "lin_acc_x": pl.Float64,
    "lin_acc_y": pl.Float64,
    "lin_acc_z": pl.Float64,
    "rot_rate_x": pl.Float64,
    "rot_rate_y": pl.Float64,
    "rot_rate_z": pl.Float64,
    "quat_w": pl.Float64,
    "quat_x": pl.Float64,
    "quat_y": pl.Float64,
    "quat_z": pl.Float64,
    "grav_x": pl.Float64,
    "grav_y": pl.Float64,
    "grav_z": pl.Float64,
}

RAW_COLUMNS: tuple[str, ...] = tuple(RAW_SCHEMA.keys())


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a raw-set DataFrame against the contract. Returns it unchanged on success.

    Checks columns, dtypes, non-emptiness, and that timestamps are strictly increasing
    (device-monotonic, §3.2). Raises ``ValueError`` on any violation — a corrupt raw
    frame must never enter storage silently.
    """
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"raw frame missing columns: {missing}")
    extra = [c for c in df.columns if c not in RAW_SCHEMA]
    if extra:
        raise ValueError(f"raw frame has unexpected columns: {extra}")
    for col, dtype in RAW_SCHEMA.items():
        if df.schema[col] != dtype:
            raise ValueError(f"column {col!r} has dtype {df.schema[col]}, expected {dtype}")
    if df.height == 0:
        raise ValueError("raw frame is empty")
    t = df["t_ns"].to_numpy()
    if (t[1:] <= t[:-1]).any():
        raise ValueError("t_ns must be strictly increasing (device-monotonic)")
    # reorder to the canonical column order
    return df.select(list(RAW_COLUMNS))


# --- relational metadata schema (§4.2) ------------------------------------------

SQLITE_DDL: str = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    device_model TEXT,
    wrist        TEXT CHECK(wrist IN ('left','right'))
);

CREATE TABLE IF NOT EXISTS sets (
    set_id          TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(session_id),
    set_index       INTEGER NOT NULL,
    exercise        TEXT NOT NULL,
    load_kg         REAL NOT NULL,
    set_type        TEXT NOT NULL,
    reported_reps   INTEGER NOT NULL,
    reported_rpe    REAL,
    reached_failure BOOLEAN NOT NULL,
    raw_path        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reps (
    rep_id      TEXT PRIMARY KEY,
    set_id      TEXT REFERENCES sets(set_id),
    rep_index   INTEGER NOT NULL,      -- 1-based
    t_start     REAL NOT NULL,         -- seconds from set start
    t_bottom    REAL NOT NULL,         -- eccentric/concentric transition
    t_end       REAL NOT NULL,
    completed   BOOLEAN NOT NULL       -- false for a failed attempt
);

CREATE TABLE IF NOT EXISTS rep_features (
    rep_id      TEXT PRIMARY KEY REFERENCES reps(rep_id),
    features    TEXT NOT NULL          -- JSON blob during prototyping (§4.2)
);
"""


# --- python-side metadata records -----------------------------------------------


@dataclass(frozen=True)
class SessionMetadata:
    """A training session (§4.2 ``sessions``)."""

    session_id: str
    user_id: str
    started_at: str  # ISO-8601
    device_model: str | None = None
    wrist: str = "right"

    def __post_init__(self) -> None:
        if self.wrist not in WRISTS:
            raise ValueError(f"wrist must be one of {WRISTS}, got {self.wrist!r}")


@dataclass(frozen=True)
class SetMetadata:
    """User-supplied per-set metadata (§3.5, §4.2 ``sets``).

    ``reached_failure`` is the single most valuable label in the system (§3.5): it,
    not RPE, determines censoring for the RIR head. It must be a deliberate label.
    """

    set_id: str
    session_id: str
    set_index: int
    exercise: str
    load_kg: float
    set_type: str
    reported_reps: int
    reached_failure: bool
    raw_path: str
    reported_rpe: float | None = None

    def __post_init__(self) -> None:
        if self.exercise not in EXERCISES:
            raise ValueError(f"exercise must be one of {EXERCISES}, got {self.exercise!r}")
        if self.set_type not in SET_TYPES:
            raise ValueError(f"set_type must be one of {SET_TYPES}, got {self.set_type!r}")
        if self.reported_rpe is not None and not (6.0 <= self.reported_rpe <= 10.0):
            raise ValueError(f"reported_rpe must be in [6.0, 10.0], got {self.reported_rpe}")


# --- ground-truth labels (synthetic only; never present for real captures) ------


@dataclass(frozen=True)
class RepTruth:
    """Ground-truth per-rep kinematics emitted by the synthetic generator.

    These are what the pipeline is *trying* to recover; they are stored alongside
    synthetic sets so phase gates can score reconstruction error against known truth.
    """

    rep_index: int
    t_start: float
    t_bottom: float
    t_end: float
    completed: bool
    rom_true_m: float  # world-frame vertical peak-to-peak displacement (m)
    conc_mean_vel_true: float  # m/s
    conc_peak_vel_true: float  # m/s


@dataclass(frozen=True)
class GroundTruth:
    """Everything the synthetic generator knows that a real capture never would.

    ``failure_rep`` is None for a censored (non-failure) set. ``rpe_bias`` is the
    virtual user's systematic over/under-reporting term — the object the RPE head's
    random effect ``b_u`` (§10.3) is meant to recover.
    """

    set_id: str
    exercise: str
    reached_failure: bool
    failure_rep: int | None
    reported_rpe: float
    mechanical_rpe_true: float  # 10 - true_RIR_at_stop
    rpe_bias: float  # reported - mechanical, the user's calibration offset
    reps: list[RepTruth] = field(default_factory=list)
