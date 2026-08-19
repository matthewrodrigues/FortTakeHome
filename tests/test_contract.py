"""Contract validation guards — a corrupt raw frame must never enter storage."""

from __future__ import annotations

import polars as pl
import pytest

from wristset.contract import RAW_SCHEMA, SetMetadata, validate_raw
from wristset.contract.schema import SessionMetadata


def _good_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {k: [0.0, 1.0, 2.0] if k != "t_ns" else [0, 10, 20] for k in RAW_SCHEMA},
        schema=dict(RAW_SCHEMA),
    )


def test_validate_raw_accepts_good_frame():
    df = validate_raw(_good_frame())
    assert list(df.columns) == list(RAW_SCHEMA)


def test_validate_raw_rejects_missing_column():
    df = _good_frame().drop("grav_z")
    with pytest.raises(ValueError, match="missing columns"):
        validate_raw(df)


def test_validate_raw_rejects_non_monotonic_time():
    df = _good_frame().with_columns(pl.Series("t_ns", [0, 10, 5]))
    with pytest.raises(ValueError, match="monotonic"):
        validate_raw(df)


def test_validate_raw_rejects_empty():
    empty = pl.DataFrame({k: [] for k in RAW_SCHEMA}, schema=dict(RAW_SCHEMA))
    with pytest.raises(ValueError, match="empty"):
        validate_raw(empty)


def test_set_metadata_rejects_bad_exercise():
    with pytest.raises(ValueError, match="exercise"):
        SetMetadata(
            set_id="s", session_id="x", set_index=1, exercise="bogus", load_kg=60,
            set_type="working", reported_reps=5, reached_failure=True, raw_path="p",
        )


def test_set_metadata_rejects_rpe_out_of_range():
    with pytest.raises(ValueError, match="reported_rpe"):
        SetMetadata(
            set_id="s", session_id="x", set_index=1, exercise="bench_press", load_kg=60,
            set_type="working", reported_reps=5, reached_failure=True, raw_path="p",
            reported_rpe=11.0,
        )


def test_session_metadata_rejects_bad_wrist():
    with pytest.raises(ValueError, match="wrist"):
        SessionMetadata(session_id="x", user_id="u", started_at="2026-08-18T10:00:00", wrist="up")
