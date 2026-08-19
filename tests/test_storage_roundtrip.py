"""Phase 0 GATE — the data contract round-trips end-to-end with zero algorithm code.

Generate a labeled set -> write raw Parquet + SQLite metadata -> read both back and
assert byte-for-byte signal equality and intact metadata.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from wristset.contract import SessionMetadata
from wristset.storage import MetadataStore, raw_path_for, read_raw_set, write_raw_set


def test_raw_parquet_roundtrip_identical(tmp_path, failure_bench_set):
    g = failure_bench_set
    path = write_raw_set(tmp_path, g.user_id, g.date, g.session_id, g.set_index, g.raw)

    assert path == raw_path_for(tmp_path, g.user_id, g.date, g.session_id, g.set_index)
    assert path.exists()

    back = read_raw_set(path)
    assert back.schema == g.raw.schema
    assert back.equals(g.raw)  # exact round-trip
    # timestamps still strictly monotonic
    t = back["t_ns"].to_numpy()
    assert (np.diff(t) > 0).all()


def test_immutable_layer_refuses_overwrite(tmp_path, failure_bench_set):
    g = failure_bench_set
    write_raw_set(tmp_path, g.user_id, g.date, g.session_id, g.set_index, g.raw)
    with pytest.raises(FileExistsError):
        write_raw_set(tmp_path, g.user_id, g.date, g.session_id, g.set_index, g.raw)


def test_metadata_roundtrip(tmp_path, failure_bench_set):
    g = failure_bench_set
    path = write_raw_set(tmp_path, g.user_id, g.date, g.session_id, g.set_index, g.raw)

    with MetadataStore(tmp_path / "meta.db") as store:
        store.add_session(
            SessionMetadata(
                session_id=g.session_id,
                user_id=g.user_id,
                started_at="2026-08-18T10:00:00",
                device_model="synthetic",
                wrist="right",
            )
        )
        store.add_set(g.set_metadata(str(path)))
        store.add_reps(g.set_id, g.reps)

        # read back
        s = store.get_set(g.set_id)
        assert s is not None
        assert s["exercise"] == g.exercise
        assert s["reported_reps"] == g.reported_reps
        assert s["reached_failure"] is True
        assert s["raw_path"] == str(path)

        reps = store.get_reps(g.set_id)
        assert len(reps) == len(g.reps)
        # the failed attempt survives as completed=False
        assert reps[-1]["completed"] is False
        assert reps[0]["completed"] is True

        # rep-features JSON blob round-trips
        rep_id = f"{g.set_id}:rep001"
        store.add_rep_features(rep_id, {"conc_mean_vel": 0.42, "rom_vertical": 0.44})
        feats = store.get_rep_features(rep_id)
        assert feats["rom_vertical"] == pytest.approx(0.44)


def test_full_session_partitioning(tmp_path):
    from wristset.synth import SetParams, generate_session

    sets = generate_session(
        [
            SetParams(set_type="warmup", capacity=6, stop_rir=3, reached_failure=False, seed=10),
            SetParams(set_type="working", capacity=8, stop_rir=0, reached_failure=True, seed=10),
        ]
    )
    paths = [
        write_raw_set(tmp_path, g.user_id, g.date, g.session_id, g.set_index, g.raw) for g in sets
    ]
    # both under the same session partition, distinct set files
    assert paths[0].parent == paths[1].parent
    assert paths[0].name == "set_001.parquet"
    assert paths[1].name == "set_002.parquet"
    for g, p in zip(sets, paths):
        assert read_raw_set(p).equals(g.raw)
