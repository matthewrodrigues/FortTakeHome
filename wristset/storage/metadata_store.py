"""Relational metadata store (SQLite) implementing the §4.2 schema.

Thin, dependency-free wrapper over ``sqlite3``. SQLite is the prototype choice
(Postgres later, §4.2). All writes are parameterised; the dataclasses in
``wristset.contract`` are the source of truth for field validation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from wristset.contract import SQLITE_DDL, SessionMetadata, SetMetadata
from wristset.contract.schema import RepTruth

__all__ = ["MetadataStore"]


class MetadataStore:
    """SQLite-backed metadata store. Use as a context manager or call ``close()``."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SQLITE_DDL)
        self._conn.commit()

    # -- context management ------------------------------------------------------
    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- writers -----------------------------------------------------------------
    def add_session(self, s: SessionMetadata) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, user_id, started_at, device_model, wrist) VALUES (?,?,?,?,?)",
            (s.session_id, s.user_id, s.started_at, s.device_model, s.wrist),
        )
        self._conn.commit()

    def add_set(self, m: SetMetadata) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sets "
            "(set_id, session_id, set_index, exercise, load_kg, set_type, "
            " reported_reps, reported_rpe, reached_failure, raw_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                m.set_id,
                m.session_id,
                m.set_index,
                m.exercise,
                m.load_kg,
                m.set_type,
                m.reported_reps,
                m.reported_rpe,
                int(m.reached_failure),
                m.raw_path,
            ),
        )
        self._conn.commit()

    def add_reps(self, set_id: str, reps: Iterable[RepTruth]) -> None:
        """Insert rep boundary rows. Accepts RepTruth (synthetic) or any object exposing
        the same rep boundary fields."""
        rows = [
            (
                f"{set_id}:rep{r.rep_index:03d}",
                set_id,
                r.rep_index,
                float(r.t_start),
                float(r.t_bottom),
                float(r.t_end),
                int(r.completed),
            )
            for r in reps
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO reps "
            "(rep_id, set_id, rep_index, t_start, t_bottom, t_end, completed) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def add_rep_features(self, rep_id: str, features: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO rep_features (rep_id, features) VALUES (?, ?)",
            (rep_id, json.dumps(features)),
        )
        self._conn.commit()

    # -- readers -----------------------------------------------------------------
    def get_session(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_set(self, set_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM sets WHERE set_id = ?", (set_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["reached_failure"] = bool(d["reached_failure"])
        return d

    def get_sets_for_session(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sets WHERE session_id = ? ORDER BY set_index", (session_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["reached_failure"] = bool(d["reached_failure"])
            out.append(d)
        return out

    def get_reps(self, set_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM reps WHERE set_id = ? ORDER BY rep_index", (set_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["completed"] = bool(d["completed"])
            out.append(d)
        return out

    def get_rep_features(self, rep_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT features FROM rep_features WHERE rep_id = ?", (rep_id,)
        ).fetchone()
        return json.loads(row["features"]) if row else None
