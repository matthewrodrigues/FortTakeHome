"""Immutable raw-timeseries storage as partitioned Parquet (§4.1).

Layout (one file per set):

    <root>/raw/user_id=<u>/date=<YYYY-MM-DD>/session_id=<s>/set_<NNN>.parquet

Files are write-once. Any reprocessing produces new derived artifacts elsewhere; the
raw layer is never modified after write (§4.1).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from wristset.contract import validate_raw

__all__ = ["raw_path_for", "write_raw_set", "read_raw_set"]


def raw_path_for(
    root: str | Path,
    user_id: str,
    date: str,
    session_id: str,
    set_index: int,
) -> Path:
    """Return the canonical Parquet path for a set. ``date`` is ``YYYY-MM-DD``."""
    return (
        Path(root)
        / "raw"
        / f"user_id={user_id}"
        / f"date={date}"
        / f"session_id={session_id}"
        / f"set_{set_index:03d}.parquet"
    )


def write_raw_set(
    root: str | Path,
    user_id: str,
    date: str,
    session_id: str,
    set_index: int,
    df: pl.DataFrame,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write a raw set to its canonical partition path.

    Refuses to overwrite an existing file unless ``overwrite=True`` — the raw layer is
    immutable (§4.1). Returns the path written.
    """
    df = validate_raw(df)
    path = raw_path_for(root, user_id, date, session_id, set_index)
    if path.exists() and not overwrite:
        raise FileExistsError(f"raw set already exists (immutable layer): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_raw_set(path: str | Path) -> pl.DataFrame:
    """Read a raw set and re-validate it against the contract."""
    return validate_raw(pl.read_parquet(path))
