"""The data contract — the boundary the capture layer and the pipeline both commit to.

Defines the raw per-set timeseries schema (§3.2), the relational metadata schema
(§4.2), and the controlled vocabularies. Importing from here (rather than hardcoding
column names downstream) is what keeps the deferred watchOS capture app and the
Python pipeline in agreement.
"""

from wristset.contract.schema import (
    EXERCISES,
    RAW_SCHEMA,
    RAW_COLUMNS,
    SET_TYPES,
    SQLITE_DDL,
    GroundTruth,
    RepTruth,
    SetMetadata,
    SessionMetadata,
    validate_raw,
)

__all__ = [
    "EXERCISES",
    "RAW_SCHEMA",
    "RAW_COLUMNS",
    "SET_TYPES",
    "SQLITE_DDL",
    "GroundTruth",
    "RepTruth",
    "SetMetadata",
    "SessionMetadata",
    "validate_raw",
]
