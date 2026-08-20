"""Synthetic IMU generator — the primary development data source (see plan Phase 0).

Emits physically-plausible wrist signals for bench/squat with ground-truth labels a
real capture never provides. Everything downstream is developed and validated against
these labels until real data collection lands.
"""

from wristset.synth.generator import (
    GeneratedSet,
    SetParams,
    generate_population,
    generate_session,
    generate_set,
    generate_training_session,
)

__all__ = ["SetParams", "GeneratedSet", "generate_set", "generate_session",
           "generate_training_session", "generate_population"]
