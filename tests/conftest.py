"""Shared pytest fixtures. Golden synthetic sets used across the suite."""

from __future__ import annotations

import pytest

from wristset.synth import SetParams, generate_set


@pytest.fixture
def failure_bench_set():
    """A clean bench set taken to failure (8 completed + 1 failed attempt)."""
    return generate_set(
        SetParams(exercise="bench_press", capacity=8, stop_rir=0, reached_failure=True, seed=1),
        set_index=1,
    )


@pytest.fixture
def censored_squat_set():
    """A back squat stopped with 2 reps in reserve (no failure)."""
    return generate_set(
        SetParams(
            exercise="back_squat",
            capacity=10,
            stop_rir=2,
            reached_failure=False,
            rom_m=0.55,
            seed=2,
        ),
        set_index=2,
    )
