"""Shared pytest fixtures. Golden synthetic sets used across the suite.

Also hosts the **corpus cache** used by the Phase 6/7 model tests. Building one 144-set
population costs ~23 s (14.5 s to generate raw signals, 8.8 s to run conditioning ->
segmentation -> features), and the RIR and RPE gates both evaluate the *same* corpora
across seeds 5/11/23. Without caching the suite rebuilds them a dozen times and runs past
ten minutes; memoizing on the generation parameters makes every repeat build free.

The cache is keyed on the full parameter tuple and the corpora are deterministic in their
seed, so a cached corpus is byte-identical to a freshly built one.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from wristset.synth import SetParams, generate_population, generate_rpe_population, generate_set


@lru_cache(maxsize=None)
def cached_population(n_users: int = 24, sets_per_user: int = 6, seed: int = 0,
                      balanced: bool = False):
    """A prepared ``[(SetFeatures, RirMeta)]`` corpus, built once per parameter set.

    ``balanced=True`` selects :func:`generate_rpe_population` (RPE labels spread across the
    scale); otherwise the RIR-oriented :func:`generate_population`. Returns the *prepared*
    corpus — the expensive Layer 2-4 pipeline is inside the cache, not just generation.
    """
    from wristset.models.rir.dataset import prepare_sets

    gen = generate_rpe_population if balanced else generate_population
    return prepare_sets(gen(n_users=n_users, sets_per_user=sets_per_user, seed=seed))


@lru_cache(maxsize=None)
def cached_rpe_corpus(n_users: int = 24, sets_per_user: int = 6, seed: int = 0,
                      balanced: bool = False):
    """A prepared ``[(SetFeatures, RpeMeta)]`` corpus, built once per parameter set.

    Same caching rationale as :func:`cached_population`; the RPE head needs its own metadata
    (reported RPE + the injected bias for the §10.5 recovery check), so it wraps
    ``prepare_rpe_sets`` rather than reusing the RIR-prepared pairs.
    """
    from wristset.models.rpe.dataset import prepare_rpe_sets

    gen = generate_rpe_population if balanced else generate_population
    return prepare_rpe_sets(gen(n_users=n_users, sets_per_user=sets_per_user, seed=seed))


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
