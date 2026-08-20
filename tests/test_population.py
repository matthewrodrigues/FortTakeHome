"""Multi-user corpus generator tests (Phase 6, §9.5).

The RIR hazard head is identified only from sets that reached the fatigue states it must
predict, so the corpus must carry many users and enough failure sets (§9.5: >=30 per
exercise) alongside naturally-censored ones — grouped by user for the by-user eval split.
"""

from __future__ import annotations

from wristset.synth import generate_population


def _summary(sets):
    users = {g.user_id for g in sets}
    failures = [g for g in sets if g.reached_failure]
    censored = [g for g in sets if not g.reached_failure]
    return users, failures, censored


def test_population_has_the_requested_users_and_sets():
    sets = generate_population(n_users=6, sets_per_user=5, seed=0)
    users, _, _ = _summary(sets)
    assert len(users) == 6
    assert len(sets) == 6 * 5


def test_population_contains_both_failure_and_censored_sets():
    sets = generate_population(n_users=8, sets_per_user=6, seed=0)
    _, failures, censored = _summary(sets)
    assert failures and censored, "need both failure and censored sets for §9.3"


def test_default_corpus_clears_the_failure_set_floor_per_exercise():
    """§9.5: at least ~30 failure sets per exercise for a usable prototype fit."""
    sets = generate_population(n_users=20, sets_per_user=8,
                              exercises=("bench_press", "back_squat"), seed=1)
    for ex in ("bench_press", "back_squat"):
        n_fail = sum(1 for g in sets if g.exercise == ex and g.reached_failure)
        assert n_fail >= 30, f"{ex}: only {n_fail} failure sets"


def test_failure_rep_is_labelled_consistently_with_reached_failure():
    sets = generate_population(n_users=5, sets_per_user=5, seed=2)
    for g in sets:
        gt = g.ground_truth
        if g.reached_failure:
            assert gt.failure_rep is not None
        else:
            assert gt.failure_rep is None


def test_population_is_deterministic_for_a_fixed_seed():
    def sig(sets):
        return [(g.user_id, g.exercise, g.reported_reps, g.reached_failure) for g in sets]
    assert sig(generate_population(n_users=4, sets_per_user=4, seed=3)) == \
           sig(generate_population(n_users=4, sets_per_user=4, seed=3))


def test_exercises_are_drawn_only_from_the_requested_set():
    sets = generate_population(n_users=6, sets_per_user=4, exercises=("bench_press",), seed=0)
    assert {g.exercise for g in sets} == {"bench_press"}
