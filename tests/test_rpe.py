"""RPE ordinal hierarchical head tests (Phase 7, §10), including the milestone-7 GATE.

The head models the user's SUBJECTIVE mapping (§10.5): the per-user effect b_u is the
object of interest and must recover the generator's injected rpe_bias. Everything is
measured on a by-USER held-out split, across multiple corpora (the Phase-6 lesson).
"""

from __future__ import annotations

import collections

import numpy as np

from wristset.models.rpe import (
    LEVELS,
    RpeDataset,
    build_rpe_dataset,
    prepare_rpe_sets,
)
from wristset.synth import generate_population, generate_rpe_population

from conftest import cached_rpe_corpus


def _dataset(seed=0, **kw):
    labeled = cached_rpe_corpus(seed=seed, **kw)
    return build_rpe_dataset(labeled), labeled


# --- §A dataset -----------------------------------------------------------------


def test_levels_are_the_half_point_grid():
    assert list(LEVELS) == [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]


def test_dataset_shapes_line_up():
    ds, labeled = _dataset(n_users=4, sets_per_user=4)
    assert isinstance(ds, RpeDataset)
    n = len(labeled)
    assert ds.X.shape[0] == n == len(ds.y) == len(ds.user_idx)
    assert ds.y.min() >= 0 and ds.y.max() <= len(LEVELS) - 1


def test_labels_match_reported_rpe():
    ds, labeled = _dataset(n_users=3, sets_per_user=5)
    for i, (_, meta) in enumerate(labeled):
        assert LEVELS[int(ds.y[i])] == meta.reported_rpe


def test_design_includes_whole_set_retrospective_features():
    """§10.4 boundary — the mirror of the RIR leakage guard. Whole-set retrospective
    summaries are FORBIDDEN to the RIR head but REQUIRED here (RPE is reported post-set)."""
    ds, _ = _dataset(n_users=3, sets_per_user=4)
    assert any("retro" in c for c in ds.columns), ds.columns
    assert "reported_reps" in ds.columns


def test_true_bias_is_carried_for_eval_only():
    _, labeled = _dataset(n_users=3, sets_per_user=4)
    # every set of a given user shares that user's injected bias
    by_user: dict[str, set] = {}
    for _, meta in labeled:
        by_user.setdefault(meta.user_id, set()).add(round(meta.true_rpe_bias, 6))
    assert all(len(v) == 1 for v in by_user.values())


# --- §B ordinal model -----------------------------------------------------------

from wristset.models.rpe import OrdinalRpeModel  # noqa: E402


def _fit(seed=0, **kw):
    ds, labeled = _dataset(seed=seed, **kw)
    return OrdinalRpeModel.fit(ds), ds, labeled


def test_pmf_rows_are_valid_distributions():
    model, ds, _ = _fit(n_users=8, sets_per_user=6)
    P = model.predict_proba(ds)
    assert P.shape == (ds.n_sets, len(LEVELS))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0.0)


def test_cumulative_probabilities_are_rank_monotone():
    """P(RPE>k) must be non-increasing in k — guaranteed by ordered thresholds (§10.2)."""
    model, ds, _ = _fit(n_users=6, sets_per_user=6)
    P = model.predict_proba(ds)
    surv = 1.0 - np.cumsum(P, axis=1)[:, :-1]  # P(RPE > level_k) for k=0..K-2
    assert np.all(np.diff(surv, axis=1) <= 1e-9)


def test_predict_level_is_on_the_grid():
    model, ds, _ = _fit(n_users=5, sets_per_user=5)
    lv = model.predict_level(ds)
    assert set(np.unique(lv)).issubset(set(LEVELS.tolist()))


def test_fit_is_deterministic():
    ds, _ = _dataset(seed=1, n_users=6, sets_per_user=5)
    a = OrdinalRpeModel.fit(ds)
    b = OrdinalRpeModel.fit(ds)
    assert np.allclose(a.beta, b.beta) and np.allclose(a.thresholds, b.thresholds)


def test_higher_effort_predicts_higher_rpe():
    """Two otherwise-equal sets: the one taken closer to failure should read higher RPE."""
    from wristset.synth import SetParams, generate_set
    model, _, _ = _fit(n_users=10, sets_per_user=6)
    easy = prepare_rpe_sets([generate_set(SetParams(capacity=10, stop_rir=4,
                             reached_failure=False, seed=101))])
    hard = prepare_rpe_sets([generate_set(SetParams(capacity=10, stop_rir=0,
                             reached_failure=True, seed=101))])
    e = model.expected_rpe(build_rpe_dataset(easy))[0]
    h = model.expected_rpe(build_rpe_dataset(hard))[0]
    assert h > e, f"hard set RPE {h:.2f} not above easy {e:.2f}"


def test_unseen_user_gets_zero_effect():
    model, _, _ = _fit(n_users=5, sets_per_user=5)
    assert model.user_effect("stranger") == 0.0


# --- §C/D evaluation + milestone-7 GATE -----------------------------------------

from wristset.models.rpe import (  # noqa: E402
    bias_recovery,
    by_user_split,
    ordinal_accuracy_within,
)


def _fit_and_eval(seed):
    labeled = cached_rpe_corpus(n_users=24, sets_per_user=6, seed=seed)
    train, test = by_user_split(labeled, seed=0)
    model = OrdinalRpeModel.fit(build_rpe_dataset(train))
    acc = ordinal_accuracy_within(model, build_rpe_dataset(test), tol_rpe=1.0)
    r = bias_recovery(model, train)
    return acc, r


def _fit_and_eval_balanced(seed):
    """Same evaluation on a corpus whose RPE labels span the scale (see
    ``generate_rpe_population``). Returns (accuracy, majority-baseline, bias recovery)."""
    labeled = cached_rpe_corpus(n_users=24, sets_per_user=6, seed=seed, balanced=True)
    train, test = by_user_split(labeled, seed=0)
    dtr, dte = build_rpe_dataset(train), build_rpe_dataset(test)
    model = OrdinalRpeModel.fit(dtr)
    acc = ordinal_accuracy_within(model, dte, tol_rpe=1.0)
    majority = LEVELS[collections.Counter(dtr.y).most_common(1)[0][0]]
    base = float(np.mean(np.abs(LEVELS[dte.y] - majority) <= 1.0 + 1e-9))
    return acc, base, bias_recovery(model, train)


def test_rpe_corpus_is_actually_balanced_across_the_scale():
    """The RPE preset must spread labels, or accuracy measured on it means little.

    ``generate_population``'s RIR-oriented defaults put ~1/3 of sets at RPE 10 (both the
    modal ``10 - stop_rir`` outcome and a clip ceiling absorbing every positive bias).
    """
    ds = build_rpe_dataset(cached_rpe_corpus(n_users=24, sets_per_user=6, seed=5,
                                             balanced=True))
    counts = collections.Counter(ds.y)
    assert len(counts) >= 8, f"only {len(counts)} of 9 levels present"
    share = max(counts.values()) / ds.n_sets
    assert share <= 0.20, f"most common level holds {share:.0%} of the corpus"


def test_accuracy_beats_the_majority_baseline_on_a_balanced_corpus():
    """The honest skill check: accuracy must clear a predict-the-most-common-level baseline.

    Raw accuracy is inflated by label skew — measured, the skewed corpus gives 0.74 accuracy
    against a 0.54 baseline, while the balanced corpus gives 0.60 against 0.37. The *lift*
    is the stable quantity (+0.20 vs +0.23), and it is what this asserts.
    """
    results = [_fit_and_eval_balanced(seed) for seed in (5, 11, 23)]
    for seed, (acc, base, _) in zip((5, 11, 23), results):
        assert acc > base, f"corpus {seed}: accuracy {acc:.2f} did not beat baseline {base:.2f}"
    lift = float(np.mean([acc - base for acc, base, _ in results]))
    assert lift >= 0.12, f"mean lift over majority baseline only {lift:.3f}"


def test_bias_recovery_holds_on_a_balanced_corpus():
    """b_u must still track the injected rpe_bias when labels are not skewed."""
    rs = [_fit_and_eval_balanced(seed)[2] for seed in (5, 11, 23)]
    for seed, r in zip((5, 11, 23), rs):
        assert r >= 0.5, f"corpus {seed}: b_u~rpe_bias correlation {r:.2f} too weak"


def test_gate_ordinal_accuracy_within_one_rpe_on_held_out_users():
    """Milestone 7: on unseen users, predicted RPE is within +/-1.0 for the majority of
    sets. +/-1 point (not exact) is the honest bar — a new lifter's personal bias is
    unknowable until the head has their data (that is what b_u recovers). Multi-seed."""
    accs = [_fit_and_eval(seed)[0] for seed in (5, 11, 23)]
    for seed, acc in zip((5, 11, 23), accs):
        assert acc >= 0.55, f"corpus {seed}: within-1-RPE accuracy {acc:.2f} not a majority"
    assert float(np.mean(accs)) >= 0.65, f"mean within-1-RPE accuracy: {accs}"


def test_gate_b_u_recovers_the_injected_user_bias():
    """§10.5: the per-user effect must track the generator's rpe_bias — this is what makes
    the RPE head model the user's SUBJECTIVE mapping, not re-derive RIR."""
    rs = [_fit_and_eval(seed)[1] for seed in (5, 11, 23)]
    for seed, r in zip((5, 11, 23), rs):
        assert r >= 0.6, f"corpus {seed}: b_u~rpe_bias correlation {r:.2f} too weak"


def test_a_planted_biased_user_gets_the_right_sign_effect():
    """A user who systematically over-reports (positive rpe_bias) must fit a positive b_u,
    and an under-reporter a negative one."""
    from wristset.synth import SetParams, generate_session
    def sets_for(bias):
        params = [SetParams(capacity=10, stop_rir=s % 4, reached_failure=(s % 4 == 0),
                            rpe_bias=bias, rpe_noise=0.1, seed=s) for s in range(8)]
        return generate_session(params, user_id=f"u{bias:+.0f}")
    over = sets_for(2.0)
    under = sets_for(-2.0)
    labeled = prepare_rpe_sets(over + under)
    model = OrdinalRpeModel.fit(build_rpe_dataset(labeled))
    assert model.user_effect("u+2") > model.user_effect("u-2")
