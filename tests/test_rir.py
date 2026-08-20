"""RIR hazard head tests (Phase 6, §9), including the milestone-6 GATE.

Discrete-time survival: person-period expansion -> logistic hazard -> RIR distribution.
The gate is measured on a by-USER held-out split (§9.6): generalization to new lifters,
not just new sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.models.rir import (
    PersonPeriod,
    build_person_period,
    prepare_sets,
)
from wristset.synth import SetParams, generate_population, generate_set


def _prep_one(**kw):
    g = generate_set(SetParams(**kw))
    return prepare_sets([g])


# --- §B person-period expansion -------------------------------------------------


def test_person_period_has_one_row_per_detected_rep():
    labeled = _prep_one(capacity=8, stop_rir=0, reached_failure=True, seed=1)
    pp = build_person_period(labeled)
    sf, _ = labeled[0]
    assert pp.n_rows == sf.n_reps


def test_failure_set_has_exactly_one_positive_outcome():
    labeled = _prep_one(capacity=9, stop_rir=0, reached_failure=True, seed=2)
    pp = build_person_period(labeled)
    assert pp.y.sum() == 1, "a failure set fails on exactly one (the last) attempt"
    # and it is the last rep
    last = pp.rep_index.max()
    assert pp.y[pp.rep_index == last][0] == 1


def test_censored_set_contributes_all_zero_rows():
    labeled = _prep_one(capacity=10, stop_rir=3, reached_failure=False, seed=3)
    pp = build_person_period(labeled)
    assert pp.y.sum() == 0, "a censored set never observed a failure (§9.3)"


def test_design_matrix_carries_no_retrospective_columns():
    """The Phase-3 leakage boundary, enforced for the head that most needs it (§9.2):
    only causal / per-rep features may reach the hazard model."""
    labeled = prepare_sets(generate_population(n_users=2, sets_per_user=2, seed=0))
    pp = build_person_period(labeled)
    assert not any("retro" in c for c in pp.columns), pp.columns
    # and the expected causal drivers ARE present
    assert any("conc_mean_vel" in c for c in pp.columns)
    assert "rep_index" in pp.columns


def test_person_period_is_a_dataclass_instance():
    labeled = _prep_one(capacity=7, stop_rir=1, reached_failure=False, seed=4)
    pp = build_person_period(labeled)
    assert isinstance(pp, PersonPeriod)
    assert pp.X.shape[0] == pp.n_rows == len(pp.user_idx) == len(pp.y)


# --- §C hazard fit --------------------------------------------------------------

from wristset.models.rir import HazardModel  # noqa: E402


def _fit_small():
    labeled = prepare_sets(generate_population(n_users=8, sets_per_user=6, seed=7))
    pp = build_person_period(labeled)
    return HazardModel.fit(pp), pp


def test_predicted_hazards_are_valid_probabilities():
    model, pp = _fit_small()
    h = model.predict_hazard_rows(pp)
    assert h.shape == (pp.n_rows,)
    assert np.all((h > 0.0) & (h < 1.0))


def test_fit_is_deterministic():
    _, pp = _fit_small()
    a = HazardModel.fit(pp)
    b = HazardModel.fit(pp)
    assert np.allclose(a.beta, b.beta)


def test_hazard_is_higher_on_failure_reps_than_completed_reps():
    """The model must learn something: the actual failure attempts should carry higher
    hazard than the completed reps."""
    model, pp = _fit_small()
    h = model.predict_hazard_rows(pp)
    assert h[pp.y == 1].mean() > h[pp.y == 0].mean()


def test_unseen_user_falls_back_to_the_population_intercept():
    model, _ = _fit_small()
    assert model.user_effect("nobody-here") == 0.0


# --- §9.3 likelihood equivalence ------------------------------------------------
#
# The design's central claim: summing the row-wise Bernoulli log-likelihood over the
# person-period table IS the §9.3 censored likelihood, written additively instead of as a
# product. Everything downstream rests on it — if it were false, the "just fit a logistic
# regression" shortcut would be estimating the wrong objective. These tests compute both
# forms independently on hand-built numbers and assert they agree.


def _censored_log_likelihood(hazards: list[float], reached_failure: bool) -> float:
    """§9.3 likelihood for ONE set, in its native product form (log of it).

    Failure set failing on the last attempted rep R:  [prod_{r<R} (1-h_r)] * h_R
    Censored set stopping at rep C:                   prod_{r<=C} (1-h_r)
    """
    logl = 0.0
    for i, h in enumerate(hazards):
        is_last = i == len(hazards) - 1
        if reached_failure and is_last:
            logl += np.log(h)          # failed on this attempt
        else:
            logl += np.log(1.0 - h)    # survived this attempt
    return logl


def _person_period_log_likelihood(hazards: list[float], y: list[int]) -> float:
    """The additive Bernoulli form actually optimized by HazardModel.fit."""
    h = np.asarray(hazards)
    yy = np.asarray(y, dtype=float)
    return float(np.sum(yy * np.log(h) + (1.0 - yy) * np.log(1.0 - h)))


def test_person_period_likelihood_equals_censored_likelihood_failure_set():
    """A 4-rep set that failed on rep 4: rows are [0,0,0,1]."""
    hazards = [0.05, 0.10, 0.25, 0.60]
    y = [0, 0, 0, 1]
    product_form = _censored_log_likelihood(hazards, reached_failure=True)
    additive_form = _person_period_log_likelihood(hazards, y)
    assert additive_form == pytest.approx(product_form, abs=1e-12)
    # and it is the value the product form literally computes
    expected = np.log(0.95) + np.log(0.90) + np.log(0.75) + np.log(0.60)
    assert additive_form == pytest.approx(expected, abs=1e-12)


def test_person_period_likelihood_equals_censored_likelihood_censored_set():
    """A censored set contributes all-zero rows: it only ever says "survived"."""
    hazards = [0.05, 0.10, 0.25]
    y = [0, 0, 0]
    product_form = _censored_log_likelihood(hazards, reached_failure=False)
    additive_form = _person_period_log_likelihood(hazards, y)
    assert additive_form == pytest.approx(product_form, abs=1e-12)
    expected = np.log(0.95) + np.log(0.90) + np.log(0.75)
    assert additive_form == pytest.approx(expected, abs=1e-12)


def test_likelihood_equivalence_holds_over_a_mixed_corpus():
    """Summed across a mix of failure and censored sets — the objective fit() minimizes."""
    sets = [
        ([0.02, 0.08, 0.30], True),    # failed on rep 3
        ([0.05, 0.10], False),         # censored after 2
        ([0.01, 0.04, 0.12, 0.55], True),
        ([0.03], False),
    ]
    product_total = sum(_censored_log_likelihood(h, f) for h, f in sets)
    flat_h, flat_y = [], []
    for h, f in sets:
        flat_h.extend(h)
        flat_y.extend([0] * (len(h) - 1) + [1 if f else 0])
    additive_total = _person_period_log_likelihood(flat_h, flat_y)
    assert additive_total == pytest.approx(product_total, abs=1e-12)


def test_fit_objective_matches_the_censored_likelihood_on_real_rows():
    """The equivalence must hold for the objective ``fit`` actually optimizes, not just for
    hand-written numbers: recompute the §9.3 product form from a fitted model's own
    predicted hazards and check it against the unpenalized person-period NLL."""
    labeled = prepare_sets([
        generate_set(SetParams(capacity=8, stop_rir=0, reached_failure=True, seed=31)),
        generate_set(SetParams(capacity=9, stop_rir=2, reached_failure=False, seed=32)),
    ])
    pp = build_person_period(labeled)
    model = HazardModel.fit(pp)
    h = model.predict_hazard_rows(pp)

    additive = float(np.sum(pp.y * np.log(h) + (1 - pp.y) * np.log(1 - h)))

    # rebuild the product form set by set, in rep order
    product = 0.0
    for sid in dict.fromkeys(pp.set_id.tolist()):
        m = pp.set_id == sid
        order = np.argsort(pp.rep_index[m])
        hs = h[m][order]
        ys = pp.y[m][order]
        product += _censored_log_likelihood(list(hs), reached_failure=bool(ys[-1] == 1))

    assert additive == pytest.approx(product, abs=1e-9)


# --- §D hazard -> RIR distribution ----------------------------------------------

from wristset.models.rir import rir_distribution  # noqa: E402


def test_rir_distribution_is_a_valid_pmf():
    model, _ = _fit_small()
    labeled = _prep_one(capacity=9, stop_rir=0, reached_failure=True, seed=11)
    sf, meta = labeled[0]
    r = sf.reps[len(sf.reps) // 2].rep_index
    pred = rir_distribution(model, sf, r, exercise=meta.exercise, K=8)
    assert pred.dist.shape == (9,)
    assert abs(pred.dist.sum() - 1.0) < 1e-9
    assert np.all(pred.dist >= 0.0)


def test_expected_rir_falls_as_the_set_progresses():
    """Deeper into a failure set, fewer reps remain — E[RIR] must decrease."""
    model, _ = _fit_small()
    labeled = _prep_one(capacity=10, stop_rir=0, reached_failure=True, seed=12)
    sf, meta = labeled[0]
    reps = [x.rep_index for x in sf.reps if x.completed]
    early = rir_distribution(model, sf, reps[1], exercise=meta.exercise).expected_rir
    late = rir_distribution(model, sf, reps[-1], exercise=meta.exercise).expected_rir
    assert late < early, f"E[RIR] did not fall: early={early:.2f} late={late:.2f}"


# --- §E/F evaluation + milestone-6 GATE -----------------------------------------

from wristset.models.rir import (  # noqa: E402
    by_user_split,
    c_index,
    EXPECTED_SIGNS,
    calibration,
    collinearity_report,
    completed_rep_c_index,
)


def _fit_and_eval(corpus_seed: int):
    """Fit on train users, evaluate on held-out users.

    Returns ``(c_index, calibration_mad, rir_mae, completed_rep_concordance)``.

    RIR MAE is measured only on FAILURE sets, where true RIR at rep r is known exactly
    (``last_rep - r``). Censored sets have no observable RIR, which is the whole reason
    §9.1 uses a hazard formulation.
    """
    corpus = prepare_sets(generate_population(n_users=24, sets_per_user=6, seed=corpus_seed))
    train, test = by_user_split(corpus, frac_test=0.3, seed=0)
    model = HazardModel.fit(build_person_period(train))
    test_pp = build_person_period(test)

    errs = []
    for sf, meta in test:
        if not (meta.reached_failure and sf.reps):
            continue
        last = max(x.rep_index for x in sf.reps)
        for x in sf.reps:
            if not x.completed:
                continue
            pred = rir_distribution(model, sf, x.rep_index,
                                    exercise=meta.exercise, user_id=meta.user_id)
            errs.append(abs(pred.expected_rir - (last - x.rep_index)))
    return (c_index(model, test_pp), calibration(model, test_pp).mad,
            float(np.mean(errs)), completed_rep_c_index(model, test))


def test_gate_c_index_and_calibration_on_a_by_user_split():
    """Milestone 6: on users the model never saw, the hazard ranks reps by failure
    proximity (C-index >> 0.5) and its probabilities stay near the diagonal.

    Evaluated across THREE independent corpora, not one. A single corpus is not evidence:
    at the original weak ridge, calibration MAD was 0.04 on seed 5 and 0.33/0.29 on seeds
    11/23 — the gate passed on a lucky draw.
    """
    for corpus_seed in (5, 11, 23):
        ci, mad, _, _ = _fit_and_eval(corpus_seed)
        assert ci >= 0.65, f"corpus {corpus_seed}: C-index {ci:.3f} not above chance"
        assert mad <= 0.30, f"corpus {corpus_seed}: calibration MAD {mad:.3f} too large"


def test_collinearity_report_flags_uninterpretable_coefficients():
    """``model.beta`` must not be read as feature importance, and the diagnostic says why.

    The causal drivers are all proxies for one latent quantity (accumulated fatigue), so
    they are strongly correlated. Under collinearity the fit can hand one feature a large
    weight and give correlated ones compensating opposite-sign weights: predictions are
    unaffected, individual coefficients are not interpretable.

    This test pins the *diagnosis*, not a particular coefficient sign — the signs are
    expected to conflict, and forcing them would trade predictive accuracy for
    interpretability the model does not need.
    """
    corpus = prepare_sets(generate_population(n_users=24, sets_per_user=6, seed=5))
    train, _ = by_user_split(corpus, frac_test=0.3, seed=0)
    pp = build_person_period(train)
    model = HazardModel.fit(pp)
    rep = collinearity_report(model, pp)

    # the design IS collinear — that is the premise of the caveat
    assert rep.max_vif > 3.0, f"expected collinear design, max VIF {rep.max_vif:.1f}"
    assert rep.vif["conc_mean_vel"] > 5.0

    # every driver's MARGINAL relationship with the outcome has the physiological sign,
    # even where the fitted coefficient does not
    for name, expected in EXPECTED_SIGNS.items():
        r = rep.marginal_corr[name]
        assert np.sign(r) == expected, (
            f"{name}: marginal corr {r:+.3f} contradicts physiology (expected {expected:+d})"
        )

    # and the report surfaces the resulting conflicts rather than hiding them
    assert isinstance(rep.sign_conflicts, list)


def test_collinear_feature_recovers_its_sign_when_fitted_alone():
    """Direct evidence that the sign conflicts are a parameterization artifact: a feature
    whose joint coefficient is negative fits positive once its correlated partners are
    removed from the design."""
    corpus = prepare_sets(generate_population(n_users=24, sets_per_user=6, seed=5))
    train, _ = by_user_split(corpus, frac_test=0.3, seed=0)
    cols = list(build_person_period(train).columns)

    for name in ("path_dtw_baseline", "tremor_change_to_date"):
        solo = build_person_period(train)
        keep = cols.index(name)
        for j in range(len(cols)):
            if j != keep:
                solo.X[:, j] = 0.0
        beta_alone = HazardModel.fit(solo).beta[1 + keep]
        assert np.sign(beta_alone) == EXPECTED_SIGNS[name], (
            f"{name}: even fitted alone the sign is wrong ({beta_alone:+.3f})"
        )


def test_completed_rep_concordance_is_the_honest_discrimination_number():
    """Ranking skill measured WITHOUT the failed attempt — the honest metric.

    The headline C-index runs ~0.99 because the generator emits a failed attempt at 40% of
    full height, making it trivially separable (``-conc_mean_vel`` alone scores ~0.98 with
    no model). Excluding it and ranking completed reps by true RIR gives ~0.82: genuinely
    well above chance, but not the near-perfection the headline number implies. Report this
    one alongside the C-index; it is what the RIR estimate actually depends on.
    """
    scores = [_fit_and_eval(seed)[3] for seed in (5, 11, 23)]
    for seed, cc in zip((5, 11, 23), scores):
        assert cc >= 0.65, f"corpus {seed}: completed-rep concordance {cc:.3f} too low"
    assert float(np.mean(scores)) >= 0.75, f"mean completed-rep concordance: {scores}"


def test_expected_rir_is_accurate_on_failure_sets():
    """E[RIR] must land near the true reps-remaining — the actual product output.

    This is the metric the C-index cannot see. The generator emits a failed attempt at 40%
    of full height, so failure reps are trivially discriminable (C-index ~1.0) while E[RIR]
    can still be badly wrong: before the §9.4 projection damping and the ridge default were
    fixed, E[RIR] read 1.0 on a rep with 6 reps remaining.
    """
    maes = [_fit_and_eval(seed)[2] for seed in (5, 11, 23)]
    for seed, mae in zip((5, 11, 23), maes):
        assert mae <= 2.5, f"corpus {seed}: E[RIR] mean abs error {mae:.2f} reps"
    assert float(np.mean(maes)) <= 2.0, f"mean E[RIR] MAE across corpora: {maes}"
