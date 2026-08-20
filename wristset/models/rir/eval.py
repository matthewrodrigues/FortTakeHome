"""RIR head evaluation (§9.6) — Layer 5b.

Split by **user**, not by set: the question is whether the hazard generalizes to a lifter
the model never saw (unseen users fall back to the population intercept). All numbers here
are synthetic-validated — measured against the generator's ground-truth failure labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from wristset.models.rir.dataset import PersonPeriod
from wristset.models.rir.hazard import HazardModel

__all__ = ["by_user_split", "c_index", "completed_rep_c_index", "Calibration",
           "calibration", "Collinearity", "collinearity_report", "EXPECTED_SIGNS"]


def by_user_split(sets: list, frac_test: float = 0.3, seed: int = 0) -> tuple[list, list]:
    """Partition ``(SetFeatures, RirMeta)`` pairs into train/test by USER (§9.6).

    Every set from a held-out user goes to test, so the test users are genuinely unseen at
    fit time. Deterministic in ``seed``.
    """
    users = sorted({meta.user_id for _, meta in sets})
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_test = max(1, int(round(len(users) * frac_test)))
    test_users = set(users[:n_test])
    train = [s for s in sets if s[1].user_id not in test_users]
    test = [s for s in sets if s[1].user_id in test_users]
    return train, test


def c_index(model: HazardModel, pp: PersonPeriod) -> float:
    """Concordance = P(hazard on a failure rep > hazard on a non-failure rep) (§9.6).

    Equivalent to the AUC of predicted hazard against the failure outcome — "does the model
    rank reps by failure proximity?" 0.5 is chance; 1.0 is perfect ranking.
    """
    h = model.predict_hazard_rows(pp)
    y = pp.y
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("C-index needs both failure and non-failure rows")
    ranks = rankdata(h)  # average ranks handle tied hazards
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def completed_rep_c_index(model: HazardModel, labeled: list) -> float:
    """Concordance over COMPLETED reps only, ranked by true reps-in-reserve (§9.6).

    The headline :func:`c_index` asks "is the failed attempt higher-hazard than the rest?".
    On this generator that is nearly free: a failed attempt is emitted at 40% of full height,
    so its concentric velocity is unlike any completed rep and ``-conc_mean_vel`` alone
    scores ~0.98 with no model at all. A C-index of ~1.0 therefore says little about the
    product question.

    This metric removes the giveaway. Using only completed reps of failure sets — where true
    RIR is known exactly as ``last_rep - r`` — it counts, over all pairs with *different*
    true RIR, how often the rep genuinely closer to failure carries the higher hazard. That
    is the ranking the RIR estimate actually depends on: "am I at 2 left or 6 left?".

    0.5 is chance. Ties in predicted hazard count as half-concordant, the standard
    convention. Pairs are formed across sets as well as within, so the model must rank
    consistently between lifters, not merely inside one set.
    """
    haz: list[float] = []
    true_rir: list[int] = []
    for sf, meta in labeled:
        if not (meta.reached_failure and sf.reps):
            continue
        last = max(r.rep_index for r in sf.reps)
        for rep in sf.reps:
            if not rep.completed:
                continue  # the failed attempt is the giveaway; exclude it
            haz.append(model.predict_hazard(
                _row_for(sf, rep), meta.exercise, meta.user_id))
            true_rir.append(last - rep.rep_index)

    h = np.asarray(haz)
    t = np.asarray(true_rir)
    if h.size < 2 or np.unique(t).size < 2:
        raise ValueError("need completed reps spanning at least two distinct true RIR values")

    # all ordered pairs where one rep is genuinely closer to failure than the other
    closer = t[:, None] < t[None, :]          # i closer to failure than j
    concordant = (h[:, None] > h[None, :]) & closer
    tied = np.isclose(h[:, None], h[None, :]) & closer
    n_pairs = int(closer.sum())
    return float((concordant.sum() + 0.5 * tied.sum()) / n_pairs)


def _row_for(sf, rep) -> dict:
    """Feature row for one rep (imported lazily to keep the module import graph flat)."""
    from wristset.models.rir.dataset import feature_row

    return feature_row(sf, rep)


#: Physiologically expected sign of each causal driver's coefficient: does MORE of this
#: value mean CLOSER to failure? Note ``conc_mean_vel`` and ``vel_loss_to_date`` are
#: negative-going with fatigue (velocity falls; ``total_change`` grows more negative), so a
#: NEGATIVE coefficient is the correct direction for both.
EXPECTED_SIGNS: dict[str, int] = {
    "rep_index": +1,
    "conc_mean_vel": -1,
    "path_dtw_baseline": +1,
    "vel_loss_to_date": -1,
    "ecc_change_to_date": -1,
    "tremor_change_to_date": +1,
}


@dataclass
class Collinearity:
    """Collinearity diagnostic for the fitted design (§9.2 interpretability caveat)."""

    vif: dict[str, float]              # variance inflation factor per feature
    marginal_corr: dict[str, float]    # each feature's own correlation with the outcome
    fitted_sign: dict[str, int]        # sign of the fitted coefficient
    sign_conflicts: list[str]          # features whose fitted sign contradicts physiology

    @property
    def max_vif(self) -> float:
        return max(self.vif.values()) if self.vif else 0.0


def collinearity_report(model: HazardModel, pp: PersonPeriod) -> Collinearity:
    """Quantify how far the fitted coefficients can be read as feature importance.

    **They largely cannot, and this says by how much.** The causal drivers are all proxies
    for one latent quantity — accumulated fatigue — so they are heavily correlated
    (``conc_mean_vel`` vs ``vel_loss_to_date``: r=0.92, VIF ~9.6). Under collinearity the
    fit is free to give one feature a large coefficient and hand the others compensating
    weights of the *opposite* sign; predictions are unaffected, but no individual
    coefficient means what it appears to.

    Measured on this corpus: every driver's MARGINAL correlation with the outcome carries
    the physiologically correct sign, yet ``path_dtw_baseline`` and ``tremor_change_to_date``
    fit negative. Each flips to positive (+1.18, +1.10) when fitted alone.

    This is a property of correlated regressors, not a defect to be corrected: forcing
    signs would trade away predictive accuracy for an interpretability the model does not
    currently need. The diagnostic exists so no caller mistakes ``model.beta`` for feature
    importance.
    """
    X = pp.X
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    n_feat = Xs.shape[1]

    vif: dict[str, float] = {}
    marginal: dict[str, float] = {}
    for i, name in enumerate(pp.columns):
        others = [j for j in range(n_feat) if j != i]
        A = np.column_stack([np.ones(len(Xs)), Xs[:, others]])
        coef, *_ = np.linalg.lstsq(A, Xs[:, i], rcond=None)
        resid = Xs[:, i] - A @ coef
        denom = float(((Xs[:, i] - Xs[:, i].mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / denom if denom > 0 else 0.0
        vif[name] = float(1.0 / max(1.0 - r2, 1e-9))
        sd = Xs[:, i].std()
        marginal[name] = (
            float(np.corrcoef(Xs[:, i], pp.y)[0, 1]) if sd > 1e-12 else 0.0
        )

    # beta layout is [intercept, numerics..., exercise dummies...]
    fitted_sign = {
        name: int(np.sign(model.beta[1 + i])) for i, name in enumerate(pp.columns)
    }
    conflicts = [
        name for name, expected in EXPECTED_SIGNS.items()
        if name in fitted_sign and fitted_sign[name] != 0
        and fitted_sign[name] != expected
    ]
    return Collinearity(vif=vif, marginal_corr=marginal,
                        fitted_sign=fitted_sign, sign_conflicts=sorted(conflicts))


@dataclass
class Calibration:
    """Calibration of predicted hazard against empirical failure rate (§9.6)."""

    bin_center: np.ndarray   # mean predicted hazard per non-empty bin
    observed: np.ndarray     # empirical failure rate per non-empty bin
    mad: float               # mean |predicted - observed| across non-empty bins


def calibration(model: HazardModel, pp: PersonPeriod, n_bins: int = 10) -> Calibration:
    """Among reps with predicted hazard ~p, did ~p fraction actually fail? (§9.6)"""
    h = model.predict_hazard_rows(pp)
    y = pp.y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(h, edges[1:-1]), 0, n_bins - 1)
    centers, observed = [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        centers.append(float(h[m].mean()))
        observed.append(float(y[m].mean()))
    centers = np.array(centers)
    observed = np.array(observed)
    mad = float(np.mean(np.abs(centers - observed))) if centers.size else float("nan")
    return Calibration(bin_center=centers, observed=observed, mad=mad)
