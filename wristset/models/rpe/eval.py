"""RPE head evaluation (§10) — Layer 5c.

Split by **user** (§9.6 rationale carries over): a held-out user is genuinely new, so the
model must predict them at the population level (``b_u = 0``). Two numbers matter:

* **Ordinal accuracy** on held-out users — measured in RPE units, because "±1 level" is
  ambiguous on a half-point grid. The honest bar is **±1.0 RPE**: a new lifter's personal
  reporting bias is, by construction, unknowable until you have their data, so exact-level
  accuracy is not the right target for unseen users (it IS what ``b_u`` buys you once they
  are seen — see :func:`bias_recovery`).
* **Bias recovery** on seen users — the correlation between the fitted ``b_u`` and the
  generator's injected ``rpe_bias``. This is the §10.5 verification: the head is recovering
  the user's *subjective calibration*, the thing that makes it non-redundant with RIR.
"""

from __future__ import annotations

import numpy as np

from wristset.models.rpe.dataset import LEVELS, RpeDataset, build_rpe_dataset
from wristset.models.rpe.ordinal import OrdinalRpeModel

__all__ = ["by_user_split", "ordinal_accuracy_within", "bias_recovery"]


def by_user_split(labeled: list, frac_test: float = 0.3, seed: int = 0) -> tuple[list, list]:
    """Partition ``(SetFeatures, RpeMeta)`` pairs into train/test by USER. Deterministic."""
    users = sorted({meta.user_id for _, meta in labeled})
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_test = max(1, int(round(len(users) * frac_test)))
    test_users = set(users[:n_test])
    train = [s for s in labeled if s[1].user_id not in test_users]
    test = [s for s in labeled if s[1].user_id in test_users]
    return train, test


def ordinal_accuracy_within(model: OrdinalRpeModel, ds: RpeDataset, tol_rpe: float = 1.0) -> float:
    """Fraction of sets whose predicted RPE is within ``tol_rpe`` of the reported RPE.

    ``tol_rpe`` is in RPE points (the grid is 0.5 apart), so ``1.0`` is +/- two levels.
    """
    pred = model.predict_level(ds)
    true = LEVELS[ds.y]
    return float(np.mean(np.abs(pred - true) <= tol_rpe + 1e-9))


def bias_recovery(model: OrdinalRpeModel, labeled: list) -> float:
    """Pearson r between fitted ``b_u`` and the generator's ``rpe_bias``, over seen users.

    The object of interest (§10.5). Computed only over users the model was fit on — a
    held-out user's ``b_u`` is 0 by design and carries no bias information.
    """
    true_bias: dict[str, float] = {}
    for _, meta in labeled:
        true_bias[meta.user_id] = meta.true_rpe_bias
    users = [u for u in true_bias if u in model._user_of]
    if len(users) < 3:
        raise ValueError("need >=3 seen users to correlate")
    fitted = np.array([model.user_effect(u) for u in users])
    injected = np.array([true_bias[u] for u in users])
    if fitted.std() < 1e-12 or injected.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(fitted, injected)[0, 1])
