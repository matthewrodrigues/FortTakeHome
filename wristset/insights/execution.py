"""Execution narrative (§8.6-8.7) — the form half.

Generated separately from the score, from the form-subscore decomposition already produced
in ``scoring/form.py``. §8.6 says to report "the largest-magnitude deviations first"; the
subtlety this layer resolves is what "largest" means.

**Ordering is by score DEFICIT, not raw pointer magnitude.** A ``Pointer.magnitude`` is a
per-channel quantity — a 0.3 ROM retention and a 2.0 tremor rise are both "large" in their
own units but are not comparable to each other. The subscore *deficit* (``100 - score``) is
already on a common 0-100 scale, so it orders subscores fairly across channels. Within a
subscore, pointer magnitude is meaningful and orders that subscore's own pointers.

This is also where the deliberate ``magnitude ≈ 0`` breakpoint pointers that Phase 4 emits
on clean subscores get suppressed: a clean subscore scores ~100, so its deficit falls below
``min_deficit`` and it never reaches the narrative.

The effort narrative (RIR/RPE, §8.6) is Phase 8 — it needs heads that do not exist yet.
"""

from __future__ import annotations

from wristset.scoring import FormSubscores

__all__ = ["execution_narrative"]


def execution_narrative(
    form: FormSubscores,
    *,
    max_points: int = 3,
    min_deficit: float = 15.0,
) -> str:
    """A short measured-change narrative for one set's form (§8.6).

    Leads with the (form-only, honestly labelled) score, then reports the worst subscores
    first — those whose deficit ``100 - score`` clears ``min_deficit`` — contributing each
    one's top pointer, up to ``max_points`` sentences. Pointer text is reused verbatim; its
    §8.7 measured-change discipline is enforced upstream in ``scoring/form.py``.
    """
    lead = _lead_line(form)

    scored = [s for s in form.subscores if s.score is not None]
    notable = [s for s in scored if (100.0 - s.score) >= min_deficit]
    # worst first; deterministic tie-break on name keeps output stable across runs
    notable.sort(key=lambda s: (-(100.0 - s.score), s.name))

    sentences: list[str] = []
    for sub in notable:
        for p in sorted(sub.pointers, key=lambda p: -p.magnitude):
            sentences.append(p.text)
            break  # one pointer per subscore in the narrative; the rest live on the object
        if len(sentences) >= max_points:
            break

    body = " ".join(sentences) if sentences else "Form held across the set."
    return f"{lead} {body}"


def _lead_line(form: FormSubscores) -> str:
    score = form.form_score
    if score is None:
        return "Form score unavailable (no reference)."
    tag = " (provisional)" if form.provisional else ""
    return f"Form score {round(score)}{tag}."
