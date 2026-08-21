"""Phase 5 demo: wire Phases 0-4 into one deterministic run (§2 flow, milestone 5).

The shared orchestrator ``analyze_set`` is the single source of the raw -> form-score
wiring; both this CLI and the Streamlit view (``ui/app.py``) call it, so the pipeline
cannot drift between them. It stays streamlit-free and on core deps so the demo runs from a
bare install.

**Composite over available components (§8.1).** ``analyze_set`` accepts an optional fitted
``rir_model``; without one the RIR term is simply absent and ``score_composite``
renormalises over what remains, rather than fabricating a value (§8.7). The CLI runs without
a model, so its composite is form + reported RPE. The Layer-1 Parquet/SQLite round-trip is
exercised by its own Phase-0 tests and skipped here to keep the run in-memory and
deterministic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from wristset.conditioning import ConditionedSet, condition_set
from wristset.features import RepSource, SetFeatures, extract_set_features
from wristset.divergence import Divergence, divergence_from_rir
from wristset.insights import effort_narrative, execution_narrative
from wristset.models.rir import rir_distribution
from wristset.scoring import CompositeScore, FormSubscores, score_composite, score_form
from wristset.segmentation import SegmentationResult, segment_reps
from wristset.synth import GeneratedSet, generate_training_session

__all__ = ["SetAnalysis", "analyze_set", "analyze_session", "fit_demo_rir_model", "main"]


@dataclass
class SetAnalysis:
    """Everything the demo derives for one set, start to finish."""

    meta: GeneratedSet
    cs: ConditionedSet
    result: SegmentationResult
    features: SetFeatures
    form: FormSubscores
    narrative: str
    #: Phase-8 additions. All optional: without a fitted hazard head the composite
    #: renormalises to form alone, which is exactly the Phase-5 behaviour (§8.1).
    composite: CompositeScore | None = None
    divergence: Divergence | None = None
    effort_text: str | None = None


def analyze_set(
    g: GeneratedSet,
    *,
    history: list[RepSource] | None = None,
    warmups: list[RepSource] | None = None,
    rir_model=None,
) -> SetAnalysis:
    """Run one set from raw frame to form score + narrative.

    ``history`` (prior sessions) and ``warmups`` (today's warmup sets) feed the §7 baseline
    hierarchy so ``path_dtw_baseline`` — and thus the path-consistency subscore — resolves
    against the best available reference rather than the current set's own opening reps.
    """
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    result = segment_reps(cs, exercise=g.exercise, reached_failure=g.reached_failure)
    features = extract_set_features(
        cs, result.reps,
        exercise=g.exercise, load_kg=g.load_kg, set_type=g.set_type,
        history=history, warmups=warmups,
    )
    form = score_form(features)
    narrative = execution_narrative(form)

    # §8.1-8.2 / §11: present only when a fitted hazard head is supplied. The composite
    # renormalises over whatever exists, so the no-model path is unchanged from Phase 5.
    expected_rir: float | None = None
    divergence: Divergence | None = None
    effort_text: str | None = None
    if rir_model is not None:
        completed = [r for r in result.reps if r.completed]
        if completed:
            pred = rir_distribution(
                rir_model, features, completed[-1].rep_index,
                exercise=g.exercise, user_id=g.user_id,
            )
            expected_rir = pred.expected_rir
            divergence = divergence_from_rir(pred.dist, g.reported_rpe)
            effort_text = effort_narrative(pred.dist, divergence)

    composite = score_composite(
        form_score=form.form_score,
        expected_rir=expected_rir,
        reported_rpe=g.reported_rpe,
        form_provisional=form.provisional,
    )
    return SetAnalysis(g, cs, result, features, form, narrative,
                       composite, divergence, effort_text)


def _rep_source(a: SetAnalysis) -> RepSource:
    return RepSource(
        cs=a.cs, reps=a.result.reps, load_kg=a.meta.load_kg,
        set_type=a.meta.set_type, exercise=a.meta.exercise,
    )


def analyze_session(sets: list[GeneratedSet], *, rir_model=None) -> list[SetAnalysis]:
    """Analyse a whole session in order, feeding each set the warmups seen before it.

    Warmup sets are analysed too (so the UI can show them) and, once analysed, become the
    §7.1 tier-2 reference for the working sets that follow — which is what makes the
    hierarchy observable in the demo. ``rir_model`` is passed through to every set.
    """
    warmups: list[RepSource] = []
    analyses: list[SetAnalysis] = []
    for g in sets:
        a = analyze_set(g, warmups=warmups or None, rir_model=rir_model)
        analyses.append(a)
        if g.set_type == "warmup":
            warmups.append(_rep_source(a))
    return analyses


#: Corpus size used by ``--with-rir``. Small on purpose: the fit itself is free (~0.02 s),
#: essentially all the cost is generating and conditioning the corpus (~7 s at this size,
#: ~22 s at the 24-user size the gates use). In-sample E[RIR] MAE is ~1.66 reps here against
#: ~1.80 held-out at the larger size, so the demo is representative without the wait.
DEMO_CORPUS_USERS: int = 10
DEMO_CORPUS_SETS: int = 5


def fit_demo_rir_model(seed: int = 0, *, n_users: int = DEMO_CORPUS_USERS):
    """Fit a hazard head on a small synthetic population, plus its §9.5 readiness.

    Returns ``(model, readiness)``. The model is ``None`` when the corpus does not clear
    §9.5's minimum failure-set count — RIR is a capability that **unlocks with data**, not
    something to show from a corpus too thin to identify it (see
    :mod:`wristset.models.rir.readiness`).

    The demo's own lifter is deliberately **not** in this corpus: it is generated with a
    different user id, so ``user_effect`` returns 0 and the RIR estimate comes from the
    population fit. That is the honest cold-start case — what a real first-time user gets —
    rather than an in-sample number that would flatter the demo.

    Deterministic: the corpus is seed-determined and the fit is convex from a zero init.
    """
    from wristset.models.rir import (
        HazardModel,
        assess_rir_readiness,
        build_person_period,
        prepare_sets,
    )
    from wristset.synth import generate_population

    corpus = generate_population(
        n_users=n_users, sets_per_user=DEMO_CORPUS_SETS, seed=seed,
    )
    labeled = prepare_sets(corpus)
    readiness = assess_rir_readiness(labeled)
    if not readiness.available:
        return None, readiness
    return HazardModel.fit(build_person_period(labeled)), readiness


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def _format_card(a: SetAnalysis) -> str:
    m = a.meta
    stop = "reached failure" if m.reached_failure else "stopped short"
    c = a.composite
    lines = [f"Set {m.set_index}  {m.exercise}  {m.load_kg:g}kg  ({m.set_type}, {stop})"]

    if c is not None and c.total is not None:
        head = f"  Set score {round(c.total)} / 100"
        if c.divergent:
            # §8.5: effort and form disagree, so the number is the least informative part
            head += "  [divergent - read the notes, not the number]"
        lines.append(head)
        lines.append(f"    components: {', '.join(c.components_used)}")

    if a.effort_text:
        lines.append(f"  {a.effort_text}")
    lines.append(f"  {a.narrative}")
    lines.append(
        f"  baseline: {a.features.baseline.source}  "
        f"| reps completed: {a.features.n_completed}/{a.features.n_reps}"
    )
    for s in a.form.subscores:
        val = "  n/a" if s.score is None else f"{round(s.score):4d}"
        lines.append(f"    {s.name:18s} {val}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic Phase-5 demo over a fresh synthetic session."""
    parser = argparse.ArgumentParser(
        prog="wristset.demo",
        description="Deterministic wristset demo: synthetic session -> form subscores.",
    )
    parser.add_argument("--seed", type=int, default=0, help="synthetic session seed")
    parser.add_argument("--exercise", default="bench_press", help="exercise name")
    parser.add_argument(
        "--with-rir", action="store_true",
        help=("supply synthetic RIR history, so the composite can include the RIR term "
              "and the effort narrative (adds ~7s)"),
    )
    parser.add_argument(
        "--rir-users", type=int, default=DEMO_CORPUS_USERS,
        help=("how many synthetic lifters of history to supply. Lower values fall below "
              "the 9.5 failure-set minimum and demonstrate RIR staying locked."),
    )
    args = parser.parse_args(argv)

    rir_model, readiness = (
        fit_demo_rir_model(seed=args.seed, n_users=args.rir_users)
        if args.with_rir else (None, None)
    )
    sets = generate_training_session(exercise=args.exercise, seed=args.seed)
    analyses = analyze_session(sets, rir_model=rir_model)

    # ASCII-only console text: the demo runs on stock Windows terminals (cp1252), where a
    # stray em-dash prints as mojibake. Pointer text is already ASCII.
    print(f"wristset demo - session seed {args.seed}, {args.exercise}")
    if readiness is None:
        print("(no RIR history in this run: the composite renormalises over the components")
        print(" present, rather than fabricating the missing one. Pass --with-rir.)\n")
    else:
        print(f"  {readiness.explain()}")
        if rir_model is None:
            print("  RIR is withheld until then; the composite uses form and RPE only.\n")
        else:
            print(f"  Fit on {args.rir_users} synthetic lifters; this session's lifter is NOT")
            print("  among them, so the estimate is the population cold-start case.\n")
    for a in analyses:
        if a.meta.set_type != "working":
            continue
        print(_format_card(a))
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m wristset.demo`
    sys.exit(main())
