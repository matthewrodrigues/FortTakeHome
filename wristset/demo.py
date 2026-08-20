"""Phase 5 demo: wire Phases 0-4 into one deterministic run (§2 flow, milestone 5).

The shared orchestrator ``analyze_set`` is the single source of the raw -> form-score
wiring; both this CLI and the Streamlit view (``ui/app.py``) call it, so the pipeline
cannot drift between them. It stays streamlit-free and on core deps so the demo runs from a
bare install.

**Form score only.** The §8.1 composite is half form, quarter RIR, quarter RPE, but the
effort heads do not exist until Phases 6-7. Showing a fabricated composite would violate
§8.7's honesty discipline, so the demo reports the form score alone, explicitly labelled.
The Layer-1 Parquet/SQLite round-trip is exercised by its own Phase-0 tests and is skipped
here to keep the run in-memory and deterministic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from wristset.conditioning import ConditionedSet, condition_set
from wristset.features import RepSource, SetFeatures, extract_set_features
from wristset.insights import execution_narrative
from wristset.scoring import FormSubscores, score_form
from wristset.segmentation import SegmentationResult, segment_reps
from wristset.synth import GeneratedSet, generate_training_session

__all__ = ["SetAnalysis", "analyze_set", "analyze_session", "main"]


@dataclass
class SetAnalysis:
    """Everything the demo derives for one set, start to finish."""

    meta: GeneratedSet
    cs: ConditionedSet
    result: SegmentationResult
    features: SetFeatures
    form: FormSubscores
    narrative: str


def analyze_set(
    g: GeneratedSet,
    *,
    history: list[RepSource] | None = None,
    warmups: list[RepSource] | None = None,
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
    return SetAnalysis(g, cs, result, features, form, narrative)


def _rep_source(a: SetAnalysis) -> RepSource:
    return RepSource(
        cs=a.cs, reps=a.result.reps, load_kg=a.meta.load_kg,
        set_type=a.meta.set_type, exercise=a.meta.exercise,
    )


def analyze_session(sets: list[GeneratedSet]) -> list[SetAnalysis]:
    """Analyse a whole session in order, feeding each set the warmups seen before it.

    Warmup sets are analysed too (so the UI can show them) and, once analysed, become the
    §7.1 tier-2 reference for the working sets that follow — which is what makes the
    hierarchy observable in the demo.
    """
    warmups: list[RepSource] = []
    analyses: list[SetAnalysis] = []
    for g in sets:
        a = analyze_set(g, warmups=warmups or None)
        analyses.append(a)
        if g.set_type == "warmup":
            warmups.append(_rep_source(a))
    return analyses


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def _format_card(a: SetAnalysis) -> str:
    m = a.meta
    stop = "reached failure" if m.reached_failure else "stopped short"
    lines = [
        f"Set {m.set_index}  {m.exercise}  {m.load_kg:g}kg  ({m.set_type}, {stop})",
        f"  {a.narrative}",
        f"  baseline: {a.features.baseline.source}  "
        f"| reps completed: {a.features.n_completed}/{a.features.n_reps}",
    ]
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
    args = parser.parse_args(argv)

    sets = generate_training_session(exercise=args.exercise, seed=args.seed)
    analyses = analyze_session(sets)

    # ASCII-only console text: the demo runs on stock Windows terminals (cp1252), where a
    # stray em-dash prints as mojibake. Pointer text is already ASCII.
    print(f"wristset demo - session seed {args.seed}, {args.exercise}")
    print("(form score only; effort half - RIR/RPE - lands in Phases 6-8)\n")
    for a in analyses:
        if a.meta.set_type != "working":
            continue
        print(_format_card(a))
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m wristset.demo`
    sys.exit(main())
