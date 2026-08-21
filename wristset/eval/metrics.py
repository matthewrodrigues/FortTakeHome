"""Cross-phase metric harness (Phase 9) — every gate number in one place.

`python -m wristset.eval` runs each layer's headline metric against the generator's ground
truth and prints a single table. This is the "model validation" step of the plan's
end-to-end verification: it re-derives the numbers the phase gates assert, but as shippable
code rather than test-only helpers, so the same figures can be quoted, tracked over time, or
regenerated after a change.

**Every number here is synthetic-validated** — measured against the synthetic generator's
own labels, not real lifting. Where a metric has an honest and a flattering form, this
reports the honest one and says so:

* Segmentation is reported on the *operating distribution* (clean/mild/moderate), with the
  degraded regime shown separately rather than averaged in.
* RIR discrimination is ``completed_rep_c_index`` (~0.82), not the headline C-index (~0.99)
  which mostly measures how separable the generator's failed attempt is.
* RPE accuracy is reported on a **label-balanced** corpus with its majority-class baseline
  alongside, because the RIR-oriented corpus concentrates a third of its labels on RPE 10.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Metric", "EvalReport", "run_evaluation"]


@dataclass
class Metric:
    """One measured number, with the context needed to read it honestly."""

    layer: str
    name: str
    value: float | None
    target: str                      # what the phase gate asks for
    passed: bool | None = None
    note: str = ""

    def format_value(self) -> str:
        if self.value is None:
            return "n/a"
        return f"{self.value:.3f}" if abs(self.value) < 100 else f"{self.value:.1f}"


@dataclass
class EvalReport:
    metrics: list[Metric] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in self.metrics if m.passed is not None)

    def to_table(self) -> str:
        width = max((len(m.layer) for m in self.metrics), default=5) + 1
        rows = [
            f"{'layer':<{width}} {'metric':<30} {'value':>8}  {'gate':<20} {'':4}",
            "-" * (width + 66),
        ]
        for m in self.metrics:
            mark = "" if m.passed is None else ("PASS" if m.passed else "FAIL")
            rows.append(
                f"{m.layer:<{width}} {m.name:<30} {m.format_value():>8}  "
                f"{m.target:<20} {mark:<4}"
            )
            if m.note:
                rows.append(f"{'':{width}} {'':30} {'':8}  {m.note}")
        return "\n".join(rows)


# --------------------------------------------------------------------------------
# per-layer metrics
# --------------------------------------------------------------------------------


def _conditioning_metrics(seeds=(1, 2, 3)) -> list[Metric]:
    """§5 gate: reconstructed per-rep ROM within 15% of the generator's true ROM."""
    from wristset.conditioning import condition_set
    from wristset.synth import SetParams, generate_set

    errs: list[float] = []
    for seed in seeds:
        g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=10,
                                   stop_rir=2, reached_failure=False, seed=seed))
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        for r in g.reps:
            if not r.completed:
                continue
            m = (cs.t >= r.t_start) & (cs.t <= r.t_end)
            if not m.any():
                continue
            est = float(cs.disp_vert[m].max() - cs.disp_vert[m].min())
            errs.append(abs(est - r.rom_true_m) / r.rom_true_m)

    mean_err = float(np.mean(errs))
    worst = float(np.max(errs))
    return [
        Metric("conditioning", "per-rep ROM error (mean)", mean_err, "< 0.15",
               mean_err < 0.15, f"n={len(errs)} reps"),
        Metric("conditioning", "per-rep ROM error (worst)", worst, "< 0.15", worst < 0.15),
    ]


def _segmentation_metrics(seeds=range(1, 4)) -> list[Metric]:
    """§6.1 gate: >=95% exact rep-count match on the operating distribution."""
    from wristset.conditioning import condition_set
    from wristset.segmentation import segment_reps
    from wristset.synth import SetParams, generate_set

    grades = {
        "clean": {},
        "mild": dict(vel_decay=0.35, rom_collapse=0.12, tremor_growth=2.5),
        "moderate": dict(vel_decay=0.45, rom_collapse=0.20, tremor_growth=3.0,
                         ecc_shorten=0.35, path_wobble_m=0.03),
        "hard": dict(vel_decay=0.55, rom_collapse=0.28, tremor_gain=0.4,
                     tremor_growth=3.5, ecc_shorten=0.4, path_wobble_m=0.04),
    }

    def rate(kw) -> float:
        match = total = 0
        for ex, rom in (("bench_press", 0.45), ("back_squat", 0.55)):
            for cap in (6, 10):
                for failure in (True, False):
                    for seed in seeds:
                        stop = 0 if failure else 2
                        g = generate_set(SetParams(exercise=ex, rom_m=rom, capacity=cap,
                                                   stop_rir=stop, reached_failure=failure,
                                                   seed=seed, **kw))
                        cs = condition_set(g.raw, reported_reps=g.reported_reps)
                        res = segment_reps(cs, exercise=ex,
                                           reached_failure=g.reached_failure)
                        match += res.n_completed == g.reported_reps
                        total += 1
        return match / total

    operating = float(np.mean([rate(grades[g]) for g in ("clean", "mild", "moderate")]))
    hard = rate(grades["hard"])
    return [
        Metric("segmentation", "rep-count exact match", operating, ">= 0.95",
               operating >= 0.95, "operating distribution (clean/mild/moderate)"),
        Metric("segmentation", "rep-count exact (degraded)", hard, "guard >= 0.75",
               hard >= 0.75, "degraded regime, reported separately not averaged in"),
    ]


def _form_metrics(seeds=(1, 2, 3)) -> list[Metric]:
    """§8.3 gate: form subscores separate clean from degraded sets."""
    from wristset.conditioning import condition_set
    from wristset.features import extract_set_features
    from wristset.scoring import score_form
    from wristset.segmentation import segment_reps
    from wristset.synth import SetParams, generate_set

    clean = dict(vel_decay=0.08, rom_collapse=0.02, tremor_growth=1.1, ecc_shorten=0.05)
    degraded = dict(vel_decay=0.5, rom_collapse=0.25, tremor_growth=3.2, ecc_shorten=0.35)

    def score(seed, kw) -> float:
        g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=10,
                                   stop_rir=0, reached_failure=True, seed=seed, **kw))
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        res = segment_reps(cs, exercise="bench_press", reached_failure=True)
        sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)
        return score_form(sf).form_score

    c = [score(s, clean) for s in seeds]
    d = [score(s, degraded) for s in seeds]
    gap = float(np.mean(c) - np.mean(d))
    return [
        Metric("form (8.3)", "clean - degraded score gap", gap, "> 0 (separates)",
               min(c) > max(d), f"clean {np.mean(c):.0f} vs degraded {np.mean(d):.0f}"),
    ]


def _rir_metrics(corpus_seeds=(5, 11, 23)) -> list[Metric]:
    """§9.6 gates, by-user held out, averaged across independent corpora."""
    from wristset.models.rir import (
        HazardModel,
        build_person_period,
        by_user_split,
        calibration,
        completed_rep_c_index,
        prepare_sets,
        rir_distribution,
    )
    from wristset.synth import generate_population

    cc, mad, mae = [], [], []
    for seed in corpus_seeds:
        corpus = prepare_sets(generate_population(n_users=24, sets_per_user=6, seed=seed))
        train, test = by_user_split(corpus, frac_test=0.3, seed=0)
        model = HazardModel.fit(build_person_period(train))
        cc.append(completed_rep_c_index(model, test))
        mad.append(calibration(model, build_person_period(test)).mad)
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
        mae.append(float(np.mean(errs)))

    m_cc, m_mad, m_mae = float(np.mean(cc)), float(np.mean(mad)), float(np.mean(mae))
    return [
        Metric("RIR (9)", "completed-rep concordance", m_cc, ">= 0.65", m_cc >= 0.65,
               "honest metric: excludes the trivially-separable failed attempt"),
        Metric("RIR (9)", "calibration MAD", m_mad, "<= 0.30", m_mad <= 0.30,
               "varies 0.09-0.24 across corpora; a known open limitation"),
        Metric("RIR (9)", "E[RIR] mean abs error (reps)", m_mae, "<= 2.0", m_mae <= 2.0,
               "dominated by the 9.4 forward projection"),
    ]


def _rpe_metrics(corpus_seeds=(5, 11, 23)) -> list[Metric]:
    """§10 gates on a LABEL-BALANCED corpus, with the majority baseline alongside."""
    import collections

    from wristset.models.rpe import (
        LEVELS,
        OrdinalRpeModel,
        bias_recovery,
        build_rpe_dataset,
        by_user_split,
        ordinal_accuracy_within,
        prepare_rpe_sets,
    )
    from wristset.synth import generate_rpe_population

    acc, base, rs = [], [], []
    for seed in corpus_seeds:
        labeled = prepare_rpe_sets(
            generate_rpe_population(n_users=24, sets_per_user=6, seed=seed))
        train, test = by_user_split(labeled, seed=0)
        dtr, dte = build_rpe_dataset(train), build_rpe_dataset(test)
        model = OrdinalRpeModel.fit(dtr)
        acc.append(ordinal_accuracy_within(model, dte, tol_rpe=1.0))
        majority = LEVELS[collections.Counter(dtr.y).most_common(1)[0][0]]
        base.append(float(np.mean(np.abs(LEVELS[dte.y] - majority) <= 1.0 + 1e-9)))
        rs.append(bias_recovery(model, train))

    m_acc, m_base, m_r = float(np.mean(acc)), float(np.mean(base)), float(np.mean(rs))
    return [
        Metric("RPE (10)", "accuracy within +/-1 RPE", m_acc, "majority of sets",
               m_acc >= 0.55, "label-BALANCED corpus (the skewed one reads ~0.74)"),
        Metric("RPE (10)", "majority-class baseline", m_base, "context",
               None, f"model lift over baseline: {m_acc - m_base:+.3f}"),
        Metric("RPE (10)", "b_u vs injected bias (r)", m_r, ">= 0.55", m_r >= 0.55,
               "balanced corpus reads 0.72 mean (0.60-0.80); the skewed one reads 0.84"),
    ]


def _divergence_metrics(corpus_seed: int = 5) -> list[Metric]:
    """§11.3: does the divergence flag fire on planted perception mismatches?"""
    from wristset.divergence import divergence_from_rir
    from wristset.models.rir import (
        HazardModel,
        build_person_period,
        by_user_split,
        prepare_sets,
        rir_distribution,
    )
    from wristset.synth import generate_population

    corpus = prepare_sets(generate_population(n_users=24, sets_per_user=6, seed=corpus_seed))
    train, test = by_user_split(corpus, frac_test=0.3, seed=0)
    model = HazardModel.fit(build_person_period(train))

    planted, controls = [], []
    for sf, meta in test:
        if not (meta.reached_failure and sf.reps):
            continue
        completed = [r for r in sf.reps if r.completed]
        if not completed:
            continue
        pred = rir_distribution(model, sf, completed[-1].rep_index,
                                exercise=meta.exercise, user_id=meta.user_id)
        # planted mismatch: a lifter reporting far below what the movement showed
        planted.append(divergence_from_rir(pred.dist, 6.5).direction == "under_reporting")
        # control: reporting in line with a near-failure set
        controls.append(divergence_from_rir(pred.dist, 10.0).direction == "under_reporting")

    recall = float(np.mean(planted))
    false_rate = float(np.mean(controls))
    return [
        Metric("divergence (11)", "planted mismatch detected", recall, ">= 0.80",
               recall >= 0.80, f"n={len(planted)} held-out failure sets"),
        Metric("divergence (11)", "same flag on aligned reports", false_rate, "< recall",
               false_rate < recall, "direction must separate the two report levels"),
    ]


def run_evaluation(*, quick: bool = False) -> EvalReport:
    """Run every layer's headline metric and collect them into one report.

    ``quick`` trims the corpus seeds for the model heads (the expensive part) so the harness
    can be smoke-tested without the full multi-corpus run.
    """
    t0 = time.perf_counter()
    seeds = (5,) if quick else (5, 11, 23)

    metrics: list[Metric] = []
    metrics += _conditioning_metrics()
    metrics += _segmentation_metrics()
    metrics += _form_metrics()
    metrics += _rir_metrics(seeds)
    metrics += _rpe_metrics(seeds)
    metrics += _divergence_metrics()

    return EvalReport(metrics=metrics, elapsed_s=time.perf_counter() - t0)
