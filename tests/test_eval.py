"""Cross-phase metric harness tests (Phase 9).

The harness re-derives every phase gate's headline number as shippable code. These tests
check that it *runs* and that its structure is honest — they deliberately do NOT re-assert
the gate values themselves, which each phase's own test file already owns. Duplicating the
thresholds here would mean two places to update and two chances to disagree.
"""

from __future__ import annotations

from wristset.eval import EvalReport, Metric, run_evaluation
from wristset.eval.metrics import (
    _conditioning_metrics,
    _divergence_metrics,
    _form_metrics,
    _segmentation_metrics,
)


def test_metric_formats_a_missing_value_as_na():
    assert Metric("l", "m", None, "target").format_value() == "n/a"


def test_report_ignores_context_only_rows_when_deciding_pass():
    """A Metric with passed=None is context (e.g. a baseline), not a gate."""
    r = EvalReport([Metric("l", "gate", 1.0, "t", True),
                    Metric("l", "context", 0.5, "context", None)])
    assert r.all_passed is True


def test_report_fails_when_any_gate_fails():
    r = EvalReport([Metric("l", "a", 1.0, "t", True), Metric("l", "b", 0.0, "t", False)])
    assert r.all_passed is False


def test_table_renders_every_metric_and_its_note():
    r = EvalReport([Metric("layer", "metric", 0.5, "gate", True, note="a note")])
    table = r.to_table()
    assert "metric" in table and "0.500" in table and "PASS" in table and "a note" in table


def test_table_is_ascii_only():
    """The harness runs on stock Windows terminals (cp1252), where a section sign or an
    em-dash prints as mojibake. Regression: the first version mojibaked every layer label."""
    report = EvalReport([
        m for fn in (_conditioning_metrics, _form_metrics) for m in fn(seeds=(1,))
    ])
    report.to_table().encode("ascii")  # raises if any non-ASCII slipped in


# --- the cheap layers run for real ----------------------------------------------


def test_conditioning_metrics_measure_rom_error():
    ms = _conditioning_metrics(seeds=(1,))
    assert [m.name for m in ms] == ["per-rep ROM error (mean)", "per-rep ROM error (worst)"]
    assert all(0.0 <= m.value < 1.0 for m in ms)


def test_segmentation_reports_the_degraded_regime_separately():
    """Averaging the degraded regime into the headline would hide it (the Phase-2 lesson)."""
    ms = _segmentation_metrics(seeds=(1,))
    names = [m.name for m in ms]
    assert "rep-count exact match" in names
    assert "rep-count exact (degraded)" in names


def test_form_metric_separates_clean_from_degraded():
    ms = _form_metrics(seeds=(1,))
    assert ms[0].value > 0, "clean sets must score above degraded ones"
    assert ms[0].passed is True


def test_divergence_metric_reports_recall_and_a_control():
    """A recall number alone is not evidence — the same flag must fire less often on
    aligned reports, or it is just always-on."""
    ms = _divergence_metrics(corpus_seed=5)
    assert len(ms) == 2
    recall, control = ms[0].value, ms[1].value
    assert control < recall


def test_run_evaluation_covers_every_layer():
    """The point of the harness: one command, every phase, no layer silently missing."""
    report = run_evaluation(quick=True)
    layers = {m.layer for m in report.metrics}
    for expected in ("conditioning", "segmentation", "form (8.3)", "RIR (9)",
                     "RPE (10)", "divergence (11)"):
        assert expected in layers, f"{expected} missing from the harness"
    assert report.elapsed_s > 0
