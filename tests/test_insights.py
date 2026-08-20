"""Execution-narrative tests (§8.6-8.7, form half) — Phase 5.

The narrative is generated separately from the score, ordering the largest deviations
first (§8.6). "Largest" is measured by score DEFICIT (100 - subscore), which is comparable
across subscores, NOT by raw pointer magnitude (whose units differ per channel).
"""

from __future__ import annotations

from wristset.scoring import FormSubscore, FormSubscores, Pointer
from wristset.insights import execution_narrative


def _sub(name, score, *pointers, provisional=True):
    return FormSubscore(name, score, provisional, pointers=list(pointers))


def _p(name, text, mag=1.0):
    return Pointer(name, text, mag, feature=name)


# --- ordering & suppression (§8.6) ----------------------------------------------


def test_worst_subscore_is_reported_first():
    form = FormSubscores([
        _sub("b", 80.0, _p("b", "BBB path note")),
        _sub("a", 40.0, _p("a", "AAA depth note")),
    ])
    text = execution_narrative(form)
    assert text.index("AAA depth note") < text.index("BBB path note")


def test_subscores_above_threshold_are_suppressed():
    form = FormSubscores([
        _sub("clean", 98.0, _p("clean", "CLEAN noise note")),
        _sub("bad", 40.0, _p("bad", "BAD real note")),
    ])
    text = execution_narrative(form, min_deficit=15.0)
    assert "BAD real note" in text
    assert "CLEAN noise note" not in text


def test_clean_set_reports_form_held():
    form = FormSubscores([
        _sub("a", 99.0, _p("a", "tiny note")),
        _sub("b", 97.0, _p("b", "tiny note 2")),
    ])
    text = execution_narrative(form)
    assert "Form held across the set." in text


def test_max_points_limits_the_number_of_pointer_sentences():
    form = FormSubscores([
        _sub("a", 30.0, _p("a", "worst note")),
        _sub("b", 50.0, _p("b", "second note")),
        _sub("c", 60.0, _p("c", "third note")),
    ])
    text = execution_narrative(form, max_points=1)
    assert "worst note" in text
    assert "second note" not in text


# --- lead line / provenance (§8.4 honesty) --------------------------------------


def test_lead_line_reports_the_form_score():
    form = FormSubscores([_sub("a", 72.0, _p("a", "note"))])
    assert "72" in execution_narrative(form)


def test_provisional_sets_are_labelled():
    form = FormSubscores([_sub("a", 72.0, _p("a", "note"), provisional=True)])
    assert "provisional" in execution_narrative(form).lower()


def test_none_scored_subscores_are_ignored():
    form = FormSubscores([
        _sub("gone", None, _p("gone", "should not appear")),
        _sub("bad", 40.0, _p("bad", "real note")),
    ])
    text = execution_narrative(form)
    assert "should not appear" not in text
    assert "real note" in text
