"""The demo page must be readable by someone who did not build it (Phase 9 polish).

These tests guard the exposition rather than the layout: every abbreviation the UI shows
has a definition, and no spec section reference ("§8.2") leaks into user-facing text.
"""

from __future__ import annotations

import pytest

from wristset.ui import glossary as gl


def test_every_core_term_has_a_real_definition():
    for term, body in gl.CORE_TERMS.items():
        assert len(body) > 40, f"{term} definition is too thin to help anyone"
        assert term.lower() not in body.lower().split()[:2], (
            f"{term} is defined circularly"
        )


def test_the_abbreviations_the_ui_shows_are_all_defined():
    """RIR/RPE/ROM/IMU appear as bare acronyms on the page; each must be spelled out."""
    for acronym, expansion in (
        ("RIR", "reps in reserve"),
        ("RPE", "rate of perceived exertion"),
        ("ROM", "range of motion"),
        ("IMU", "inertial measurement unit"),
    ):
        assert acronym in gl.CORE_TERMS
        assert expansion in gl.CORE_TERMS[acronym].lower()


def test_every_feature_column_is_explained():
    """The per-rep table is the densest jargon on the page — sparc, dtw_base, stick_pos."""
    from wristset.features import FEATURE_NAMES  # noqa: F401  (import guards the contract)

    for opaque in ("sparc", "dtw_base", "stick_pos", "tremor", "path_eff", "tempo"):
        assert opaque in gl.FEATURE_TABLE, f"{opaque} appears in the UI with no definition"
        what, _ = gl.FEATURE_TABLE[opaque]
        assert len(what) > 15


def test_opaque_acronyms_are_expanded_where_they_are_used():
    assert "dynamic time warping" in gl.FEATURE_TABLE["dtw_base"][1].lower()
    assert "spectral arc length" in gl.FEATURE_TABLE["sparc"][0].lower()
    assert "zero-velocity" in gl.CHART_GUIDE.lower()


def test_chart_guide_explains_peaks_and_troughs():
    """The most common question about the signal plot."""
    guide = gl.CHART_GUIDE.lower()
    assert "trough" in guide and "peak" in guide
    assert "bottom of each rep" in guide


def test_all_four_subscores_are_described():
    from wristset.scoring import SUBSCORE_NAMES

    for name in SUBSCORE_NAMES:
        assert name in gl.SUBSCORES, f"{name} shown in the UI with no explanation"


def test_glossary_prose_is_ascii():
    """Rendered into a Windows console in the CLI path and copied into docs; keep it safe."""
    for block in (gl.WHAT_THIS_IS, gl.CHART_GUIDE, gl.SCORE_GUIDE):
        block.encode("ascii")


def test_renderers_produce_markdown():
    assert gl.definition_list(gl.CORE_TERMS).startswith("- **IMU**")
    table = gl.as_markdown_table(gl.FEATURE_TABLE)
    assert table.startswith("| column |") and "`sparc`" in table


@pytest.mark.parametrize("attr", ["WHAT_THIS_IS", "CHART_GUIDE", "SCORE_GUIDE"])
def test_orientation_prose_avoids_spec_references(attr):
    """A reader coming cold has not read the architecture doc; section numbers mean
    nothing to them and read as noise."""
    assert "§" not in getattr(gl, attr)
