"""End-to-end demo tests (Phase 5, milestone-5 GATE).

The gate: one command runs a fresh synthetic session raw -> form subscores + narrative
with no manual steps, deterministically. These tests exercise the shared orchestrator and
the CLI entry point directly.
"""

from __future__ import annotations

from wristset.demo import SetAnalysis, analyze_session, main
from wristset.features.baseline import BaselineSource
from wristset.synth import generate_training_session


def _session(seed=1):
    return generate_training_session(exercise="bench_press", seed=seed)


# --- shared orchestrator --------------------------------------------------------


def test_analyze_session_scores_every_working_set():
    analyses = analyze_session(_session())
    working = [a for a in analyses if a.meta.set_type == "working"]
    assert working, "expected working sets in the session"
    for a in working:
        assert isinstance(a, SetAnalysis)
        assert a.form.form_score is not None
        assert isinstance(a.narrative, str) and a.narrative


def test_warmups_feed_the_baseline_hierarchy():
    """A working set analysed after the session's warmups must resolve its §7 baseline to
    the warmup tier, not fall through to the early-set fallback."""
    analyses = analyze_session(_session())
    working = [a for a in analyses if a.meta.set_type == "working"]
    assert working[0].features.baseline.source == BaselineSource.WARMUP


def test_session_degradation_shows_up_across_working_sets():
    """generate_training_session degrades later working sets harder; form score should
    trend down across the working sets."""
    working = [a for a in analyze_session(_session()) if a.meta.set_type == "working"]
    scores = [a.form.form_score for a in working]
    assert scores[-1] < scores[0], f"form score did not degrade across sets: {scores}"


# --- CLI gate -------------------------------------------------------------------


def test_cli_runs_end_to_end_and_reports_a_score(capsys):
    rc = main(["--seed", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Form score" in out


def test_cli_output_is_deterministic_for_a_fixed_seed(capsys):
    main(["--seed", "1"])
    first = capsys.readouterr().out
    main(["--seed", "1"])
    second = capsys.readouterr().out
    assert first == second and first, "demo output must be byte-identical across runs"


def test_cli_output_states_measured_change_not_coaching_cues(capsys):
    banned = ("keep your", "chest up", "you should", "tuck", "brace", "squeeze",
              "elbows", "sit back", "drive through")
    main(["--seed", "2"])
    out = capsys.readouterr().out.lower()
    for phrase in banned:
        assert phrase not in out, f"coaching cue {phrase!r} in demo output"


# --- Phase 9: planted-bias demo path --------------------------------------------


def test_planted_rpe_bias_changes_the_reported_rpe(capsys):
    """``--rpe-bias`` gives the divergence signal something real to detect: the same
    session, reported differently by a lifter who systematically under-reports."""
    main(["--seed", "1", "--rpe-bias", "0"])
    neutral = capsys.readouterr().out
    main(["--seed", "1", "--rpe-bias", "-2.5"])
    biased = capsys.readouterr().out
    assert neutral != biased, "planted bias did not reach the reported RPE"


def test_demo_stays_deterministic_with_the_new_flags(capsys):
    main(["--seed", "2", "--rpe-bias", "-1.5"])
    first = capsys.readouterr().out
    main(["--seed", "2", "--rpe-bias", "-1.5"])
    assert capsys.readouterr().out == first
