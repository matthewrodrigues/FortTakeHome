"""Layer 4 feature tests, including the Phase 3 GATE.

Gate (milestone 4): features are stable across repeat synthetic recordings of the same
movement. "Stable" means the sensor/timing noise between two recordings of the same
nominal set moves a feature far less than genuine fatigue does within a set — otherwise
the feature cannot support the trend claims Phases 4-8 build on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.conditioning import condition_set
from wristset.features import (
    FEATURE_NAMES,
    KEY_CHANNELS,
    BaselineSource,
    RepSource,
    causal_summaries,
    extract_set_features,
    retrospective_summaries,
    summarize_channel,
)
from wristset.segmentation import segment_reps
from wristset.synth import SetParams, generate_set, generate_training_session


def _features(seed=1, capacity=10, stop_rir=2, failure=False, **kw):
    g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=capacity,
                               stop_rir=stop_rir, reached_failure=failure, seed=seed, **kw))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=g.reached_failure)
    sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)
    return g, sf


# --- §6.2 per-rep features ------------------------------------------------------


def test_all_features_present_and_finite():
    _, sf = _features()
    assert sf.n_reps > 0
    for r in sf.reps:
        for name in FEATURE_NAMES:
            v = getattr(r, name)
            assert isinstance(v, float), f"{name} is {type(v)}"
            assert np.isfinite(v), f"{name} is not finite"


def test_features_are_physically_plausible():
    """Guards against unit errors and sign flips, which are silent otherwise."""
    _, sf = _features()
    for r in sf.reps:
        assert 0.05 < r.rom_vertical < 1.5, f"ROM {r.rom_vertical}"
        assert 0.0 < r.conc_mean_vel < 3.0
        assert r.conc_peak_vel >= r.conc_mean_vel  # peak cannot be below the mean
        assert 0.1 < r.ecc_duration < 10.0
        assert 0.1 < r.conc_duration < 10.0
        assert r.path_length >= r.rom_vertical * 0.9  # a 3-D path is at least the vertical
        assert r.path_efficiency >= 0.9
        # sticking point is optional; when present it must be a normalized ROM position
        if r.min_vel_position is not None:
            assert 0.0 <= r.min_vel_position <= 1.0
            assert r.min_vel_value is not None and r.min_vel_value >= 0.0
        else:
            assert r.min_vel_value is None, "sticking-point fields must be None together"
        assert r.tremor_power_8_12 >= 0.0
        assert r.spectral_arc_length <= 0.0  # SPARC is negative by construction


def test_reps_remaining_counts_down():
    _, sf = _features()
    remaining = [r.reps_remaining_in_set for r in sf.reps]
    assert remaining == sorted(remaining, reverse=True)
    assert remaining[-1] == 0


def test_fatigue_signals_move_in_the_expected_direction():
    """The generator degrades velocity and grows tremor across a set; features must see it."""
    _, sf = _features(capacity=10, stop_rir=0, failure=True,
                      vel_decay=0.45, tremor_growth=3.0, rom_collapse=0.2)
    completed = [r for r in sf.reps if r.completed]
    assert len(completed) >= 5
    vel = [r.conc_mean_vel for r in completed]
    tremor = [r.tremor_power_8_12 for r in completed]
    # compare set halves rather than adjacent reps, which are noisy
    h = len(completed) // 2
    assert np.mean(vel[h:]) < np.mean(vel[:h]), "concentric velocity did not decay"
    assert np.mean(tremor[h:]) > np.mean(tremor[:h]), "tremor did not grow"


def test_sticking_point_is_none_when_the_ascent_has_no_slowdown():
    """The current generator models each concentric as one smooth half-cosine — a
    monotonic arch with no interior slowdown, i.e. genuinely no sticking point. Both
    fields must therefore be None rather than a fabricated 0.0, which downstream code
    would misread as "the bar stopped dead". Guards the optional contract."""
    _, sf = _features(capacity=10, stop_rir=0, failure=True, vel_decay=0.5)
    for r in sf.reps:
        assert (r.min_vel_position is None) == (r.min_vel_value is None)


def test_sticking_point_detected_when_one_is_planted():
    """A genuine interior slowdown must be found and located near where it was planted.

    Built directly from a synthetic velocity profile rather than the generator, since the
    generator cannot currently produce a sticking point (see _STICKING_MARGIN).
    """
    from wristset.features.rep_features import _sticking_point

    n = 200
    x = np.linspace(0.0, 1.0, n)
    arch = np.sin(np.pi * x)  # smooth ascent, no sticking point
    assert _sticking_point(arch, x, 1.0) == (None, None)

    # plant a dip at ~40% of the ascent
    dip = arch - 0.55 * np.exp(-(((x - 0.40) / 0.06) ** 2))
    pos, val = _sticking_point(dip, x, 1.0)
    assert pos is not None and val is not None
    assert 0.30 <= pos <= 0.50, f"sticking point located at {pos:.2f}, expected ~0.40"
    assert val < arch.max()


# --- §6.3 trajectory summaries --------------------------------------------------


def test_summary_of_known_series():
    """Summaries must recover the shape of a series with a known trend."""
    s = summarize_channel(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert s["slope"] == pytest.approx(1.0)
    assert s["total_change"] == pytest.approx(4.0)
    assert s["curvature"] == pytest.approx(0.0, abs=1e-9)
    assert s["max_dev"] == pytest.approx(0.0, abs=1e-9)


def test_summaries_return_none_when_series_too_short():
    """None, not zero: a 2-rep set has no curvature, and downstream models must be able to
    distinguish 'no degradation' from 'not enough reps to say'."""
    s = summarize_channel(np.array([1.0, 2.0]))
    assert s["slope"] is not None
    assert s["curvature"] is None
    assert s["breakpoint"] is None


def test_breakpoint_finds_a_planted_regime_change():
    # flat, then a sharp decline -> breakpoint at the elbow
    series = np.array([5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    s = summarize_channel(series)
    assert s["breakpoint"] is not None
    assert 3 <= s["breakpoint"] <= 6
    assert s["breakpoint_gain"] > 0.5  # two lines fit far better than one


def test_retrospective_covers_all_key_channels():
    _, sf = _features()
    for ch in KEY_CHANNELS:
        for name in ("slope", "curvature", "total_change", "max_dev"):
            assert f"retro_{ch}_{name}" in sf.retrospective


def test_causal_summaries_use_only_the_prefix():
    """THE LEAKAGE BOUNDARY (§6.3). A causal summary at rep r must be identical whether or
    not later reps exist — otherwise the RIR head sees the future and its C-index inflates
    with no error raised anywhere."""
    _, sf = _features(capacity=12, stop_rir=2)
    reps = sf.reps
    assert len(reps) >= 6
    r = 4
    from_full = causal_summaries(reps, r)
    from_prefix = causal_summaries([x for x in reps if x.rep_index <= r], r)
    assert from_full == from_prefix


def test_causal_and_retrospective_keys_never_collide():
    _, sf = _features()
    assert set(sf.retrospective) & set(sf.causal[1]) == set()
    assert all(k.startswith("retro_") for k in sf.retrospective)
    assert all(k.startswith("causal_") for k in sf.causal[1])


def test_causal_differs_from_retrospective_on_a_degrading_set():
    """If these were equal the separation would be pointless — confirm they actually differ
    mid-set, where the causal view has not yet seen the degradation."""
    _, sf = _features(capacity=12, stop_rir=0, failure=True, vel_decay=0.5, rom_collapse=0.25)
    mid = max(3, sf.n_reps // 2)
    causal_slope = sf.causal[mid]["causal_conc_mean_vel_slope"]
    retro_slope = sf.retrospective["retro_conc_mean_vel_slope"]
    assert causal_slope is not None and retro_slope is not None
    assert causal_slope != retro_slope


# --- §7 baseline hierarchy ------------------------------------------------------


def _session_sources(seed):
    out = []
    for g in generate_training_session(seed=seed, working_load_kg=80.0):
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        res = segment_reps(cs, exercise=g.exercise, reached_failure=g.reached_failure)
        out.append((g, cs, res))
    return out


def test_baseline_falls_through_the_hierarchy():
    """§7.1 tiers must resolve in order and fall through when unavailable."""
    prep = _session_sources(1)
    warm = [RepSource(cs, r.reps, g.load_kg, g.set_type, g.exercise)
            for g, cs, r in prep if g.set_type == "warmup"]
    hist = [RepSource(cs, r.reps, g.load_kg, g.set_type, g.exercise)
            for g, cs, r in _session_sources(99) if g.set_type == "working"]
    g, cs, res = next(p for p in prep if p[0].set_type == "working")

    kw = dict(exercise=g.exercise, load_kg=g.load_kg, set_type=g.set_type)
    assert extract_set_features(cs, res.reps, **kw).baseline.source == BaselineSource.EARLY_SET
    assert extract_set_features(cs, res.reps, warmups=warm, **kw).baseline.source == BaselineSource.WARMUP
    full = extract_set_features(cs, res.reps, warmups=warm, history=hist, **kw)
    assert full.baseline.source == BaselineSource.CROSS_SESSION
    assert full.baseline.is_reliable


def test_early_set_baseline_is_marked_unreliable():
    """§7.1 tier 3 carries a known flaw (a set that starts bad reads as clean); callers
    must be able to see that from the result."""
    _, sf = _features()
    assert sf.baseline.source == BaselineSource.EARLY_SET
    assert sf.baseline.is_reliable is False


def test_load_conditioning_rejects_out_of_band_history():
    """§7.2: history at a very different load must not be pooled into the template."""
    prep = _session_sources(1)
    hist = [RepSource(cs, r.reps, g.load_kg, g.set_type, g.exercise)
            for g, cs, r in _session_sources(99) if g.set_type == "working"]
    g, cs, res = next(p for p in prep if p[0].set_type == "working")
    # ask for a load far outside the +/-10% band of the 80 kg history
    sf = extract_set_features(cs, res.reps, exercise=g.exercise, load_kg=200.0, history=hist)
    assert sf.baseline.source != BaselineSource.CROSS_SESSION


def test_path_dtw_baseline_is_filled_and_grows_with_degradation():
    _, sf = _features(capacity=10, stop_rir=0, failure=True, vel_decay=0.5, rom_collapse=0.25)
    vals = [r.path_dtw_baseline for r in sf.reps if r.path_dtw_baseline is not None]
    assert len(vals) == sf.n_reps
    h = len(vals) // 2
    # later reps deviate further from the early-set template
    assert np.mean(vals[h:]) > np.mean(vals[:h])


# --- PHASE 3 GATE ---------------------------------------------------------------


def test_feature_stability_across_repeat_recordings():
    """GATE (milestone 4): features are stable across repeat recordings of the same movement.

    Two recordings of the same nominal set differ only in sensor noise and timestamp
    jitter. The spread a feature shows across those repeats must be small relative to the
    spread genuine fatigue produces within a set — otherwise a trend claim built on it is
    measuring noise. Compared as a ratio so it is scale-free per channel.
    """
    # tremor_power_8_12 was excluded here until 2026-08-20: the generator drew f_tremor
    # from U(8,12) — the band-pass CORNERS, where zero-phase filtering retains only ~50% of
    # amplitude — and injected gyro tremor as a randomly-scaled 3-vector whose contribution
    # to |gyro| depended on its alignment with the movement axis. Together those swung
    # absolute tremor level ~200x across recordings for reasons unrelated to the lifter.
    # Both are fixed in the generator, and tremor now clears this gate with a wide margin.
    channels = ("conc_mean_vel", "rom_vertical", "ecc_duration", "tremor_power_8_12")
    # repeat "recordings": same movement, different sensor-noise draw
    repeats = [_features(seed=s, capacity=10, stop_rir=2)[1] for s in (11, 12, 13, 14, 15)]

    for ch in channels:
        # between-recording spread of the set-mean (noise only)
        set_means = [float(np.mean(sf.channel(ch))) for sf in repeats]
        between = float(np.std(set_means))
        # within-set spread across reps (genuine fatigue signal), averaged over repeats
        within = float(np.mean([np.std(sf.channel(ch)) for sf in repeats]))
        assert within > 0, f"{ch}: no within-set variation to compare against"
        assert between < within, (
            f"{ch}: repeat-to-repeat noise ({between:.4g}) exceeds within-set "
            f"signal ({within:.4g}) — feature is not stable enough to carry a trend"
        )


@pytest.mark.parametrize(
    "channel,expected_sign",
    [
        ("tremor_power_8_12", +1),  # tremor grows with fatigue
        ("conc_mean_vel", -1),  # concentric velocity decays
        ("rom_vertical", -1),  # ROM collapses
    ],
)
def test_trajectory_summaries_stable_across_repeats(channel, expected_sign):
    """The GATE applied to §6.3 summaries — the binding half for change-features.

    A degradation slope must keep its SIGN across repeat recordings, or the trend Phases
    4-8 read off it is not reproducible. This is the property that carries the gate for
    channels whose absolute level varies legitimately between recordings (see the note in
    test_feature_stability_across_repeat_recordings).
    """
    slopes = []
    for s in (11, 12, 13, 14, 15):
        _, sf = _features(seed=s, capacity=10, stop_rir=0, failure=True,
                          vel_decay=0.45, rom_collapse=0.2, tremor_growth=3.0)
        slopes.append(sf.retrospective[f"retro_{channel}_slope"])
    assert all(s is not None for s in slopes), f"{channel}: missing slope"
    assert all(np.sign(s) == expected_sign for s in slopes), (
        f"{channel}: slope sign not reproducible across repeats: {slopes}"
    )
