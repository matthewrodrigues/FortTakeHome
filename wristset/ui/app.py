"""Streamlit demo + annotation UI (§12).

Two audiences, one page. For a **reader coming cold**, every section carries plain-language
exposition (see :mod:`wristset.ui.glossary`) explaining what the abbreviations mean, what
the peaks and troughs on the chart are, and how to read each score. For **debugging**, the
same page overlays detected rep boundaries and ZUPT anchors on the conditioned signal and
shows detected-vs-reported rep counts — segmentation errors silently corrupt every
downstream feature, so this view exists to catch them (§6.1, §12).

Run:  ``streamlit run wristset/ui/app.py``  (requires the ``ui`` extra: ``uv sync --extra ui``)
"""

from __future__ import annotations

import numpy as np

try:
    import plotly.graph_objects as go
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - import-safety guard
    raise SystemExit(
        "The annotation UI needs the 'ui' extra. Install with: uv sync --extra ui"
    ) from exc

from wristset.demo import SetAnalysis, analyze_set, fit_demo_rir_model
from wristset.features import KEY_CHANNELS
from wristset.ui import glossary as gl
from wristset.synth import SetParams, generate_set


def _synthetic(seed: int, exercise: str, capacity: int, failure: bool, degraded: bool):
    kw = (
        dict(vel_decay=0.5, rom_collapse=0.22, tremor_growth=3.0, ecc_shorten=0.35,
             path_wobble_m=0.03)
        if degraded
        else {}
    )
    rom = 0.55 if exercise == "back_squat" else 0.45
    return generate_set(
        SetParams(
            exercise=exercise, rom_m=rom, capacity=capacity,
            stop_rir=0 if failure else 2, reached_failure=failure, seed=seed, **kw,
        )
    )


@st.cache_resource(show_spinner="Building training history (first run only)...")
def _rir_head(seed: int, n_users: int):
    """Fit (and cache) the hazard head for the sidebar's history slider.

    Two layers of caching, because this is the only slow step on the page:

    * ``st.cache_resource`` keeps it out of Streamlit's per-widget reruns.
    * ``fit_demo_rir_model`` additionally caches the fitted head to disk, so the ~9 s of
      generating and conditioning the corpus is paid once *ever* rather than once per app
      start. The corpus is fully determined by ``(seed, n_users)`` and the fit is convex,
      so a cached head is byte-identical to a freshly fitted one.

    Returns ``(model, readiness)``, with ``model=None`` when there is not yet enough
    history to fit a trustworthy estimate.
    """
    if n_users <= 0:
        return None, None
    return fit_demo_rir_model(seed=seed, n_users=n_users)


def main() -> None:
    st.set_page_config(page_title="wristset", layout="wide")
    st.title("wristset · what your wrist says about a set")
    st.markdown(gl.WHAT_THIS_IS)
    with st.expander("Glossary - the terms used on this page"):
        st.markdown(gl.definition_list(gl.CORE_TERMS))

    with st.sidebar:
        st.header("Simulate a set")
        exercise = st.selectbox("Exercise", ["bench_press", "back_squat"])
        st.caption("These sliders change the simulated lift the app then analyses.")
        capacity = st.slider("Max reps this lifter could do", 4, 14, 8)
        failure = st.checkbox("Taken to failure", value=True)
        degraded = st.checkbox("Heavy fatigue (form breaks down)", value=False)
        seed = st.number_input("Random seed (changes the simulated lifter)", 0, 9999, 1, step=1)

        st.header("Training history")
        st.caption(
            "The reps-in-reserve estimate is a learned model, so it only appears once "
            "enough past sets taken to failure exist to fit it. Drag this down to see it "
            "stay locked."
        )
        rir_users = st.slider("Simulated lifters of past data", 0, 14, 10)

    rir_model, readiness = _rir_head(int(seed), int(rir_users))
    if readiness is not None:
        (st.success if rir_model is not None else st.info)(readiness.explain())

    g = _synthetic(int(seed), exercise, capacity, failure, degraded)
    # Single source for the raw -> form-score wiring, shared with the CLI demo.
    analysis = analyze_set(g, rir_model=rir_model)
    cs, res = analysis.cs, analysis.result

    c1, c2, c3 = st.columns(3)
    c1.metric("Reported reps", g.reported_reps)
    c2.metric("Detected completed", res.n_completed,
              delta=res.n_completed - g.reported_reps)
    c3.metric("Signal quality", "low-confidence" if cs.low_confidence else "ok")
    if cs.quality.get("flags"):
        st.warning("Flags: " + ", ".join(cs.quality["flags"]))
    with st.expander("What 'Signal quality' checks"):
        st.markdown(gl.QUALITY_GUIDE)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs.t, y=cs.disp_vert, name="Height (m)"))
    fig.add_trace(go.Scatter(x=cs.t, y=cs.v_vert, name="Speed (m/s)",
                             yaxis="y2", line=dict(width=1)))

    # stationary anchors
    fig.add_trace(go.Scatter(
        x=cs.t[cs.anchor_idx], y=cs.disp_vert[cs.anchor_idx], mode="markers",
        name="Still moments", marker=dict(color="gray", size=6, symbol="x")))

    # rep boundaries
    for r in res.reps:
        color = "green" if r.completed else "red"
        fig.add_vrect(x0=r.t_start, x1=r.t_end, fillcolor=color, opacity=0.08,
                      line_width=0)
        fig.add_vline(x=r.t_bottom, line=dict(color=color, width=1, dash="dot"))

    fig.update_layout(
        height=560, hovermode="x unified",
        yaxis=dict(title="height (m)"),
        yaxis2=dict(title="speed (m/s)", overlaying="y", side="right"),
        # Legend above the plot rather than inside it: horizontal entries were
        # colliding. `itemwidth` sets a minimum px per entry, which is what
        # actually forces the gap; `y=1.08` lifts it clear of the plot area.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08,
            xanchor="left", x=0, itemwidth=90, tracegroupgap=30,
        ),
        margin=dict(t=90),
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander("How to read this chart"):
        st.markdown(gl.CHART_GUIDE)

    st.subheader("Detected reps")
    st.caption(
        "One row per rep the app found in the signal above. It was not told where the "
        "reps were - it located them from the motion alone."
    )
    st.dataframe(
        [
            {
                "rep": r.rep_index,
                "completed": r.completed,
                "t_start": round(r.t_start, 2),
                "t_bottom": round(r.t_bottom, 2),
                "t_end": round(r.t_end, 2),
                "rom_vertical_m": round(r.rom_vertical, 3),
                "recovery": round(r.recovery, 2),
            }
            for r in res.reps
        ],
        width="stretch",
    )
    with st.expander("What these columns mean"):
        st.markdown(gl.definition_list(gl.REP_TABLE))
    st.caption(
        f"Ground truth (simulator's correct answer): failure_rep="
        f"{g.ground_truth.failure_rep}, "
        f"true ROMs (completed) = "
        f"{[round(r.rom_true_m, 3) for r in g.reps if r.completed]}"
    )

    _scoring_view(analysis)
    _features_view(analysis)


def _scoring_view(a: SetAnalysis) -> None:
    """§8.1-8.6 set-score panel: composite, divergence, and both narratives.

    On a **divergent** set (§8.5) the layout inverts — the narratives lead and the number
    is demoted to a caption — because that is exactly the case where a single number
    collides two different sets and is least informative.
    """
    if not a.result.reps:
        return
    fs, c = a.form, a.composite
    st.divider()
    st.subheader("Set score")
    with st.expander("How the set score works"):
        st.markdown(gl.SCORE_GUIDE)

    divergent = c is not None and c.divergent

    if divergent:
        st.warning(
            "**Effort and execution disagree on this set.** A single number averages them "
            "into something misleading — read the notes below, not the score."
        )
        _narratives(a)

    if c is not None and c.total is not None:
        label = "Set score" + (" (de-emphasised — see above)" if divergent else "")
        if divergent:
            st.caption(f"{label}: {round(c.total)} / 100")
        else:
            st.metric(label, f"{round(c.total)} / 100")
        cols = st.columns(3)
        cols[0].metric("Effort", "—" if c.effort is None else f"{round(c.effort)}")
        cols[1].metric("Form", "—" if c.form is None else f"{round(c.form)}")
        cols[2].metric("Components", ", ".join(c.components_used))
        if "rir" not in c.components:
            st.caption(
                "Reps-in-reserve not included — needs enough sets taken to failure "
                "before it can be estimated (§9.5). The composite is renormalised over "
                "the components present rather than filling in a default."
            )
        if c.provisional:
            st.caption(
                "Provisional: this lifter has no past data, so scores are scaled against "
                "generic thresholds rather than their own history."
            )

    st.markdown("**Form subscores** - four equal-weight views of how the movement held up")
    with st.expander("What each subscore measures"):
        st.markdown(gl.definition_list(gl.SUBSCORES))
    for s in fs.subscores:
        if s.score is None:
            st.write(f"**{s.name}** · n/a (no reference)")
            continue
        st.write(f"**{s.name}** · {round(s.score)}")
        st.progress(min(max(s.score / 100.0, 0.0), 1.0))

    if not divergent:
        _narratives(a)


def _narratives(a: SetAnalysis) -> None:
    """§8.6 effort + execution narratives, in that order (effort leads when present)."""
    if a.effort_text:
        st.markdown("**How hard was it?**")
        st.info(a.effort_text)
        if a.divergence is not None and a.divergence.alert:
            st.caption(
                f"The reported effort sits at the {a.divergence.percentile:.0%} percentile "
                f"of what the movement implied - i.e. only {a.divergence.percentile:.0%} of "
                f"the model's estimates were this easy or easier. Flagged when it falls "
                f"outside the middle 80%."
            )
    st.markdown("**How did the movement hold up?**")
    st.info(a.narrative)


def _features_view(a: SetAnalysis) -> None:
    """§6.2-6.3 feature panel — the Phase 3 gate's visual sanity check (milestone 4)."""
    cs, res, g = a.cs, a.result, a.meta
    if not res.reps:
        return
    sf = a.features

    st.divider()
    st.subheader("Per-rep measurements")
    st.caption(
        "The raw measurements every score above is built from - one row per rep. This is "
        "what makes a score debuggable: if a number looks wrong, it traces back to here."
    )
    b = sf.baseline
    st.caption(
        f"Compared against: **{b.source}** ({b.n_reps} reps). "
        + ("A trustworthy reference." if b.is_reliable else
           "Fallback only - this set is compared against its OWN opening reps, so a set "
           "that started badly will still look consistent.")
    )

    with st.expander("What every column means"):
        st.markdown(gl.as_markdown_table(gl.FEATURE_TABLE))
    st.dataframe(
        [
            {
                "rep": r.rep_index,
                "done": r.completed,
                "conc_vel": round(r.conc_mean_vel, 3),
                "peak_vel": round(r.conc_peak_vel, 3),
                "ecc_vel": round(r.ecc_mean_vel, 3),
                "rom_m": round(r.rom_vertical, 3),
                "ecc_s": round(r.ecc_duration, 2),
                "conc_s": round(r.conc_duration, 2),
                "tempo": round(r.tempo_ratio, 2),
                "pause_s": round(r.bottom_pause, 2),
                "path_eff": round(r.path_efficiency, 2),
                "horiz_m": round(r.horiz_excursion, 3),
                # None when the ascent has no interior slowdown (§6.2 sticking point)
                "stick_pos": round(r.min_vel_position, 2) if r.min_vel_position is not None else None,
                "stick_vel": round(r.min_vel_value, 3) if r.min_vel_value is not None else None,
                "tremor": f"{r.tremor_power_8_12:.2e}",
                "sparc": round(r.spectral_arc_length, 2),
                "dtw_base": round(r.path_dtw_baseline, 4) if r.path_dtw_baseline else None,
            }
            for r in sf.reps
        ],
        width="stretch",
    )

    # Degradation trajectories — each channel on its own normalized axis, since they carry
    # different units. Normalizing to rep 1 shows SHAPE, which is what §6.3 summarises.
    st.subheader("How each measurement changed across the set")
    st.caption(
        "Each line is one measurement divided by its own value on rep 1, so they share an "
        "axis despite having different units. **1.0 = unchanged from the first rep.** "
        "Lines falling below 1.0 are degrading (speed, depth); lines rising above it are "
        "growing (tremor). The steeper the line, the faster that aspect broke down."
    )
    tfig = go.Figure()
    for ch in ("conc_mean_vel", "rom_vertical", "ecc_duration", "tremor_power_8_12"):
        series = np.array([getattr(r, ch) for r in sf.reps], dtype=float)
        if series.size == 0 or not np.isfinite(series).all() or abs(series[0]) < 1e-12:
            continue
        tfig.add_trace(go.Scatter(
            x=[r.rep_index for r in sf.reps], y=series / series[0],
            mode="lines+markers", name=ch,
        ))
    tfig.update_layout(
        height=360, hovermode="x unified",
        xaxis=dict(title="rep index"),
        yaxis=dict(title="value relative to rep 1"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08,
            xanchor="left", x=0, itemwidth=90, tracegroupgap=30,
        ),
        margin=dict(t=90),
    )
    st.plotly_chart(tfig, width="stretch")

    st.subheader("Trend summaries")
    st.caption(
        "Each trajectory above, reduced to numbers. **slope** = average change per rep "
        "(negative = declining). **curvature** = whether the decline is accelerating. "
        "**total_change** = fractional change from first rep to last. **breakpoint** = "
        "the rep index where a two-piece trend fits better than one straight line, i.e. "
        "roughly where things started going wrong."
    )
    rows = []
    for ch in KEY_CHANNELS:
        rows.append({
            "channel": ch,
            **{
                name: (round(v, 4) if isinstance(v, float) else v)
                for name in ("slope", "curvature", "total_change", "max_dev",
                             "breakpoint", "breakpoint_gain")
                for v in [sf.retrospective.get(f"retro_{ch}_{name}")]
            },
        })
    st.dataframe(rows, width="stretch")
    st.caption(
        "These summarise the whole set, so they are only used for scores computed after "
        "the set ends. The reps-in-reserve estimate, which predicts DURING the set, gets "
        "a separate version built only from reps seen so far - otherwise it would be "
        "using the future to predict the future."
    )


if __name__ == "__main__":
    main()
