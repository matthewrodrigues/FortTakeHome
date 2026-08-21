"""Streamlit annotation UI for debugging segmentation (§12).

Build the annotation UI early — it is used constantly to debug segmentation, and
segmentation errors silently corrupt every downstream feature (§12). This view plots the
conditioned vertical displacement / velocity with detected rep boundaries and stationary
anchors overlaid, and reports detected-vs-reported rep counts (the primary segmentation
metric, §6.1).

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


@st.cache_resource(show_spinner="Fitting RIR head on synthetic history…")
def _rir_head(seed: int, n_users: int):
    """Fit (and cache) the hazard head for the sidebar's history slider.

    Streamlit reruns the whole script on every widget change, and building the corpus costs
    seconds — so this is cached on ``(seed, n_users)``. Returns ``(model, readiness)``, with
    ``model=None`` when §9.5's failure-set minimum is not met.
    """
    if n_users <= 0:
        return None, None
    return fit_demo_rir_model(seed=seed, n_users=n_users)


def main() -> None:
    st.set_page_config(page_title="wristset — segmentation", layout="wide")
    st.title("wristset · rep segmentation debugger")

    with st.sidebar:
        st.header("Synthetic set")
        exercise = st.selectbox("Exercise", ["bench_press", "back_squat"])
        capacity = st.slider("Capacity (max reps)", 4, 14, 8)
        failure = st.checkbox("Reached failure", value=True)
        degraded = st.checkbox("Degraded (high fatigue)", value=False)
        seed = st.number_input("Seed", 0, 9999, 1, step=1)

        st.header("RIR history (§9.5)")
        st.caption(
            "Reps-in-reserve unlocks once enough sets have been taken to failure. "
            "Fewer lifters means fewer failure sets, so the estimate stays locked."
        )
        rir_users = st.slider("Synthetic lifters of history", 0, 14, 10)

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
    c3.metric("Quality", "low-confidence" if cs.low_confidence else "ok")
    if cs.quality.get("flags"):
        st.warning("Flags: " + ", ".join(cs.quality["flags"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs.t, y=cs.disp_vert, name="vertical displacement (m)"))
    fig.add_trace(go.Scatter(x=cs.t, y=cs.v_vert, name="vertical velocity (m/s)",
                             yaxis="y2", line=dict(width=1)))

    # stationary anchors
    fig.add_trace(go.Scatter(
        x=cs.t[cs.anchor_idx], y=cs.disp_vert[cs.anchor_idx], mode="markers",
        name="ZUPT anchors", marker=dict(color="gray", size=6, symbol="x")))

    # rep boundaries
    for r in res.reps:
        color = "green" if r.completed else "red"
        fig.add_vrect(x0=r.t_start, x1=r.t_end, fillcolor=color, opacity=0.08,
                      line_width=0)
        fig.add_vline(x=r.t_bottom, line=dict(color=color, width=1, dash="dot"))

    fig.update_layout(
        height=520, hovermode="x unified",
        yaxis=dict(title="displacement (m)"),
        yaxis2=dict(title="velocity (m/s)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected reps")
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
        use_container_width=True,
    )
    st.caption(
        f"Ground truth: failure_rep={g.ground_truth.failure_rep}, "
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
    st.subheader("Layer 5a · set score (§8.1)")

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
            st.caption("Provisional — no personal history; §8.4 fixed-scale normalization.")

    st.markdown("**Form subscores (§8.3)**")
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
        st.markdown("**Effort narrative (§8.6)**")
        st.info(a.effort_text)
        if a.divergence is not None and a.divergence.alert:
            st.caption(
                f"Reported RPE sits at the {a.divergence.percentile:.0%} percentile of the "
                f"mechanically-implied distribution (§11.3 alerts outside 10-90%)."
            )
    st.markdown("**Execution narrative (§8.6)**")
    st.info(a.narrative)


def _features_view(a: SetAnalysis) -> None:
    """§6.2-6.3 feature panel — the Phase 3 gate's visual sanity check (milestone 4)."""
    cs, res, g = a.cs, a.result, a.meta
    if not res.reps:
        return
    sf = a.features

    st.divider()
    st.subheader("Layer 4 · per-rep features (§6.2)")
    b = sf.baseline
    st.caption(
        f"Baseline: **{b.source}** (n={b.n_reps} reps, "
        f"{'reliable' if b.is_reliable else 'provisional — §7.1 early-set fallback'})"
    )

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
        use_container_width=True,
    )

    # Degradation trajectories — each channel on its own normalized axis, since they carry
    # different units. Normalizing to rep 1 shows SHAPE, which is what §6.3 summarises.
    st.subheader("Degradation trajectories (§6.3)")
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
        height=320, hovermode="x unified",
        xaxis=dict(title="rep index"),
        yaxis=dict(title="value relative to rep 1"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(tfig, use_container_width=True)

    st.subheader("Set-level summaries (§6.3, retrospective)")
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
    st.dataframe(rows, use_container_width=True)
    st.caption(
        "Causal summaries (reps 1..r) are computed separately for the RIR head and are "
        "never mixed with these retrospective values — §6.3 leakage boundary."
    )


if __name__ == "__main__":
    main()
