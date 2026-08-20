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

from wristset.demo import SetAnalysis, analyze_set
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

    g = _synthetic(int(seed), exercise, capacity, failure, degraded)
    # Single source for the raw -> form-score wiring, shared with the CLI demo.
    analysis = analyze_set(g)
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
    """§8.3-8.4 form-score panel (Phase 5). Form half only — the effort half (RIR/RPE) and
    the composite arrive in Phase 8, so this never fabricates a composite (§8.7)."""
    if not a.result.reps:
        return
    fs = a.form
    st.divider()
    st.subheader("Layer 5a · form subscores (§8.3)")

    score = fs.form_score
    head = "—" if score is None else f"{round(score)} / 100"
    tag = " · provisional" if fs.provisional else ""
    st.metric("Form score (effort half pending)", head)
    if fs.provisional:
        st.caption("Provisional — no personal history; §8.4 fixed-scale normalization" + tag)

    for s in fs.subscores:
        if s.score is None:
            st.write(f"**{s.name}** · n/a (no reference)")
            continue
        st.write(f"**{s.name}** · {round(s.score)}")
        st.progress(min(max(s.score / 100.0, 0.0), 1.0))

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
