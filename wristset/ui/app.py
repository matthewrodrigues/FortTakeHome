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

from wristset.conditioning import condition_set
from wristset.segmentation import segment_reps
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
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise=exercise, reached_failure=g.reached_failure)

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


if __name__ == "__main__":
    main()
