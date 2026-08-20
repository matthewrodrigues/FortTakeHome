# wristset

Prototype pipeline that turns wrist-worn smartwatch IMU data into per-set resistance-training
analysis: deterministic **form subscores**, an estimated **RIR** (reps-in-reserve) hazard head,
a hierarchical **RPE** head, and the **divergence** between reported and mechanically-estimated
exertion.

See `wrist-imu-set-analysis-architecture.md` for the full system architecture, and
`~/.claude/plans/hey-claude-can-you-lovely-dongarra.md` for the phased development plan.

## Status

Under construction — **Phase 0 (Foundation)**. Capture (watchOS) is deferred; the system is
developed against a **synthetic IMU generator** that emits ground-truth labels. Every
model-derived number is therefore *synthetic-validated* until real data collection lands.

## Layout

```
wristset/
  common/        # shared math (quaternion rotation, etc.)
  contract/      # raw-timeseries + metadata schemas (the capture<->pipeline boundary)
  synth/         # synthetic IMU generator (ground-truth labeled sets)
  storage/       # Layer 1: Parquet + SQLite stores
  conditioning/  # Layer 2  (Phase 1)
  segmentation/  # Layer 3  (Phase 2)
  features/      # Layers 4 & 7 (Phase 3) — per-rep features, trajectories, baselines
  models/rir/    # Layer 5b (Phase 6)
  models/rpe/    # Layer 5c (Phase 7)
  scoring/       # Layer 5a (Phase 8)
  divergence/    # Layer 6  (Phase 8)
  insights/      # narratives (Phase 8)
  ui/            # Streamlit annotation + demo (Phase 2+)
  eval/          # metric harness (Phase 9)
```

## Setup

```bash
uv sync                 # core deps
uv sync --extra dev     # + pytest
uv run pytest           # run the test suite
```
