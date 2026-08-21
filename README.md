# wristset

Prototype pipeline that turns wrist-worn smartwatch IMU data into per-set resistance-training
analysis: deterministic **form subscores**, an estimated **RIR** (reps-in-reserve) hazard head,
a hierarchical **RPE** head, and the **divergence** between reported and mechanically-estimated
exertion.

See `wrist-imu-set-analysis-architecture.md` for the full system architecture, and
`~/.claude/plans/hey-claude-can-you-lovely-dongarra.md` for the phased development plan.

## Status

Capture (watchOS) is deferred; the system is developed against a **synthetic IMU generator**
that emits ground-truth labels. Every model-derived number below is therefore
*synthetic-validated* — measured against the generator's own ground truth, not real lifting.

### How to read this project

**Phases 1–5 are the setup: a well-validated deterministic pipeline.** Raw IMU → conditioned
signals → rep segmentation → per-rep features → form subscores → narrative. Everything here
is signal processing and arithmetic, validated against ground truth on held-out seeds:

| Layer | Gate | Result |
|---|---|---|
| Conditioning (§5) | ROM within 15% of true | 2.4% mean error, 80/80 reps |
| Segmentation (§6.1) | ≥95% exact rep count | 100% operating dist., 98.7% held-out |
| Features (§6.2–6.3) | stable across repeat recordings | passes level + trend stability |
| Form subscores (§8.3) | separate clean from degraded | 87–91 vs 45–51, no overlap |

These have **known design limitations that cost real-world accuracy**, documented rather
than hidden. The largest is rep counting: segmentation scores 100% on synthetic but 16.8%
exact on an external real corpus, because the approach (prominent minima of vertical
displacement + rhythmicity) assumes clean set boundaries that real recordings do not have,
and because that corpus contains only out-of-vocabulary exercises. With more time the right
move is a different formulation — the evidence points at a periodicity-first or learned
approach over the current peak-finding with accumulated guards. See §14 of the architecture
doc for the full list.

**Phases 6+ are the actual product features**, and their *formulation* still needs work:

- **Phase 6 — RIR hazard head (§9).** The most solid of the three. Discrete-time survival
  with a censored likelihood, verified equivalent to the person-period form to 1e-12.
  Honest discrimination (completed reps only, excluding the trivially-separable failed
  attempt) is **0.819**; E[RIR] mean absolute error is **1.80 reps**. The dominant error
  source is §9.4's linear forward projection, which is a documented approximation.
- **Phase 7 — RPE ordinal head (§10).** Weaker. Accuracy within ±1 RPE on held-out users is
  **0.595** on a label-balanced corpus against a 0.365 majority baseline (**+0.23 lift**).
  The headline number is 0.738 on the RIR-oriented corpus, but that is inflated by label
  skew — the same model, honestly measured. `b_u` does recover the injected per-user bias
  (r = **0.72–0.84**), which is the §10.5 claim that makes the head non-redundant with RIR.
  It regresses toward the middle of the scale, so low-RPE sets are served poorly.
- **Phase 8 — composite score, divergence signal, insights (§8.1-8.6, §11).** Combines the
  three heads into one set score and adds the reported-vs-mechanical comparison. Built:

  - **Composite (§8.1-8.2)** — `scoring/composite.py`. Effort and form split the total
    50/50 (`0.25·RIR + 0.25·RPE + 0.5·form`); §8.1's rationale is that `RPE ≈ 10 − RIR`, so
    treating the three as equal categories would hand the effort construct 2/3 of the score.
    Both effort terms are scored as **proximity to a target**, not "more is better" —
    overshooting is penalised symmetrically, because a set pushed past failure with
    assistance is not superior to one that hit its target. Targets and tolerances are
    module-level parameters from day one.
  - **Missing components renormalise, never default.** Any of the three inputs may be
    absent; the remaining §8.1 weights are renormalised over what exists. With form alone
    the composite equals the form score, so the Phase-5 path is the degenerate case of the
    new one rather than a separate branch.
  - **Divergence signal (§11)** — `divergence/signal.py`. Locates the reported RPE as a
    **percentile within the RIR-implied distribution** (`RPE_mech = 10 − RIR`) rather than
    as a raw difference, so model uncertainty is built into the threshold: when the
    distribution is wide, fewer sets trigger (§11.3). Alerts outside the 10th/90th
    percentile as `under_reporting` / `over_reporting`.
  - **Divergence flag (§8.5)** — a composite where `|effort − form| > 30` is collapsing two
    genuinely different sets onto one number (effort 100/form 40 and effort 40/form 100
    both score 70). When it fires, both the CLI and the UI **lead with the narrative and
    de-emphasise the number** — in the UI the large score metric is suppressed entirely and
    demoted to a caption.
  - **Effort narrative (§8.6)** — `insights/effort.py`. Reports a **range**, never a
    decimal: `E[RIR]` carries ~1.8 reps of error, so "estimated 3-5 reps left at set end"
    is the honest summary and "4.2 reps left" would imply resolution the model lacks.
  - **RIR as a data-gated feature (§9.5)** — `models/rir/readiness.py`. Reps-in-reserve is
    a capability that **unlocks as data accumulates**, not something shown from a corpus too
    thin to identify it. Below 15 failure sets it is withheld and the composite renormalises
    over form and RPE; 15-29 it is shown labelled provisional; at 30+ it is settled.
    Censored sets deliberately do not count — they inform the hazard where they reached but
    never observe a failure, so they cannot identify the high-fatigue end of the curve.
    This mirrors §7.3's baseline cold start: capability appears with data, and absence is
    reported as absence.

  Gate met: the composite ranks good above degraded sets and above sets that missed their
  effort target; the divergence signal fires on planted perception-vs-mechanics mismatches
  using real fitted-model distributions on held-out users. 30 tests.

  **Demo and UI** were updated rather than forked: `analyze_set` takes an optional
  `rir_model`, `python -m wristset.demo --with-rir [--rir-users N]` fits a small population
  first (the demo's own lifter is deliberately *not* in it, so the estimate is the honest
  population cold-start case), and the Streamlit panel gained the composite, both
  narratives, a history slider that exercises the readiness gate, and the §8.5 layout
  inversion.

- **Phase 9 — integration and eval harness.** `python -m wristset.eval` re-derives every
  phase gate's headline number in one table, as shippable code rather than test-only
  helpers, so the same figures can be quoted or regenerated after a change. Where a metric
  has an honest and a flattering form it prints the honest one and says so: segmentation on
  the operating distribution with the degraded regime shown *separately* rather than
  averaged in; RIR discrimination as completed-rep concordance rather than the headline
  C-index; RPE accuracy on the label-balanced corpus with its majority baseline alongside.

  Running it surfaced one thing the phase gates had not: **`b_u` bias recovery is weaker on
  a balanced corpus** (0.72 mean, 0.60-0.80 across seeds) than on the skewed one (0.84).
  Recovering a per-user bias is easier when the mechanical reference barely moves; once RPE
  spans 6-10 the model must fit the effort mapping and separate per-user bias at the same
  time. Same pattern as the accuracy figure — the balanced number is the real one.

  The demo also gained `--rpe-bias`, which plants a per-lifter reporting bias so the
  divergence signal has a genuine mismatch to detect rather than an incidental one.

  **Streamlit demo, written for a reader who did not build it.** Every abbreviation
  (RIR, RPE, ROM, IMU, DTW, SPARC, ZUPT), every feature column, every subscore and every
  chart element now has a plain-language definition, in collapsed expanders next to the
  thing they explain rather than in a separate glossary nobody opens. The chart guide says
  what the peaks and troughs actually are - troughs are the bottom of each rep, the
  velocity line crosses zero at every direction change - and names the check to make: if
  the shaded rep blocks line up with the troughs, detection worked; if not, every number
  below is built on a bad count. Definitions live in `ui/glossary.py` so they can be
  reviewed as prose, with tests asserting every acronym shown is expanded and that no spec
  section reference leaks into user-facing text.

  **Load time: ~9.7s to ~0.4s.** The only slow step is fitting the RIR head, which needs a
  synthetic corpus generated and conditioned. That corpus is fully determined by
  `(seed, n_users)` and the fit is convex from a zero init, so the fitted head (958 bytes)
  is cached to disk and reused across app restarts - previously `st.cache_resource` only
  helped within one running process.

### Deferred: confidence-gated RIR (§8.2)

§8.2 specifies dropping the RIR term from the composite when `rir_predictive_std` is too
wide. **That gate is deferred, deliberately.** Measured on 527 held-out predictions across
three corpora, the predicted distribution's spread is overconfident ~3× (median 0.5 against
median error 1.6), covers only 20–30% of errors within 1σ (a calibrated distribution gives
~68%), and is *anti*-correlated with error (r = −0.08) — the predictions it calls most
certain are the least accurate. Every alternative signal tested (entropy, tail mass, max
probability, projected-hazard saturation, forecast horizon) correlated at |r| ≤ 0.15.

Shipping a gate on any of those would imply a reliability guarantee the model cannot make,
so Phase 8 keeps the RIR term unconditionally. `RIRPrediction.is_confident` remains as a
weak *advisory* horizon annotation, explicitly not a gate. The real fix is a calibrated
uncertainty estimate, which needs the forward projection reworked first.

### A note on the numbers

Several figures here are lower than earlier versions of this project reported. That is
deliberate: each was replaced after measurement showed the flattering version was
measuring an artifact — a C-index of 1.000 that turned out to be reading the generator's
40%-height failed attempt, a gate that passed on a single lucky corpus seed, an RPE
accuracy inflated by a third of labels sitting on one level. The lower numbers are the
real ones.

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
  models/rir/    # Layer 5b (Phase 6) — hazard head, RIR distribution, §9.5 readiness gate
  models/rpe/    # Layer 5c (Phase 7) — CORAL ordinal head, per-user calibration b_u
  scoring/       # Layer 5a — form subscores (Phase 4) + composite/effort (Phase 8)
  divergence/    # Layer 6  (Phase 8) — reported-vs-mechanical percentile (§11)
  insights/      # execution narrative (Phase 5); effort narrative (Phase 8)
  demo.py        # wires raw -> composite + narratives (CLI + UI orchestrator)
  ui/            # Streamlit annotation + demo (Phase 2+)
  eval/          # metric harness (Phase 9)
```

## Demo

```bash
uv run python -m wristset.demo --seed 1              # form + RPE composite (fast)
uv run python -m wristset.demo --seed 1 --with-rir   # + fitted RIR head (~10s)
uv run python -m wristset.demo --seed 1 --with-rir --rir-users 3   # RIR stays locked (§9.5)
uv run python -m wristset.demo --seed 1 --with-rir --rpe-bias -2.5  # planted under-reporter
uv run --extra ui streamlit run wristset/ui/app.py   # interactive

uv run python -m wristset.eval            # every phase gate's number, one table
uv run python -m wristset.eval --quick    # same, one corpus seed (~85s)
```

## Setup

```bash
uv sync                 # core deps
uv sync --extra dev     # + pytest
uv run pytest           # run the test suite
```
