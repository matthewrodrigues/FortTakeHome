# Phase 7 — RPE ordinal hierarchical head (Layer 5c, §10)

Status: approved 2026-08-20. Adds `wristset/models/rpe/`. Reuses Phase-6 pipeline prep
(`rir.prepare_sets`) and Phase-3 `SetFeatures.retrospective`. Consumes
`GroundTruth.reported_rpe` (label) and `GroundTruth.rpe_bias` (eval only).

## Goal

Predict reported RPE as an **ordinal, hierarchical** target (§10). Two structural facts
rule out plain regression: RPE is ordinal with unequal intervals (§10.1), and calibration is
per-user with thin data (§10.3). The head's real job (§10.5) is the user's **subjective
mapping** — the per-user effect `b_u` is the object of interest, capturing systematic over-
or under-reporting relative to what the movement shows. That is what makes it non-redundant
with the RIR head and what §11's divergence signal needs.

The generator makes this concrete: `reported_rpe = clip(10 − stop_rir, 6, 10) + rpe_bias +
N(0, rpe_noise)`, half-point rounded. So the population `x·β` chases the mechanical part and
`b_u` recovers `rpe_bias` — a falsifiable gate.

## Resolved decisions

1. **Backend: scipy hand-rolled**, mirroring Phase 6 — no modelling deps, deterministic.
   The point head yields a PMF over levels, which suffices for Phase 8 (the §11 divergence
   percentile is computed in the RIR head's distribution, not the RPE posterior). NumPyro's
   full posterior is a documented future upgrade.
2. **Ordinal form: CORAL / proportional-odds.** A single shared effort score `η = x·β + b_u`
   with ordered thresholds `τ_1 ≤ … ≤ τ_{K-1}`; `P(RPE > k) = σ(η − τ_k)`. Rank-monotonicity
   is guaranteed by construction; few parameters (thin-data friendly); `b_u` is a clean
   scalar bias term to recover (§10.5). Chosen over CORN, whose per-threshold classifiers
   lack a single `b_u` and need a post-hoc monotonicity fix.
3. **User effect: ridge-penalized `b_u`** = convex partial pooling (Phase-6 precedent):
   `l2_user` shrinks thin/new users toward 0 (population), so a zero-set user defaults to
   the population model and a heavy user personalizes (§10.3 shrinkage, deterministically).

## §A — Data (`models/rpe/dataset.py`)

`LEVELS = [6.0, 6.5, …, 10.0]` (9 levels, 8 thresholds). Model is **set-level** (one RPE per
set). Feature columns (§10.4 — whole-set retrospective is legitimate because RPE is reported
*after* the set):

- `reported_reps`, `load_kg`
- `retro_{conc_mean_vel,rom_vertical,ecc_duration,tremor_power_8_12}_total_change`
  from `SetFeatures.retrospective` (`None → 0.0` on sets too short to summarize)
- exercise → drop-first dummies (in the model, as Phase 6)

`RpeMeta`: `set_id, user_id, exercise, load_kg, reported_reps, reported_rpe,
true_rpe_bias` (last is eval-only, never a feature). `RpeDataset`: `X, y` (level index 0..8),
`user_idx, users, exercise, columns, levels`. `prepare_rpe_sets(generated)` calls
`rir.prepare_sets` for `SetFeatures` (single pipeline wiring, reused) and builds each
`RpeMeta` from the paired `GeneratedSet.ground_truth`.

## §B — Model (`models/rpe/ordinal.py`)

`OrdinalRpeModel.fit(ds, *, l2_user=1.0, l2_ridge=0.5)`:

- Standardize numerics (store `mu/sd`); design `= [z(numerics), exercise dummies]` (no own
  intercept — the thresholds carry the offset). `η_i = design_i·β + b_{u_i}`.
- Ordered thresholds `τ_k = τ_1 + Σ_{j≤k} softplus(δ_j)` (monotone increasing by
  construction). `P(y_i > k) = σ(η_i − τ_k)`.
- `NLL = Σ_i Σ_k BCE(1[y_i > k], σ(η_i − τ_k)) + ½·l2_ridge·Σβ² + ½·l2_user·Σb_u²`
  (thresholds unpenalized). scipy `L-BFGS-B`; convex in `(β, τ, b_u)` under the ordered
  parameterization ⇒ deterministic from a fixed init (`τ` seeded from empirical cumulative
  level frequencies; `β, b_u = 0`).
- Predict: `predict_proba(X, exercise, user) -> PMF (n×9)` via cumulative differences;
  `predict_level` (argmax → LEVELS value); `expected_rpe` (Σ level·PMF); `user_effect(user)`
  → `b_u` or 0.0 for unseen.

## §C — Eval (`models/rpe/eval.py`)

- `ordinal_accuracy_within(model, ds, tol_levels)` — fraction of sets with `|pred_idx −
  true_idx| ≤ tol_levels`. **The gate tolerance is set from a measured, disclosed
  definition** ("level" = half-point grid step; ±1 level = ±0.5 RPE unless measurement shows
  that is unreasonably strict, in which case ±2 grid steps = ±1.0 RPE, stated explicitly).
- `bias_recovery(model, meta) -> r` — Pearson correlation of fitted `b_u` against
  `true_rpe_bias` across users (§10.5 verification).
- `by_user_split` — small local helper (users, not sets), deterministic.

## §D — Tests / gate (`tests/test_rpe.py`)

- **Milestone-7 gate**, by-user, **multi-seed** (Phase-6 lesson — never one corpus): ordinal
  accuracy within tol on the majority of held-out sets, and `bias_recovery` correlation
  strongly positive.
- **Units:** PMF rows sum to 1; `P(y>k)` monotone decreasing in k (rank consistency); a
  higher-effort set predicts higher RPE; a planted large-positive-`rpe_bias` user fits a
  positive `b_u`; determinism (two fits identical); unseen user → `b_u = 0`.
- **§10.4 boundary (mirror of the RIR guard):** assert `retro_` columns ARE present in the
  RPE design — the whole-set summaries the RIR head is forbidden are legitimate here.

## Out of scope (later phases)

Composite/divergence use of RPE (§8, §11 → Phase 8); NumPyro posterior; CORN; per-block RPE
targets; cross-validation beyond one by-user split.

## Verification

1. `uv run pytest tests/test_rpe.py -q` — units + gate green across seeds.
2. `uv run pytest` — full suite still green.
3. REPL: fit on a population, print predicted vs reported RPE on a held-out user and the
   `b_u`↔`rpe_bias` scatter correlation.
