# Phase 6 — RIR hazard head (Layer 5b, §9)

Status: approved 2026-08-20. Adds `wristset/models/rir/` and a multi-user corpus generator
to `synth`. Consumes Phase-3 `SetFeatures.causal` (leakage-safe) and the generator's
`GroundTruth.failure_rep`. First model in the system; the composite (Phase 8) depends on it.

## Goal

Estimate **reps in reserve** as a discrete-time survival problem (§9.1). RIR is unobserved
on any set not taken to failure; right-censoring lets those sets still inform the hazard.
Output a full `P(RIR=k)` distribution per rep (§9.4), not a point estimate.

## Resolved decisions

1. **Fit backend: scipy hand-rolled** (`scipy.optimize`, already installed) — no modelling
   deps, deterministic, and it optimizes the §9.3 censored likelihood *directly*. The
   person-period Bernoulli NLL and the §9.3 product-form likelihood are the same objective
   written two ways; the code and a test make that equivalence explicit.
2. **User effect: ridge-penalized per-user intercept.** A convex, deterministic stand-in
   for the §9.2 random intercept — the L2 penalty *is* partial pooling (λ→∞ ⇒ every user at
   the population intercept ⇒ new-user behavior; small λ ⇒ heavy users personalize). Not a
   true GLMM; documented as the prototype approximation. The user's *subjective* calibration
   is the RPE head's job (§10.5), so the RIR head only needs the user term to absorb
   within-user correlation.

## §A — Multi-user corpus generator (`synth/generator.py`)

`generate_population(*, n_users, sets_per_user, exercises, seed) -> list[GeneratedSet]`.
Each virtual user draws latent traits: `capacity` (strength), working `load_kg`, and
`rpe_bias` (unused until Phase 7 but planted now). Each set samples a **failure/censored**
mix — `stop_rir=0, reached_failure=True` for failures, `stop_rir∈1..4` for censored — sized
so the corpus clears §9.5's **≥30 failure sets per exercise**. `user_id` groups sets for
the by-user split. Fully seeded/deterministic. Reuses `generate_set`; does not touch the
existing single-user helpers.

## §B — Person-period dataset (`models/rir/dataset.py`)

`build_person_period(labeled_sets) -> PersonPeriod`. `labeled_sets` = `(SetFeatures, meta)`
where `meta` carries `exercise, user_id, load_pct_e1rm, failure_rep, reached_failure`.

One row per **attempted** rep. Outcome `y=1` only on the failed attempt (`rep_index ==
failure_rep`); every rep of a censored set, and every completed rep of a failure set, is
`y=0`. Row features — **causal only** (§9.2), asserted by the leakage guard:

- `rep_index`, `load_pct_e1rm`
- `conc_mean_vel(r)`, `path_dtw_baseline(r)` — from `RepFeatures`
- to-date changes = `causal_{conc_mean_vel,ecc_duration,tremor_power_8_12}_total_change` at
  rep r, read from `SetFeatures.causal[r]`; `None → 0.0` ("no change yet" by construction)
- exercise → fixed-effect dummies; `user_id` → ridge-penalized intercept index

`PersonPeriod` holds `X` (numeric design incl. exercise dummies), `user_idx`, `y`, column
names, the ordered user list, and `set_id`/`rep_index` per row for eval. A short helper maps
a `SetFeatures` + rep index to a feature row so predict/projection reuse the same builder.

## §C — Hazard model (`models/rir/hazard.py`)

`HazardModel.fit(pp, *, l2_user=1.0, l2_ridge=1e-3) -> HazardModel`.
`logit h(r) = Xβ + u[user]`. Objective = person-period Bernoulli NLL
`−Σ[y·log h + (1−y)·log(1−h)]` `+ ½·l2_user·Σu²` `+ ½·l2_ridge·Σβ²` (intercept unpenalized).
Minimize with L-BFGS-B from zero init — convex ⇒ unique optimum ⇒ deterministic. Analytic
gradient for speed/stability. `predict_hazard(rows) -> h`; an **unseen user → u=0**
(population fit). Stores `β`, `u`, column order, user index.

## §D — Hazard → RIR (`models/rir/rir.py`)

`rir_distribution(model, set_feats, r, *, K=8) -> RIRPrediction`. Future hazards
`h(r+1..r+K)` come from **linear forward-projection** of the causal trends (increment
`rep_index`; extrapolate `conc_mean_vel` and the to-date changes along their fitted causal
slopes; hold load/exercise/user). Then, with `p_j = h(r+j)`:

```
P(RIR = m) = [∏_{j=1..m} (1 − p_j)] · p_{m+1}      for m = 0..K−1
P(RIR ≥ K) = ∏_{j=1..K} (1 − p_j)                  (tail mass at the cap)
E[RIR]     = Σ m · P(RIR = m)
```

`K=8` cap (§9.4); the projection is a documented approximation and the dominant error
source. `RIRPrediction` carries the distribution vector, `expected_rir`, and `K`.

## §E — Evaluation (`models/rir/eval.py`)

- `c_index(model, sets)` — concordance: over comparable rep pairs, does higher predicted
  hazard match closer-to-failure? Ranks reps within/across failure sets by true RIR.
- `calibration(model, rows, n_bins)` — predicted-hazard bins vs empirical failure rate;
  returns bin centers + observed rates + a mean-abs-deviation-from-diagonal scalar.
- `by_user_split(sets, frac, seed)` — partitions **users** (not sets) into train/test; the
  gate fits on train users and evaluates on held-out users (u=0), measuring generalization
  to new lifters (§9.6). Outputs are labeled "synthetic-validated."

## §F — Tests / gate (`tests/test_rir.py`)

- **Milestone-6 gate:** build population, by-user split, fit; assert C-index meaningfully
  > 0.5 (target ≥ 0.65) and calibration mean-abs-deviation small (near diagonal).
- **Units:** person-period row counts & outcome placement (failure row `y=1`, all else 0;
  censored all-zero); RIR distribution sums to 1 and `E[RIR]` falls monotonically as fatigue
  advances within a failure set; unseen user reproduces the population hazard; the §9.3↔
  person-period likelihood equivalence on a tiny hand-built set.
- **Leakage guard:** assert no `retro_` / whole-set column ever enters `PersonPeriod.X`
  (the Phase-3 boundary, now enforced for the model that most needs it).

## Out of scope (later phases)

RPE head (§10 → Phase 7); composite/effort scoring and confidence gate (§8.1-8.2 → Phase 8);
divergence signal (§11 → Phase 8); cross-validation, post-hoc calibration correction, and
non-linear forward projection.

## Verification

1. `uv run pytest tests/test_rir.py -q` — units + gate green.
2. `uv run pytest` — full suite still green.
3. Quick REPL: fit on a population, print a failure set's per-rep `E[RIR]` — it should
   descend toward 0 at the failure rep; print C-index and the calibration deviation.
