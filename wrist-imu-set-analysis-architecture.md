# Wrist-IMU Set Analysis — System Architecture

**Version:** 0.1 (prototype scope)
**Input:** wrist-worn smartwatch motion data + user-reported exercise, load, and RPE
**Outputs:** per-set form score with subscores, estimated RIR, estimated RPE, and the divergence between estimated and reported exertion

---

## 1. Scope

### 1.1 What the system does

Ingests raw inertial data from a single wrist-worn device during a resistance-training set, segments it into reps, extracts per-rep kinematic features and set-level trajectory summaries, and produces three outputs:

1. **Estimated RIR** — inferred. How many reps the lifter could likely still have completed.
2. **Estimated RPE** — inferred. What the lifter would likely report, given their personal calibration history.
3. **Composite set score** — ¼ RIR proximity to target, ¼ RPE proximity to target, ½ form. Form components are deterministic; effort components consume the two heads above.
4. **Textual insights** — separate effort and execution narratives. These carry the meaning; the score exists for glanceability.

The comparison between estimated and reported RPE is the primary novel signal.

### 1.2 Non-goals for the prototype

- No absolute anatomical claims (e.g. "you hit parallel"). All ROM statements are relative to a user-defined or historical reference.
- No prescriptive coaching cues. Output is descriptive measurement, not causal diagnosis.
- No injury prediction. The system observes deviation, not pathology.
- No multi-device fusion. Single wrist, dominant hand.
- No exercise inference. The user declares the exercise; the system may optionally *verify* it.

### 1.3 Exercise vocabulary (prototype)

Restrict to movements where the wrist is on the implement and therefore acts as a genuine bar/handle path sensor:

| Exercise | Wrist informative? | Notes |
|---|---|---|
| Barbell bench press | Yes | Best case — bar path directly observed |
| Barbell back squat | Yes | Hands fixed on bar; bar path ≈ wrist path |
| Conventional deadlift | Yes | Bar in hand throughout |
| Overhead press | Yes | Clean vertical path, large ROM |
| Barbell row | Yes | Path shape informative |
| Dumbbell curl | Partial | Short ROM; tempo/tremor still usable |
| Push-up | **No** | Hand is the fixed end of the chain — no path, no depth. Tempo and tremor only. Excluded from prototype. |

Start with **bench press and back squat only.** Both have long ROM, clear top/bottom pauses (needed for ZUPT), and well-documented velocity-loss behavior.

---

## 2. System overview

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 0 — CAPTURE                                           │
│ watchOS / WearOS app · IMU @ 100 Hz · session metadata      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1 — STORAGE                                           │
│ Raw immutable timeseries (Parquet) + relational metadata    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2 — SIGNAL CONDITIONING                               │
│ Resample · gravity separation · frame rotation · filtering  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3 — SEGMENTATION                                      │
│ Set boundaries → rep boundaries → phase boundaries          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4 — FEATURE EXTRACTION                                │
│ Per-rep feature vector + set-level trajectory summaries     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌───────────────────┐              ┌───────────────────┐
│ 5b  RIR HEAD      │              │ 5c  RPE HEAD      │
│ discrete-time     │              │ ordinal +         │
│ hazard, censored  │              │ hierarchical      │
│ MLE               │              │ per-user          │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          ├──────────────┬───────────────────┤
          ▼              │                   ▼
┌───────────────────┐    │         ┌───────────────────┐
│ 6  DIVERGENCE     │    │         │ reported_rpe      │
│ reported vs.      │    │         │ (metadata)        │
│ mechanical RPE    │    │         └─────────┬─────────┘
└─────────┬─────────┘    │                   │
          │              ▼                   │
          │   ┌──────────────────────────────┴──────┐
          │   │ 5a  SET SCORE                       │
          │   │ ¼ RIR + ¼ RPE + ½ form              │
          │   │ + form subscores (deterministic,    │
          │   │   direct from Layer 4 features)     │
          │   │ + divergence flag                   │
          │   └──────────────────┬──────────────────┘
          ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 7 — PRESENTATION                                      │
│ Composite score · effort narrative · execution narrative    │
│ Divergence flag suppresses the number, leads with text      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 0 — Capture

### 3.1 Platform

| Platform | API | Notes |
|---|---|---|
| watchOS | `CoreMotion` → `CMDeviceMotion` | Provides fused attitude quaternion, gravity vector, and user (linear) acceleration already separated. Strongly preferred — removes the need to implement your own sensor fusion. |
| WearOS | `SensorManager` + `Health Services` | `TYPE_ROTATION_VECTOR` gives fused orientation; `TYPE_LINEAR_ACCELERATION` gives gravity-removed accel. Availability varies by device. |

**Verify achievable sample rates on your target hardware before committing.** Nominal maxima are often not sustained under battery/thermal constraints, and the delivered rate may be irregular.

### 3.2 Channels to record

| Channel | Units | Purpose |
|---|---|---|
| Linear acceleration (x,y,z) | m/s² | Velocity/displacement integration |
| Rotation rate (x,y,z) | rad/s | Orientation change, tremor |
| Attitude quaternion (w,x,y,z) | — | Frame rotation |
| Gravity vector (x,y,z) | m/s² | Vertical reference |
| Timestamp | monotonic ns | **Device-monotonic, not wall clock** |

### 3.3 Sampling

- **Target 100 Hz.** Rationale: rep fundamental frequency is ~0.3–1 Hz, but physiological tremor sits at 8–12 Hz and jerk estimation needs headroom above that. 50 Hz is a workable floor; below 50 Hz you lose the tremor band to aliasing risk and jerk becomes unreliable.
- **Record actual delivered timestamps.** Do not assume uniform spacing. Watch sensor delivery is frequently irregular, and silently treating irregular samples as uniform corrupts every downstream integration.

### 3.4 Buffering and transfer

- Write to a local ring buffer on device; flush to persistent local storage per set.
- Upload in batch at session end, not in real time. Removes latency and connectivity from the critical path.
- Prototype fallback: dump to file, transfer manually. Do not build sync infrastructure before the algorithm works.

### 3.5 User-supplied metadata (per set)

Required at set close:

| Field | Type | Notes |
|---|---|---|
| `exercise` | enum | From the fixed vocabulary |
| `load_kg` | float | Needed for load-conditioned baselines |
| `reported_reps` | int | Weak supervision for the segmenter |
| `reported_rpe` | float | 6.0–10.0, half-point increments |
| `reached_failure` | bool | **Critical.** Determines censoring for the RIR head. |
| `set_type` | enum | `warmup` \| `working` \| `backoff` — warmups serve as the clean-form reference |

`reached_failure` must be a deliberate, unambiguous prompt ("Did you attempt a rep you could not complete?"), not inferred from RPE. It is the single most valuable label in the system.

---

## 4. Layer 1 — Storage

### 4.1 Raw timeseries

Immutable Parquet, partitioned by `user_id / date / session_id`. One file per set. Never modified after write — all processing produces new derived artifacts.

```
raw/
  user_id=001/
    date=2026-08-18/
      session_id=abc123/
        set_001.parquet
        set_002.parquet
```

### 4.2 Relational metadata

SQLite for the prototype (Postgres later). Minimal schema:

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    device_model TEXT,
    wrist        TEXT CHECK(wrist IN ('left','right'))
);

CREATE TABLE sets (
    set_id          TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(session_id),
    set_index       INTEGER NOT NULL,
    exercise        TEXT NOT NULL,
    load_kg         REAL NOT NULL,
    set_type        TEXT NOT NULL,
    reported_reps   INTEGER NOT NULL,
    reported_rpe    REAL,
    reached_failure BOOLEAN NOT NULL,
    raw_path        TEXT NOT NULL
);

CREATE TABLE reps (
    rep_id      TEXT PRIMARY KEY,
    set_id      TEXT REFERENCES sets(set_id),
    rep_index   INTEGER NOT NULL,     -- 1-based
    t_start     REAL NOT NULL,        -- seconds from set start
    t_bottom    REAL NOT NULL,        -- eccentric/concentric transition
    t_end       REAL NOT NULL,
    completed   BOOLEAN NOT NULL      -- false for a failed attempt
);

CREATE TABLE rep_features (
    rep_id      TEXT PRIMARY KEY REFERENCES reps(rep_id),
    -- one column per feature in §6.2, or a JSON blob during prototyping
    features    TEXT NOT NULL
);
```

The `reps.completed` flag matters: a failed final attempt is a rep *attempt* that supplies the failure event but should be excluded from feature-trend fitting, since its kinematics are qualitatively different.

---

## 5. Layer 2 — Signal conditioning

Ordered pipeline. Each stage is a pure function over the previous stage's output, which keeps it unit-testable.

### 5.1 Resample to uniform grid

Interpolate onto a uniform 100 Hz grid using the recorded timestamps. Linear interpolation is adequate; cubic if gaps are large. **Flag any gap > 50 ms** and mark the affected reps as low-confidence.

### 5.2 Gravity separation

If the platform supplies separated gravity and linear acceleration, use it. Otherwise apply a low-pass filter (~0.3 Hz cutoff) to estimate gravity, subtract for linear acceleration. Platform fusion is better — it uses the gyro, whereas naive low-pass fails under sustained acceleration.

> **Implementation note (Phase 1).** Only the platform-supplied path is built; the ~0.3 Hz
> low-pass fallback is unimplemented, since both the synthetic generator and CoreMotion
> supply the split. The 0.3 Hz constant that exists in `conditioning/frame.py` smooths an
> *already-separated* gravity channel to estimate the steady up direction — it is not the
> fallback separator. Build the fallback only if a source without platform fusion appears.
>
> **Ingest must normalize units and gravity sign.** A smoke test over the 164-file Kaggle
> CoreMotion corpus (2026-08-19) found the corpus is *not* internally consistent: 161 files
> report acceleration and gravity in **g**, 3 in **m/s²**, and those same 3 carry a
> **sign-flipped gravity channel** (gravity rotates to world `+Z`, not `−Z`; conjugating the
> quaternion does not fix it, so this is a source-recording convention, not a handedness
> mismatch). Neither defect raises an error. A blind `× 9.80665` inflates the SI files ~10×,
> and because `up_hat` is normalized the magnitude error re-emerges as a *sign* error rather
> than a scale error — silently plausible output. Any real-data ingest path must therefore
> **measure** `‖gravity‖` per file (a physical constant, so it is a free unit detector)
> rather than assume, and record the gravity sign for the polarity issue noted in §5.3.

### 5.3 Rotate into world frame

**Non-negotiable step.** Using the attitude quaternion `q`, rotate linear acceleration from device frame into a gravity-aligned world frame:

```
a_world = q ⊗ a_device ⊗ q*
```

Without this, the model learns watch orientation on the wrist rather than movement. Yaw is unconstrained without a magnetometer — resolve it per set by defining the horizontal axes relative to the mean horizontal displacement direction of the first rep, so "forward/back" is consistent within a set even if not globally referenced.

**Vertical axis** (`z_world`) is defined by gravity and is reliable. Most high-value features depend only on vertical displacement, so build those first.

> **Implementation note (Phase 1) — two open items for the feature layer (§6.2).**
>
> **1. Per-set yaw resolution is deferred, not done.** `to_world_frame()` leaves the
> horizontal plane in the quaternion's arbitrary-yaw world frame; the first-rep alignment
> prescribed above is unimplemented. Phase 1's gate is vertical-only, and the first consumer
> of horizontal path is the feature layer, so this was deferred rather than guessed at.
> **Consequence:** `a_horiz`, `v_horiz`, and `disp_horiz` on `ConditionedSet` are currently
> **not comparable across sets** and have no accuracy test. Any path-consistency feature
> (§6.2) must resolve yaw first or it will compare incommensurable frames.
>
> **2. `a_vert` polarity is per-file, not global.** `up_hat` is derived from *measured*
> gravity (`−ĝ`) rather than a hardcoded `[0,0,1]`. This is deliberate and it works: on the
> 3 sign-flipped Kaggle files (§5.2) it returned `up_hat = [0,0,−1]`, the true up for that
> file's convention, and produced physically correct ROM with no code change — the
> portability property held on real data. But it means `a_vert` is positive-up *in each
> file's own frame*. **Consequence:** concentric/eccentric direction reads inverted between
> those files and the rest, so any feature keyed on the *sign* of vertical motion — tempo
> phase split, eccentric/concentric duration, peak-velocity direction — must normalize
> polarity at ingest (§5.2) before use. Magnitude-only features (ROM, path length) are
> unaffected.

### 5.4 Filtering

Two parallel branches from the same conditioned signal:

| Branch | Filter | Feeds |
|---|---|---|
| Kinematic | 4th-order zero-phase Butterworth low-pass, ~10–15 Hz | Velocity, displacement, path, tempo |
| Tremor | Band-pass 8–12 Hz | Tremor power features |

Use `scipy.signal.filtfilt` for zero-phase — a causal filter introduces lag that corrupts rep boundary timing.

### 5.5 Velocity and displacement via ZUPT

Naive double integration of accelerometer data diverges quadratically. Bound it with **zero-velocity updates**:

1. Detect stationary intervals: windows where `|a_linear|` and `|ω|` are both below threshold for ≥ 100 ms. These occur at the top of each rep (lockout) and often at the bottom.
2. Force `v = 0` at each stationary interval.
3. Integrate acceleration → velocity **within each rep only**, then apply linear drift correction so velocity returns to zero at the closing stationary interval.
4. Integrate corrected velocity → displacement, same drift correction.

Integration horizon is now ~1–3 s per rep instead of ~60 s per set. Error is bounded and roughly constant across reps, which is what makes rep-to-rep *comparison* valid even if absolute displacement carries bias.

**Known limitation:** touch-and-go reps with no bottom pause give you only one ZUPT anchor per rep instead of two. Accuracy degrades. Detect and flag this; consider requiring a brief pause during data collection for the prototype.

> **Implementation note (Phase 1) — three deviations from the steps above.**
>
> **Gyro is the primary stationary gate, not `|a_linear|`.** Step 1 above treats accel and
> gyro symmetrically; the implementation does not. At the bottom turnaround velocity is zero
> but linear acceleration is at its *maximum* (the bar is being reversed), so an accel-led
> test rejects exactly the bottom anchors it most needs. `GYRO_STATIONARY_THRESH = 0.6 rad/s`
> is the discriminator; `ACC_STATIONARY_THRESH = 3.0 m/s²` is a loose outlier gate only.
> Both magnitudes are low-passed at 3 Hz *before* thresholding, because tremor persists
> during isometric holds and grows with fatigue — unfiltered, it lifts pause-time magnitude
> above threshold and drops anchors on the latest, most diagnostic reps. Validated on real
> data (2026-08-19): 160/161 single-set Kaggle files yielded ≥1 anchor per reported rep,
> median 3.6, **zero files with zero anchors** — the threshold transferred untuned.
>
> **Integration is set-wide, not per-rep.** Step 3 says "within each rep only," which is not
> yet possible: rep boundaries do not exist until Phase 2. `zupt_integrate()` instead
> integrates continuously across the set and subtracts a piecewise-linear drift curve fit
> through *all* anchors (`np.interp`, held flat beyond the outer anchors). This yields the
> same bounded-horizon property — drift is pinned at every anchor, ~1–3 s apart — without
> needing segmentation. Revisit when Phase 2 lands only if per-rep integration measurably
> improves on it; the set-wide form is strictly more general.
>
> **The bounded quantity is per-anchor-segment, not whole-set.** When validating
> displacement, measure excursion *between adjacent anchors*. Whole-file `max − min`
> additionally accumulates residual inter-anchor drift and will look alarming while the
> pipeline is healthy. On the Kaggle corpus: whole-file median 2.32 m vs per-segment median
> 0.192 m (p90 0.49 m, max 0.53 m — real curl/press/row ROMs). The signature that confirms
> correct behaviour is `corr(whole-file, duration) = +0.38` against
> `corr(per-segment, duration) = −0.01`: error grows with recording length, the bounded
> horizon does not. This is the §5.5 claim — absolute displacement carries bias, rep-to-rep
> comparison stays valid — measured on real data.

### 5.6 Quality gates

Reject or flag a set if:
- Sample gaps exceed threshold
- Fewer stationary intervals detected than `reported_reps`
- Detected rep count differs from `reported_reps` by more than 1
- Total set duration is implausible for the rep count

Flagged sets are excluded from model training but still shown to the user with a confidence caveat.

> **Implementation note (Phase 1).** Three of the four gates are live: `sample_gaps`,
> `insufficient_stationary` (`n_stationary < reported_reps`), and `no_stationary_anchors`
> (`n_stationary == 0`, checked independently of `reported_reps` — which is absent whenever
> a set arrives without trusted metadata, as all real-data files do). The remaining two
> gates — detected rep count vs `reported_reps`, and implausible duration-for-rep-count —
> require segmentation and land in Phase 2.
>
> **Touch-and-go detection (§5.5) is deliberately not implemented.** Gyro-led ZUPT recovers
> a near-zero-velocity anchor at each reversal whether or not the lifter dwells, so accuracy
> degrades gracefully rather than failing; and the current synthetic generator models smooth
> reversals only, so the flag **could not be validated even if written**. An unvalidatable
> flag is worse than no flag — it reads as coverage. Revisit when the generator models sharp
> (non-zero-velocity) reversals, or when real touch-and-go data is labelled.

---

## 6. Layers 3–4 — Segmentation and features

### 6.1 Segmentation

**Set detection.** Rolling-window energy threshold on `|a_linear|` distinguishes active from rest. For the prototype, the user explicitly starts/stops the set, so this is a validation check rather than a detection problem.

**Rep segmentation.** Three-stage approach, cheapest first:

1. **Vertical velocity zero-crossings.** For bench and squat, reps are a clean down-up cycle. Sign changes in `v_z` bracketed by stationary intervals give boundaries directly. This alone should work for a majority of clean sets.
2. **Autocorrelation for period estimate.** Gives expected rep duration, used to reject spurious boundaries.
3. **DTW template matching.** Build a template from reps 1–3 of the set (or the user's historical clean template), match forward. Handles degraded late reps where velocity zero-crossings get noisy.

**Validation via reported count.** `reported_reps` is free weak supervision on every set. Track segmentation accuracy as exact-match rate against it. This is your primary segmentation metric and it requires no annotation.

**Phase boundaries within a rep:** `t_start` (top, begin eccentric) → `t_bottom` (velocity sign change) → `t_end` (top, end concentric). For deadlift the order inverts (concentric first); handle per-exercise.

### 6.2 Per-rep features

Computed for every rep. This is the core representation.

#### Kinematic

| Feature | Definition | Rationale |
|---|---|---|
| `conc_mean_vel` | Mean \|v_z\| during concentric | Primary fatigue indicator (velocity-based training) |
| `conc_peak_vel` | Max \|v_z\| during concentric | Sensitive to intent/effort |
| `ecc_mean_vel` | Mean \|v_z\| during eccentric | Control indicator |
| `rom_vertical` | Peak-to-peak vertical displacement | ROM proxy |
| `min_vel_position` | Normalized ROM position of concentric velocity minimum | Sticking point location |
| `min_vel_value` | Velocity at that minimum | Sticking point severity |

#### Tempo

| Feature | Definition | Rationale |
|---|---|---|
| `ecc_duration` | `t_bottom − t_start` | Degrades before concentric velocity does |
| `conc_duration` | `t_end − t_bottom` | Direct effort measure |
| `bottom_pause` | Stationary duration at `t_bottom` | Technique consistency |
| `tempo_ratio` | `ecc_duration / conc_duration` | Normalizes out overall pace |

#### Path

| Feature | Definition | Rationale |
|---|---|---|
| `path_length` | Total 3D arc length | Numerator of efficiency |
| `path_efficiency` | `path_length / rom_vertical` | Wobble/inefficiency scalar |
| `horiz_excursion` | Max horizontal deviation from vertical line through start | Forward drift, bench arc collapse |
| `path_dtw_baseline` | DTW distance to baseline template (see §7) | Direct form-consistency measure |

#### Smoothness / tremor

| Feature | Definition | Rationale |
|---|---|---|
| `jerk_rms` | RMS of d(a)/dt over the rep | Movement smoothness |
| `spectral_arc_length` | SPARC metric on velocity profile | Smoothness measure robust to duration differences |
| `tremor_power_8_12` | Band power, 8–12 Hz, gyro magnitude | Motor unit recruitment shift proxy |

#### Metadata attached to each rep

`rep_index`, `reps_remaining_in_set` (known post hoc), `load_kg`, `load_pct_e1rm` (if a 1RM estimate exists), `exercise`, `set_type`.

### 6.3 Set-level trajectory summaries

Per-rep features treated as independent lose the shape of degradation, which is the informative part. For each of the key per-rep features `f`, compute across the set:

| Summary | Definition |
|---|---|
| `{f}_slope` | OLS slope of `f` vs. `rep_index` |
| `{f}_curvature` | Quadratic coefficient — is degradation accelerating? |
| `{f}_total_change` | `(f_last − f_first) / f_first` |
| `{f}_max_dev` | Largest single-rep deviation from the set's fitted trend |
| `{f}_breakpoint` | Rep index of best two-segment piecewise-linear fit, plus the fit improvement over a single line |

Apply to at minimum: `conc_mean_vel`, `ecc_duration`, `rom_vertical`, `path_dtw_baseline`, `tremor_power_8_12`.

**Note on `breakpoint`:** this doubles as the regime diagnostic. If breakpoints across the five channels cluster at the same rep index across many sets, discrete regime structure exists and an HMM becomes worth revisiting. If they scatter, degradation is smooth and the continuous model is correct. Free answer — the feature is being computed regardless.

Trajectory summaries are also computed **causally** (using only reps 1..r) for the per-rep RIR head, and **retrospectively** (whole set) for the set score and RPE head. Keep these strictly separate to avoid leakage.

---

## 7. Baseline management

Every comparison-based feature requires a reference for "this user's clean form on this exercise at this load."

### 7.1 Baseline hierarchy

Resolve in order, falling through when unavailable:

1. **Cross-session personal template** — pooled reps from the user's historical low-fatigue reps on this exercise, load-conditioned. Best, requires history.
2. **Warmup-set reference** — reps from today's `set_type = warmup` sets. Low load, low fatigue, most likely genuine clean form. Good default.
3. **Early-set reference** — reps 1–3 of the current set. Fallback only. **Known flaw:** if the set starts bad, the whole set reads as clean.

### 7.2 Load conditioning

Form at 90% 1RM differs legitimately from form at 60%. Comparing across loads generates false positives. Either restrict baselines to a load band (±10%), or include `load_pct_e1rm` as a covariate in a regression that predicts expected feature values, and measure deviation as residual.

### 7.3 Cold start

A new user has no baseline. Options:
- Ask for a **calibration set**: 5 reps at a self-declared comfortable load with self-declared correct form. Establishes the template immediately.
- Fall back to population-level expectations for the RPE head (see §9.3), which is what partial pooling handles automatically.
- Suppress baseline-dependent features and show only self-contained ones (tempo, velocity) until enough history exists.

---

## 8. Layer 5a — Set score and insights

Two outputs, deliberately separated:

1. **A composite score (0–100)** — one glanceable number.
2. **Textual insights** — separate effort and execution narratives, which carry the actual meaning.

The form components are deterministic (computed directly from features, no model). The effort components consume the outputs of Layers 5b and 5c.

**Dependency note:** this layer is numbered 5a for continuity with the original design, but it now runs *after* 5b and 5c, not in parallel with them. See the revised flow in §2.

### 8.1 Composite weighting

| Component | Weight | Source |
|---|---|---|
| RIR proximity to target | 1/4 | `E[RIR]` from hazard head (§9.4) |
| RPE proximity to target | 1/4 | `reported_rpe` from set metadata |
| Form | 1/2 | Deterministic subscores (§8.3) |

Effort and execution split 50/50. This deliberately avoids the double-count that arises from `RPE ≈ 10 − RIR`: treating RIR and RPE as two of three equally weighted categories would give the effort construct 2/3 of the total.

```python
effort = 0.5 * (rir_score + rpe_score)
form   = form_score
total  = 0.5 * effort + 0.5 * form
```

### 8.2 Effort components

Both are scored as **proximity to a target**, not as monotone "more is better." Overshooting is penalized — a set taken past failure with assistance is not superior to one that hit the target.

```python
TARGET_RIR = 0.0        # prototype default; later per-user / per-block / per-exercise
TARGET_RPE = 10.0       # prototype default
RIR_TOLERANCE = 3.0
RPE_TOLERANCE = 3.0

rir_score = clamp(100 * (1 - abs(E_rir - TARGET_RIR) / RIR_TOLERANCE), 0, 100)
rpe_score = clamp(100 * (1 - abs(reported_rpe - TARGET_RPE) / RPE_TOLERANCE), 0, 100)
```

**Targets are parameterized from day one even though the prototype hardcodes them.** Hypertrophy blocks target RIR 1–3, technique work targets RIR 4–5, peaking targets RIR 0. When those are added it must be a config change, not a refactor.

**Confidence-gated RIR.** `E[RIR]` is a model output sitting next to two direct measurements, and its forward-projection step (§9.4) is the weakest link in the system. When the RIR predictive distribution is too wide, drop the term and renormalize:

```python
if rir_predictive_std > RIR_CONFIDENCE_THRESHOLD:
    total = 0.5 * rpe_score + 0.5 * form_score      # RIR dropped, effort = RPE alone
else:
    total = 0.25 * rir_score + 0.25 * rpe_score + 0.5 * form_score
```

This keeps the composite from silently inheriting model error. Surface the effort portion with a confidence band derived from the same distribution width.

### 8.3 Form subscores

Equal weighting within the form half. There is no data to fit weights on, and uniform is the defensible prior.

| Subscore | Driven by | Example pointer |
|---|---|---|
| **ROM completeness** | `mean(rom_r) / rom_baseline`, or fraction of reps within 10% of baseline | "3 of 8 reps fell short of your usual depth" |
| **Path consistency** | `path_dtw_baseline` trend, `horiz_excursion` trend | "Bar path diverged from baseline starting rep 6" |
| **Tempo control** | `ecc_duration_total_change`, `tempo_ratio` trend | "Eccentric shortened 40% across the set" |
| **Stability** | `jerk_rms` trend, `tremor_power_8_12` trend | "Tremor rose sharply after rep 7" |

**Completeness vs. retention.** These are different measurements and both are useful:

- **Completeness** is a *level* measure — `mean(rom_r) / rom_baseline`. Feeds the score.
- **Retention** is a *change* measure — `(rom_last − rom_first) / rom_first`. Feeds the pointers.

A set can be uniformly shallow (poor completeness, perfect retention) or start deep and collapse (good completeness, poor retention). The score wants completeness; the text wants retention.

`rom_baseline` resolves through the §7 hierarchy. There is no absolute anatomical reference available from a wrist sensor, so "full ROM" always means "full relative to this user's established baseline."

### 8.4 Normalization

Raw feature changes are not comparable across users or exercises. Convert each form subscore to a per-user, per-exercise z-score against that user's historical distribution of that summary statistic, then map through a monotone squashing function to 0–100.

**During cold start** there is no personal distribution. Use a fixed reference scale (e.g. "10% ROM loss = notable") and label the score as provisional. Do not present a personalized-looking number computed from a population prior.

### 8.5 Divergence flag

A single composite necessarily collides distinct sets. A failure set with form collapse (effort 100, form 40) and a conservative set with clean form (effort 40, form 100) both score 70. This is inherent to projecting two orthogonal axes onto one number.

The mitigation is to detect the collision and change what gets emphasized:

```python
divergent = abs(effort - form) > DIVERGENCE_THRESHOLD    # ~30
```

When `divergent` fires, **lead with the textual insights and de-emphasize the number.** This is precisely the case where the composite is least informative and the underlying situation most warrants attention — in particular the high-effort / low-form quadrant, which is the pattern the system exists to surface.

### 8.6 Textual insights

Generated separately from the score, from sources already present in the pipeline.

**Effort narrative** — sourced from the RIR predictive distribution (§9.4) and the divergence percentile (§11.1):

> "Estimated 1–2 reps left at set end. You reported RPE 7; movement suggested closer to 9."

**Execution narrative** — sourced from the form subscore decomposition, reporting the largest-magnitude deviations first:

> "Depth held through rep 5, then dropped ~15%. Bar path diverged from your baseline starting rep 6."

Each subscore must decompose into at least one pointer. This is what makes the system debuggable — when a score looks wrong, it traces to a specific feature.

### 8.7 Language discipline

Pointers state measured change: *"horizontal excursion increased 6 cm between rep 1 and rep 8."*
Not causal claims: *"keep your chest up."*

The system observes change, not correctness. A path that looks unusual may be optimal for a given lifter's proportions. Deviation from the user's own baseline is a defensible claim; deviation from an ideal is not, because the system has no access to their ideal.

---

## 9. Layer 5b — RIR head (discrete-time hazard)

### 9.1 Why this formulation

RIR is unobserved on any set not taken to failure. Discarding those sets throws away most of the data. Discrete-time survival analysis with right censoring uses them correctly: a set stopped at rep 8 without failure is evidence that reps 1–8 were all completable, which is real information about the hazard function even though the failure point is unknown.

### 9.2 Model

Define the **hazard** at rep `r`:

```
h(r) = P(rep r cannot be completed | reps 1..r−1 were completed)
```

Parameterize with logistic regression on causally-available features:

```
logit h(r) = β₀ + β₁·rep_index
                + β₂·load_pct_e1rm
                + β₃·conc_mean_vel(r)
                + β₄·velocity_loss_to_date(r)
                + β₅·ecc_duration_change_to_date(r)
                + β₆·tremor_change_to_date(r)
                + β₇·path_dtw_baseline(r)
                + exercise fixed effects
                + user random intercept
```

All features must use **only reps 1..r**. No whole-set summaries.

### 9.3 Censored likelihood

For a set that **reached failure** at attempt `R` (reps 1..R−1 completed, rep R failed):

```
L = h(R) · ∏_{r=1}^{R−1} (1 − h(r))
```

For a set **stopped at rep C without failure** (right-censored):

```
L = ∏_{r=1}^{C} (1 − h(r))
```

Total log-likelihood sums over all sets. This is the key equation — it is what allows non-failure sets to contribute.

**Implementation shortcut:** discrete-time hazard models can be fit as ordinary logistic regression on a "person-period" expanded dataset — one row per rep, binary outcome = "was this the failure rep," censored sets contributing all-zero rows. Any logistic regression implementation then works directly, including with a random intercept via `statsmodels` mixed effects or a small PyTorch model.

### 9.4 Converting hazard to RIR

Expected additional completable reps after rep `r`:

```
E[RIR | r] = Σ_{k=1}^{K} ∏_{j=1}^{k} (1 − h(r+j))
```

Future-rep hazards require projecting features forward. For the prototype, extrapolate the fitted per-set trends (velocity slope, tempo slope) linearly. This is a real approximation and a known source of error — document it, and consider capping `K` at ~8 since the projection degrades with horizon.

Report a **distribution**, not a point estimate: `P(RIR = 0), P(RIR = 1), ...` follows directly from the survival products and is more honest than a single number.

### 9.5 Data requirements

The hazard at high rep-fatigue states is only identified from sets that actually reached those states. **Sets to failure are the binding constraint.** Every failure set retroactively labels all of its own reps, so they are disproportionately valuable.

Prototype target: **≥30 failure sets per exercise**, plus as many censored sets as accumulate naturally. Below ~15 failure sets, expect the model to be unstable and treat outputs as illustrative only.

### 9.6 Evaluation

- **Concordance index (C-index)** on held-out sets — does the model rank reps by failure proximity correctly?
- **Calibration curve** — among reps assigned hazard ≈ 0.2, did ≈ 20% actually fail?
- **Held-out failure-point error** — on failure sets, |predicted RIR at rep r − true RIR at rep r|.
- Split by **user**, not by set, to measure generalization to new lifters.

---

## 10. Layer 5c — RPE head (ordinal, hierarchical)

### 10.1 Why not plain regression

Two structural properties break MSE regression:

1. **RPE is ordinal with unequal intervals.** The gap from 6→7 is not the gap from 9→10; the top of the scale compresses because everyone's 10 is failure.
2. **Calibration is user-specific and data is thin per user.** A novice's 8 and an experienced lifter's 8 are different internal states. You will have few sets per user.

### 10.2 Ordinal formulation

Use **CORN / CORAL**-style ordinal encoding: for a scale of RPE ∈ {6, 6.5, ..., 10} with `K` levels, train `K−1` binary outputs predicting `P(RPE > k)`, with a rank-monotonicity constraint. Recover the predicted level from the cumulative probabilities.

This respects ordering, produces a distribution over levels rather than a point estimate, and does not assume equal spacing.

### 10.3 Hierarchical partial pooling

Model per-user calibration as a random effect around a population mean:

```
logit P(RPE > k | user u, set s) = α_k + x_sᵀβ + b_u
b_u ~ N(0, σ²_user)
```

Behavior falls out correctly at both extremes:
- **New user, zero sets** → `b_u` shrinks to 0, predictions default to the population model.
- **Established user, many sets** → `b_u` dominates, predictions become personal.

The shrinkage is automatic and principled. This is the correct treatment of the small-N-per-user problem, not a workaround for it.

Implementation options: `statsmodels` mixed effects for a linear approximation, PyMC/NumPyro for full Bayesian with posterior uncertainty (preferred — the divergence signal in §11 needs the predictive distribution), or a PyTorch model with a regularized user-embedding term.

### 10.4 Features

Same causal per-rep features as the RIR head, plus **whole-set retrospective summaries** — the user reports RPE after the set, so the model may use the full set. Also include `load_pct_e1rm`, `reported_reps`, and exercise fixed effects.

### 10.5 Relationship to the RIR head — important design note

In common practice, RPE and RIR are near-deterministically linked: `RPE ≈ 10 − RIR`. If the RPE head were fit only on movement features, it would largely re-derive the RIR head.

**The RPE head's job is different: it models the user's *subjective mapping*, not the objective state.** The user random effect `b_u` is the object of interest — it captures systematic over- or under-reporting relative to what the movement shows. Keep this framing explicit, because it is what makes the two heads non-redundant and makes §11 meaningful.

---

## 11. Layer 6 — Divergence signal

### 11.1 Definition

Two routes to a predicted exertion level:

- **Mechanical:** from the RIR head, `RPE_mech = 10 − E[RIR]`, with a distribution inherited from the survival probabilities.
- **Subjective-calibrated:** from the RPE head, `RPE_subj`, incorporating the user's personal reporting tendency.

Compare **reported RPE** against the RIR-derived mechanical estimate. Because the RIR head has a full predictive distribution, express the comparison as a percentile rather than a raw difference:

```
divergence = percentile of reported_rpe within the predictive distribution of RPE_mech
```

### 11.2 Interpretation

| Pattern | Reading | Surfaced as |
|---|---|---|
| Reported ≪ mechanical | Movement shows near-failure, lifter reports comfortable — under-perception of fatigue | "Felt like a 7, moved like a 9" |
| Reported ≫ mechanical | Movement clean, lifter reports high effort — non-mechanical factor (sleep, stress, illness, novelty) | "Moved well, but it felt hard — worth noting" |
| Aligned | Calibration is good | Reinforce; no alert |

### 11.3 Thresholding

Alert only when reported RPE falls outside a percentile band (e.g. below the 10th or above the 90th) of the mechanical predictive distribution. Uncertainty is built in — when the RIR model is unsure, the distribution is wide and few sets trigger. That is the desired behavior.

### 11.4 Longitudinal use

`b_u` drifting over time is itself informative: a lifter whose reported RPE increasingly under-shoots the mechanical estimate is losing calibration, which is arguably more actionable than any single-set output. Track it, but only surface it once enough sets exist for the estimate to be stable.

---

## 12. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Capture | Swift / watchOS (`CoreMotion`) | Free sensor fusion; strongly preferred for prototype |
| Transfer | File export → manual, initially | Do not build sync before the algorithm works |
| Storage | Parquet + SQLite | Postgres/TimescaleDB later if needed |
| Processing | Python: NumPy, SciPy, Polars | `scipy.signal` for filtering, `scipy.spatial` / `dtaidistance` for DTW |
| Classical models | statsmodels, scikit-learn | Hazard model as person-period logistic regression |
| Bayesian / hierarchical | PyMC or NumPyro | Needed for RPE posterior |
| Deep models (later) | PyTorch | Only once data justifies it |
| Annotation UI | Streamlit | Plot signal, click rep boundaries, correct segmentation |
| Serving (later) | FastAPI | Deferred |
| On-device (later) | CoreML / TFLite via ONNX | Deferred |

Build the **annotation UI early.** It will be used constantly for debugging segmentation, and segmentation errors silently corrupt every downstream feature.

---

## 13. Build order

Each milestone has a pass/fail criterion. Do not proceed past a failing one.

| # | Milestone | Pass criterion |
|---|---|---|
| 1 | Capture app writing labeled sets to file | 20 sets recorded with complete metadata, no dropped samples |
| 2 | Conditioning pipeline (§5) | Vertical displacement of a known-ROM movement within ~15% of tape measurement |
| 3 | Rep segmentation | ≥95% exact match against `reported_reps` on held-out sets |
| 4 | Feature extraction | Features stable across repeat recordings of the same movement; visual sanity check in the annotation UI |
| 5 | Form subscores (§8.3) | Subscore trends match subjective read on sets you know were good vs. degraded |
| 6 | RIR hazard model | C-index meaningfully above 0.5 on held-out sets; calibration curve near diagonal |
| 7 | RPE head | Ordinal accuracy within ±1 level on the majority of held-out sets |
| 8 | Composite score + insights (§8.1, §8.6) | Score separates known-good from known-degraded sets; divergence flag fires on high-effort/low-form sets |
| 9 | Divergence signal (§11) | Flags fire on sets where you know perception and mechanics diverged |

Milestone 3 is the gate. If segmentation fails, nothing downstream is meaningful.

**Note the reordering:** the composite score now depends on the RIR and RPE heads, so milestone 8 follows 6 and 7. The form subscores (milestone 5) remain independent and can be built and validated early — they are useful on their own, and shipping them before the heads exist gives you a working product to test against.

---

## 14. Known limitations and open questions

**Sensor scope.** The wrist observes one endpoint of the kinematic chain. It says nothing about spine, hip, or knee mechanics, where most lifting injury actually occurs. All claims must stay within what the wrist can support.

**Integration drift.** ZUPT bounds it but does not eliminate it. Absolute displacement carries bias; only rep-to-rep comparison within a set is trustworthy. Touch-and-go reps degrade this further.

**Watch shift.** If the device moves on the strap mid-set, path features change for a non-physiological reason. No current mitigation — consider a sudden orientation-offset detector as a flag.

**DTW template matching cannot discriminate reps (§6.1 stage 3).** Wired in as a
conservative outlier guard, but measurement (2026-08-19) shows it cannot do the job §6.1
assigns it. Two findings, both from synthetic ground truth: (1) in the degraded regime
20.8% of *genuine* completed reps exceed 3× the set's median template distance, because
fatigue legitimately deforms rep shape — cutting on that fights the exact signal the
product measures, and collapsed the hard-regime gate 83.3% → 38.3%; (2) spurious
detections that survive set-detection trimming have a *median* distance ratio of 1.34 and
none exceed 6.0 — after trimming they are shape-indistinguishable from real reps. The two
distributions overlap in the wrong direction, so no threshold separates them. The stage is
retained at a conservative 6.0 (verified identical to disabling it) for gross outliers
only. Rep-shape similarity is the wrong axis; the discriminating signal is rhythmicity
(§6.1 set detection), not shape.

**Horizontal plane is unresolved (§5.3).** Per-set yaw alignment is deferred, so `a_horiz` /
`v_horiz` / `disp_horiz` are in an arbitrary-yaw frame, are not comparable across sets, and
carry no accuracy test. Vertical-axis results are unaffected. This blocks path-consistency
features (§6.2) and must be closed in Phase 3.

**Source-convention drift across capture sources (§5.2).** Real corpora are not internally
consistent. The Kaggle CoreMotion set mixes g and m/s² units *and* gravity sign within a
single directory, and neither defect raises an error — a unit error becomes a sign error
after normalization, and a sign error yields self-consistent but polarity-inverted vertical
motion. Ingest must measure `‖gravity‖` and gravity sign per file rather than trust a
source-wide assumption. Any sign-keyed feature (tempo phase split, eccentric/concentric
duration) inherits this risk; magnitude-only features do not.

**§5.1 resampling is validated only against synthetic data.** The Kaggle corpus was
pre-resampled before publication — exactly 100 Hz, `dt` min = max = 0.01000, zero gaps
> 50 ms across all 164 files. It therefore exercises §5.2–5.6 but says *nothing* about the
jitter/gap handling §5.1 exists for. The synthetic generator's irregular-delivery model
remains the stronger test of that stage. Cleaning that happened upstream of publication is
invisible in the file, so real data can look like a harder test than it is.

**Baseline contamination.** If early reps are already poor, deviation-based features under-report. Mitigated but not solved by warmup-referenced and cross-session baselines.

**Feature projection in §9.4.** Forward-extrapolating features to compute expected RIR is the weakest link in the hazard model. Worth revisiting once data exists.

**No external validation of kinematic claims.** Without a reference measurement, statements like "wrist flexion increased" are assertions about what the IMU signature means, not verified measurements. A one-time calibration study against video would convert these from assumptions to measured relationships. Optional, but it is the only way to check the proxy assumptions.

**Regime structure unresolved.** The `breakpoint` features in §6.3 will answer whether degradation is smooth or piecewise. If breakpoints cluster consistently across feature channels and across sets, a switching state-space model becomes justified as a future component. Until then the continuous representation stands, and the diagnostic costs nothing since the features are computed anyway.
