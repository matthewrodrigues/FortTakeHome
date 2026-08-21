# wristset — development log

A record of how this project was built: the decisions, the things that turned out to be
wrong, and the measurements that settled them. The [README](README.md) says what the system
*is*; this says how it got there and what I'd do differently.

Built collaboratively with Claude (Opus), working from a written architecture document
(`wrist-imu-set-analysis-architecture.md`) and a phased plan. Every phase has a gate; every
gate is measured against a synthetic generator with known ground truth.

**Final state:** 217 tests, ~6,700 lines of source and ~2,700 of tests, Phases 0–9 complete.

---

## The shape of the problem

Turn wrist-worn IMU data into per-set resistance-training analysis: form subscores, an
estimated reps-in-reserve (RIR), a reported-RPE model, and the divergence between what the
lifter *reported* and what their movement *showed*. No camera, no bar sensor, no manual
annotation — wrist motion alone.

Capture (watchOS) was deferred by design. Everything is developed against a **synthetic IMU
generator** that emits ground-truth labels, so each layer can be validated against a known
correct answer. That choice is what made the whole project measurable — and, as it turned
out, it was also the source of several of the hardest bugs.

---

## Phase-by-phase, with what actually happened

### Phases 1–5: the deterministic pipeline

Signal conditioning → rep segmentation → per-rep features → form subscores → demo.
Everything here is signal processing and arithmetic. All gates met.

| Layer | Gate | Result |
|---|---|---|
| Conditioning | per-rep ROM within 15% of true | 2.0% mean error |
| Segmentation | ≥95% exact rep count | 100% operating distribution |
| Features | stable across repeat recordings | passes level + trend stability |
| Form subscores | separate clean from degraded | 90 vs 49, no overlap |

**The largest single finding in this stretch** came from running segmentation against a real
external corpus (164 Kaggle CoreMotion files) rather than only synthetic data. It scored
**16.8% exact** against 100% on synthetic. Diagnosing that took the project in a direction
the plan hadn't anticipated:

- `corr(detected_reps, recording_duration) = 0.84` versus `corr(detected, true_count) = 0.43`
  — the segmenter was tracking *how long the recording was*, not how many reps happened.
- Real recordings contain ~25% non-lifting time (setup, racking, rest). Synthetic sets begin
  at rep 1 and end at the last rep.
- The spec prescribed an **energy threshold** to find the active window. Measured, energy
  during racking (1.48–1.60) was indistinguishable from energy during lifting (1.97).
- What *did* separate them was **rhythmicity**: inter-rep gaps inside a set were 2.06–2.46 s,
  edges gave 1.38, 2.50, 3.53 s.

So set detection was rebuilt around periodicity rather than energy. That improved things
(mean error 3.91 → 3.10) but did not rescue the exact-match rate, because the corpus also
contains only out-of-vocabulary exercises (cable and dumbbell work; no bench, squat, or
deadlift — I verified this against the dataset's own exercise key).

**Honest conclusion:** rep counting via prominent minima of vertical displacement is the
weakest formulation in the project. With more time the evidence points at a
periodicity-first or learned approach — `reported_reps` is free weak supervision on every
set, which makes this a learnable problem rather than a heuristic one.

### Phase 6: RIR hazard head

Discrete-time survival. One row per attempted rep, outcome "was this the failure rep."
Censored sets (stopped early) contribute all-zero rows, which is what lets them inform the
model without inventing an unobserved label.

The gate initially reported a **C-index of 1.000** on held-out users. That is not a good
result — it is a sign the task is degenerate, and digging in confirmed it: the generator
emits a failed attempt at 40% of full height, so its velocity signature is unlike any
completed rep. `-conc_mean_vel` alone scores **0.983 with no model at all**.

Replaced with `completed_rep_c_index` — ranking *completed* reps by true RIR, excluding the
giveaway. Honest number: **0.819**.

Three further fixes came out of measurement, not intuition:

- **E[RIR] was collapsing.** At rep 4 of an 11-rep set it predicted 1.0 when the truth was 6.
  Cause: the fitted `conc_mean_vel` coefficient was −21 (a near step-function), and the
  forward projection walked features straight across that cliff, saturating every projected
  hazard. Fixed by damping the projection **and** — the larger effect — raising the ridge
  from 1e-3 to 0.5. MAE 2.22 → **1.80 reps**, improved on all three corpus seeds.
- **The gate was passing on a lucky seed.** Calibration MAD was 0.037 on seed 5 and
  0.327 / 0.294 on seeds 11 and 23. The gate now runs three corpora.
- **`model.beta` is not feature importance.** VIF 9.6 between correlated fatigue proxies
  means the fit hands one feature a large weight and the others compensating opposite-sign
  weights. Two coefficients are physiologically backwards; both flip to correct when fitted
  alone. Documented via `collinearity_report` rather than "fixed" by forcing signs, which
  would trade real predictive accuracy for interpretability nothing downstream needs.

### Phase 7: RPE ordinal head

Proportional-odds (CORAL) with a ridge-penalised per-user effect. Two choices worth stating:

- **CORAL over CORN** — §10.5 makes the per-user term `b_u` the object of interest, and
  CORN's per-threshold classifiers give no single scalar bias to recover.
- **scipy over NumPyro** — the plan justified NumPyro because "§11 divergence needs a
  posterior," but §11's percentile is computed in the *RIR* head's distribution, not the
  RPE posterior. The plan misread its own spec.

The user asked whether the 0.738 accuracy was real or an artifact of label skew. It was
partly skew: the RIR-oriented corpus puts ~⅓ of sets at RPE 10 (both the modal `10 −
stop_rir` outcome *and* a clip ceiling absorbing every positive bias). Built
`generate_rpe_population` to spread labels; accuracy dropped to **0.595** — but **lift over
the majority baseline rose** from +0.198 to +0.230. Lift is the stable quantity; raw
accuracy was partly reading "RPE 10 is common."

`b_u` recovers the injected per-user bias at r = **0.72–0.84**, which is the claim that
makes this head non-redundant with RIR.

### Phase 8: composite, divergence, insights

Composite is ¼ RIR + ¼ RPE + ½ form, with effort terms scored as **proximity to a target**
— overshooting is penalised, because a set pushed past failure with assistance is not
superior to one that hit the mark.

**The divergence signal (§11) is the feature this system exists for.** It locates the
reported RPE as a *percentile* within the RIR-implied distribution rather than as a raw
difference, so model uncertainty is built into the threshold: when the model is unsure the
distribution is wide and fewer sets trigger. That produces the intended output —
*"You reported RPE 7; movement suggested closer to 9."*

**Missing components renormalise rather than defaulting.** Any of the three inputs may be
absent; remaining weights are reweighted over what exists. This meant Phase 8 added
capability without forking a single code path — the Phase 5 demo is the degenerate case of
the new formula.

### Phase 9: eval harness

`python -m wristset.eval` re-derives every phase gate in one table, as shippable code rather
than test-only helpers. It immediately found something the per-phase tests had not: `b_u`
recovery is weaker on a *balanced* corpus (0.72) than the skewed one (0.84), because
recovering a per-user bias is easier when the mechanical reference barely moves.

That is the argument for a cross-phase harness. Per-phase tests inherit each phase's own
framing; a harness is where inconsistencies between framings surface.

---

## The deferred confidence gate — and a conversation worth recording

§8.2 specifies dropping the RIR term from the composite when its predictive uncertainty is
too wide. Before building it, I measured whether that uncertainty means anything. Across 527
held-out predictions:

| Property | Measured | Should be |
|---|---|---|
| median std vs median error | 0.5 vs 1.6 | comparable |
| coverage within 1σ | 20–30% | ~68% |
| corr(std, \|error\|) | **−0.08** | strongly positive |

The distribution's spread is **anti-correlated with error** — it calls its worst predictions
its most certain. Seven alternative signals were tested (entropy, tail mass, max probability,
projected-hazard saturation, forecast horizon); none exceeded |r| = 0.15.

**The user's reaction was that these weak correlations undermined confidence in the whole
project.** That is a reasonable reading of the number in isolation, and the resolution was
worth having explicitly: an uncertainty estimate says *"the model cannot tell you when it's
wrong,"* not *"the model is wrong."* Self-knowledge is a strictly harder problem than
prediction, and it is the last thing to come together. Measured accuracy — 0.819
concordance, 1.80 rep MAE, +0.23 RPE lift — is established separately and holds.

The agreed resolution was to **scope the weak part down honestly rather than tune it
upward**: the gate is deferred and documented, the RIR term is kept unconditionally, and
`is_confident` survives only as a weak advisory annotation explicitly labelled as not a gate.

---

## Decisions that came from the user

- **RIR as a data-gated feature.** Rather than RIR appearing based on whether a caller
  passed a model — an implementation detail — it now unlocks based on whether enough sets
  have been taken to failure to identify the hazard (§9.5: <15 withheld, 15–29 provisional,
  30+ settled). This turned the composite's renormalisation from a fallback into a product
  rule with a user-facing meaning, and it mirrors the baseline cold-start pattern already in
  the system.
- **Balanced-label RPE testing.** Asking whether 0.738 was signal or skew produced the
  balanced corpus, the baseline-lift metric, and a more honest headline number.
- **Framing for presentation.** Phases 1–5 as a validated deterministic setup with known
  design limitations; Phases 6+ as the product features whose formulation still needs work.
- **UI exposition.** The demo page now defines every abbreviation, feature column, subscore,
  chart element, and the signal-quality check, in expanders placed next to the thing they
  explain.

## Decisions that came from me

- Replacing flattering metrics with honest ones (`completed_rep_c_index`, balanced-corpus
  RPE accuracy, lift-over-baseline).
- Multi-seed gates after the single-seed calibration result proved to be luck.
- Deferring rather than shipping the confidence gate, the touch-and-go flag, and DTW-based
  rep rejection — each measured as unable to do its job, and documented as such.
- Diagnosing the generator artifacts that were leaking into results (below).

---

## Generator artifacts — the recurring lesson

The synthetic generator is what makes everything measurable, and three times its *model* of
a phenomenon leaked a shortcut reality wouldn't provide:

1. **Tremor varied ~200× between recordings** for reasons unrelated to the lifter. Two
   compounding causes: `f_tremor` drawn from the *corners* of the analysis band (where a
   zero-phase filter retains only ~50% of amplitude), and — dominant — gyro tremor injected
   as a randomly-scaled 3-vector whose contribution to `|gyro|` depended on its alignment
   with the movement axis. This had leaked into three phases: a Phase 3 gate exclusion, the
   Phase 4 stability subscore, and Phase 5 demo output where a set taken to failure scored a
   perfect 100 for stability.
2. **The failed attempt is too distinctive** (emitted at 40% of full height), making RIR
   discrimination nearly free and the C-index meaningless.
3. **Label skew** concentrated a third of RPE labels on one level, inflating accuracy.

The pattern: a generator's simplifications become the model's shortcuts. Every headline
number is worth asking "could a trivial baseline do this?" — twice here, the answer was yes.

---

## What I would do differently

- **Rep counting needs a different formulation.** Single-channel vertical displacement
  ignores the gyro and quaternion entirely; a cable lateral raise is mostly rotational and
  nearly invisible to it. Rhythmicity outperformed both energy and shape as a discriminator,
  which points at a periodicity-first or learned approach.
- **Calibrate the forward projection.** It is the documented dominant error source in RIR,
  and both the 1.80-rep MAE and the uncertainty miscalibration trace back to it. Fixing it is
  the prerequisite for re-enabling §8.2's gate.
- **Get in-vocabulary real data.** No bench, squat, or deadlift exists in the external
  corpus, so the strongest real-data claim available is a 5-file barbell/machine incline
  subset.

---

## On the numbers in this project

Several figures here are lower than earlier versions reported. That is deliberate. A C-index
of 1.000 became 0.819 once it was measured without the generator's giveaway; an RPE accuracy
of 0.738 became 0.595 on balanced labels; a confidence gate was deferred after it measured
backwards; a form score of 100 on a 3-rep set became `None` after the guard was added.

None of those were caught by tests passing — they were caught by not trusting the tests. The
lower numbers are the real ones.
