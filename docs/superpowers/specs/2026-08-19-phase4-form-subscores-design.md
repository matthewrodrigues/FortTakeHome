# Phase 4 — Deterministic form subscores (Layer 5a form half, §8.3–8.4)

Status: approved 2026-08-19. Builds only on stable Phase-3 interfaces; adds a new
`wristset/scoring/` package and alters no existing interface. First stand-alone shippable
product (milestone 5).

## Goal

Turn the Phase-3 feature layer (`SetFeatures`) into four deterministic **form subscores**
(0–100) plus an equal-weight composite **form score**, with each subscore decomposing into
≥1 structured **pointer** for the Phase-8 execution narrative. No training data; everything
computed directly from features (§8.3).

## Resolved decisions (the design tension of this phase)

1. **Normalization (§8.4).** Ship the cold-start **fixed reference scale** as the working
   default, flagged `provisional`. A **z-score path is parameterized from day one** via an
   optional `NormRef` (per-channel `mean`/`std`); when supplied, score via z→logistic and
   drop `provisional`. Mirrors the Baseline / `TARGET_RIR` "parameterize now, hardcode the
   prototype value" pattern. We do **not** fabricate a personalized-looking number from a
   population prior (§8.4).

2. **ROM reference.** `rom_baseline` is computed **inside the scorer** as the mean
   `rom_vertical` over the current set's first `n_ref_reps` (=`EARLY_SET_REPS`=3) completed
   reps — tier-3 style, self-contained in `scoring/`, §7 untouched. This deliberately
   carries §7.1's known flaw (a set that *starts* shallow reads as complete) — already a
   preserved invariant — so ROM completeness is `provisional` by default. Overridable by an
   optional arg so history/warmup references can feed it in a later phase.

3. **Score = level, pointer = trend.** §8.3's "driven by" column lists trends for three of
   four subscores, but its completeness-vs-retention note says the *score wants the level
   and the text wants the change*. We generalize that split to **all four**: the **score**
   reads a magnitude/level of deviation from the reference; the **pointer** reads the
   within-set **trend** (slope / `total_change` / `breakpoint`) straight off Phase-3
   `retrospective_summaries`. This resolves the tension coherently: a uniformly-bad set
   (high level, flat trend) is still penalized, which a trend-only score would miss.

## Module & interface (§A)

New package `wristset/scoring/` (Layer 5a). One module `form.py` for now (split if it
grows). Exports via `scoring/__init__.py`.

```python
score_form(sf: SetFeatures, *, norm_ref: NormRef | None = None,
           n_ref_reps: int = EARLY_SET_REPS) -> FormSubscores
```

- `Pointer` — `{subscore: str, text: str, magnitude: float, onset_rep: int | None,
  feature: str}`. Structured, not prose; Phase-8 `insights/` orders (largest-magnitude
  first, §8.6) and renders them. Phase 4 only produces them, ≥1 per subscore (§8.3).
- `FormSubscore` — `{name, score: float | None, provisional: bool, drivers: dict,
  pointers: list[Pointer]}`. `score is None` ⇒ excluded from the composite (see §None).
- `NormRef` — optional `{mean: dict[str,float], std: dict[str,float]}` keyed by driver
  channel. Absent ⇒ fixed-scale + provisional.
- `FormSubscores` — the four `FormSubscore`s + `form_score: float | None` (equal-weight
  mean of the non-None subscores, §8.3) + `provisional: bool` (any component provisional).

## The four subscores (§B)

| Subscore | Score driver (level) | Pointer driver (trend / decomposition) |
|---|---|---|
| **rom_completeness** | `mean(rom_vertical over completed) / rom_baseline` | `retro_rom_vertical_total_change` (retention); count of reps > 10% below baseline; `retro_rom_vertical_breakpoint` for onset |
| **path_consistency** | mean `path_dtw_baseline` (already §7-baseline-relative) | `retro_path_dtw_baseline` slope + breakpoint; `horiz_excursion` change |
| **tempo_control** | \|`retro_ecc_duration_total_change`\| + `tempo_ratio` drift | `retro_ecc_duration_total_change` |
| **stability** | rise in `tremor_power_8_12` (+ `jerk_rms`) vs early-rep reference | `retro_tremor_power_8_12` breakpoint + slope |

## Normalization & squash (§C, §8.4)

One monotone squash shared by both paths, `deviation → [0,100]`, 100 at deviation 0,
strictly decreasing:

- **Fixed-scale (default):** natural-unit deviation `d ≥ 0` (e.g. ROM fractional shortfall)
  divided by a per-channel "notable" constant (`ROM_LOSS_NOTABLE = 0.10`, etc.), then
  logistic. Constants are named module-level parameters from day one.
- **z-score (NormRef supplied):** driver → `z = (x − mean)/std`, same logistic, sign
  oriented so "worse" lowers the score. Clears `provisional`.

## `None` semantics (invariant)

A driver of `None` (cold start / too few reps — e.g. `path_dtw_baseline` with no template,
or `total_change` on a 1-rep set) yields `score = None` for that subscore and it is
**dropped from the composite mean**, never counted as 0 (Phase-3 "None means no reference,
not zero" invariant). If every subscore is None, `form_score` is None.

## Testing (§D, §E)

`tests/test_scoring.py`:

- **Milestone-5 gate:** across seeds, clean set vs degraded set (regime dict mirroring
  `test_segmentation`), assert `form_score(clean) > form_score(degraded)` and each subscore
  separates in the correct direction.
- **Units:** squash monotonicity & endpoints; None-exclusion from the composite; provisional
  flag flips off when `NormRef` supplied; pointer `onset_rep` matches the generator's planted
  degradation breakpoint.
- **§8.7 language discipline:** scan every pointer's `text` against a prohibited-phrase set
  ("keep your", "chest up", "should", causal cues); pointers state measured change only.

## Out of scope (later phases)

Composite score / effort half / confidence-gated RIR / divergence flag (§8.1–8.2, 8.5 →
Phase 8); narrative assembly and ordering (`insights/` → Phase 8); wired per-user historical
`NormRef` and warmup/cross-session ROM references (needs history plumbing).
