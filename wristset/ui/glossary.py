"""Plain-language definitions for everything the demo UI puts on screen.

The rest of the codebase is written for someone who has read the architecture doc. This
module is written for someone who has not: every abbreviation, feature name, score and
chart element that appears in the UI gets one sentence saying what it *is* and, where it
matters, what a high or low value means.

Kept separate from ``app.py`` so the explanations can be reviewed as prose, and so the
layout code stays readable.
"""

from __future__ import annotations

__all__ = [
    "CORE_TERMS", "REP_TABLE", "FEATURE_TABLE", "SUBSCORES", "CHART_GUIDE",
    "SCORE_GUIDE", "QUALITY_GUIDE", "WHAT_THIS_IS", "definition_list",
    "as_markdown_table",
]


#: One-paragraph orientation for someone opening the app cold.
WHAT_THIS_IS: str = """
This demo takes the motion signal from a **smartwatch-like device worn during a set of lifitng** and determines: how many reps you did, how your form
deteriorated across the set, how many reps you had left in the tank (RIR), and whether that matches how
hard you said it felt (RPE).

Everything in this demo is computed from wrist motion alone - there is no camera, no bar
sensor, and no manual annotation. The data here is **simulated**, so the correct answers
are known and every number can be checked against them.
""".strip()


#: The vocabulary the whole app assumes. Everything else builds on these.
CORE_TERMS: dict[str, str] = {
    "IMU": (
        "Inertial Measurement Unit - the motion sensor in a smartwatch. It reports "
        "acceleration (how fast the wrist is speeding up or slowing down) and rotation "
        "rate, about 100 times a second."
    ),
    "Rep": (
        "One repetition: a single down-and-up cycle of the lift. The app finds these in "
        "the motion signal rather than being told where they are."
    ),
    "Set": "A group of reps performed back to back without resting.",
    "RIR": (
        "Reps In Reserve - how many more reps you could have done before failing. "
        "RIR 0 means you had nothing left; RIR 3 means you stopped three reps early. "
        "The app *estimates* this from how your movement changed."
    ),
    "RPE": (
        "Rate of Perceived Exertion - how hard the set felt, on a 6-10 scale. This is "
        "*reported by the lifter*, not measured. RPE 10 means maximal effort."
    ),
    "Failure": (
        "Attempting a rep you cannot complete. The app treats the failed attempt as a "
        "distinct event, not as another rep."
    ),
    "Concentric": "The lifting phase - pushing the weight up, away from the bottom.",
    "Eccentric": "The lowering phase - controlling the weight down.",
    "ROM": (
        "Range Of Motion - how far the weight travelled on a rep, in metres. Shrinking "
        "ROM across a set is a classic sign of fatigue."
    ),
    "Baseline": (
        "The reference this lifter is compared against - ideally their own past clean "
        "reps. Everything the app calls a deviation is deviation from this, never from an "
        "ideal, because the app has no access to what is ideal for a given body."
    ),
    "Synthetic-validated": (
        "Every number here was measured against a simulator that generates fake but "
        "physically plausible lifting data with known correct answers. None of it has "
        "been validated against real lifting yet."
    ),
}


#: Columns of the "Detected reps" table.
REP_TABLE: dict[str, str] = {
    "rep": "Rep number within the set, in order.",
    "completed": (
        "True if the lifter finished the rep. A False row is a failed attempt - the "
        "weight went down but did not come back up."
    ),
    "t_start": "Time (seconds into the recording) the rep began, at the top.",
    "t_bottom": "Time the weight reached its lowest point and reversed.",
    "t_end": "Time the rep finished, back at the top.",
    "rom_vertical_m": "How far the wrist travelled vertically on this rep, in metres.",
    "recovery": (
        "How much of the descent the lifter got back. 1.0 means they returned fully to "
        "the top; a low value means they stalled partway, which is what a failed attempt "
        "looks like."
    ),
}


#: Columns of the per-rep feature table, in the order the UI shows them.
FEATURE_TABLE: dict[str, tuple[str, str]] = {
    "rep": ("Rep number", ""),
    "done": ("Whether the rep was completed", "False = a failed attempt"),
    "conc_vel": (
        "Average lifting speed, metres per second",
        "Falls as you fatigue - the strongest single fatigue signal in the system",
    ),
    "peak_vel": ("Fastest moment of the lift", "Sensitive to how hard you tried"),
    "ecc_vel": (
        "Average lowering speed",
        "A control measure: rushing the descent shows up here",
    ),
    "rom_m": ("Range of motion, metres", "Shrinking = the reps are getting shallower"),
    "ecc_s": (
        "Seconds spent lowering",
        "Often shortens BEFORE lifting speed drops - an early fatigue warning",
    ),
    "conc_s": ("Seconds spent lifting", "Grows as the weight slows down"),
    "tempo": (
        "Lowering time divided by lifting time",
        "Cancels out overall pace, so it shows the SHAPE of the rep changing",
    ),
    "pause_s": ("Seconds paused at the bottom", "A technique-consistency measure"),
    "path_eff": (
        "Distance the wrist actually travelled, divided by the vertical distance",
        "1.0 would be a perfectly straight line; higher means more wobble",
    ),
    "horiz_m": (
        "How far the wrist drifted sideways, metres",
        "The bar drifting forward on bench is a common fatigue pattern",
    ),
    "stick_pos": (
        "Where in the lift the bar slowed most (0 = bottom, 1 = top)",
        "Blank when the lift had no distinct sticking point",
    ),
    "stick_vel": ("How slow the bar got at that point", "Lower = more of a grind"),
    "tremor": (
        "Power in the 8-12 Hz band of wrist rotation - physiological shake",
        "Rises steeply near failure as fine motor control degrades",
    ),
    "sparc": (
        "Spectral arc length: a smoothness score, always negative",
        "More negative = jerkier, less controlled movement",
    ),
    "dtw_base": (
        "How different this rep's shape is from the lifter's baseline reps",
        "DTW (Dynamic Time Warping) compares shapes while ignoring speed, so a slow rep "
        "is not penalised for being slow - only for being a different SHAPE",
    ),
}


#: The four form subscores.
SUBSCORES: dict[str, str] = {
    "rom_completeness": (
        "Are you still hitting your normal depth? Drops when reps get shallower than the "
        "opening reps of the set."
    ),
    "path_consistency": (
        "Is the bar still travelling the path it did at the start? Drops when the "
        "movement shape drifts away from the lifter's baseline."
    ),
    "tempo_control": (
        "Is the rhythm holding? Mostly driven by the lowering phase shortening, which "
        "tends to degrade before lifting speed does."
    ),
    "stability": (
        "How steady is the movement? Driven by tremor and jerkiness rising through the "
        "set."
    ),
}


#: What the signal chart is showing.
CHART_GUIDE: str = """
**How to read this chart**

- **"Height (m)" line.** How high the wrist is. **Troughs are the bottom of each rep**
  (weight at its lowest); **peaks are the top**, where the lifter locks out. One
  peak-trough-peak cycle is one rep, so you can count reps by counting troughs.
- **"Speed (m/s)" line** (right-hand axis). How fast the wrist is moving vertically. It
  crosses zero at every peak and trough, because the weight has to stop to change
  direction. Positive = moving up, negative = moving down. **Watch the peaks flatten
  across the set** - that is fatigue slowing the lift down.
- **"Still moments" (grey X markers).** Moments the app detected the wrist was momentarily
  still - the top lockout and the bottom pause of each rep. Working out position from an
  accelerometer accumulates error fast, so the app pins velocity back to zero at these
  known-still points to keep the height line honest. (Called ZUPT anchors in the code, for
  Zero-velocity UPdaTe.)
- **Green / red shading - detected reps.** Green is a completed rep, red is a failed
  attempt. The dotted line inside each is that rep's lowest point.

**The check to make:** if the shaded blocks line up with the troughs, rep detection worked.
If they do not, every number further down the page is built on a bad count.
""".strip()


#: What the "Quality" indicator at the top of the page means.
QUALITY_GUIDE: str = """
**What "Quality" is checking**

This is a check on the **incoming sensor data**, not on the analysis. It asks one question:
did the watch deliver a clean enough signal to work with? It runs before reps are detected
and before any score is computed.

Three things can flag a set:

- **`sample_gaps`** - the watch dropped samples somewhere, leaving a hole longer than
  50 milliseconds. The app fills such holes by interpolating, which produces
  plausible-looking data that was never actually measured, so it flags them instead of
  hiding them. Detected on the original timestamps, before the signal is resampled - once
  it has been resampled the gap is invisible.
- **`no_stationary_anchors`** - the app never found a moment where the wrist was still.
  This matters more than it sounds. Working out position from an accelerometer accumulates
  error very fast, and the fix is to pin velocity back to zero at moments the wrist was
  genuinely still (the grey X markers on the chart). With no such moments there is nothing
  to pin to, so the displacement trace drifts and every distance-based number is
  unreliable.
- **`insufficient_stationary`** - fewer still moments than reps. A normal set pauses at
  least once per rep, at the top or the bottom, so fewer means some reps were measured
  across a longer uncorrected stretch.

**"ok" does not mean the analysis is correct.** It means the input was clean. A set can
have perfect signal quality and still have its reps miscounted - that is what the chart
above is for.
""".strip()


#: How to read the score panel.
SCORE_GUIDE: str = """
**How the set score works**

It combines two independent things, each worth half:

- **Effort** - how close the set came to its target intensity, from the estimated
  reps-in-reserve and the reported RPE. Scored as *closeness to a target*, so overshooting
  counts against you too: a set pushed well past failure is not better than one that hit
  the mark.
- **Form** - how well the movement held up, from the four subscores below.

Both run 0-100, higher is better.

When the two **disagree sharply**, the app says so and pushes the number aside. That is the
case a single average hides: a gruelling set with collapsing form and a comfortable set
with clean form can land on the same score, and they call for opposite advice.
""".strip()


def definition_list(table: dict) -> str:
    """Render a term dictionary as a markdown bullet list."""
    lines = []
    for term, body in table.items():
        if isinstance(body, tuple):
            what, means = body
            lines.append(f"- **`{term}`** - {what}." + (f" _{means}._" if means else ""))
        else:
            lines.append(f"- **{term}** - {body}")
    return "\n".join(lines)


def as_markdown_table(table: dict[str, tuple[str, str]]) -> str:
    """Render the feature dictionary as a markdown table with a 'what to look for' column."""
    rows = ["| column | what it is | what to look for |", "|---|---|---|"]
    for term, (what, means) in table.items():
        rows.append(f"| `{term}` | {what} | {means or '-'} |")
    return "\n".join(rows)
