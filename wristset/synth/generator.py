"""Parametric synthetic wrist-IMU generator with ground-truth labels.

Pipeline of the generator (the inverse of the Layer-2 conditioning pipeline):

    1. Build a smooth world-frame vertical trajectory z(t) from per-rep half-cosine
       segments. Velocity is exactly zero at every top/bottom pause -> genuine ZUPT
       anchors (§5.5). Fatigue knobs make late reps slower, shorter-eccentric,
       shallower, and more tremulous (§6.2 degradation).
    2. Add horizontal wobble/drift (path inefficiency, §6.2 path features).
    3. Differentiate z twice -> world-frame linear acceleration; add 8-12 Hz tremor
       and white noise.
    4. Rotate world-frame accel + gravity INTO the device frame with the attitude
       quaternion (world->device via q*). Phase 1 undoes exactly this (§5.3).
    5. Resample onto jittered, occasionally-gapped timestamps (§3.3 irregular delivery).

Ground truth emitted: per-rep boundaries + true world ROM + true concentric velocity,
the set's true failure rep (or None if censored), and the virtual user's RPE bias.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from scipy.signal import butter, sosfiltfilt

from wristset.common.quaternion import rotate_vectors_inverse
from wristset.contract import RAW_SCHEMA, GroundTruth, SetMetadata, validate_raw
from wristset.contract.schema import RepTruth

G = 9.80665  # gravitational acceleration magnitude (m/s^2)
_FINE_HZ = 1000  # continuous-signal grid before resampling to the sensor timestamps

__all__ = ["SetParams", "GeneratedSet", "generate_set", "generate_session"]


@dataclass(frozen=True)
class SetParams:
    """Knobs for one synthetic set. Defaults describe a clean bench set taken to failure."""

    exercise: str = "bench_press"
    set_type: str = "working"
    load_kg: float = 60.0

    #: True max completable reps (latent capacity C). See stop_rir for where the set stops.
    capacity: int = 8
    #: RIR at the moment the lifter stops. 0 with reached_failure=True means the last
    #: completed rep is C and there is one failed attempt at C+1.
    stop_rir: int = 0
    reached_failure: bool = True

    # --- kinematics ---
    rom_m: float = 0.45
    top_pause_s: float = 0.35
    bottom_pause_s: float = 0.15
    base_ecc_s: float = 0.9
    base_conc_s: float = 0.8
    vel_decay: float = 0.35  # fractional concentric slowdown across the set
    ecc_shorten: float = 0.25  # eccentric duration shrinks with fatigue (§6.2)
    rom_collapse: float = 0.12  # fractional ROM loss by the last rep
    path_wobble_m: float = 0.02  # random horizontal wobble amplitude
    horiz_drift_m: float = 0.05  # systematic forward drift by end of set (bench arc)

    # --- tremor / smoothness ---
    tremor_gain: float = 0.35  # base 8-12 Hz accel amplitude (m/s^2)
    tremor_growth: float = 2.5  # multiplier on tremor by end of set

    # --- user calibration (RPE) ---
    rpe_bias: float = 0.0  # reported - mechanical; negative = under-reports effort
    rpe_noise: float = 0.3

    # --- recording edges (§6.1 set detection) ---
    # Real captures are rarely trimmed to the working set: the recording starts while the
    # lifter is still setting up and ends after racking. That motion is large and SLOW
    # (~0.2-0.4 Hz, below the 0.3-1 Hz rep band) — energetic enough to pass an energy
    # threshold but not rhythmic at rep cadence. Defaults are 0.0 so every existing set
    # and gate is unchanged; opt in to exercise set detection.
    lead_s: float = 0.0  # non-lifting motion before the first rep
    tail_s: float = 0.0  # non-lifting motion after the last rep
    edge_motion_m: float = 0.25  # amplitude of that motion (m, vertical)
    edge_hz: float = 0.3  # its dominant frequency (Hz)

    # --- sensor model ---
    sample_hz: int = 100
    jitter: float = 0.15  # timestamp jitter as fraction of nominal dt
    gap_prob: float = 0.0  # per-sample probability of a delivery gap
    gap_ms: float = 80.0  # gap size when one occurs
    noise_acc: float = 0.05  # white accel noise std (m/s^2)
    noise_gyro: float = 0.02  # white gyro noise std (rad/s)

    # --- orientation (device tilt on wrist, degrees, XYZ euler) ---
    tilt_deg: tuple[float, float, float] = (20.0, -12.0, 6.0)

    seed: int = 0

    def __post_init__(self) -> None:
        if self.stop_rir < 0:
            raise ValueError("stop_rir must be >= 0")
        if self.reached_failure and self.stop_rir != 0:
            raise ValueError("reached_failure=True requires stop_rir=0")
        if self.stop_rir >= self.capacity:
            raise ValueError("stop_rir must be < capacity")


@dataclass
class GeneratedSet:
    """Bundle returned by :func:`generate_set`."""

    raw: pl.DataFrame
    user_id: str
    session_id: str
    date: str
    set_index: int
    set_id: str
    exercise: str
    load_kg: float
    set_type: str
    reported_reps: int
    reported_rpe: float
    reached_failure: bool
    ground_truth: GroundTruth
    reps: list[RepTruth] = field(default_factory=list)

    def set_metadata(self, raw_path: str) -> SetMetadata:
        """Build the SetMetadata record once the raw file's path is known."""
        return SetMetadata(
            set_id=self.set_id,
            session_id=self.session_id,
            set_index=self.set_index,
            exercise=self.exercise,
            load_kg=self.load_kg,
            set_type=self.set_type,
            reported_reps=self.reported_reps,
            reached_failure=self.reached_failure,
            raw_path=raw_path,
            reported_rpe=self.reported_rpe,
        )


# --------------------------------------------------------------------------------
# trajectory construction
# --------------------------------------------------------------------------------


def _half_cos(z0: float, z1: float, n: int) -> np.ndarray:
    """Smooth transition z0->z1 over n samples with zero velocity at both ends."""
    s = np.linspace(0.0, 1.0, n)
    return z0 + (z1 - z0) * 0.5 * (1.0 - np.cos(np.pi * s))


def _const(z: float, n: int) -> np.ndarray:
    return np.full(n, z, dtype=np.float64)


def _euler_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ intrinsic euler angles (radians) -> scalar-first quaternion."""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    # q = qx * qy * qz
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z], dtype=np.float64)


def _edge_motion(dur_s: float, p: SetParams, rng: np.random.Generator) -> np.ndarray:
    """Non-lifting motion for a recording edge (§6.1): setup, unracking, walking, racking.

    Deliberately OUT of the rep band — a couple of slow components near ``edge_hz`` plus a
    wander term. It must be energetic (so an energy threshold cannot trivially reject it)
    while not rhythmic at rep cadence (so it is not a legitimate rep). Tapered at the join
    so the trajectory stays continuous into the set.
    """
    n = max(int(round(dur_s * _FINE_HZ)), 2)
    t = np.arange(n) / _FINE_HZ
    z = np.zeros(n)
    for k in (1.0, 1.7):  # incommensurate -> no clean periodicity
        z += (p.edge_motion_m / k) * np.sin(2 * np.pi * p.edge_hz * k * t + rng.uniform(0, 2 * np.pi))
    z += p.edge_motion_m * 0.4 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 2 * np.pi))
    return z


def _build_rep_segments(p: SetParams) -> tuple[np.ndarray, list[RepTruth], int | None]:
    """Return (z_fine, rep_truths, failure_rep). z on the _FINE_HZ grid, t=0 at start.

    When ``lead_s``/``tail_s`` are set, the rep block is wrapped in non-lifting motion and
    every rep truth time is shifted by the lead duration so ground truth continues to
    address the emitted trajectory (§6.1).
    """
    dt = 1.0 / _FINE_HZ
    completed = p.capacity - p.stop_rir  # completed reps before stopping
    n_denom = max(p.capacity - 1, 1)

    z_chunks: list[np.ndarray] = []
    reps: list[RepTruth] = []
    t_cursor = 0.0

    def n_for(dur: float) -> int:
        return max(int(round(dur * _FINE_HZ)), 2)

    def emit_pause(z: float, dur: float) -> None:
        nonlocal t_cursor
        n = n_for(dur)
        z_chunks.append(_const(z, n))
        t_cursor += (n - 1) * dt

    def emit_move(z0: float, z1: float, dur: float) -> float:
        nonlocal t_cursor
        n = n_for(dur)
        z_chunks.append(_half_cos(z0, z1, n))
        t0 = t_cursor
        t_cursor += (n - 1) * dt
        return t0

    # leading top pause
    emit_pause(0.0, p.top_pause_s)

    def add_rep(i: int, *, completed_flag: bool, conc_frac: float = 1.0) -> None:
        """Add one rep (eccentric, bottom pause, concentric). conc_frac<1 => failed attempt."""
        nonlocal t_cursor
        fatigue = min(i / n_denom, 1.0)
        ecc_dur = p.base_ecc_s * (1.0 - p.ecc_shorten * fatigue)
        vel_scale = max(1.0 - p.vel_decay * fatigue, 0.3)
        conc_dur = p.base_conc_s / vel_scale
        rom_i = p.rom_m * (1.0 - p.rom_collapse * fatigue)
        achieved_rom = rom_i * conc_frac  # failed attempt rises only part-way

        t_start = t_cursor
        emit_move(0.0, -rom_i, ecc_dur)  # eccentric: top -> bottom
        t_bottom = t_cursor
        emit_pause(-rom_i, p.bottom_pause_s)
        # concentric: bottom -> (top or stall point)
        top_target = -rom_i + achieved_rom
        emit_move(-rom_i, top_target, conc_dur)
        t_end = t_cursor
        if completed_flag:
            emit_pause(0.0, p.top_pause_s)
        else:
            # failed attempt stalls, sags slightly, then racked (held then down)
            emit_pause(top_target, 0.4)
            emit_move(top_target, -rom_i * 0.2, 0.5)
            emit_pause(-rom_i * 0.2, p.top_pause_s)

        conc_mean_vel = achieved_rom / conc_dur
        conc_peak_vel = conc_mean_vel * (np.pi / 2)  # half-cosine peak/mean ratio
        reps.append(
            RepTruth(
                rep_index=i + 1,
                t_start=t_start,
                t_bottom=t_bottom,
                t_end=t_end,
                completed=completed_flag,
                rom_true_m=achieved_rom,
                conc_mean_vel_true=conc_mean_vel,
                conc_peak_vel_true=conc_peak_vel,
            )
        )

    for i in range(completed):
        add_rep(i, completed_flag=True)

    failure_rep: int | None = None
    if p.reached_failure:
        add_rep(completed, completed_flag=False, conc_frac=0.4)
        failure_rep = completed + 1  # 1-based attempt index

    z_fine = np.concatenate(z_chunks)

    if p.lead_s > 0 or p.tail_s > 0:
        # Edge motion is generated from a dedicated stream so adding edges does not
        # perturb the rep trajectory drawn for this seed (a set with lead_s>0 has the
        # same reps as the same seed without it).
        edge_rng = np.random.default_rng(p.seed + 9973)
        parts = []
        if p.lead_s > 0:
            lead = _edge_motion(p.lead_s, p, edge_rng)
            parts.append(lead - lead[-1])  # join continuously at z=0 (top position)
        parts.append(z_fine)
        if p.tail_s > 0:
            tail = _edge_motion(p.tail_s, p, edge_rng)
            parts.append(tail - tail[0] + z_fine[-1])
        z_fine = np.concatenate(parts)

        # shift rep truth times past the prepended lead (ground truth must stay aligned)
        if p.lead_s > 0:
            shift = (max(int(round(p.lead_s * _FINE_HZ)), 2)) / _FINE_HZ
            reps = [
                replace(r, t_start=r.t_start + shift, t_bottom=r.t_bottom + shift,
                        t_end=r.t_end + shift)
                for r in reps
            ]

    return z_fine, reps, failure_rep


def _fatigue_envelope(z_fine: np.ndarray) -> np.ndarray:
    """0->1 ramp over the length of the set, used to grow tremor/drift with time."""
    n = z_fine.shape[0]
    return np.linspace(0.0, 1.0, n)


# --------------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------------


def generate_set(
    params: SetParams,
    *,
    user_id: str = "user_synth",
    session_id: str | None = None,
    date: str = "2026-08-18",
    set_index: int = 1,
) -> GeneratedSet:
    """Generate one synthetic set with raw signals + ground truth."""
    rng = np.random.default_rng(params.seed)
    session_id = session_id or uuid.uuid4().hex[:12]
    set_id = f"{session_id}:set{set_index:03d}"

    # 1-2. world-frame position trajectory + horizontal wobble/drift
    z_fine, reps, failure_rep = _build_rep_segments(params)
    n = z_fine.shape[0]
    t_fine = np.arange(n) / _FINE_HZ
    env = _fatigue_envelope(z_fine)

    def smooth_noise(scale: float) -> np.ndarray:
        # Band-limited (~1.5 Hz) wobble so that DOUBLE differentiation to acceleration
        # stays physiological. A boxcar/moving-average leaves high-frequency residue that
        # d^2/dt^2 amplifies by (2*pi*f)^2 into absurd accelerations.
        raw = rng.standard_normal(n)
        if n > 30:
            sos = butter(4, 1.5 / (0.5 * _FINE_HZ), btype="low", output="sos")
            s = sosfiltfilt(sos, raw)
        else:
            s = raw
        s = s / (s.std() + 1e-9)
        return scale * s

    x_fine = params.horiz_drift_m * env + smooth_noise(params.path_wobble_m)
    y_fine = 0.4 * params.horiz_drift_m * env + smooth_noise(0.5 * params.path_wobble_m)

    # 3. world-frame linear acceleration = second derivative of position
    def d2(sig: np.ndarray) -> np.ndarray:
        return np.gradient(np.gradient(sig, t_fine), t_fine)

    a_world = np.stack([d2(x_fine), d2(y_fine), d2(z_fine)], axis=1)

    # tremor: 8-12 Hz, amplitude grows with fatigue
    f_tremor = rng.uniform(8.0, 12.0)
    tremor_amp = params.tremor_gain * (1.0 + params.tremor_growth * env)
    tremor = (tremor_amp * np.sin(2 * np.pi * f_tremor * t_fine))[:, None]
    a_world = a_world + tremor * rng.uniform(0.6, 1.0, size=(1, 3))
    a_world = a_world + rng.normal(0.0, params.noise_acc, size=a_world.shape)

    # gravity in world frame points down (-z)
    g_world = np.tile(np.array([0.0, 0.0, -G]), (n, 1))

    # 4. attitude quaternion (device->world): fixed tilt + slow drift + tremor rotation
    tilt = np.radians(np.array(params.tilt_deg))
    drift = np.radians(2.0) * env[:, None] * rng.uniform(-1, 1, size=(1, 3))
    wobble = np.radians(0.5) * tremor / max(params.tremor_gain, 1e-6)
    eul = tilt[None, :] + drift + wobble * np.array([1.0, 1.0, 0.3])
    q_fine = np.stack([_euler_to_quat(*eul[i]) for i in range(n)], axis=0)

    # rotate world -> device
    lin_acc_dev = rotate_vectors_inverse(q_fine, a_world)
    grav_dev = rotate_vectors_inverse(q_fine, g_world)

    # gyro: magnitude tracks |v_z| (moving => rotating), ~0 during pauses (ZUPT-friendly)
    v_z = np.gradient(z_fine, t_fine)
    omega_mag = np.abs(v_z) * 1.5
    gyro_axis = rng.standard_normal(3)
    gyro_axis /= np.linalg.norm(gyro_axis)
    gyro_dev = omega_mag[:, None] * gyro_axis[None, :]
    gyro_dev = gyro_dev + 0.3 * tremor * rng.uniform(0.6, 1.0, size=(1, 3))
    gyro_dev = gyro_dev + rng.normal(0.0, params.noise_gyro, size=gyro_dev.shape)

    # 5. resample onto jittered / gapped sensor timestamps
    t_samples = _sample_timestamps(t_fine[-1], params, rng)

    def interp(col: np.ndarray) -> np.ndarray:
        return np.interp(t_samples, t_fine, col)

    def interp3(arr: np.ndarray) -> np.ndarray:
        return np.stack([interp(arr[:, j]) for j in range(3)], axis=1)

    lin_s = interp3(lin_acc_dev)
    gyro_s = interp3(gyro_dev)
    grav_s = interp3(grav_dev)
    q_s = np.stack([interp(q_fine[:, j]) for j in range(4)], axis=1)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)  # renormalise after interpolation

    t_ns = _to_monotonic_ns(t_samples)

    raw = pl.DataFrame(
        {
            "t_ns": t_ns,
            "lin_acc_x": lin_s[:, 0],
            "lin_acc_y": lin_s[:, 1],
            "lin_acc_z": lin_s[:, 2],
            "rot_rate_x": gyro_s[:, 0],
            "rot_rate_y": gyro_s[:, 1],
            "rot_rate_z": gyro_s[:, 2],
            "quat_w": q_s[:, 0],
            "quat_x": q_s[:, 1],
            "quat_y": q_s[:, 2],
            "quat_z": q_s[:, 3],
            "grav_x": grav_s[:, 0],
            "grav_y": grav_s[:, 1],
            "grav_z": grav_s[:, 2],
        },
        schema={k: RAW_SCHEMA[k] for k in RAW_SCHEMA},
    )
    raw = validate_raw(raw)

    reported_reps = sum(1 for r in reps if r.completed)
    mech_rpe = float(np.clip(10.0 - params.stop_rir, 6.0, 10.0))
    reported_rpe = _round_half(
        float(np.clip(mech_rpe + params.rpe_bias + rng.normal(0, params.rpe_noise), 6.0, 10.0))
    )

    gt = GroundTruth(
        set_id=set_id,
        exercise=params.exercise,
        reached_failure=params.reached_failure,
        failure_rep=failure_rep,
        reported_rpe=reported_rpe,
        mechanical_rpe_true=mech_rpe,
        rpe_bias=params.rpe_bias,
        reps=reps,
    )

    return GeneratedSet(
        raw=raw,
        user_id=user_id,
        session_id=session_id,
        date=date,
        set_index=set_index,
        set_id=set_id,
        exercise=params.exercise,
        load_kg=params.load_kg,
        set_type=params.set_type,
        reported_reps=reported_reps,
        reported_rpe=reported_rpe,
        reached_failure=params.reached_failure,
        ground_truth=gt,
        reps=reps,
    )


def _sample_timestamps(duration_s: float, p: SetParams, rng: np.random.Generator) -> np.ndarray:
    """Irregular, occasionally-gapped sample times over [0, duration] (§3.3)."""
    dt = 1.0 / p.sample_hz
    times = [0.0]
    while times[-1] < duration_s:
        step = dt * (1.0 + rng.uniform(-p.jitter, p.jitter))
        if p.gap_prob > 0 and rng.random() < p.gap_prob:
            step += p.gap_ms / 1000.0
        times.append(times[-1] + step)
    return np.asarray(times[:-1], dtype=np.float64)  # drop the one past the end


def _to_monotonic_ns(t_samples: np.ndarray) -> np.ndarray:
    """Convert float seconds to strictly-increasing int64 nanoseconds."""
    ns = np.round(t_samples * 1e9).astype(np.int64)
    # guarantee strict monotonicity after rounding
    for i in range(1, ns.shape[0]):
        if ns[i] <= ns[i - 1]:
            ns[i] = ns[i - 1] + 1
    return ns


def _round_half(x: float) -> float:
    return round(x * 2) / 2


def generate_session(
    param_list: list[SetParams],
    *,
    user_id: str = "user_synth",
    session_id: str | None = None,
    date: str = "2026-08-18",
) -> list[GeneratedSet]:
    """Generate a whole session of sets sharing one session_id. Set indices are 1-based."""
    session_id = session_id or uuid.uuid4().hex[:12]
    out: list[GeneratedSet] = []
    for i, params in enumerate(param_list, start=1):
        # ensure distinct RNG streams even if callers reuse the same seed
        params_i = replace(params, seed=params.seed + i)
        out.append(
            generate_set(
                params_i,
                user_id=user_id,
                session_id=session_id,
                date=date,
                set_index=i,
            )
        )
    return out
