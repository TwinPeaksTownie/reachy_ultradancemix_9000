#!/usr/bin/env python3
# coding: utf-8
"""
reachy_rhythm_controller.py - v12.0 (Simplified, no keyboard, no smart correction)

Real-time robot choreography driven by live BPM from the microphone.
- No manual controls (no keyboard listener).
- No "smart correction" phase alignment. The beat clock follows detected BPM directly.
- Auto-advances dance moves every N beats.

Dependencies: numpy, librosa, pyaudio, reachy_mini
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import librosa
import numpy as np
import pyaudio

from reachy_mini import ReachyMini, utils
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


# ───────────────────────────── Audio Device Selection ─────────────────────────
def find_audio_device(name_pattern: str = "Reachy Mini Audio") -> int | None:
    """Find audio input device index by name pattern."""
    pa = pyaudio.PyAudio()
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and name_pattern in info['name']:
            device_index = i
            print(f"Found audio device: [{i}] {info['name']}")
            break
    pa.terminate()
    return device_index


# ───────────────────────────── Move Amplitude Overrides ──────────────────────
# Per-move amplitude scaling (1.0 = full, 0.8 = 20% reduction, etc.)
# Moves not listed here use 1.0 (full amplitude)
MOVE_AMPLITUDE_OVERRIDES = {
    "headbanger_combo": 0.3,
    "dizzy_spin": 0.8,
    "pendulum_swing": 0.4,
    "jackson_square": 0.6,
    "side_to_side_sway": 0.65,
    "sharp_side_tilt": 0.35,
    "grid_snap": 0.5,
    "side_peakaboo": 0.4,
    "simple_nod": 0.5,
    "chin_lead": 0.9,
}


# ───────────────────────────────── Config ────────────────────────────────────
@dataclass
class Config:
    # Main control loop period (s). Lower for tighter control, higher for lower CPU.
    control_ts: float = 0.01

    # Audio analysis window (s). Short window = lives in the "now", reacts instantly.
    # 1.6s is ~1 bar at moderate tempo. Robot loses beat easily, finds it instantly.
    audio_win: float = 1.6

    # Microphone sample rate (Hz). Reachy Mini Audio runs at 16000 Hz.
    audio_rate: int = 16000

    # Mic buffer size per read. Smaller reduces latency; too small increases CPU/underruns.
    audio_chunk_size: int = 2048

    # Noise calibration duration (s). Records noise while robot does breathing.
    # 14s covers 2 full breathing cycles (slowest is roll at 6.67s/cycle).
    noise_calibration_duration: float = 14.0

    # Dance noise calibration duration (s). Records noise while robot does dance moves.
    dance_noise_calibration_duration: float = 8.0

    # How aggressively to subtract noise (1.0 = full subtraction, 0.5 = half).
    noise_subtraction_strength: float = 1.0

    # How many most recent BPM estimates to average for stability.
    bpm_stability_buffer: int = 6

    # BPM range clamping - forces BPM into this range by halving/doubling.
    # This fixes half-time/double-time detection issues.
    bpm_min: float = 70.0
    bpm_max: float = 140.0

    # Max allowed standard deviation over the stability buffer to consider "Locked".
    # Lower = stricter, higher = looser. 5.0 allows coasting through brief confusion.
    bpm_stability_threshold: float = 5.0

    # If BPM becomes Unstable, how many consecutive unstable periods we tolerate before pausing motion.
    # Higher = more patience, lets robot coast through momentary analysis glitches.
    unstable_periods_before_stop: int = 4

    # If we haven't seen audio events for this many seconds, consider silence and stop motion.
    silence_tmo: float = 2.0

    # Volume gate threshold (RMS). Audio quieter than this is treated as silence/motor noise.
    # Prevents self-excitation where robot dances to its own motor sounds.
    # Tune based on your setup - check RawAmp in debug output.
    volume_gate_threshold: float = 0.005

    # Minimum ratio of cleaned audio to threshold to consider "confident" music presence.
    # Prevents dancing to residual noise that barely passes the gate.
    # Value of 1.5 means cleaned_rms must be 1.5x the threshold to dance.
    music_confidence_ratio: float = 1.5

    # Buffer of recent accepted beat times used by the graph and control.
    beat_buffer_size: int = 20

    # Beat deduplication: beats closer than this fraction of the expected interval are considered duplicates.
    # Increase to remove double triggers; decrease if valid syncopations are dropped.
    min_interval_factor: float = 0.5

    # Move duration in beats. One full move executes for this many beats.
    beats_per_sequence: int = 8

    # Minimum breathing time (seconds) between moves. Forces a pause after each move completes.
    min_breathing_between_moves: float = 0.2

    # How often the terminal UI refreshes (Hz).
    ui_update_rate: float = 1.0

    # Neutral pose (position in meters and Euler orientation in radians).
    # Adjust to fit your neutral posture in your setup.
    # Z offset of 0.01m added to prevent head-body collision.
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # The following two are kept for completeness but are not used (smart correction removed).
    offset_correction_rate: float = 0.02
    max_phase_correction_per_frame: float = 0.005

    # Derived: number of samples in the rolling audio window.
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────────────── Runtime State ─────────────────────────────────
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = "Init"
        self.beats: collections.deque[float] = collections.deque(maxlen=512)
        self.unstable_period_count = 0
        self.has_ever_locked = False  # Require stable lock before dancing
        self.bpm_std = 0.0  # Debug: track BPM standard deviation
        self.cleaned_amplitude = 0.0  # Debug: track audio amplitude after noise subtraction
        self.raw_amplitude = 0.0  # Debug: track raw audio amplitude (used for presence detection)
        self.is_breathing = True  # True when robot is in breathing/idle mode (set by main loop)
        self.music_confident = False  # True when cleaned signal is well above threshold


class Choreographer:
    def __init__(self):
        # Registry of available moves (name -> (fn, base_params, meta))
        self.move_names = list(AVAILABLE_MOVES.keys())
        random.shuffle(self.move_names)  # Randomize move order
        # Default single waveform; moves that support it will consume it.
        self.waveforms = ["sin"]
        self.move_idx = 0
        self.waveform_idx = 0
        self.amplitude_scale = 1.0
        self.beat_counter_for_cycle = 0.0

    def current_move_name(self):
        return self.move_names[self.move_idx]

    def current_waveform(self):
        return self.waveforms[self.waveform_idx]

    def advance(self, beats_this_frame, config: Config):
        self.beat_counter_for_cycle += beats_this_frame
        if self.beat_counter_for_cycle >= config.beats_per_sequence:
            self.move_idx += 1
            # Reshuffle when we've played all moves
            if self.move_idx >= len(self.move_names):
                random.shuffle(self.move_names)
                self.move_idx = 0
            self.beat_counter_for_cycle = 0.0


# ───────────────────────────── Idle Breathing Motion ─────────────────────────
def compute_breathing_pose(t: float, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute breathing/idle pose when music state is unstable.

    Based on pi-500-reachy-mini-client-1 BreathingMove with modifications:
    - Y sway: 8mm amplitude, 0.2 Hz (5 second cycle)
    - Head roll: 15% intensity sin wave at 0.15 Hz
    - Antenna sway: 24° amplitude (2x original 12°), 0.5 Hz

    Returns:
        (position, orientation, antennas) - all as numpy arrays
    """
    # Y sway - gentle side-to-side breathing motion (doubled from original 8mm)
    y_amplitude = 0.016  # 16mm (100% increase from 8mm)
    y_freq = 0.2  # 0.2 Hz = 5 second cycle
    y_offset = y_amplitude * np.sin(2.0 * np.pi * y_freq * t)

    # Head roll - 30 degree amplitude
    roll_amplitude = 0.222  # 30 degrees in radians
    roll_freq = 0.15  # Slow, gentle roll
    roll_offset = roll_amplitude * np.sin(2.0 * np.pi * roll_freq * t)

    # No antenna movement during breathing - motors are too close to microphone
    position = config.neutral_pos + np.array([0.0, y_offset, 0.0])
    orientation = config.neutral_eul + np.array([roll_offset, 0.0, 0.0])
    antennas = np.zeros(2)

    return position, orientation, antennas


# ───────────────────────────── Noise Calibration ─────────────────────────────
# Audio capture runs in a separate thread so motion can be smooth during calibration.
# Without this, audio read() blocks for ~128ms per chunk, making motion jerky.


def compute_noise_profile(
    audio: np.ndarray,
    method: str = "median",
    exclude_transients: bool = True,
    transient_threshold: float = 2.0,
) -> tuple[np.ndarray, float, dict]:
    """Compute spectral noise profile from audio samples.

    Uses median (robust to outliers) and can exclude transient frames (collisions/knocks).

    Args:
        audio: Raw audio samples
        method: "mean", "median", or "percentile_25"
        exclude_transients: If True, detect and exclude collision/impact frames
        transient_threshold: Frames with RMS > threshold * median_rms are excluded

    Returns:
        (profile, rms, stats) where stats contains analysis metadata
    """
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Compute per-frame RMS to detect transients
    frame_rms = np.sqrt(np.mean(magnitude**2, axis=0))
    median_frame_rms = np.median(frame_rms)

    stats = {
        "total_frames": magnitude.shape[1],
        "excluded_frames": 0,
        "transient_threshold": transient_threshold,
        "median_frame_rms": float(median_frame_rms),
        "method": method,
    }

    if exclude_transients and median_frame_rms > 0:
        # Find frames that are likely transients (collisions, knocks)
        threshold = median_frame_rms * transient_threshold
        quiet_mask = frame_rms < threshold

        stats["excluded_frames"] = int(np.sum(~quiet_mask))
        stats["max_frame_rms"] = float(np.max(frame_rms))

        if np.sum(quiet_mask) > 10:  # Need at least some quiet frames
            magnitude = magnitude[:, quiet_mask]
            if stats["excluded_frames"] > 0:
                print(f"      Excluded {stats['excluded_frames']} transient frames (collisions/knocks)")

    # Compute profile using selected method
    if method == "median":
        profile = np.median(magnitude, axis=1)
    elif method == "percentile_25":
        profile = np.percentile(magnitude, 25, axis=1)
    else:  # mean
        profile = np.mean(magnitude, axis=1)

    rms = float(np.sqrt(np.mean(audio**2)))
    return profile, rms, stats


def _audio_capture_thread(
    config: Config,
    audio_device_index: int | None,
    duration: float,
    collected: list,
    stop_event: threading.Event,
) -> None:
    """Background thread to capture audio without blocking motion."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=config.audio_rate,
        input=True,
        input_device_index=audio_device_index,
        frames_per_buffer=config.audio_chunk_size,
    )

    samples_needed = int(config.audio_rate * duration)

    while len(collected) * config.audio_chunk_size < samples_needed and not stop_event.is_set():
        try:
            stereo = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            chunk = (stereo[0::2] + stereo[1::2]) / 2.0
            collected.append(chunk)
        except (IOError, ValueError):
            continue

    stream.stop_stream()
    stream.close()
    pa.terminate()


def calibrate_background_silence(
    mini: ReachyMini,
    config: Config,
    audio_device_index: int | None = None
) -> np.ndarray:
    """Record background silence with robot completely still.

    This captures the ambient noise floor (room noise, mic hiss, etc.)
    that should be subtracted from all readings.
    """
    silence_duration = 4.0  # 4 seconds of silence
    print(f"\n{'='*60}")
    print(f"PHASE 1: Background Silence ({silence_duration:.0f}s)")
    print(f"{'='*60}")
    print("   Robot is still. Please ensure room is quiet (no music).")

    # Move robot to neutral and hold still
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )
    time.sleep(0.5)  # Let robot settle

    # Start audio capture in background thread
    collected: list = []
    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=_audio_capture_thread,
        args=(config, audio_device_index, silence_duration, collected, stop_event),
        daemon=True,
    )
    audio_thread.start()

    # Wait and show progress (robot stays still)
    start_time = time.time()
    while time.time() - start_time < silence_duration:
        remaining = silence_duration - (time.time() - start_time)
        print(f"\r   Recording silence: {remaining:.1f}s remaining...", end="", flush=True)
        time.sleep(0.1)

    # Wait for audio thread to finish
    stop_event.set()
    audio_thread.join(timeout=2.0)

    if not collected:
        print("\r   WARNING: No audio collected during silence calibration!")
        return np.zeros(1025)  # Default profile shape for n_fft=2048

    audio = np.concatenate(collected)

    # Silence shouldn't have transients, but use median for consistency
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)
    print(f"\r✅ Phase 1 complete. Silence RMS: {rms:.6f}                    ")
    return profile


def calibrate_noise_with_breathing(
    mini: ReachyMini,
    config: Config,
    audio_device_index: int | None = None,
    silence_profile: np.ndarray | None = None
) -> np.ndarray:
    """Record noise while robot does breathing motion to capture motor sounds.

    Audio capture runs in a separate thread so breathing motion stays smooth.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Breathing Motor Noise ({config.noise_calibration_duration:.0f}s)")
    print(f"{'='*60}")
    print("   Robot will do smooth breathing motion.")

    # Start audio capture in background thread
    collected: list = []
    stop_event = threading.Event()
    audio_thread_handle = threading.Thread(
        target=_audio_capture_thread,
        args=(config, audio_device_index, config.noise_calibration_duration, collected, stop_event),
        daemon=True,
    )
    audio_thread_handle.start()

    # Run breathing motion in main thread (smooth 100Hz updates)
    breathing_time = 0.0
    start_time = time.time()
    last_loop_time = start_time

    while time.time() - start_time < config.noise_calibration_duration:
        loop_start = time.time()
        dt = loop_start - last_loop_time
        last_loop_time = loop_start

        remaining = config.noise_calibration_duration - (loop_start - start_time)
        print(f"\r   Recording breathing noise: {remaining:.1f}s remaining...", end="", flush=True)

        # Smooth breathing motion
        breathing_time += dt
        breath_pos, breath_ori, breath_ant = compute_breathing_pose(breathing_time, config)
        mini.set_target(
            utils.create_head_pose(*breath_pos, *breath_ori, degrees=False),
            antennas=breath_ant,
        )

        # Maintain 100Hz control loop
        elapsed_loop = time.time() - loop_start
        sleep_time = max(0.0, config.control_ts - elapsed_loop)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Stop audio capture and wait
    stop_event.set()
    audio_thread_handle.join(timeout=2.0)

    # Return to neutral
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    if not collected:
        print("\r   WARNING: No audio collected during breathing calibration!")
        return silence_profile if silence_profile is not None else np.zeros(1025)

    audio = np.concatenate(collected)

    # Use median but do NOT exclude transients - antenna twangs are normal motor noise
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)

    # Combine with silence profile: take the maximum of both
    if silence_profile is not None:
        noise_profile = np.maximum(profile, silence_profile)
    else:
        noise_profile = profile

    print(f"\r✅ Phase 2 complete. Breathing RMS: {rms:.6f}                    ")
    return noise_profile


def calibrate_dance_noise(
    mini: ReachyMini,
    config: Config,
    audio_device_index: int | None = None,
    silence_profile: np.ndarray | None = None
) -> np.ndarray:
    """Record noise while robot does dance moves to capture dance motor sounds.

    Uses the SAME amplitude as actual dancing (from MOVE_AMPLITUDE_OVERRIDES)
    to avoid collisions and capture realistic motor noise.
    Audio capture runs in a separate thread so dance motion stays smooth.
    """
    # Use same amplitude as actual dancing
    dance_amplitude = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 0.3)

    print(f"\n{'='*60}")
    print(f"PHASE 3: Dance Motor Noise ({config.dance_noise_calibration_duration:.0f}s)")
    print(f"{'='*60}")
    print(f"   Robot will do headbanger combo at {dance_amplitude:.0%} amplitude.")

    # Start audio capture in background thread
    collected: list = []
    stop_event = threading.Event()
    audio_thread_handle = threading.Thread(
        target=_audio_capture_thread,
        args=(config, audio_device_index, config.dance_noise_calibration_duration, collected, stop_event),
        daemon=True,
    )
    audio_thread_handle.start()

    # Get headbanger move function
    move_fn, base_params, _ = AVAILABLE_MOVES["headbanger_combo"]
    params = base_params.copy()
    amp_scale = dance_amplitude  # Same amplitude as actual dancing

    # Run dance motion in main thread (smooth 100Hz updates)
    t_beats = 0.0
    start_time = time.time()
    last_loop_time = start_time

    while time.time() - start_time < config.dance_noise_calibration_duration:
        loop_start = time.time()
        dt = loop_start - last_loop_time
        last_loop_time = loop_start

        remaining = config.dance_noise_calibration_duration - (loop_start - start_time)
        print(f"\r   Recording dance noise: {remaining:.1f}s remaining...", end="", flush=True)

        # Smooth dance motion at 120 BPM
        t_beats += dt * (120.0 / 60.0)
        offsets = move_fn(t_beats, **params)

        scaled_pos = offsets.position_offset * amp_scale
        scaled_ori = offsets.orientation_offset * amp_scale
        scaled_ant = offsets.antennas_offset * amp_scale

        mini.set_target(
            utils.create_head_pose(
                *(config.neutral_pos + scaled_pos),
                *(config.neutral_eul + scaled_ori),
                degrees=False,
            ),
            antennas=scaled_ant,
        )

        # Maintain 100Hz control loop
        elapsed_loop = time.time() - loop_start
        sleep_time = max(0.0, config.control_ts - elapsed_loop)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Stop audio capture and wait
    stop_event.set()
    audio_thread_handle.join(timeout=2.0)

    # Return to neutral
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    if not collected:
        print("\r   WARNING: No audio collected during dance calibration!")
        return silence_profile if silence_profile is not None else np.zeros(1025)

    audio = np.concatenate(collected)

    # Use median, no transient exclusion needed at reduced amplitude (no collisions)
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)

    # Combine with silence profile: take the maximum of both
    if silence_profile is not None:
        noise_profile = np.maximum(profile, silence_profile)
    else:
        noise_profile = profile

    print(f"\r✅ Phase 3 complete. Dance RMS: {rms:.6f}                    ")
    return noise_profile


def load_environment_profile(profile_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load saved environment profile from .npz file.

    Returns:
        Tuple of (silence_profile, breathing_profile, dance_profile) or None if loading fails.
    """
    path = Path(profile_path)
    if not path.exists():
        print(f"WARNING: Profile file not found: {path}")
        return None

    try:
        data = np.load(path, allow_pickle=True)
        silence_profile = data["silence_profile"]
        breathing_profile = data["breathing_profile"]
        dance_profile = data["dance_profile"]

        # Parse metadata for display
        metadata = json.loads(str(data["metadata"]))
        print(f"\n{'='*60}")
        print("LOADED ENVIRONMENT PROFILE")
        print(f"{'='*60}")
        print(f"   File: {path}")
        print(f"   Created: {metadata.get('created', 'unknown')}")
        print(f"   Silence RMS: {metadata.get('silence_rms', 0):.6f}")
        print(f"   Breathing RMS: {metadata.get('breathing_rms', 0):.6f}")
        print(f"   Dance RMS: {metadata.get('dance_rms', 0):.6f}")
        print(f"{'='*60}\n")

        return silence_profile, breathing_profile, dance_profile

    except Exception as e:
        print(f"WARNING: Failed to load profile: {e}")
        return None


def subtract_noise(audio: np.ndarray, noise_profile: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Subtract noise profile from audio using spectral subtraction."""
    n_fft = 2048
    hop_length = 512

    # Compute STFT of input audio
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Subtract noise profile (broadcast across time frames)
    # Use strength parameter to control how aggressive the subtraction is
    cleaned_magnitude = magnitude - (noise_profile[:, np.newaxis] * strength)

    # Floor at zero (no negative magnitudes)
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

    # Reconstruct complex STFT and invert
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(audio))

    return cleaned_audio.astype(np.float32)


def clamp_bpm(bpm: float, bpm_min: float, bpm_max: float) -> float:
    """Force BPM into range by halving or doubling.

    Fixes librosa's half-time/double-time confusion where it flip-flops
    between e.g. 83 and 166 BPM for the same song.
    """
    if bpm <= 0:
        return bpm

    # Double until we're at or above minimum
    while bpm < bpm_min:
        bpm *= 2.0

    # Halve until we're at or below maximum
    while bpm > bpm_max:
        bpm /= 2.0

    return bpm


# ───────────────────────────── Worker Threads ────────────────────────────────
def audio_thread(
    state: MusicState,
    config: Config,
    stop_event: threading.Event,
    breathing_noise_profile: np.ndarray | None = None,
    dance_noise_profile: np.ndarray | None = None,
    audio_device_index: int | None = None,
) -> None:
    """Continuously read microphone audio and update BPM + beat times."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=2,  # Reachy Mini Audio requires stereo
        rate=config.audio_rate,
        input=True,
        input_device_index=audio_device_index,
        frames_per_buffer=config.audio_chunk_size,
    )

    buf = np.empty(0, dtype=np.float32)
    bpm_hist = collections.deque(maxlen=config.bpm_stability_buffer)

    while not stop_event.is_set():
        try:
            stereo = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            # Convert stereo to mono by averaging channels
            audio_chunk = (stereo[0::2] + stereo[1::2]) / 2.0
            buf = np.append(buf, audio_chunk)
        except (IOError, ValueError):
            continue

        if len(buf) < config.audio_buffer_len:
            continue

        # === 1. NOISE SUBTRACTION FIRST (Clean Gate approach) ===
        # Remove motor noise BEFORE checking volume, so we gate on music not motors
        analysis_buf = buf
        with state.lock:
            is_breathing = state.is_breathing
        if is_breathing and breathing_noise_profile is not None:
            analysis_buf = subtract_noise(buf, breathing_noise_profile, config.noise_subtraction_strength)
        elif not is_breathing and dance_noise_profile is not None:
            analysis_buf = subtract_noise(buf, dance_noise_profile, config.noise_subtraction_strength)

        # === 2. VOLUME GATE ON CLEANED AUDIO ===
        # Gate on cleaned signal - if motor noise was all there was, this will be near zero
        rms_amplitude = np.sqrt(np.mean(buf**2))  # Raw for debug display
        cleaned_rms = np.sqrt(np.mean(analysis_buf**2))  # Cleaned for gate decision

        if cleaned_rms < config.volume_gate_threshold:
            # After noise removal, nothing left - treat as silence
            with state.lock:
                state.state = "Gathering"
                state.librosa_bpm = 0.0
                state.raw_amplitude = rms_amplitude
                state.last_event_time = 0.0
                state.music_confident = False
            buf = buf[-int(config.audio_rate * config.audio_win):]
            continue

        # Check if signal is confidently music (well above threshold, not marginal)
        confidence_threshold = config.volume_gate_threshold * config.music_confidence_ratio
        is_confident = cleaned_rms > confidence_threshold

        # === 3. BEAT DETECTION ON CLEANED AUDIO ===
        tempo, beat_frames = librosa.beat.beat_track(
            y=analysis_buf, sr=config.audio_rate, units="frames", tightness=80
        )
        now = time.time()
        raw_tempo = float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo)

        # Clamp BPM to expected range (fixes half-time/double-time confusion)
        tempo_val = clamp_bpm(raw_tempo, config.bpm_min, config.bpm_max)

        # Track amplitudes for debug
        cleaned_amplitude = np.abs(analysis_buf).mean()
        has_audio = cleaned_rms > config.volume_gate_threshold and len(beat_frames) > 0

        with state.lock:
            # Only update last_event_time if we actually detected audio
            if has_audio:
                state.last_event_time = now
            state.raw_librosa_bpm = raw_tempo  # Show raw for debugging

            if tempo_val > 40:
                bpm_hist.append(tempo_val)  # Use clamped value for averaging
                state.librosa_bpm = float(np.mean(bpm_hist))

            win_dur = len(buf) / config.audio_rate
            abs_times = [
                now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate))
                for f in beat_frames
            ]
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05:
                    state.beats.append(t)

            # Track debug values
            state.raw_amplitude = rms_amplitude
            state.cleaned_amplitude = cleaned_amplitude
            state.bpm_std = float(np.std(bpm_hist)) if len(bpm_hist) > 1 else 0.0
            state.music_confident = is_confident

            if len(bpm_hist) < config.bpm_stability_buffer:
                state.state = "Gathering"
                state.unstable_period_count = 0
            elif np.std(bpm_hist) < config.bpm_stability_threshold:
                state.state = "Locked"
                state.unstable_period_count = 0
                state.has_ever_locked = True
            else:
                state.state = "Unstable"
                state.unstable_period_count += 1

        # Keep a rolling audio buffer matching the analysis window
        buf = buf[-int(config.audio_rate * config.audio_win) :]

    stream.stop_stream()
    stream.close()
    pa.terminate()


def ui_thread(data_queue: Queue, config: Config, stop_event: threading.Event):
    """Lightweight terminal UI; refresh rate controlled by config.ui_update_rate."""
    last_ui_print_time, last_data = time.time(), None
    while not stop_event.is_set():
        try:
            while True:
                last_data = data_queue.get_nowait()
        except Empty:
            pass

        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate):
            time.sleep(0.1)
            continue
        last_ui_print_time = now

        paused_status = " | PAUSED (Music Unstable)" if last_data["unstable_pause"] else ""
        locked_status = "YES" if last_data.get("has_ever_locked", False) else "NO"
        confident_status = "YES" if last_data.get("music_confident", False) else "NO"

        print(
            "\n" + "─" * 80 + "\n"
            f"🎵 Music State: {last_data['state']:<10} | BPM (Active/Raw): "
            f"{last_data['active_bpm']:.1f}/{last_data['raw_bpm']:.1f}{paused_status}\n"
            f"📊 Debug: StdDev={last_data.get('bpm_std', 0):.2f} (need <{config.bpm_stability_threshold}) | "
            f"RawAmp={last_data.get('raw_amp', 0):.4f} (need >{config.volume_gate_threshold})\n"
            f"🔒 Locks: EverLocked={locked_status} | MusicConfident={confident_status} (need both YES to dance)\n"
            f"🕺 Dance State: {last_data['move_name']:<25} | Wave: {last_data['waveform']:<8} | Amp: {last_data['amp_scale']:.1f}x\n"
            + "─" * 80
        )
        sys.stdout.flush()


# ───────────────────────────── Main Control Loop ────────────────────────────
def main(config: Config, profile_path: str | Path | None = None) -> None:
    data_queue, stop_event = Queue(), threading.Event()
    music, choreographer = MusicState(), Choreographer()

    # Find the Reachy Mini audio device
    audio_device_index = find_audio_device("Reachy Mini Audio")
    if audio_device_index is None:
        print("WARNING: Reachy Mini Audio device not found, using default input")

    print("Connecting to Reachy Mini...")
    with ReachyMini(media_backend="no_media") as mini:
        # Enable motors so robot responds to commands
        mini.enable_motors()

        # Try to load saved profile, otherwise run live calibration
        breathing_noise_profile = None
        dance_noise_profile = None

        if profile_path:
            loaded = load_environment_profile(profile_path)
            if loaded:
                _, breathing_noise_profile, dance_noise_profile = loaded
                print("Using saved environment profile - skipping calibration!")
            else:
                print("Failed to load profile, falling back to live calibration...")

        if breathing_noise_profile is None or dance_noise_profile is None:
            # Full 3-phase calibration sequence:
            # 1. Background silence (robot still) - captures ambient room noise
            # 2. Breathing motion - captures motor noise during idle breathing
            # 3. Dance motion - captures motor noise during active dancing
            print("\n" + "=" * 60)
            print("STARTING 3-PHASE NOISE CALIBRATION")
            print("Please ensure NO MUSIC is playing during calibration!")
            print("=" * 60)

            # Phase 1: Background silence (robot completely still)
            silence_profile = calibrate_background_silence(mini, config, audio_device_index)

            # Phase 2: Breathing motion noise (combined with silence)
            breathing_noise_profile = calibrate_noise_with_breathing(
                mini, config, audio_device_index, silence_profile
            )

            # Phase 3: Dance motion noise (combined with silence)
            dance_noise_profile = calibrate_dance_noise(
                mini, config, audio_device_index, silence_profile
            )

            print("\n" + "=" * 60)
            print("✅ All calibration phases complete!")
            print("=" * 60)
            print("\nTIP: Run 'python calibrate_environment.py' to save this profile")
            print("     for faster startup next time.")

        # Start audio and UI threads after calibration
        threading.Thread(
            target=audio_thread,
            args=(music, config, stop_event, breathing_noise_profile, dance_noise_profile, audio_device_index),
            daemon=True,
        ).start()
        threading.Thread(target=ui_thread, args=(data_queue, config, stop_event), daemon=True).start()

        last_loop = time.time()
        processed_beats, active_bpm = 0, 0.0
        filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
        t_beats = 0.0
        breathing_time = 0.0  # Track time for idle breathing animation
        is_executing_move = False  # True while a move is in progress
        move_beats_elapsed = 0.0  # Beats since move started
        force_breathing_until = 0.0  # Timestamp until which we must breathe
        last_active_bpm = 0.0  # Remember BPM for completing moves if beat is lost

        print("\nRobot ready — play music!\n")

        try:
            while True:
                loop_start_time = time.time()
                dt = loop_start_time - last_loop
                last_loop = loop_start_time

                with music.lock:
                    librosa_bpm, raw_bpm = music.librosa_bpm, music.raw_librosa_bpm
                    state, last_event_time = music.state, music.last_event_time
                    unstable_count = music.unstable_period_count
                    has_ever_locked = music.has_ever_locked
                    bpm_std = music.bpm_std
                    raw_amp = music.raw_amplitude
                    cleaned_amp = music.cleaned_amplitude
                    music_confident = music.music_confident
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)

                active_bpm = librosa_bpm if time.time() - last_event_time < config.silence_tmo else 0.0

                # Beat filtering and deduplication
                accepted_this_frame = []
                if new_beats and active_bpm > 0:
                    expected_interval = 60.0 / active_bpm
                    min_interval = expected_interval * config.min_interval_factor
                    i = 0
                    while i < len(new_beats):
                        last_beat = (
                            filtered_beat_times[-1]
                            if filtered_beat_times
                            else new_beats[i] - expected_interval
                        )
                        current_beat = new_beats[i]
                        if i + 1 < len(new_beats) and (new_beats[i + 1] - current_beat) < min_interval:
                            c = new_beats[i + 1]
                            e1 = abs((current_beat - last_beat) - expected_interval)
                            e2 = abs((c - last_beat) - expected_interval)
                            accepted_this_frame.append(current_beat if e1 <= e2 else c)
                            i += 2
                        else:
                            if (current_beat - last_beat) > min_interval:
                                accepted_this_frame.append(current_beat)
                            i += 1
                filtered_beat_times.extend(accepted_this_frame)

                # Separate criteria for STARTING vs CONTINUING a move
                # START: Must be Locked AND music_confident (strict) - prevents ego-noise dancing
                # CONTINUE: Can coast through Unstable using last_active_bpm (loose)
                can_start_new_move = active_bpm > 0 and has_ever_locked and state == "Locked" and music_confident
                can_continue_move = has_ever_locked and (state == "Locked" or (state == "Unstable" and unstable_count < config.unstable_periods_before_stop))

                # Tell audio thread whether we're breathing (so it knows to filter motor noise)
                with music.lock:
                    music.is_breathing = not (is_executing_move or can_start_new_move)

                # One-shot execution model:
                # 1. Wait for beat lock (can_dance)
                # 2. Execute ONE full move (beats_per_sequence beats)
                # 3. Force return to breathing
                # 4. Wait minimum breathing time before next move

                now = time.time()
                in_forced_breathing = now < force_breathing_until

                if is_executing_move:
                    # === EXECUTING A MOVE ===
                    # Use last known BPM if current drops to 0 (lets move complete)
                    bpm_for_move = active_bpm if active_bpm > 0 else last_active_bpm
                    beats_this_frame = dt * (bpm_for_move / 60.0)
                    move_beats_elapsed += beats_this_frame
                    t_beats += beats_this_frame

                    # Check if move is complete
                    if move_beats_elapsed >= config.beats_per_sequence:
                        # Move finished - force breathing
                        is_executing_move = False
                        force_breathing_until = now + config.min_breathing_between_moves
                        choreographer.move_idx = (choreographer.move_idx + 1) % len(choreographer.move_names)
                        print(f"   Move complete. Breathing for {config.min_breathing_between_moves}s...")
                    else:
                        # Continue executing move
                        move_name = choreographer.current_move_name()
                        move_fn, base_params, _ = AVAILABLE_MOVES[move_name]
                        params = base_params.copy()

                        if "waveform" in params:
                            params["waveform"] = choreographer.current_waveform()

                        offsets = move_fn(t_beats, **params)

                        amp_scale = MOVE_AMPLITUDE_OVERRIDES.get(move_name, 1.0)
                        scaled_pos = offsets.position_offset * amp_scale
                        scaled_ori = offsets.orientation_offset * amp_scale
                        scaled_ant = offsets.antennas_offset * amp_scale

                        mini.set_target(
                            utils.create_head_pose(
                                *(config.neutral_pos + scaled_pos),
                                *(config.neutral_eul + scaled_ori),
                                degrees=False,
                            ),
                            antennas=scaled_ant,
                        )

                elif can_start_new_move and not in_forced_breathing:
                    # === START A NEW MOVE ===
                    is_executing_move = True
                    t_beats = 0.0
                    move_beats_elapsed = 0.0
                    last_active_bpm = active_bpm  # Remember BPM for this move
                    move_name = choreographer.current_move_name()
                    print(f"\n🔥 BEAT DROP! Starting: {move_name}")

                    # Execute first frame of the move
                    move_fn, base_params, _ = AVAILABLE_MOVES[move_name]
                    params = base_params.copy()

                    if "waveform" in params:
                        params["waveform"] = choreographer.current_waveform()

                    offsets = move_fn(t_beats, **params)

                    amp_scale = MOVE_AMPLITUDE_OVERRIDES.get(move_name, 1.0)
                    scaled_pos = offsets.position_offset * amp_scale
                    scaled_ori = offsets.orientation_offset * amp_scale
                    scaled_ant = offsets.antennas_offset * amp_scale

                    mini.set_target(
                        utils.create_head_pose(
                            *(config.neutral_pos + scaled_pos),
                            *(config.neutral_eul + scaled_ori),
                            degrees=False,
                        ),
                        antennas=scaled_ant,
                    )

                else:
                    # === BREATHING ===
                    breathing_time += dt
                    breath_pos, breath_ori, breath_ant = compute_breathing_pose(breathing_time, config)
                    mini.set_target(
                        utils.create_head_pose(*breath_pos, *breath_ori, degrees=False),
                        antennas=breath_ant,
                    )

                # UI update
                ui_data = {
                    "state": state,
                    "active_bpm": active_bpm,
                    "raw_bpm": raw_bpm,
                    "move_name": choreographer.current_move_name(),
                    "waveform": choreographer.current_waveform(),
                    "amp_scale": choreographer.amplitude_scale,
                    "unstable_pause": not can_continue_move and state == "Unstable",
                    "bpm_std": bpm_std,
                    "raw_amp": raw_amp,
                    "cleaned_amp": cleaned_amp,
                    "has_ever_locked": has_ever_locked,
                    "music_confident": music_confident,
                }
                data_queue.put(ui_data)

                time.sleep(max(0.0, config.control_ts - (time.time() - loop_start_time)))

        except KeyboardInterrupt:
            print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set()
            print("Putting robot to sleep and cleaning up...")
            print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time robot choreography driven by live BPM from the microphone"
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        help="Path to saved environment profile (.npz) to skip calibration"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip calibration entirely (use with caution - may cause self-excitation)"
    )
    args = parser.parse_args()

    cfg = Config()

    # Check for default profile in same directory if no profile specified
    default_profile = Path(__file__).parent / "environment_profile.npz"
    profile_path = args.profile

    if profile_path is None and default_profile.exists():
        print(f"Found default profile: {default_profile}")
        profile_path = str(default_profile)

    if args.no_calibration:
        print("WARNING: Running without calibration - may dance to motor noise!")
        profile_path = None  # Force no profile loading

    main(cfg, profile_path)
