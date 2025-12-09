#!/usr/bin/env python3
"""
calibrate_environment.py - Standalone microphone/environment calibration for Reachy Mini

Creates a reusable "environment signature" that captures:
1. Background silence (ambient room noise)
2. Breathing motor noise (idle servo sounds)
3. Dance motor noise (active movement servo sounds)

Saves profiles to a .npz file that can be loaded by live_groove.py
to skip the calibration phase on subsequent runs.

Usage:
    python calibrate_environment.py                    # Save to default location
    python calibrate_environment.py --output my_env.npz  # Custom output file
    python calibrate_environment.py --silence-duration 6  # Longer silence phase
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pyaudio

from reachy_mini import ReachyMini, utils
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CalibrationConfig:
    """Configuration for environment calibration."""

    # Audio settings (must match live_groove.py)
    audio_rate: int = 16000
    audio_chunk_size: int = 2048

    # Calibration durations
    silence_duration: float = 4.0      # Phase 1: background silence
    breathing_duration: float = 14.0   # Phase 2: breathing motion
    dance_duration: float = 8.0        # Phase 3: dance motion

    # Motion control
    control_ts: float = 0.01  # 100Hz control loop

    # Robot neutral pose
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ─────────────────────────────────────────────────────────────────────────────
# Audio Device
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Breathing Pose (copied from live_groove.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_breathing_pose(t: float, config: CalibrationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute breathing/idle pose."""
    # Y sway
    y_amplitude = 0.016
    y_freq = 0.2
    y_offset = y_amplitude * np.sin(2.0 * np.pi * y_freq * t)

    # Head roll
    roll_amplitude = 0.222
    roll_freq = 0.15
    roll_offset = roll_amplitude * np.sin(2.0 * np.pi * roll_freq * t)

    position = config.neutral_pos + np.array([0.0, y_offset, 0.0])
    orientation = config.neutral_eul + np.array([roll_offset, 0.0, 0.0])
    antennas = np.zeros(2)

    return position, orientation, antennas


# ─────────────────────────────────────────────────────────────────────────────
# Audio Capture Thread
# ─────────────────────────────────────────────────────────────────────────────
def audio_capture_thread(
    config: CalibrationConfig,
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


def compute_noise_profile(
    audio: np.ndarray,
    method: str = "median",
    exclude_transients: bool = True,
    transient_threshold: float = 2.0,
) -> tuple[np.ndarray, float, dict]:
    """Compute spectral noise profile from audio samples.

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


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Phases
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_silence(
    mini: ReachyMini,
    config: CalibrationConfig,
    audio_device_index: int | None,
) -> tuple[np.ndarray, float, dict]:
    """Phase 1: Record background silence with robot still."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Background Silence ({config.silence_duration:.0f}s)")
    print(f"{'='*60}")
    print("Robot is still. Please ensure room is quiet (no music).")
    print()

    # Move to neutral and settle
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )
    time.sleep(0.5)

    # Start audio capture
    collected: list = []
    stop_event = threading.Event()
    thread = threading.Thread(
        target=audio_capture_thread,
        args=(config, audio_device_index, config.silence_duration, collected, stop_event),
        daemon=True,
    )
    thread.start()

    # Wait with progress
    start = time.time()
    last_print = 0
    while time.time() - start < config.silence_duration:
        elapsed = time.time() - start
        if elapsed - last_print >= 0.5:
            remaining = config.silence_duration - elapsed
            print(f"   Recording silence: {remaining:.1f}s remaining...")
            last_print = elapsed
        time.sleep(0.1)

    stop_event.set()
    thread.join(timeout=2.0)

    if not collected:
        print("   WARNING: No audio collected!")
        return np.zeros(1025), 0.0, {}

    audio = np.concatenate(collected)
    # Silence shouldn't have transients, but use median for consistency
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)
    print(f"   Silence RMS: {rms:.6f}")
    print(f"   Profile shape: {profile.shape}")
    return profile, rms, stats


def calibrate_breathing(
    mini: ReachyMini,
    config: CalibrationConfig,
    audio_device_index: int | None,
) -> tuple[np.ndarray, float, dict]:
    """Phase 2: Record motor noise during breathing motion.

    NOTE: We do NOT exclude transients here because antenna "twangs"
    (rapid oscillations that sound like door stopper springs) are
    normal motor behavior that we want to capture in the profile.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Breathing Motor Noise ({config.breathing_duration:.0f}s)")
    print(f"{'='*60}")
    print("Robot will do smooth breathing motion.")
    print("(Capturing antenna servo noise as normal motor sound)")
    print()

    # Start audio capture
    collected: list = []
    stop_event = threading.Event()
    thread = threading.Thread(
        target=audio_capture_thread,
        args=(config, audio_device_index, config.breathing_duration, collected, stop_event),
        daemon=True,
    )
    thread.start()

    # Run breathing motion at 100Hz
    breathing_time = 0.0
    start = time.time()
    last_loop = start
    last_print = 0

    while time.time() - start < config.breathing_duration:
        loop_start = time.time()
        dt = loop_start - last_loop
        last_loop = loop_start

        # Progress (every 0.5s)
        elapsed = loop_start - start
        if elapsed - last_print >= 0.5:
            remaining = config.breathing_duration - elapsed
            print(f"   Recording breathing: {remaining:.1f}s remaining...")
            last_print = elapsed

        # Smooth breathing motion
        breathing_time += dt
        pos, ori, ant = compute_breathing_pose(breathing_time, config)
        mini.set_target(
            utils.create_head_pose(*pos, *ori, degrees=False),
            antennas=ant,
        )

        # Maintain 100Hz
        sleep_time = max(0.0, config.control_ts - (time.time() - loop_start))
        if sleep_time > 0:
            time.sleep(sleep_time)

    stop_event.set()
    thread.join(timeout=2.0)

    # Return to neutral
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    if not collected:
        print("   WARNING: No audio collected!")
        return np.zeros(1025), 0.0, {}

    audio = np.concatenate(collected)
    # Use median but do NOT exclude transients - antenna twangs are normal motor noise
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)
    print(f"   Breathing RMS: {rms:.6f}")
    print(f"   Profile shape: {profile.shape}")
    return profile, rms, stats


# Amplitude override for headbanger during calibration - must match live_groove.py
HEADBANGER_AMPLITUDE = 0.3


def calibrate_dance(
    mini: ReachyMini,
    config: CalibrationConfig,
    audio_device_index: int | None,
) -> tuple[np.ndarray, float, dict]:
    """Phase 3: Record motor noise during dance motion.

    Uses the SAME amplitude as actual dancing (0.3) to avoid collisions
    and capture realistic motor noise. No need to exclude transients
    since the robot won't be banging into its body at this amplitude.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Dance Motor Noise ({config.dance_duration:.0f}s)")
    print(f"{'='*60}")
    print(f"Robot will do headbanger combo at {HEADBANGER_AMPLITUDE:.0%} amplitude.")
    print("(Same amplitude as actual dancing - no collisions expected)")
    print()

    # Start audio capture
    collected: list = []
    stop_event = threading.Event()
    thread = threading.Thread(
        target=audio_capture_thread,
        args=(config, audio_device_index, config.dance_duration, collected, stop_event),
        daemon=True,
    )
    thread.start()

    # Get headbanger move
    move_fn, base_params, _ = AVAILABLE_MOVES["headbanger_combo"]
    params = base_params.copy()
    amp_scale = HEADBANGER_AMPLITUDE  # Same as live_groove.py MOVE_AMPLITUDE_OVERRIDES

    # Run dance motion at 100Hz
    t_beats = 0.0
    start = time.time()
    last_loop = start
    last_print = 0

    while time.time() - start < config.dance_duration:
        loop_start = time.time()
        dt = loop_start - last_loop
        last_loop = loop_start

        # Progress (every 0.5s)
        elapsed = loop_start - start
        if elapsed - last_print >= 0.5:
            remaining = config.dance_duration - elapsed
            print(f"   Recording dance: {remaining:.1f}s remaining...")
            last_print = elapsed

        # Dance motion at 120 BPM
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

        # Maintain 100Hz
        sleep_time = max(0.0, config.control_ts - (time.time() - loop_start))
        if sleep_time > 0:
            time.sleep(sleep_time)

    stop_event.set()
    thread.join(timeout=2.0)

    # Return to neutral
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    if not collected:
        print("   WARNING: No audio collected!")
        return np.zeros(1025), 0.0, {}

    audio = np.concatenate(collected)
    # Use median, no transient exclusion needed at reduced amplitude (no collisions)
    profile, rms, stats = compute_noise_profile(audio, method="median", exclude_transients=False)
    print(f"   Dance RMS: {rms:.6f}")
    print(f"   Profile shape: {profile.shape}")
    return profile, rms, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Calibrate microphone environment for Reachy Mini dance"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="environment_profile.npz",
        help="Output file path (default: environment_profile.npz)"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=4.0,
        help="Duration of silence calibration in seconds (default: 4)"
    )
    parser.add_argument(
        "--breathing-duration",
        type=float,
        default=14.0,
        help="Duration of breathing calibration in seconds (default: 14)"
    )
    parser.add_argument(
        "--dance-duration",
        type=float,
        default=8.0,
        help="Duration of dance calibration in seconds (default: 8)"
    )
    args = parser.parse_args()

    config = CalibrationConfig(
        silence_duration=args.silence_duration,
        breathing_duration=args.breathing_duration,
        dance_duration=args.dance_duration,
    )

    print("\n" + "=" * 60)
    print("REACHY MINI ENVIRONMENT CALIBRATION")
    print("=" * 60)
    print(f"Output file: {args.output}")
    print(f"Silence duration: {config.silence_duration}s")
    print(f"Breathing duration: {config.breathing_duration}s")
    print(f"Dance duration: {config.dance_duration}s")
    print()
    print("Please ensure NO MUSIC is playing during calibration!")
    print("=" * 60)

    # Find audio device
    audio_device_index = find_audio_device("Reachy Mini Audio")
    if audio_device_index is None:
        print("WARNING: Reachy Mini Audio device not found, using default input")

    # Connect to robot
    print("\nConnecting to Reachy Mini...")
    with ReachyMini(media_backend="no_media") as mini:
        mini.enable_motors()

        # Run calibration phases
        silence_profile, silence_rms, silence_stats = calibrate_silence(mini, config, audio_device_index)
        breathing_profile, breathing_rms, breathing_stats = calibrate_breathing(mini, config, audio_device_index)
        dance_profile, dance_rms, dance_stats = calibrate_dance(mini, config, audio_device_index)

        # Combine profiles (breathing/dance include silence as baseline)
        breathing_combined = np.maximum(breathing_profile, silence_profile)
        dance_combined = np.maximum(dance_profile, silence_profile)

        # Save to file
        output_path = Path(args.output)
        metadata = {
            "created": datetime.now().isoformat(),
            "silence_duration": config.silence_duration,
            "breathing_duration": config.breathing_duration,
            "dance_duration": config.dance_duration,
            "audio_rate": config.audio_rate,
            "audio_chunk_size": config.audio_chunk_size,
            "n_fft": 2048,
            "hop_length": 512,
            "silence_rms": silence_rms,
            "breathing_rms": breathing_rms,
            "dance_rms": dance_rms,
            # Analysis stats
            "profile_method": "median",
            "transient_exclusion": True,
            "silence_stats": silence_stats,
            "breathing_stats": breathing_stats,
            "dance_stats": dance_stats,
        }

        np.savez(
            output_path,
            silence_profile=silence_profile,
            breathing_profile=breathing_combined,
            dance_profile=dance_combined,
            metadata=json.dumps(metadata),
        )

        print(f"\n{'='*60}")
        print("CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Saved environment profile to: {output_path.absolute()}")
        print()
        print("Profile summary:")
        print(f"  Silence RMS:   {silence_rms:.6f}")
        print(f"  Breathing RMS: {breathing_rms:.6f}")
        print(f"  Dance RMS:     {dance_rms:.6f}")
        print()
        print("To use this profile with live_groove.py:")
        print(f"  python live_groove.py --profile {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
