"""
Mode A: Live Groove - BPM-driven dance mode using microphone input.

Listens to live audio via the robot's USB microphone, detects BPM,
and executes pre-recorded dance moves from the library.

This is the "Listener" - social, present, organic. Designed to exist
in a room with people and account for the messiness of the real world.

Key features:
- Noise calibration (captures motor noise profile during breathing/dancing)
- Spectral noise subtraction to filter out motor sounds
- Librosa BPM detection with half/double-time clamping
- BPM stability tracking: Gathering → Locked → Unstable
- Pre-recorded library moves (reachy_mini_dances_library)
- Breathing idle motion when no music detected
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import librosa
import numpy as np
import pyaudio

from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

from .base import DanceMode
from ..core.safety_mixer import MovementIntent
from ..config import MODE_A_CONFIG, AUDIO_CONFIG
from .. import move_config
from .. import mode_settings

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


# Initialize move_config with available moves
move_config.init_moves(list(AVAILABLE_MOVES.keys()))


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


@dataclass
class LiveGrooveConfig:
    """Configuration for Live Groove mode."""
    # Main control loop period (s)
    control_ts: float = 0.01

    # Audio
    audio_rate: int = 16000
    audio_chunk_size: int = 2048
    audio_win: float = 1.6  # Analysis window in seconds

    # Noise calibration
    silence_calibration_duration: float = 4.0  # Background silence phase
    noise_calibration_duration: float = 14.0  # Breathing phase
    dance_noise_calibration_duration: float = 8.0  # Dance phase
    noise_subtraction_strength: float = 1.0

    # BPM detection
    bpm_min: float = 70.0
    bpm_max: float = 140.0
    bpm_stability_buffer: int = 6
    bpm_stability_threshold: float = 5.0

    # Timing
    silence_tmo: float = 2.0
    volume_gate_threshold: float = 0.005  # Lowered from 0.008 for better sensitivity
    music_confidence_ratio: float = 1.5  # Signal must be 1.5x threshold to dance
    beats_per_sequence: int = 8
    min_breathing_between_moves: float = 0.2
    unstable_periods_before_stop: int = 4

    # Neutral pose
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Derived
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


class MusicState:
    """Thread-safe state for audio analysis."""
    def __init__(self):
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = "Init"
        self.beats: collections.deque = collections.deque(maxlen=512)
        self.unstable_period_count = 0
        self.has_ever_locked = False
        self.bpm_std = 0.0
        self.cleaned_amplitude = 0.0
        self.raw_amplitude = 0.0
        self.is_breathing = True
        self.music_confident = False  # True when cleaned signal is well above threshold


def compute_noise_profile(
    audio: np.ndarray,
    method: str = "median",
    exclude_transients: bool = False,
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
                print(f"      Excluded {stats['excluded_frames']} transient frames")

    # Compute profile using selected method
    if method == "median":
        profile = np.median(magnitude, axis=1)
    elif method == "percentile_25":
        profile = np.percentile(magnitude, 25, axis=1)
    else:  # mean
        profile = np.mean(magnitude, axis=1)

    rms = float(np.sqrt(np.mean(audio**2)))
    return profile, rms, stats


def load_environment_profile(profile_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load saved environment profile from .npz file.

    Returns:
        Tuple of (silence_profile, breathing_profile, dance_profile) or None if loading fails.
    """
    path = Path(profile_path)
    if not path.exists():
        print(f"[LiveGroove] WARNING: Profile file not found: {path}")
        return None

    try:
        data = np.load(path, allow_pickle=True)
        silence_profile = data["silence_profile"]
        breathing_profile = data["breathing_profile"]
        dance_profile = data["dance_profile"]

        # Parse metadata for display
        metadata = json.loads(str(data["metadata"]))
        print(f"[LiveGroove] Loaded profile from {path}")
        print(f"   Created: {metadata.get('created', 'unknown')}")
        print(f"   Silence RMS: {metadata.get('silence_rms', 0):.6f}")
        print(f"   Breathing RMS: {metadata.get('breathing_rms', 0):.6f}")
        print(f"   Dance RMS: {metadata.get('dance_rms', 0):.6f}")

        return silence_profile, breathing_profile, dance_profile

    except Exception as e:
        print(f"[LiveGroove] WARNING: Failed to load profile: {e}")
        return None


class MoveChoreographer:
    """Manages dance move selection and sequencing."""
    def __init__(self):
        self.base_moves = list(AVAILABLE_MOVES.keys())
        self.move_names = []
        self.waveforms = ["sin"]
        self.move_idx = 0
        self.waveform_idx = 0
        self.amplitude_scale = 1.0
        self.beat_counter_for_cycle = 0.0
        # Build initial move list with mirrored versions
        self.rebuild_move_list()

    def rebuild_move_list(self) -> None:
        """Rebuild move list including enabled mirrored versions."""
        moves = list(self.base_moves)

        # Add mirrored versions for moves that have mirror enabled
        mirror_settings = move_config.get_all_mirror()
        for move_name, is_mirrored in mirror_settings.items():
            if is_mirrored and move_name in self.base_moves:
                moves.append(f"{move_name}_mirrored")

        random.shuffle(moves)
        self.move_names = moves
        self.move_idx = 0
        print(f"[MoveChoreographer] Rebuilt move list: {len(moves)} moves ({len(self.base_moves)} base + {len(moves) - len(self.base_moves)} mirrored)")

    def current_move_name(self) -> str:
        return self.move_names[self.move_idx]

    def current_waveform(self) -> str:
        return self.waveforms[self.waveform_idx]

    def advance_move(self):
        """Advance to next move."""
        self.move_idx = (self.move_idx + 1) % len(self.move_names)
        if self.move_idx == 0:
            random.shuffle(self.move_names)
        self.beat_counter_for_cycle = 0.0


class LiveGroove(DanceMode):
    """Live Groove: Real-time BPM-driven dancing with pre-recorded moves."""

    MODE_ID = "live_groove"
    MODE_NAME = "Live Groove"

    # Default profile location
    DEFAULT_PROFILE_PATH = Path(__file__).parent.parent / "environment_profile.npz"

    def __init__(
        self,
        safety_mixer: SafetyMixer,
        profile_path: Optional[str] = None,
        skip_calibration: bool = False,
        force_calibration: bool = False,
    ):
        super().__init__(safety_mixer)

        self.config = LiveGrooveConfig()
        self.music_state = MusicState()
        self.choreographer = MoveChoreographer()

        # Profile settings
        self.profile_path = profile_path
        self.skip_calibration = skip_calibration
        self.force_calibration = force_calibration  # If True, ignore default profile

        # Audio state
        self.pa: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.audio_device_index: Optional[int] = None

        # Noise profiles (3-phase)
        self.silence_noise_profile: Optional[np.ndarray] = None
        self.breathing_noise_profile: Optional[np.ndarray] = None
        self.dance_noise_profile: Optional[np.ndarray] = None

        # Threading
        self.stop_event = threading.Event()
        self.audio_thread: Optional[threading.Thread] = None
        self.control_thread: Optional[threading.Thread] = None

        # Status
        self._status = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "bpm": 0.0,
            "move": "",
            "calibrated": False,
            "music_confident": False,
            "volume_threshold": self.config.volume_gate_threshold,
        }

        # Load settings from mode_settings
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("live_groove")
        # Live Groove's intensity is applied via SafetyMixer
        if "intensity" in settings:
            self.mixer.update_config(intensity=settings["intensity"])
        # Load volume threshold if specified
        if "volume_gate_threshold" in settings:
            self.config.volume_gate_threshold = settings["volume_gate_threshold"]
            self._status["volume_threshold"] = settings["volume_gate_threshold"]

    def apply_settings(self, updates: dict[str, float]) -> None:
        """Apply settings updates (called from API for live tuning)."""
        if "intensity" in updates:
            self.mixer.update_config(intensity=updates["intensity"])
        if "volume_gate_threshold" in updates:
            self.config.volume_gate_threshold = updates["volume_gate_threshold"]
            self._status["volume_threshold"] = updates["volume_gate_threshold"]
            print(f"[{self.MODE_NAME}] Volume threshold updated to {updates['volume_gate_threshold']:.4f}")

    def refresh_moves(self) -> None:
        """Rebuild the move list with current mirror settings."""
        self.choreographer.rebuild_move_list()

    async def start(self) -> None:
        """Start Live Groove mode."""
        if self.running:
            return

        print(f"[{self.MODE_NAME}] Starting...")

        # Find audio device
        self.audio_device_index = self._find_audio_device("Reachy Mini Audio")
        if self.audio_device_index is None:
            print(f"[{self.MODE_NAME}] Warning: Reachy Mini Audio not found, using default")

        # Try to load saved profile, otherwise run live calibration
        profile_loaded = False

        if self.profile_path:
            # Explicit profile path provided
            loaded = load_environment_profile(self.profile_path)
            if loaded:
                self.silence_noise_profile, self.breathing_noise_profile, self.dance_noise_profile = loaded
                profile_loaded = True
                print(f"[{self.MODE_NAME}] Using saved environment profile!")
        elif self.DEFAULT_PROFILE_PATH.exists() and not self.skip_calibration and not self.force_calibration:
            # Try default profile (unless force_calibration is True)
            loaded = load_environment_profile(self.DEFAULT_PROFILE_PATH)
            if loaded:
                self.silence_noise_profile, self.breathing_noise_profile, self.dance_noise_profile = loaded
                profile_loaded = True
                print(f"[{self.MODE_NAME}] Using default environment profile!")

        if not profile_loaded and not self.skip_calibration:
            # Run calibration (this takes ~26 seconds)
            if self.force_calibration:
                print(f"[{self.MODE_NAME}] Force calibration requested...")
            print(f"[{self.MODE_NAME}] Running 3-phase noise calibration...")
            self._status["state"] = "calibrating"
            await self._run_calibration()
        elif self.skip_calibration:
            print(f"[{self.MODE_NAME}] WARNING: Skipping calibration - may dance to motor noise!")

        self._status["calibrated"] = True

        # Start threads
        self.stop_event.clear()
        self.running = True

        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()

        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        self._status["running"] = True
        self._status["state"] = "listening"
        print(f"[{self.MODE_NAME}] Started - play music!")

    async def stop(self) -> None:
        """Stop Live Groove mode."""
        if not self.running:
            return

        print(f"[{self.MODE_NAME}] Stopping...")
        self.running = False
        self.stop_event.set()

        # Abort audio stream to unblock the blocking read() call
        if self.audio_stream:
            try:
                self.audio_stream.abort()  # Immediate stop, unblocks read()
            except Exception:
                pass

        # Wait for threads - audio thread handles final cleanup in its finally block
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
            if self.audio_thread.is_alive():
                print(f"[{self.MODE_NAME}] Warning: audio thread didn't stop cleanly")
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        # Return to neutral
        self.mixer.reset()

        self._status["running"] = False
        self._status["state"] = "idle"
        print(f"[{self.MODE_NAME}] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        with self.music_state.lock:
            self._status["bpm"] = self.music_state.librosa_bpm
            self._status["state"] = self.music_state.state
            self._status["music_confident"] = self.music_state.music_confident
            self._status["has_ever_locked"] = self.music_state.has_ever_locked
            self._status["raw_amplitude"] = self.music_state.raw_amplitude
        self._status["move"] = self.choreographer.current_move_name()
        self._status["volume_threshold"] = self.config.volume_gate_threshold

        # Convert any numpy types to Python native types for JSON serialization
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, 'item'):  # numpy scalar
                status[key] = value.item()
        return status

    def _find_audio_device(self, name_pattern: str) -> Optional[int]:
        """Find audio input device by name."""
        pa = pyaudio.PyAudio()
        device_index = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and name_pattern in info["name"]:
                device_index = i
                print(f"[{self.MODE_NAME}] Found audio device: [{i}] {info['name']}")
                break
        pa.terminate()
        return device_index

    async def _run_calibration(self) -> None:
        """Run 3-phase noise calibration sequence with threaded audio capture."""
        print(f"\n{'='*60}")
        print("STARTING 3-PHASE NOISE CALIBRATION")
        print("Please ensure NO MUSIC is playing during calibration!")
        print(f"{'='*60}")

        # Phase 1: Background silence (robot still)
        print(f"\n[{self.MODE_NAME}] PHASE 1: Background Silence ({self.config.silence_calibration_duration}s)")
        print("   Robot is still. Please ensure room is quiet.")
        self.silence_noise_profile = await self._calibrate_silence()

        # Phase 2: Breathing motion noise
        print(f"\n[{self.MODE_NAME}] PHASE 2: Breathing Motor Noise ({self.config.noise_calibration_duration}s)")
        print("   Robot will do smooth breathing motion.")
        self.breathing_noise_profile = await self._calibrate_with_motion(
            duration=self.config.noise_calibration_duration,
            movement_type="breathing"
        )

        # Phase 3: Dance motion noise
        dance_amplitude = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 0.3)
        print(f"\n[{self.MODE_NAME}] PHASE 3: Dance Motor Noise ({self.config.dance_noise_calibration_duration}s)")
        print(f"   Robot will do headbanger combo at {dance_amplitude:.0%} amplitude.")
        self.dance_noise_profile = await self._calibrate_with_motion(
            duration=self.config.dance_noise_calibration_duration,
            movement_type="dancing"
        )

        print(f"\n{'='*60}")
        print("All calibration phases complete!")
        print(f"{'='*60}\n")

    def _audio_capture_thread(
        self,
        duration: float,
        collected: list,
        stop_event: threading.Event,
    ) -> None:
        """Background thread to capture audio without blocking motion."""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.config.audio_rate,
            input=True,
            input_device_index=self.audio_device_index,
            frames_per_buffer=self.config.audio_chunk_size,
        )

        samples_needed = int(self.config.audio_rate * duration)

        while len(collected) * self.config.audio_chunk_size < samples_needed and not stop_event.is_set():
            try:
                stereo = np.frombuffer(
                    stream.read(self.config.audio_chunk_size, exception_on_overflow=False),
                    dtype=np.float32,
                )
                chunk = (stereo[0::2] + stereo[1::2]) / 2.0
                collected.append(chunk)
            except (IOError, ValueError):
                continue

        stream.stop_stream()
        stream.close()
        pa.terminate()

    async def _calibrate_silence(self) -> np.ndarray:
        """Record background silence with robot completely still."""
        duration = self.config.silence_calibration_duration

        # Move robot to neutral and hold still
        intent = MovementIntent(
            position=self.config.neutral_pos.copy(),
            orientation=self.config.neutral_eul.copy(),
            antennas=np.zeros(2),
        )
        self.mixer.send_intent(intent)
        await asyncio.sleep(0.5)  # Let robot settle

        # Start audio capture in background thread
        collected: list = []
        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            args=(duration, collected, stop_event),
            daemon=True,
        )
        audio_thread.start()

        # Wait and show progress (robot stays still)
        start_time = time.time()
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            print(f"\r   Recording silence: {remaining:.1f}s remaining...", end="", flush=True)
            await asyncio.sleep(0.1)

        # Wait for audio thread to finish
        stop_event.set()
        audio_thread.join(timeout=2.0)

        if not collected:
            print("\r   WARNING: No audio collected during silence calibration!")
            return np.zeros(1025)  # Default profile shape for n_fft=2048

        audio = np.concatenate(collected)
        profile, rms, _ = compute_noise_profile(audio, method="median", exclude_transients=False)
        print(f"\r   Phase 1 complete. Silence RMS: {rms:.6f}                    ")
        return profile

    async def _calibrate_with_motion(self, duration: float, movement_type: str) -> np.ndarray:
        """Record noise profile while robot moves. Audio runs in separate thread."""
        # Start audio capture in background thread
        collected: list = []
        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            args=(duration, collected, stop_event),
            daemon=True,
        )
        audio_thread.start()

        # Run motion in main thread (smooth updates)
        motion_time = 0.0
        start_time = time.time()
        last_loop_time = start_time

        while time.time() - start_time < duration:
            loop_start = time.time()
            dt = loop_start - last_loop_time
            last_loop_time = loop_start

            remaining = duration - (loop_start - start_time)
            phase_name = "breathing" if movement_type == "breathing" else "dance"
            print(f"\r   Recording {phase_name} noise: {remaining:.1f}s remaining...", end="", flush=True)

            motion_time += dt

            if movement_type == "breathing":
                intent = self._compute_breathing_pose(motion_time)
            else:
                # Use headbanger with SAME amplitude as actual dancing (0.3)
                intent = self._compute_calibration_dance_pose(motion_time)

            self.mixer.send_intent(intent)

            # Maintain smooth control loop
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0.001, self.config.control_ts - elapsed_loop)
            await asyncio.sleep(sleep_time)

        # Stop audio capture and wait
        stop_event.set()
        audio_thread.join(timeout=2.0)

        # Return to neutral
        self.mixer.reset()

        if not collected:
            print(f"\r   WARNING: No audio collected during {movement_type} calibration!")
            return self.silence_noise_profile if self.silence_noise_profile is not None else np.zeros(1025)

        audio = np.concatenate(collected)

        # Use median, no transient exclusion (capture all motor noise)
        profile, rms, _ = compute_noise_profile(audio, method="median", exclude_transients=False)

        # Combine with silence profile: take the maximum of both
        if self.silence_noise_profile is not None:
            noise_profile = np.maximum(profile, self.silence_noise_profile)
        else:
            noise_profile = profile

        phase_num = 2 if movement_type == "breathing" else 3
        print(f"\r   Phase {phase_num} complete. {movement_type.capitalize()} RMS: {rms:.6f}                    ")
        return noise_profile

    def _compute_calibration_dance_pose(self, t_beats: float) -> MovementIntent:
        """Compute dance pose for calibration using reduced amplitude."""
        move_fn, base_params, _ = AVAILABLE_MOVES["headbanger_combo"]
        params = base_params.copy()

        # Use SAME amplitude as actual dancing to avoid collisions
        amp_scale = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 0.3)

        # Run at 120 BPM
        t = t_beats * (120.0 / 60.0)
        offsets = move_fn(t, **params)

        return MovementIntent(
            position=self.config.neutral_pos + offsets.position_offset * amp_scale,
            orientation=self.config.neutral_eul + offsets.orientation_offset * amp_scale,
            antennas=offsets.antennas_offset * amp_scale,
        )

    def _compute_breathing_pose(self, t: float) -> MovementIntent:
        """Compute breathing/idle pose."""
        # Y sway
        y_amplitude = 0.016
        y_freq = 0.2
        y_offset = y_amplitude * np.sin(2.0 * np.pi * y_freq * t)

        # Head roll
        roll_amplitude = 0.222
        roll_freq = 0.15
        roll_offset = roll_amplitude * np.sin(2.0 * np.pi * roll_freq * t)

        return MovementIntent(
            position=self.config.neutral_pos + np.array([0.0, y_offset, 0.0]),
            orientation=self.config.neutral_eul + np.array([roll_offset, 0.0, 0.0]),
            antennas=np.array([-0.15, 0.15]),
        )

    def _compute_dance_pose(self, t_beats: float, move_name: str, bpm: float) -> MovementIntent:
        """Compute dance pose from library move."""
        # Check if this is a mirrored version
        is_mirrored = move_name.endswith("_mirrored")
        base_move_name = move_name[:-9] if is_mirrored else move_name  # Strip "_mirrored"

        move_fn, base_params, _ = AVAILABLE_MOVES[base_move_name]
        params = base_params.copy()

        if "waveform" in params:
            params["waveform"] = self.choreographer.current_waveform()

        offsets = move_fn(t_beats, **params)
        amp_scale = move_config.get_dampening(base_move_name)

        # Apply amplitude scaling
        pos_offset = offsets.position_offset * amp_scale
        ori_offset = offsets.orientation_offset * amp_scale
        ant_offset = offsets.antennas_offset * amp_scale

        # Apply Y-axis mirroring for mirrored moves
        if is_mirrored:
            pos_offset = pos_offset.copy()  # Don't modify original
            ori_offset = ori_offset.copy()
            pos_offset[1] = -pos_offset[1]  # Mirror Y position
            ori_offset[2] = -ori_offset[2]  # Mirror yaw (rotation around Z)

        return MovementIntent(
            position=self.config.neutral_pos + pos_offset,
            orientation=self.config.neutral_eul + ori_offset,
            antennas=ant_offset,
        )

    def _subtract_noise(self, audio: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
        """Subtract noise profile from audio."""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        cleaned_magnitude = magnitude - (noise_profile[:, np.newaxis] * self.config.noise_subtraction_strength)
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=512, length=len(audio))

        return cleaned_audio.astype(np.float32)

    def _clamp_bpm(self, bpm: float) -> float:
        """Force BPM into range by halving/doubling."""
        if bpm <= 0:
            return bpm
        while bpm < self.config.bpm_min:
            bpm *= 2.0
        while bpm > self.config.bpm_max:
            bpm /= 2.0
        return bpm

    def _audio_loop(self) -> None:
        """Audio analysis thread."""
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.config.audio_rate,
            input=True,
            input_device_index=self.audio_device_index,
            frames_per_buffer=self.config.audio_chunk_size,
        )

        buf = np.empty(0, dtype=np.float32)
        bpm_hist: collections.deque = collections.deque(maxlen=self.config.bpm_stability_buffer)

        try:
            while not self.stop_event.is_set():
                try:
                    stereo = np.frombuffer(
                        self.audio_stream.read(self.config.audio_chunk_size, exception_on_overflow=False),
                        dtype=np.float32,
                    )
                    chunk = (stereo[0::2] + stereo[1::2]) / 2.0
                    buf = np.append(buf, chunk)
                except (IOError, ValueError, OSError):
                    # Stream was closed or error - check if we should stop
                    if self.stop_event.is_set():
                        break
                    continue

                if len(buf) < self.config.audio_buffer_len:
                    continue

                # Select noise profile based on current state
                with self.music_state.lock:
                    is_breathing = self.music_state.is_breathing

                if is_breathing and self.breathing_noise_profile is not None:
                    analysis_buf = self._subtract_noise(buf, self.breathing_noise_profile)
                elif not is_breathing and self.dance_noise_profile is not None:
                    analysis_buf = self._subtract_noise(buf, self.dance_noise_profile)
                else:
                    analysis_buf = buf

                # Volume gate on cleaned audio
                rms = np.sqrt(np.mean(buf**2))
                cleaned_rms = np.sqrt(np.mean(analysis_buf**2))

                if cleaned_rms < self.config.volume_gate_threshold:
                    with self.music_state.lock:
                        self.music_state.state = "Gathering"
                        self.music_state.librosa_bpm = 0.0
                        self.music_state.raw_amplitude = rms
                        self.music_state.last_event_time = 0.0
                        self.music_state.music_confident = False
                    buf = buf[-self.config.audio_buffer_len:]
                    continue

                # Check if signal is confidently music (well above threshold)
                confidence_threshold = self.config.volume_gate_threshold * self.config.music_confidence_ratio
                is_confident = cleaned_rms > confidence_threshold

                # BPM detection
                tempo, beat_frames = librosa.beat.beat_track(
                    y=analysis_buf, sr=self.config.audio_rate, units="frames", tightness=80
                )
                now = time.time()
                raw_tempo = float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo)
                tempo_val = self._clamp_bpm(raw_tempo)

                has_audio = cleaned_rms > self.config.volume_gate_threshold and len(beat_frames) > 0

                with self.music_state.lock:
                    if has_audio:
                        self.music_state.last_event_time = now
                    self.music_state.raw_librosa_bpm = raw_tempo
                    self.music_state.raw_amplitude = rms
                    self.music_state.cleaned_amplitude = np.abs(analysis_buf).mean()
                    self.music_state.music_confident = is_confident

                    if tempo_val > 40:
                        bpm_hist.append(tempo_val)
                        self.music_state.librosa_bpm = float(np.mean(bpm_hist))

                    self.music_state.bpm_std = float(np.std(bpm_hist)) if len(bpm_hist) > 1 else 0.0

                    if len(bpm_hist) < self.config.bpm_stability_buffer:
                        self.music_state.state = "Gathering"
                        self.music_state.unstable_period_count = 0
                    elif np.std(bpm_hist) < self.config.bpm_stability_threshold:
                        self.music_state.state = "Locked"
                        self.music_state.unstable_period_count = 0
                        self.music_state.has_ever_locked = True
                    else:
                        self.music_state.state = "Unstable"
                        self.music_state.unstable_period_count += 1

                    buf = buf[-self.config.audio_buffer_len:]

        finally:
            # Clean up audio resources
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None
            if self.pa:
                try:
                    self.pa.terminate()
                except Exception:
                    pass
                self.pa = None

    def _control_loop(self) -> None:
        """Movement control thread."""
        last_time = time.time()
        t_beats = 0.0
        breathing_time = 0.0
        is_executing_move = False
        move_beats_elapsed = 0.0
        force_breathing_until = 0.0
        last_active_bpm = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            with self.music_state.lock:
                librosa_bpm = self.music_state.librosa_bpm
                state = self.music_state.state
                last_event_time = self.music_state.last_event_time
                unstable_count = self.music_state.unstable_period_count
                has_ever_locked = self.music_state.has_ever_locked
                music_confident = self.music_state.music_confident

            active_bpm = librosa_bpm if now - last_event_time < self.config.silence_tmo else 0.0

            # Separate criteria for STARTING vs CONTINUING a move:
            # START: Must be Locked AND music_confident (strict) - prevents ego-noise dancing
            # CONTINUE: Can coast through Unstable using last_active_bpm (loose)
            can_start_new_move = active_bpm > 0 and has_ever_locked and state == "Locked" and music_confident
            can_continue_move = has_ever_locked and (
                state == "Locked" or
                (state == "Unstable" and unstable_count < self.config.unstable_periods_before_stop)
            )

            # Update breathing state for audio thread
            with self.music_state.lock:
                self.music_state.is_breathing = not (is_executing_move or can_start_new_move)

            in_forced_breathing = now < force_breathing_until

            if is_executing_move:
                # Continue executing move
                bpm_for_move = active_bpm if active_bpm > 0 else last_active_bpm
                beats_this_frame = dt * (bpm_for_move / 60.0)
                move_beats_elapsed += beats_this_frame
                t_beats += beats_this_frame

                if move_beats_elapsed >= self.config.beats_per_sequence:
                    # Move complete
                    is_executing_move = False
                    force_breathing_until = now + self.config.min_breathing_between_moves
                    self.choreographer.advance_move()
                else:
                    # Execute move frame
                    intent = self._compute_dance_pose(
                        t_beats,
                        self.choreographer.current_move_name(),
                        bpm_for_move
                    )
                    self.mixer.send_intent(intent)

            elif can_start_new_move and not in_forced_breathing:
                # Start new move
                is_executing_move = True
                t_beats = 0.0
                move_beats_elapsed = 0.0
                last_active_bpm = active_bpm
                print(f"[{self.MODE_NAME}] Starting: {self.choreographer.current_move_name()}")

            else:
                # Breathing
                breathing_time += dt
                intent = self._compute_breathing_pose(breathing_time)
                self.mixer.send_intent(intent)

            time.sleep(0.01)
