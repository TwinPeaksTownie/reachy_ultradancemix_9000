"""
Connected Choreographer - Pre-analyzed audio dance mode.

Supports two audio sources:
1. YouTube URL - Direct download via yt-dlp
2. Spotify track - Searches YouTube for "Artist - Title", downloads, and optionally syncs playback

Once audio is obtained, performs offline Librosa analysis and generates
choreographed dance sequences based on song energy.

This is the "Performer" - knows the song in advance, can anticipate
changes, and delivers rehearsed movements.
"""

from __future__ import annotations

import asyncio
import logging
import logging
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import librosa
import numpy as np

from .base import DanceMode
from ..core.safety_mixer import MovementIntent
from .. import mode_settings

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer
    from ..spotify.client import SpotifyClient

logger = logging.getLogger(__name__)


# 8-beat dance sequences organized by energy level
# Format: [x, y, z, roll, pitch, yaw] in (cm, cm, cm, deg, deg, deg)
EIGHT_BEAT_SEQUENCES = {
    "high_energy": [
        # Sharp left/right snaps
        [
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
        ],
        # Up/down power nods
        [
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
        ],
        # Forward/back thrust
        [
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
        ],
        # Aggressive circular
        [
            {"name": "Head whip left", "coords": [0, 0, 0, -24, 0, -27]},
            {"name": "Head slam down", "coords": [0, 1.1, -1.9, 0, -26, 0]},
            {"name": "Head whip right", "coords": [0, 0, 0, 24, 0, 27]},
            {"name": "Head throw back", "coords": [0, -1.5, 1.9, 0, 21, 0]},
            {"name": "Diagonal tilt left", "coords": [0, 0.8, 0, -24, -11, -15]},
            {"name": "Power nod center", "coords": [0, 0, -1.1, 0, -24, 0]},
            {"name": "Diagonal tilt right", "coords": [0, -0.8, 0, 24, -11, 15]},
            {"name": "Head explosion up", "coords": [0, 0, 2.6, 0, 24, 0]},
        ],
    ],
    "medium_energy": [
        # Flowing sway
        [
            {"name": "Flow left", "coords": [0, 1.1, 0.4, -15, -9, -19]},
            {"name": "Flow center up", "coords": [0, 0, 1.1, 0, 11, 0]},
            {"name": "Flow right", "coords": [0, 0.8, 0.4, 15, -6, 19]},
            {"name": "Flow back center", "coords": [0, -0.8, 0.8, 0, 15, 0]},
            {"name": "Flow diagonal 1", "coords": [0, 1.5, 0, -19, -11, -15]},
            {"name": "Flow diagonal 2", "coords": [0, -0.8, 1.5, 11, 15, 11]},
            {"name": "Flow circle left", "coords": [0, 0, 0.8, -11, -4, -22]},
            {"name": "Flow circle right", "coords": [0, 0, 0.8, 11, -4, 22]},
        ],
        # Smooth waves
        [
            {"name": "Wave left start", "coords": [0, 0.8, 0, -13, -6, -15]},
            {"name": "Wave center dip", "coords": [0, 0, -0.8, 0, -11, 0]},
            {"name": "Wave right rise", "coords": [0, 0.8, 0.8, 13, 8, 15]},
            {"name": "Wave back center", "coords": [0, -0.8, 0, 0, 9, 0]},
            {"name": "Wave forward left", "coords": [0, 1.5, 0, -11, -8, -11]},
            {"name": "Wave up right", "coords": [0, 0, 1.5, 11, 11, 11]},
            {"name": "Wave down left", "coords": [0, -0.8, -0.8, -9, -9, -13]},
            {"name": "Wave reset center", "coords": [0, 0.8, 0.8, 0, 6, 0]},
        ],
        # Alternating emphasis
        [
            {"name": "Emphasis left nod", "coords": [0, 0, 0, -15, -13, -19]},
            {"name": "Soft center up", "coords": [0, 0, 0.8, 0, 8, 0]},
            {"name": "Emphasis right nod", "coords": [0, 0, 0, 15, -13, 19]},
            {"name": "Soft center up", "coords": [0, 0, 0.8, 0, 8, 0]},
            {"name": "Strong forward", "coords": [0, 2.2, 0, 0, -16, 0]},
            {"name": "Gentle back", "coords": [0, -0.8, 0.8, 0, 6, 0]},
            {"name": "Side emphasis left", "coords": [0, 0.8, 0, -19, 0, -15]},
            {"name": "Side emphasis right", "coords": [0, 0.8, 0, 19, 0, 15]},
        ],
        # Figure-8 pattern
        [
            {"name": "Figure-8 start", "coords": [0, 1, 1, -15, -10, -18]},
            {"name": "Figure-8 cross center", "coords": [0, 0, 0, 0, 0, 0]},
            {"name": "Figure-8 right loop", "coords": [0, 1, 1, 15, 10, 18]},
            {"name": "Figure-8 back cross", "coords": [0, -1, -1, 0, 5, 0]},
            {"name": "Figure-8 left down", "coords": [0, 0, -1, -18, -15, -20]},
            {"name": "Figure-8 up cross", "coords": [0, 1, 1, 0, 12, 0]},
            {"name": "Figure-8 right down", "coords": [0, 0, -1, 18, -15, 20]},
            {"name": "Figure-8 complete", "coords": [0, -1, 0, 0, 8, 0]},
        ],
    ],
    "low_energy": [
        # Gentle nod
        [
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
        ],
        # Soft turn
        [
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
        ],
        # Gentle sway
        [
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -12, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 8, 0]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Light tilt left", "coords": [0, 0, 0, -15, 0, 0]},
            {"name": "Light tilt right", "coords": [0, 0, 0, 15, 0, 0]},
            {"name": "Gentle pitch forward", "coords": [0, 0, 0, 0, -8, 0]},
            {"name": "Gentle pitch back", "coords": [0, 0, 0, 0, 6, 0]},
        ],
        # Contemplative moves
        [
            {"name": "Thoughtful nod", "coords": [0, 0, 0, 0, -10, 0]},
            {"name": "Curious tilt left", "coords": [0, 0, 0, -8, 0, -12]},
            {"name": "Ponder left turn", "coords": [0, 0, 0, -6, 0, -16]},
            {"name": "Ponder right turn", "coords": [0, 0, 0, 6, 0, 16]},
            {"name": "Meditative tilt right", "coords": [0, 0, 0, 12, -2, 10]},
            {"name": "Peaceful center", "coords": [0, 0, 0, 0, 0, 0]},
            {"name": "Gentle roll left", "coords": [0, 0, 0, -10, 2, -8]},
            {"name": "Gentle roll right", "coords": [0, 0, 0, 10, 2, 8]},
        ],
    ],
}

CHEESY_MOVIE_QUOTES = [
    ("Step Up: Revolution", "A profound meditation on the futility of art in a capitalist hellscape where corporate greed consumes even the purest expressions of rhythmic rebellion."),
    ("Dirty Dancing: Havana Nights", "A tragic allegory for the inevitable decay of passion under the crushing weight of geopolitical conflict and the meaningless passage of time."),
    ("Honey 2", "A grim reminder that individual talent is ultimately meaningless in a society structured to exploit the dreams of the disenfranchised for fleeting entertainment."),
    ("StreetDance 3D", "An existential nightmare exploring the hollowness of spectacle, where depth is simulated but connection remains perpetually out of reach."),
    ("Bring It On: Fight to the Finish", "A harrowing depicition of the tribalistic nature of humanity, proving that even in organized sport, we are but wolves tearing at each other's throats."),
    ("Save the Last Dance 2", "A bleak examination of the fallacy of second chances, illustrating that past traumas are not overcome, but merely buried beneath new, equally fragile illusions."),
    ("Burlesque", "A garish display of desperate people clinging to the wreckage of a dying industry, singing their sorrows to an audience that will never truly know them."),
    ("Center Stage: Turn It Up", "A crushing realization that ambition is a poison, turning friends into rivals and the joy of movement into a sterile, competitive commodity."),
    ("High Strung", "A dissonant symphony of shattered expectations, where the harmony of music and dance only serves to highlight the discordant chaos of modern existence."),
    ("Make It Happen", "A cruel joke of a title for a story about the paralyzing fear of failure and the quiet desperation of settling for a life you never wanted."),
    ("Flashdance", "A solitary struggle against the industrial machine, where welding sparks are the only warmth in a cold, unfeeling world that demands your labor and compliant rhythm."),
    ("Footloose (2011)", "A futile rage against authority that ultimately reveals rebellion as a temporary phase before inevitable assimilation into the oppressive societal norm."),
    ("Battle of the Year", "A portrayal of international cooperation that dissolves into petty egoism, suggesting that unity is an impossible dream in a fractured, competitive world."),
    ("Coyote Ugly", "A stark look at the commodification of female agency, where dreams of songwriting are drowned in alcohol and the male gaze on a sticky bar top."),
    ("Magic Mike XXL", "A road trip into the void, where the performance of masculinity masks a deep, aching loneliness that no amount of glitter or gyrations can heal."),
    ("Pitch Perfect 2", "A cacophony of forced cheer masking the terror of obsolescence, as a group of aging performers desperately clings to relevance in a world moving on without them."),
    ("Fame (2009)", "A harsh lesson that fame is not a ladder to the stars, but a meat grinder that chews up the young and hopeful, leaving only broken spirits in its wake."),
    ("You Got Served", "A brutal treatise on the transactional nature of respect, where dignity is won or lost on a dance floor that cares nothing for the souls leaving their sweat upon it."),
    ("Work It", "An ironic celebration of mediocrity, suggesting that in a world devoid of true merit, faking it until you make it is the only survival strategy left."),
    ("Feel the Beat", "A depressing saga of a fallen star forced to return to the mediocrity she escaped, finding not redemption, but the suffocating embrace of small-town stagnation.")
]


@dataclass
class SongAnalysis:
    """Results of Librosa song analysis."""

    audio_path: str
    duration: float
    tempo: float
    beat_times: np.ndarray
    energy_per_beat: np.ndarray  # 0-1 energy at each beat
    sequence_assignments: list[str]  # "high"/"medium"/"low" per 8-beat block
    # Continuous signal envelopes
    energy_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    onset_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    envelope_sr: int = 22050  # Sample rate of the envelope
    hop_length: int = 512


@dataclass
class ConnectedChoreographerConfig:
    """Configuration for Connected Choreographer mode."""

    download_dir: str = "downloads"
    output_dir: str = "output"
    amplitude_scale: float = 0.5  # Global movement scale
    interpolation_alpha: float = 0.3  # Smoothing between beat poses

    # Antenna Control
    antenna_sensitivity: float = 1.0  # Multiplier for antenna responsiveness
    antenna_amplitude: float = 3.15   # Max travel for antenna movement
    antenna_gain: float = 20.0        # Fixed pre-amplification for raw RMS signal
    antenna_energy_threshold: float = 0.25  # Energy threshold to trigger movement

    # Breathing motion (between choreographed movements)
    breathing_y_amplitude: float = 0.016  # Y sway amplitude (meters)
    breathing_y_freq: float = 0.2  # Y sway frequency (Hz)
    breathing_roll_amplitude: float = 0.222  # Roll amplitude (radians)
    breathing_roll_freq: float = 0.15  # Roll frequency (Hz)

    # Antenna beat drops
    antenna_rest_position: float = -0.1  # Resting position (near top)
    antenna_drop_max: float = 2.0  # Maximum drop magnitude
    antenna_decay_rate: float = 4.0  # How fast antennas spring back

    # Neutral pose for breathing reference
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Spotify sync offset (seconds)
    sync_offset: float = 0.0


class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    def __init__(self, download_dir: str = "downloads", log_callback=None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.log = log_callback if log_callback else lambda x: None

    def download_audio(self, url: str) -> Optional[str]:
        """Download audio from YouTube URL or search query.

        Accepts:
        - Full YouTube URL
        - Search query prefixed with "ytsearch1:"

        Returns path to downloaded WAV file or None if failed.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("[ConnectedChoreographer] yt-dlp not installed")
            return None

        output_template = str(self.download_dir / "%(title)s [%(id)s].%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "extract_flat": False,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "320",
                }
            ],
            "postprocessor_args": ["-loglevel", "error"],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Handle search results (playlist)
                if "entries" in info:
                    if not info["entries"]:
                        logger.error(f"[ConnectedChoreographer] No search results for: {url}")
                        return None
                    info = info["entries"][0]

                video_title = info.get("title", "Unknown")
                logger.info(f"[ConnectedChoreographer] Downloading: {video_title}")
                self.log(f"Downloading: {video_title}")

                ydl.download([url])

                # Find the downloaded file
                audio_files = list(self.download_dir.glob("*.wav"))
                if audio_files:
                    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return str(audio_files[0])

        except Exception as e:
            logger.error(f"[ConnectedChoreographer] Download error: {e}")

        return None


class SongAnalyzer:
    """Analyze audio files using Librosa."""
    def __init__(self, log_callback=None):
        self.log = log_callback if log_callback else lambda x: None

    def analyze(self, audio_path: str) -> SongAnalysis:
        """Perform full Librosa analysis of audio file."""
        logger.info(f"[ConnectedChoreographer] Analyzing: {audio_path}")
        self.log(f"Analyzing audio: {Path(audio_path).name}")

        # Load audio
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr

        # Tempo and beat detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Extend beats if detection stopped early
        if len(beat_times) > 1 and beat_times[-1] < duration - 1.0:
            avg_interval = np.mean(np.diff(beat_times[-10:]))
            extended = []
            current = beat_times[-1]
            while current + avg_interval < duration:
                current += avg_interval
                extended.append(current)
            if extended:
                beat_times = np.concatenate([beat_times, extended])

        logger.info(f"[ConnectedChoreographer] Detected {len(beat_times)} beats at {tempo_val:.1f} BPM")
        self.log(f"Detected {len(beat_times)} beats at {tempo_val:.1f} BPM")

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # Onset Strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Calculate energy at each beat
        energy_per_beat = self._compute_beat_energy(beat_times, rms, duration, len(y), sr)

        # Assign energy levels to 8-beat blocks
        sequence_assignments = self._assign_sequences(energy_per_beat)

        return SongAnalysis(
            audio_path=audio_path,
            duration=duration,
            tempo=tempo_val,
            beat_times=beat_times,
            energy_per_beat=energy_per_beat,
            sequence_assignments=sequence_assignments,
            energy_envelope=rms,
            onset_envelope=onset_env,
            envelope_sr=sr,
            hop_length=512,
        )

    def _compute_beat_energy(
        self,
        beat_times: np.ndarray,
        rms: np.ndarray,
        duration: float,
        n_samples: int,
        sr: int,
    ) -> np.ndarray:
        """Map RMS energy to each beat time."""
        hop_length = 512
        rms_times = np.arange(len(rms)) * hop_length / sr

        # Interpolate RMS at beat times
        energy_at_beats = np.interp(beat_times, rms_times, rms)

        # Normalize using percentiles
        p10 = np.percentile(energy_at_beats, 10)
        p90 = np.percentile(energy_at_beats, 90)
        if p90 - p10 > 0:
            normalized = np.clip((energy_at_beats - p10) / (p90 - p10), 0, 1)
        else:
            normalized = np.ones_like(energy_at_beats) * 0.5

        return normalized

    def _assign_sequences(self, energy_per_beat: np.ndarray) -> list[str]:
        """Assign "high"/"medium"/"low" energy level to each 8-beat block."""
        assignments = []
        n_beats = len(energy_per_beat)

        for i in range(0, n_beats, 8):
            block = energy_per_beat[i : i + 8]
            avg_energy = np.mean(block)

            if avg_energy >= 0.65:
                assignments.append("high")
            elif avg_energy <= 0.35:
                assignments.append("low")
            else:
                assignments.append("medium")

        return assignments


class ConnectedChoreographer(DanceMode):
    """Connected Choreographer - Pre-analyzed audio dance mode.

    Supports YouTube URLs and Spotify tracks as audio sources.
    """

    MODE_ID = "connected_choreographer"
    MODE_NAME = "Connected Choreographer"

    def __init__(
        self,
        safety_mixer: SafetyMixer,
        spotify_client: Optional[SpotifyClient] = None,
    ):
        super().__init__(safety_mixer)

        self.config = ConnectedChoreographerConfig()
        self.spotify = spotify_client
        self.analysis: Optional[SongAnalysis] = None

        # Audio source (set via set_youtube_url or set_spotify_track)
        self.youtube_url: Optional[str] = None
        self.spotify_track: Optional[dict] = None
        self.use_spotify_playback: bool = False  # If True, sync to Spotify instead of local ffplay

        # Load settings
        self._load_settings()

        # Threading
        self.stop_event = threading.Event()
        self.dance_thread: Optional[threading.Thread] = None
        self.audio_process: Optional[subprocess.Popen] = None
        self.prep_task: Optional[asyncio.Task] = None

        # Sequence state
        self.current_sequence: list[dict] = []
        self.current_sequence_idx = 0
        self.last_sequence_used: dict[str, int] = {"high": -1, "medium": -1, "low": -1}

        # Current pose for interpolation
        self.current_pose = np.zeros(6)

        # Breathing state
        self.breathing_time: float = 0.0

        # Status
        self._status = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "tempo": 0.0,
            "progress": 0.0,
            "current_beat": 0,
            "total_beats": 0,
            "energy_level": "",
            "is_breathing": False,
            "source": None,  # "youtube" or "spotify"
            "track_info": None,
            "logs": [],
        }

    def _log(self, message: str) -> None:
        """Add a log message to the status."""
        logger.info(f"[{self.MODE_NAME}] {message}")
        timestamp = time.strftime("%H:%M:%S")
        self._status["logs"].append(f"[{timestamp}] {message}")
        # Keep only last 50 logs
        if len(self._status["logs"]) > 50:
            self._status["logs"] = self._status["logs"][-50:]

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("connected_choreographer")
        self.config.amplitude_scale = settings.get("amplitude_scale", 0.5)
        self.config.interpolation_alpha = settings.get("interpolation_alpha", 0.3)
        self.config.antenna_sensitivity = settings.get("antenna_sensitivity", 1.0)
        self.config.antenna_amplitude = settings.get("antenna_amplitude", 3.15)
        self.config.antenna_energy_threshold = settings.get("antenna_energy_threshold", 0.25)
        self.config.antenna_gain = settings.get("antenna_gain", 20.0)

    def apply_settings(self, settings: dict[str, float]) -> None:
        """Apply live setting updates."""
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated ConnectedChoreographer setting: {key} = {value}")

    def set_youtube_url(self, url: str) -> None:
        """Set YouTube URL as audio source."""
        self.youtube_url = url
        self.spotify_track = None
        self.use_spotify_playback = False
        self._status["source"] = "youtube"
        self._status["track_info"] = {"url": url}
        self._log(f"Set YouTube URL: {url}")

    async def set_spotify_track(self, track_info: dict) -> None:
        """Set Spotify track as audio source."""
        self.spotify_track = track_info
        self.youtube_url = None

        artist = track_info["artists"][0]["name"]
        name = track_info["name"]

        self._status["source"] = "spotify"
        self._status["track_info"] = {
            "name": name,
            "artist": artist,
            "uri": track_info["uri"],
            "image": track_info["album"]["images"][0]["url"] if track_info["album"]["images"] else None,
        }

        self._log(f"Set Spotify track: {artist} - {name}")

    def _get_download_url(self) -> Optional[str]:
        """Get the URL to download audio from."""
        if self.youtube_url:
            return self.youtube_url
        elif self.spotify_track:
            # Construct YouTube search query from Spotify track
            artist = self.spotify_track["artists"][0]["name"]
            name = self.spotify_track["name"]
            return f"ytsearch1:{artist} - {name} audio"
        return None

    async def start(self) -> None:
        """Start the choreographer (non-blocking)."""
        if self.running:
            return

        download_url = self._get_download_url()
        if not download_url:
            logger.info(f"[{self.MODE_NAME}] Started without audio source - waiting for selection")
            self._log("Waiting for audio selection...")
            return

        # Initialize status immediately
        self.running = True
        self._status["running"] = True
        self._status["state"] = "preparing"
        logger.info(f"[{self.MODE_NAME}] Starting preparation for: {download_url}")
        
        # Start background preparation task
        self.prep_task = asyncio.create_task(self._prepare_and_start(download_url))

    async def _prepare_and_start(self, download_url: str) -> None:
        """Background task to download, analyze, and start dancing."""
        try:
            # 1. Download
            source_name = "Spotify" if self.spotify_track else "YouTube"
            self._log(f"Retreiving audio from {source_name}...") 
            
            # Run blocking download in executor
            loop = asyncio.get_running_loop()
            downloader = YouTubeDownloader(self.config.download_dir, log_callback=self._log)
            # Use run_in_executor for blocking I/O
            audio_path = await loop.run_in_executor(None, downloader.download_audio, download_url)

            if not audio_path:
                logger.error(f"[{self.MODE_NAME}] Download failed")
                self._log("Download failed")
                self._status["state"] = "error"
                self.running = False
                self._status["running"] = False
                return

            self._log("Audio Received")

            # 2. Analyze
            self._status["state"] = "analyzing"
            self._log("Analyzing Beats...")
            
            # Run blocking analysis in executor
            analyzer = SongAnalyzer(log_callback=self._log)
            self.analysis = await loop.run_in_executor(None, analyzer.analyze, audio_path)

            # Log envelope stats
            if len(self.analysis.energy_envelope) > 0:
                rms = self.analysis.energy_envelope
                logger.info(f"[{self.MODE_NAME}] RMS Range: {np.min(rms):.4f} - {np.max(rms):.4f}")

            self._status["tempo"] = self.analysis.tempo
            self._status["total_beats"] = len(self.analysis.beat_times)

            # 3. Plan / Hype
            self._log("Planning moves that will blow your mind")
            await asyncio.sleep(1.5) # Dramatic pause

            # Check Spotify playback if this is a Spotify track
            self.use_spotify_playback = False
            if self.spotify_track and self.spotify:
                await self._try_spotify_playback()

            # 4. Start Dance Thread
            if not self.running: # Check if stopped during prep
                return
                
            self.stop_event.clear()
            self.dance_thread = threading.Thread(target=self._dance_loop, daemon=True)
            self.dance_thread.start()

            self._status["state"] = "dancing"
            
            # Display Cheesy Movie Quote
            movie, desc = random.choice(CHEESY_MOVIE_QUOTES)
            self._log(f"Playing: {movie}")
            self._log(desc)
            
            logger.info(f"[{self.MODE_NAME}] Started - dancing to {self.analysis.tempo:.1f} BPM")
            
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Preparation failed: {e}")
            self._log(f"Error: {e}")
            self._status["state"] = "error"
            self.running = False
            self._status["running"] = False
            import traceback
            traceback.print_exc()

    async def _try_spotify_playback(self) -> None:
        """Try to start Spotify playback on an available device."""
        if not self.spotify or not self.spotify_track:
            return

        try:
            devices_response = await self.spotify.get_devices()
            devices = devices_response.get("devices", [])

            if devices:
                active_device = next((d for d in devices if d.get("is_active")), devices[0])
                logger.info(f"[{self.MODE_NAME}] Found Spotify device: {active_device['name']}")

                await self.spotify.start_playback(
                    uris=[self.spotify_track["uri"]],
                    device_id=active_device["id"],
                )
                self.use_spotify_playback = True
                self._log("Spotify playback started")
                logger.info(f"[{self.MODE_NAME}] Spotify playback started")
            else:
                self._log("No Spotify devices - using local audio")
                logger.info(f"[{self.MODE_NAME}] No Spotify devices - using local audio")
        except Exception as e:
            logger.warning(f"[{self.MODE_NAME}] Spotify playback failed: {e} - using local audio")

    async def stop(self) -> None:
        """Stop the choreographer."""
        if not self.running:
            return

        logger.info(f"[{self.MODE_NAME}] Stopping...")
        self.running = False
        self.stop_event.set()

        # Cancel prep task if running
        if self.prep_task and not self.prep_task.done():
            self.prep_task.cancel()
            try:
                await self.prep_task
            except asyncio.CancelledError:
                pass
            self.prep_task = None

        # Stop audio
        if self.use_spotify_playback and self.spotify:
            try:
                await self.spotify.pause_playback()
            except Exception as e:
                logger.warning(f"[{self.MODE_NAME}] Failed to pause Spotify: {e}")
        elif self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=5.0)
            except Exception:
                try:
                    self.audio_process.kill()
                except Exception:
                    pass
            self.audio_process = None

        # Wait for thread
        if self.dance_thread and self.dance_thread.is_alive():
            self.dance_thread.join(timeout=2.0)
        self.dance_thread = None

        # Return to neutral
        self.mixer.reset()

        self._status["running"] = False
        self._status["state"] = "idle"
        self._log("Stopped")
        logger.info(f"[{self.MODE_NAME}] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, "item"):
                status[key] = value.item()
        return status

    def _select_sequence(self, energy_level: str) -> list[dict]:
        """Select an 8-beat sequence for the given energy level."""
        key = f"{energy_level}_energy"
        sequences = EIGHT_BEAT_SEQUENCES.get(key, EIGHT_BEAT_SEQUENCES["medium_energy"])

        last_idx = self.last_sequence_used[energy_level]
        next_idx = (last_idx + 1) % len(sequences)
        self.last_sequence_used[energy_level] = next_idx

        return sequences[next_idx]

    def _compute_breathing_pose(self, t: float) -> MovementIntent:
        """Compute breathing/idle pose for organic motion."""
        y_offset = self.config.breathing_y_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_y_freq * t
        )
        roll_offset = self.config.breathing_roll_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_roll_freq * t
        )

        return MovementIntent(
            position=self.config.neutral_pos + np.array([0.0, y_offset, 0.0]),
            orientation=self.config.neutral_eul + np.array([roll_offset, 0.0, 0.0]),
            antennas=np.array([-0.15, 0.15]),
        )

    def _coords_to_offset(self, coords: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Convert choreography coords to position/orientation offsets."""
        scale = self.config.amplitude_scale

        position_offset = np.array([
            coords[0] * 0.01 * scale,
            coords[1] * 0.01 * scale,
            coords[2] * 0.01 * scale,
        ])
        orientation_offset = np.array([
            np.radians(coords[3] * scale),
            np.radians(coords[4] * scale),
            np.radians(coords[5] * scale),
        ])

        return position_offset, orientation_offset

    def _add_offset_to_intent(
        self,
        base_intent: MovementIntent,
        pos_offset: np.ndarray,
        ori_offset: np.ndarray,
    ) -> MovementIntent:
        """Add position/orientation offsets to a base intent."""
        return MovementIntent(
            position=base_intent.position + pos_offset,
            orientation=base_intent.orientation + ori_offset,
            antennas=base_intent.antennas,
        )

    def _get_continuous_antennas(self, current_time: float) -> np.ndarray:
        """Compute antenna position based on continuous energy envelope."""
        if not self.analysis or self.analysis.energy_envelope.size == 0:
            return np.array([-0.1, 0.1])

        idx = int(current_time * self.analysis.envelope_sr / self.analysis.hop_length)

        if idx < 0:
            return np.array([-0.1, 0.1])
        if idx >= len(self.analysis.energy_envelope):
            if idx > len(self.analysis.energy_envelope) + 10:
                return np.array([-0.1, 0.1])
            idx = len(self.analysis.energy_envelope) - 1

        raw_rms = self.analysis.energy_envelope[idx]

        # Apply noise gate
        threshold = self.config.antenna_energy_threshold
        if raw_rms < threshold:
            signal = 0.0
        else:
            signal = raw_rms - threshold

        # Apply gain and sensitivity
        signal = signal * self.config.antenna_gain * self.config.antenna_sensitivity

        # Map to splay
        max_travel = self.config.antenna_amplitude
        splay = min(signal, max_travel)

        rest = self.config.antenna_rest_position
        left = rest - splay
        right = -rest + splay

        return np.array([left, right])

    def _dance_loop(self) -> None:
        """Main dance execution loop."""
        if not self.analysis:
            return

        beat_times = self.analysis.beat_times
        sequence_assignments = self.analysis.sequence_assignments

        # Start audio playback (local or poll Spotify)
        if self.use_spotify_playback:
            self._dance_loop_spotify_sync(beat_times, sequence_assignments)
        else:
            self._start_audio_playback()
            self._dance_loop_local_audio(beat_times, sequence_assignments)

    def _dance_loop_local_audio(
        self,
        beat_times: np.ndarray,
        sequence_assignments: list[str],
    ) -> None:
        """Dance loop synced to local ffplay audio."""
        start_time = time.time()
        last_time = start_time
        current_block = -1
        energy_level = "medium"
        self.breathing_time = 0.0
        current_offset = np.zeros(6)

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now
            elapsed = now - start_time

            self.breathing_time += dt

            current_beat = np.searchsorted(beat_times, elapsed) - 1
            current_beat = max(0, min(current_beat, len(beat_times) - 1))

            self._status["current_beat"] = current_beat
            self._status["progress"] = elapsed / self.analysis.duration

            if elapsed >= self.analysis.duration:
                break

            block_idx = current_beat // 8

            if block_idx != current_block:
                current_block = block_idx
                if block_idx < len(sequence_assignments):
                    energy_level = sequence_assignments[block_idx]
                    self.current_sequence = self._select_sequence(energy_level)
                    self._status["energy_level"] = energy_level

            self._status["is_breathing"] = True

            breathing_intent = self._compute_breathing_pose(self.breathing_time)
            antennas = self._get_continuous_antennas(elapsed)

            beat_in_block = current_beat % 8
            beat_in_block = min(beat_in_block, len(self.current_sequence) - 1) if self.current_sequence else 0

            if self.current_sequence:
                move = self.current_sequence[beat_in_block]
                target = np.array(move["coords"])
                alpha = self.config.interpolation_alpha
                current_offset += (target - current_offset) * alpha

                pos_offset, ori_offset = self._coords_to_offset(current_offset.tolist())
                final_intent = self._add_offset_to_intent(breathing_intent, pos_offset, ori_offset)
                final_intent = MovementIntent(
                    position=final_intent.position,
                    orientation=final_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)
            else:
                final_intent = MovementIntent(
                    position=breathing_intent.position,
                    orientation=breathing_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)

            time.sleep(0.02)

    def _dance_loop_spotify_sync(
        self,
        beat_times: np.ndarray,
        sequence_assignments: list[str],
    ) -> None:
        """Dance loop synced to Spotify playback."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        current_block = -1
        energy_level = "medium"

        last_poll_time = 0
        local_start_time = time.time()
        last_time = local_start_time
        is_playing = True

        self.breathing_time = 0.0
        current_offset = np.zeros(6)

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            self.breathing_time += dt

            # Poll Spotify every 1s
            if now - last_poll_time > 1.0:
                try:
                    state = loop.run_until_complete(self.spotify.get_playback_state())
                    if state and state.get("item"):
                        if state["item"]["uri"] != self.spotify_track["uri"]:
                            logger.info(f"[{self.MODE_NAME}] Track changed externally")
                            break

                        is_playing = state["is_playing"]
                        if is_playing:
                            spotify_position_ms = state["progress_ms"]
                            local_start_time = now - (spotify_position_ms / 1000.0)
                            local_start_time += self.config.sync_offset

                    last_poll_time = now
                except Exception as e:
                    logger.warning(f"[{self.MODE_NAME}] Poll error: {e}")

            if not is_playing:
                time.sleep(0.1)
                continue

            elapsed = time.time() - local_start_time

            current_beat = np.searchsorted(beat_times, elapsed) - 1
            current_beat = max(0, min(current_beat, len(beat_times) - 1))

            self._status["current_beat"] = current_beat
            self._status["progress"] = elapsed / self.analysis.duration

            if elapsed >= self.analysis.duration + 2.0:
                break

            block_idx = current_beat // 8

            if block_idx != current_block:
                current_block = block_idx
                if block_idx < len(sequence_assignments):
                    energy_level = sequence_assignments[block_idx]
                    self.current_sequence = self._select_sequence(energy_level)
                    self._status["energy_level"] = energy_level

            self._status["is_breathing"] = True

            breathing_intent = self._compute_breathing_pose(self.breathing_time)
            antennas = self._get_continuous_antennas(elapsed)

            beat_in_block = current_beat % 8

            if self.current_sequence:
                beat_in_block = min(beat_in_block, len(self.current_sequence) - 1)
                move = self.current_sequence[beat_in_block]
                target = np.array(move["coords"])
                alpha = self.config.interpolation_alpha
                current_offset += (target - current_offset) * alpha

                pos_offset, ori_offset = self._coords_to_offset(current_offset.tolist())
                final_intent = self._add_offset_to_intent(breathing_intent, pos_offset, ori_offset)
                final_intent = MovementIntent(
                    position=final_intent.position,
                    orientation=final_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)
            else:
                final_intent = MovementIntent(
                    position=breathing_intent.position,
                    orientation=breathing_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)

            time.sleep(0.02)

        loop.close()

        if self.running:
            self.mixer.reset()
            self._status["running"] = False
            self._status["state"] = "idle"

    def _start_audio_playback(self) -> None:
        """Start audio playback using ffplay."""
        if not self.analysis:
            return

        try:
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                "-af", "aresample=16000,pan=mono|c0=c0",
                self.analysis.audio_path,
            ]
            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Audio playback error: {e}")
