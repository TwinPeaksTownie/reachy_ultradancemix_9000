# Reachy UltraDanceMix 9000

An app that makes Reachy Mini dance.

Built to be the "Wii Sports" of Reachy apps - the fun, immediately accessible experience that ships with every robot.

## Three Dance Modes

### Live Groove - *The Listener*
Listens to music playing in the room via the robot's USB microphone. Uses Librosa for BPM detection with spectral noise subtraction (3-phase calibration filters out motor noise).

**Key addition:** A "recovery expert" system - gentle sine wave breathing motion with slight head roll between dance moves. This gives balance to the upbeat choreography and works surprisingly well for sustained dancing.

### Bluetooth Streamer - *The Reactor*
Real-time audio-to-motion with <50ms latency via system audio loopback (BlackHole on Mac).

- Bass → body sway + bounce
- Highs → head pitch
- Vocals → antenna spread

Raw, visceral, arcade-like response.

### Connected Choreographer - *The Performer*
Pre-analyzes a song (YouTube URL or Spotify track), then delivers rehearsed choreography synced to the music's energy profile.

**Built on Xavier's YT mini MCP** with additions:
- Per-move dampening sliders (amplitude adjustment per dance move)
- Spotify integration (search, select, sync playback)

## Quick Start

```bash
# From the dance_app directory
cd dance_app

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m reachy_dance_suite.app
```

Open `http://localhost:8080` in your browser.

## Requirements

- Python 3.10+
- Reachy Mini daemon running (real robot or simulator)
- **Live Groove:** Robot's USB microphone ("Reachy Mini Audio")
- **Bluetooth Streamer:** BlackHole audio loopback (Mac)
- **Connected Choreographer:** `yt-dlp` and `ffmpeg` for YouTube downloads

### Optional: Spotify Integration

Requires Spotify Premium. The app includes a setup wizard:
1. Create a Spotify Developer app
2. Paste your Client ID
3. Complete OAuth login (one-time setup with refresh token)

## Open Questions

**Antenna dampening system** - The current implementation is complex. Open to suggestions for simplification.

## Architecture

```
dance_app/
├── reachy_dance_suite/
│   ├── app.py                 # FastAPI server
│   ├── behaviors/
│   │   ├── base.py            # DanceMode abstract base
│   │   ├── live_groove.py     # BPM detection + breathing
│   │   ├── bluetooth_streamer.py
│   │   └── connected_choreographer.py
│   ├── core/
│   │   ├── safety_mixer.py    # Movement intent → robot
│   │   └── audio_stream.py    # Platform-agnostic audio
│   ├── spotify/               # Spotify OAuth + API
│   └── static/
│       └── index.html         # Web UI
```

## License

Apache 2.0

## Contributors

- Carson ([@TwinPeaksTownie](https://github.com/TwinPeaksTownie))
- Remi Fabre ([@RemiFabre](https://github.com/RemiFabre)) - Pollen Robotics
- Haixuan Xavier Tao ([@haixuanTao](https://github.com/haixuanTao)) - Pollen Robotics
