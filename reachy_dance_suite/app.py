"""
FastAPI application for Reachy Dance Suite.

Provides REST API and serves config UI for controlling dance modes.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from reachy_mini import ReachyMini

from .core.safety_mixer import SafetyMixer, SafetyConfig

from . import move_config
from . import mode_settings
from .behaviors.base import DanceMode
from .behaviors.live_groove import LiveGroove
from .behaviors.bluetooth_streamer import BluetoothStreamer
from .behaviors.connected_choreographer import ConnectedChoreographer
from .spotify import SpotifyAuth, SpotifyClient
from .config import get_default_safety_config, APP_CONFIG, SPOTIFY_CONFIG


# Global state
class AppState:
    """Global application state."""
    mini: Optional[ReachyMini] = None
    safety_mixer: Optional[SafetyMixer] = None
    spotify_auth: Optional[SpotifyAuth] = None
    spotify_client: Optional[SpotifyClient] = None
    current_mode: Optional[DanceMode] = None
    modes: dict[str, type[DanceMode]] = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - connects to robot on startup."""
    print("Connecting to Reachy Mini...")

    try:
        state.mini = ReachyMini()
        print("Connected to Reachy Mini!")

        # Initialize SafetyMixer
        safety_config = get_default_safety_config()
        state.safety_mixer = SafetyMixer(safety_config, state.mini)
        print("SafetyMixer initialized")

        # Initialize Spotify
        import os
        client_id = os.getenv("SPOTIFY_CLIENT_ID", SPOTIFY_CONFIG["client_id"])
        # Passed as default_client_id (optional now)
        state.spotify_auth = SpotifyAuth(SPOTIFY_CONFIG["redirect_uri"], default_client_id=client_id)
        state.spotify_client = SpotifyClient(state.spotify_auth)
        print("Spotify Client initialized")

        # Register available modes
        state.modes = {
            "live_groove": LiveGroove,
            "bluetooth_streamer": BluetoothStreamer,
            "connected_choreographer": ConnectedChoreographer,
        }
        print(f"Registered modes: {list(state.modes.keys())}")

        yield

    finally:
        # Cleanup
        print("Shutting down...")
        if state.current_mode and state.current_mode.running:
            await state.current_mode.stop()

        if state.safety_mixer:
            state.safety_mixer.reset()

        state.mini = None
        state.safety_mixer = None
        print("Shutdown complete")


app = FastAPI(
    title="Reachy Dance Suite",
    description="Unified dance application for Reachy Mini",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic models for API
class SafetyConfigUpdate(BaseModel):
    """Model for updating safety config."""
    z_threshold: Optional[float] = None
    max_pitch_at_low_z: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    intensity: Optional[float] = None


class ModeStartRequest(BaseModel):
    """Model for mode start request."""
    url: Optional[str] = None  # For Mode B
    profile_path: Optional[str] = None  # For Mode A - "default" to load saved profile, path for custom
    skip_calibration: bool = False  # For Mode A - skip calibration entirely
    force_calibration: bool = False  # For Mode A - force fresh calibration (ignore default profile)


# Static file path
STATIC_DIR = Path(__file__).parent / "static"


# API Endpoints
@app.get("/")
async def root():
    """Serve the config UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(content="<h1>Reachy Dance Suite</h1><p>Static files not found.</p>")


@app.get("/api/status")
async def get_status():
    """Get current application status."""
    mode_status = None
    if state.current_mode:
        mode_status = state.current_mode.get_status()

    return {
        "connected": state.mini is not None,
        "current_mode": state.current_mode.MODE_ID if state.current_mode else None,
        "available_modes": list(state.modes.keys()),
        "mode_status": mode_status,
    }


@app.post("/api/mode/{mode_id}/start")
async def start_mode(mode_id: str, request: ModeStartRequest = None):
    """Start a dance mode."""
    mode_id = mode_id.lower()

    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    if mode_id not in state.modes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: {mode_id}. Available: {list(state.modes.keys())}",
        )

    # Stop current mode if running
    if state.current_mode and state.current_mode.running:
        await state.current_mode.stop()

    # Create and start new mode
    mode_class = state.modes[mode_id]

    # Live Groove accepts profile_path, skip_calibration, and force_calibration
    if mode_id == "live_groove":
        profile_path = request.profile_path if request else None
        skip_calibration = request.skip_calibration if request else False
        force_calibration = request.force_calibration if request else False
        # "default" means use the default profile path (let LiveGroove handle it)
        if profile_path == "default":
            profile_path = None  # LiveGroove will check DEFAULT_PROFILE_PATH
        state.current_mode = mode_class(
            state.safety_mixer,
            profile_path=profile_path,
            skip_calibration=skip_calibration,
            force_calibration=force_calibration,
        )
    # Connected Choreographer accepts URL and spotify client
    elif mode_id == "connected_choreographer":
        state.current_mode = mode_class(state.safety_mixer, state.spotify_client)
        # If URL provided, set it directly and start
        if request and request.url:
            state.current_mode.set_youtube_url(request.url)
    else:
        # Bluetooth Streamer - no special params
        state.current_mode = mode_class(state.safety_mixer)

    await state.current_mode.start()

    return {
        "status": "started",
        "mode": mode_id,
        "mode_status": state.current_mode.get_status(),
    }


@app.post("/api/mode/stop")
async def stop_mode():
    """Stop the current dance mode."""
    if state.current_mode is None or not state.current_mode.running:
        return {"status": "already_stopped"}

    await state.current_mode.stop()

    return {
        "status": "stopped",
        "mode_status": state.current_mode.get_status() if state.current_mode else None,
    }


@app.get("/api/safety")
async def get_safety_config():
    """Get current safety configuration."""
    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    config = state.safety_mixer.config
    return {
        "z_threshold": config.z_threshold,
        "max_pitch_at_low_z": config.max_pitch_at_low_z,
        "smoothing_alpha": config.smoothing_alpha,
        "intensity": config.intensity,
        "max_position": config.max_position.tolist(),
        "min_position": config.min_position.tolist(),
        "max_orientation": config.max_orientation.tolist(),
        "min_orientation": config.min_orientation.tolist(),
    }


@app.post("/api/safety")
async def update_safety_config(update: SafetyConfigUpdate):
    """Update safety configuration (live tuning)."""
    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    # Apply updates
    updates = update.model_dump(exclude_none=True)
    state.safety_mixer.update_config(**updates)

    return {
        "status": "updated",
        "updated_fields": list(updates.keys()),
    }


# Move dampening endpoints
@app.get("/api/moves")
async def get_moves():
    """Get all moves and their dampening values."""
    return {
        "moves": move_config.get_all_dampening(),
    }


@app.post("/api/moves")
async def update_moves(updates: dict[str, float]):
    """Update dampening values for moves."""
    move_config.set_all_dampening(updates)
    return {
        "status": "updated",
        "moves": move_config.get_all_dampening(),
    }


@app.post("/api/moves/reset")
async def reset_moves():
    """Reset all moves to default dampening values."""
    move_config.reset_to_defaults()
    return {
        "status": "reset",
        "moves": move_config.get_all_dampening(),
    }


# Move mirror endpoints
@app.get("/api/moves/mirror")
async def get_move_mirror():
    """Get mirror settings for all mirrorable moves."""
    return {
        "mirrorable": move_config.get_mirrorable_moves(),
        "mirror": move_config.get_all_mirror(),
    }


@app.post("/api/moves/mirror")
async def update_move_mirror(updates: dict[str, bool]):
    """Update mirror settings for moves."""
    move_config.set_all_mirror(updates)

    # If Live Groove is running, refresh its move list
    if state.current_mode and state.current_mode.MODE_ID == "live_groove":
        if hasattr(state.current_mode, 'refresh_moves'):
            state.current_mode.refresh_moves()

    return {
        "status": "updated",
        "mirror": move_config.get_all_mirror(),
    }


@app.post("/api/moves/mirror/reset")
async def reset_move_mirror():
    """Reset all mirror settings to defaults (all false)."""
    move_config.reset_mirror_to_defaults()

    # If Live Groove is running, refresh its move list
    if state.current_mode and state.current_mode.MODE_ID == "live_groove":
        if hasattr(state.current_mode, 'refresh_moves'):
            state.current_mode.refresh_moves()

    return {
        "status": "reset",
        "mirror": move_config.get_all_mirror(),
    }


# Mode settings endpoints
@app.get("/api/mode-settings")
async def get_all_mode_settings():
    """Get all mode settings."""
    return {
        "settings": mode_settings.get_all_settings(),
    }


@app.get("/api/mode-settings/{mode_id}")
async def get_mode_settings(mode_id: str):
    """Get settings for a specific mode."""
    mode_id = mode_id.lower()  # Normalize to lowercase
    settings = mode_settings.get_mode_settings(mode_id)
    return {
        "mode": mode_id,
        "settings": settings,
    }


@app.post("/api/mode-settings/{mode_id}")
async def update_mode_settings(mode_id: str, updates: dict[str, float]):
    """Update settings for a mode and apply to running mode if active."""
    mode_id = mode_id.lower()  # Normalize to lowercase
    mode_settings.update_mode_settings(mode_id, updates)

    # If this mode is currently running, apply settings live
    if state.current_mode and state.current_mode.MODE_ID == mode_id:
        if hasattr(state.current_mode, 'apply_settings'):
            state.current_mode.apply_settings(updates)

    return {
        "status": "updated",
        "mode": mode_id,
        "settings": mode_settings.get_mode_settings(mode_id),
    }


@app.post("/api/mode-settings/sync")
async def sync_mode_settings():
    """Reload all settings from JSON file."""
    settings = mode_settings.sync_from_file()
    return {
        "status": "synced",
        "settings": settings,
    }


@app.post("/api/mode-settings/reset")
async def reset_mode_settings():
    """Reset all mode settings to defaults."""
    settings = mode_settings.reset_to_defaults()
    return {
        "status": "reset",
        "settings": settings,
    }


# Profile Management Endpoints
PROFILE_PATH = Path(__file__).parent / "environment_profile.npz"


@app.get("/api/profile/status")
async def get_profile_status():
    """Check if a saved environment profile exists."""
    if not PROFILE_PATH.exists():
        return {
            "exists": False,
            "path": str(PROFILE_PATH),
        }

    try:
        # Load metadata from profile
        data = np.load(PROFILE_PATH, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        return {
            "exists": True,
            "path": str(PROFILE_PATH),
            "created": metadata.get("created", "unknown"),
            "silence_rms": metadata.get("silence_rms", 0),
            "breathing_rms": metadata.get("breathing_rms", 0),
            "dance_rms": metadata.get("dance_rms", 0),
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(PROFILE_PATH),
            "error": str(e),
        }


@app.post("/api/profile/save")
async def save_profile():
    """Save current calibration profile to disk."""
    if not state.current_mode or state.current_mode.MODE_ID != "live_groove":
        raise HTTPException(status_code=400, detail="Live Groove must be active to save profile")

    mode = state.current_mode

    # Check if calibration data exists
    if mode.silence_noise_profile is None or mode.breathing_noise_profile is None:
        raise HTTPException(status_code=400, detail="No calibration data - run calibration first")

    try:
        # Compute RMS values for metadata
        silence_rms = float(np.sqrt(np.mean(mode.silence_noise_profile**2)))
        breathing_rms = float(np.sqrt(np.mean(mode.breathing_noise_profile**2)))
        dance_rms = float(np.sqrt(np.mean(mode.dance_noise_profile**2))) if mode.dance_noise_profile is not None else 0.0

        metadata = {
            "created": datetime.now().isoformat(),
            "silence_rms": silence_rms,
            "breathing_rms": breathing_rms,
            "dance_rms": dance_rms,
        }

        np.savez(
            PROFILE_PATH,
            silence_profile=mode.silence_noise_profile,
            breathing_profile=mode.breathing_noise_profile,
            dance_profile=mode.dance_noise_profile if mode.dance_noise_profile is not None else mode.breathing_noise_profile,
            metadata=json.dumps(metadata),
        )

        print(f"[API] Saved environment profile to {PROFILE_PATH}")
        return {
            "status": "saved",
            "path": str(PROFILE_PATH),
            "created": metadata["created"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {e}")


# Spotify Endpoints
@app.get("/api/spotify/status")
async def get_spotify_status():
    """Check if authenticated with Spotify."""
    if not state.spotify_auth:
        return {"authenticated": False}
    return {"authenticated": state.spotify_auth.is_authenticated()}


@app.get("/api/spotify/login")
async def spotify_login(client_id: Optional[str] = None):
    """Get Spotify login URL."""
    if not state.spotify_auth:
        raise HTTPException(status_code=503, detail="Spotify not configured")
    try:
        return {"url": state.spotify_auth.get_auth_url(client_id=client_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/spotify/callback")
async def spotify_callback(code: str, state: str):
    """Handle Spotify auth callback."""
    app_state = globals()["state"]
    if not app_state.spotify_auth:
         raise HTTPException(status_code=503, detail="Spotify not configured")
    
    try:
        await app_state.spotify_auth.exchange_code(code, state)
        return HTMLResponse(content="<h1>Authenticated!</h1><p>You can close this window.</p><script>window.close()</script>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=400)


@app.get("/api/spotify/search")
async def spotify_search(q: str):
    """Search Spotify tracks."""
    if not state.spotify_client:
        raise HTTPException(status_code=503, detail="Spotify not configured")
    
    try:
        results = await state.spotify_client.search(q)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mode/connected_choreographer/track")
async def set_spotify_track(track: dict):
    """Set the Spotify track for Connected Choreographer."""
    print(f"[API] Received track selection: {track.get('name', 'Unknown')}")

    if not state.current_mode or state.current_mode.MODE_ID != "connected_choreographer":
        print(f"[API] Error: Connected Choreographer not active (current: {state.current_mode.MODE_ID if state.current_mode else None})")
        raise HTTPException(status_code=400, detail="Connected Choreographer is not active")

    try:
        # If running, stop it first so we can restart with new track
        if state.current_mode.running:
            print(f"[API] Stopping current playback...")
            await state.current_mode.stop()

        print(f"[API] Setting track...")
        await state.current_mode.set_spotify_track(track)

        print(f"[API] Starting playback...")
        await state.current_mode.start()

        print(f"[API] Track set successfully: {track['name']}")
        return {"status": "track_set", "track": track["name"]}
    except Exception as e:
        print(f"[API] Error setting track: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mode/connected_choreographer/youtube")
async def set_youtube_url(data: dict):
    """Set YouTube URL for Connected Choreographer."""
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    print(f"[API] Received YouTube URL: {url}")

    if not state.current_mode or state.current_mode.MODE_ID != "connected_choreographer":
        print(f"[API] Error: Connected Choreographer not active")
        raise HTTPException(status_code=400, detail="Connected Choreographer is not active")

    try:
        # If running, stop it first
        if state.current_mode.running:
            await state.current_mode.stop()

        state.current_mode.set_youtube_url(url)
        await state.current_mode.start()

        return {"status": "started", "url": url}
    except Exception as e:
        print(f"[API] Error setting URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time status streaming
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates."""
    await websocket.accept()

    try:
        while True:
            # Build status
            mode_status = None
            if state.current_mode:
                mode_status = state.current_mode.get_status()

            status = {
                "connected": state.mini is not None,
                "current_mode": state.current_mode.MODE_ID if state.current_mode else None,
                "available_modes": list(state.modes.keys()),
                "mode_status": mode_status,
            }

            await websocket.send_json(status)
            await asyncio.sleep(0.1)  # 10Hz updates

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def create_app() -> FastAPI:
    """Factory function to create the app."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "reachy_dance_suite.app:app",
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        reload=False,
    )
