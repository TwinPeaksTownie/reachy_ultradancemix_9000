"""
Reachy Ultra Dance Mix 9000 - Main Entry Point

A comprehensive dance application for Reachy Mini with three modes:
- Live Groove: Real-time BPM-driven dancing from audio input
- Connected Choreographer: Spotify/YouTube integration with beat analysis
- Bluetooth Streamer: Streaming audio control

This wraps the dance suite as a ReachyMiniApp for the Reachy Mini dashboard.
"""

import asyncio
import threading
import uvicorn

from reachy_mini import ReachyMini, ReachyMiniApp

from .app import app, state, initialize_with_robot


class ReachyUltradancemix9000(ReachyMiniApp):
    """Ultra Dance Mix 9000 - Full dance suite for Reachy Mini.

    Features:
    - Live Groove: Dance to any music playing nearby using microphone input
    - Connected Choreographer: Play Spotify tracks or YouTube videos with synchronized dancing
    - Bluetooth Streamer: Stream audio for reactive movement

    The app provides a web UI for configuration and control.
    """

    # URL for the custom settings page (served by our FastAPI app)
    custom_app_url: str | None = "http://localhost:9000"
    # Prevent daemon from starting its own basic server - we handle it ourselves
    dont_start_webserver: bool = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """
        Run the dance suite.

        This starts the FastAPI server which provides:
        - Web UI for mode selection and configuration
        - REST API for controlling dance modes
        - WebSocket for real-time status updates
        """
        # Initialize app state with the provided robot
        initialize_with_robot(reachy_mini)

        # Debug: Print registered routes
        print(f"[UltraDanceMix9000] Routes registered: {len(app.routes)}")
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                print(f"  {route.methods} {route.path}")

        # Configure the server port
        port = 9000

        print(f"[UltraDanceMix9000] App object id: {id(app)}, routes: {len(app.routes)}")

        # Create uvicorn config - use the app object directly
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",  # Enable info logging to see what's happening
        )

        # Create the server
        server = uvicorn.Server(config)

        # Run the server in a separate thread so we can monitor stop_event
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        print(f"[UltraDanceMix9000] Started on http://0.0.0.0:{port}")

        # Wait for stop signal
        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)

        # Cleanup
        print("[UltraDanceMix9000] Stopping...")

        # Stop any running dance mode
        if state.current_mode and state.current_mode.running:
            # Run async stop in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(state.current_mode.stop())
            finally:
                loop.close()

        # Reset safety mixer
        if state.safety_mixer:
            state.safety_mixer.reset()

        # Signal server to shutdown
        server.should_exit = True
        server_thread.join(timeout=5.0)

        print("[UltraDanceMix9000] Stopped")

if __name__ == "__main__":
    instance = ReachyUltradancemix9000()
    instance.wrapped_run()
