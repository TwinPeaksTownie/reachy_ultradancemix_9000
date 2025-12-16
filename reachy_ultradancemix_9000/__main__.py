"""
Entry point for Reachy Dance Suite.

Usage:
    # UI Mode - starts idle, control via web interface
    python -m reachy_dance_suite

    # CLI Mode - starts dancing immediately
    python -m reachy_dance_suite --mode bluetooth_streamer
    python -m reachy_dance_suite --mode live_groove
    python -m reachy_dance_suite --mode connected_choreographer --url "https://youtube.com/..."
"""

import argparse
import asyncio
import sys

import uvicorn

from .config import APP_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reachy Dance Suite - Unified dance application for Reachy Mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m reachy_dance_suite                                    # Start in UI mode (idle)
    python -m reachy_dance_suite --mode bluetooth_streamer          # Start Bluetooth Streamer
    python -m reachy_dance_suite --mode live_groove                 # Start Live Groove
    python -m reachy_dance_suite --mode connected_choreographer --url "https://youtube.com/watch?v=..."
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["live_groove", "bluetooth_streamer", "connected_choreographer"],
        help="Start in this mode immediately (skip idle state)",
    )
    parser.add_argument(
        "--url",
        help="YouTube URL for Connected Choreographer mode",
    )
    parser.add_argument(
        "--host",
        default=APP_CONFIG["host"],
        help=f"Host to bind to (default: {APP_CONFIG['host']})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=APP_CONFIG["port"],
        help=f"Port to bind to (default: {APP_CONFIG['port']})",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run without web UI (CLI mode only)",
    )

    return parser.parse_args()


async def run_cli_mode(mode: str, url: str | None = None):
    """Run in CLI mode - start dancing immediately without web UI."""
    from reachy_mini import ReachyMini
    from .core.safety_mixer import SafetyMixer
    from .config import get_default_safety_config
    from .behaviors.bluetooth_streamer import BluetoothStreamer
    from .behaviors.live_groove import LiveGroove
    from .behaviors.connected_choreographer import ConnectedChoreographer

    print(f"Starting Reachy Dance Suite in CLI mode ({mode})")
    print("Press Ctrl+C to stop")

    try:
        print("Connecting to Reachy Mini...")
        mini = ReachyMini()
        print("Connected!")

        safety_config = get_default_safety_config()
        safety_mixer = SafetyMixer(safety_config, mini)
        print("SafetyMixer initialized")

        # Create and start the requested mode
        if mode == "bluetooth_streamer":
            dance_mode = BluetoothStreamer(safety_mixer)
        elif mode == "live_groove":
            dance_mode = LiveGroove(safety_mixer)
        elif mode == "connected_choreographer":
            if not url:
                print("Connected Choreographer requires --url argument")
                return
            dance_mode = ConnectedChoreographer(safety_mixer)
            dance_mode.set_youtube_url(url)
        else:
            print(f"Unknown mode: {mode}")
            return

        await dance_mode.start()
        print(f"{mode} started. Dancing...")

        # Keep running until interrupted
        try:
            while dance_mode.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")

        await dance_mode.stop()
        safety_mixer.reset()
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode and args.no_ui:
        # CLI-only mode
        asyncio.run(run_cli_mode(args.mode, args.url))
    elif args.mode:
        # Start web server but also auto-start the mode
        print(f"Starting web UI with auto-start of {args.mode}")
        print(f"Web UI available at http://{args.host}:{args.port}")
        print("Note: Auto-start not yet implemented. Use API to start mode.")
        uvicorn.run(
            "reachy_dance_suite.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    else:
        # Normal UI mode - start idle
        print(f"Starting Reachy Dance Suite web UI")
        print(f"Web UI available at http://{args.host}:{args.port}")
        uvicorn.run(
            "reachy_dance_suite.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
