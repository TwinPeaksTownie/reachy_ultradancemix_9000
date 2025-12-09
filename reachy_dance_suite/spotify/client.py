"""
Spotify Client for Reachy Dance Suite.

Handles OAuth 2.0 (PKCE) flow and API interactions.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import urllib.parse
from typing import Any, Dict, Optional

import httpx


class SpotifyAuth:
    """Handles Spotify OAuth 2.0 with PKCE and Dynamic Client ID."""

    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    TOKEN_FILE = Path(".spotify_tokens.json")
    
    SCOPES = [
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
        "streaming",
        "app-remote-control",
        "user-read-email",
        "user-read-private",
    ]

    def __init__(self, redirect_uri: str, default_client_id: Optional[str] = None):
        self.redirect_uri = redirect_uri
        self.default_client_id = default_client_id
        # In-memory store for verifiers keyed by state
        self._pending_auths: Dict[str, Dict[str, str]] = {}
        
        # Active session state
        self.active_client_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: float = 0
        
        # Load saved tokens if they exist
        self._load_tokens()

    def get_auth_url(self, client_id: Optional[str] = None) -> str:
        """Generate auth URL with client_id embedded in state."""
        use_id = client_id or self.default_client_id
        if not use_id:
            raise ValueError("No Client ID provided")

        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip("=")

        # Encode Client ID into state parameter
        state_data = json.dumps({"cid": use_id, "nonce": secrets.token_hex(4)})
        state = base64.urlsafe_b64encode(state_data.encode()).decode().rstrip("=")

        self._pending_auths[state] = {
            "verifier": code_verifier,
            "client_id": use_id
        }

        params = {
            "client_id": use_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
            "scope": " ".join(self.SCOPES),
            "state": state
        }
        
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange code for token using the client_id recovered from state."""
        auth_context = self._pending_auths.pop(state, None)
        if not auth_context:
            raise ValueError("Invalid or expired state parameter")

        code_verifier = auth_context["verifier"]
        client_id = auth_context["client_id"]

        data = {
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.TOKEN_URL, data=data)
            response.raise_for_status()
            tokens = response.json()
            
            self._save_tokens(tokens, client_id)
            return tokens

    async def refresh_access_token(self) -> str:
        if not self.refresh_token or not self.active_client_id:
            raise ValueError("No refresh token available")

        data = {
            "client_id": self.active_client_id,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.TOKEN_URL, data=data)
            response.raise_for_status()
            tokens = response.json()
            self._save_tokens(tokens, self.active_client_id)
            return self.access_token

    def _save_tokens(self, tokens: Dict[str, Any], client_id: str):
        """Save tokens to memory and disk."""
        self.active_client_id = client_id
        self.access_token = tokens["access_token"]
        if "refresh_token" in tokens:
            self.refresh_token = tokens["refresh_token"]
        self.expires_at = time.time() + tokens["expires_in"]
        
        # Persist to disk
        try:
            token_data = {
                "client_id": self.active_client_id,
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": self.expires_at,
            }
            self.TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
            print(f"[Spotify] Tokens saved to {self.TOKEN_FILE}")
        except Exception as e:
            print(f"[Spotify] Failed to save tokens: {e}")

    def _load_tokens(self):
        """Load tokens from disk if they exist."""
        if not self.TOKEN_FILE.exists():
            return
        
        try:
            token_data = json.loads(self.TOKEN_FILE.read_text())
            self.active_client_id = token_data.get("client_id")
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            self.expires_at = token_data.get("expires_at", 0)
            print(f"[Spotify] Loaded saved tokens for client: {self.active_client_id[:8]}...")
        except Exception as e:
            print(f"[Spotify] Failed to load tokens: {e}")

    def is_authenticated(self) -> bool:
        return bool(self.access_token) or bool(self.refresh_token)

    async def get_valid_token(self) -> str:
        if not self.access_token:
            raise ValueError("Not authenticated")
        if time.time() > self.expires_at - 60:
            await self.refresh_access_token()
        return self.access_token


class SpotifyClient:
    """Client for Spotify Web API."""

    API_BASE = "https://api.spotify.com/v1"

    def __init__(self, auth: SpotifyAuth):
        self.auth = auth

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to Spotify API."""
        token = await self.auth.get_valid_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        url = f"{self.API_BASE}/{endpoint.lstrip('/')}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            
            if response.status_code == 204:
                return {}
                
            response.raise_for_status()
            return response.json()

    async def search(self, query: str, type: str = "track", limit: int = 10) -> Dict[str, Any]:
        """Search for tracks, albums, etc."""
        params = {"q": query, "type": type, "limit": limit}
        return await self._request("GET", "search", params=params)

    async def get_playback_state(self) -> Dict[str, Any]:
        """Get information about the user's current playback state."""
        return await self._request("GET", "me/player")

    async def start_playback(self, uris: list[str], device_id: Optional[str] = None) -> None:
        """Start or resume playback."""
        data = {"uris": uris}
        params = {}
        if device_id:
            params["device_id"] = device_id
            
        await self._request("PUT", "me/player/play", json=data, params=params)

    async def pause_playback(self) -> None:
        """Pause playback."""
        try:
            await self._request("PUT", "me/player/pause")
        except httpx.HTTPStatusError as e:
            # Ignore if already paused (403 or 404 sometimes)
            if e.response.status_code not in [403, 404, 502]:
                raise

    async def seek(self, position_ms: int) -> None:
        """Seek to position in current track."""
        await self._request("PUT", "me/player/seek", params={"position_ms": position_ms})

    async def get_devices(self) -> Dict[str, Any]:
        """Get information about available devices."""
        return await self._request("GET", "me/player/devices")

