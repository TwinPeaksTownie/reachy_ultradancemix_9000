# Amazon Music Integration & Spotify Restructuring Recap

## Part 1: Amazon Music Integration Proposal

This section outlines the architectural plan for integrating Amazon Music functionality into the Reachy Dance Suite, mirroring the pattern established for Spotify.

### Architecture Diagram

```mermaid
graph TD
    subgraph Frontend ["Web UI (Frontend)"]
        UI_Tab[Amazon Music Tab]
        UI_Auth[Login / Auth Flow]
        UI_Search[Search Interface]
    end

    subgraph Backend ["Reachy Dance Suite (Backend)"]
        API[FastAPI App]
        
        subgraph Amazon_Pkg ["reachy_dance_suite.amazon"]
            Auth[AmazonAuth Class]
            Client[AmazonClient Class]
            Choreographer[AmazonChoreographer (Mode M)]
        end
        
        subgraph Core ["Core Systems"]
            Mixer[Safety Mixer]
            Mini[Reachy Mini Interface]
        end
        
        subgraph Helpers ["Helpers"]
            YTDL[YouTube Downloader]
            Librosa[Audio Analyzer]
        end
    end

    subgraph External ["External Services"]
        AM_API[Amazon Music API]
        AM_Auth[Login with Amazon (LWA)]
        YouTube[YouTube (Fallback Audio)]
    end

    %% Interactions
    UI_Tab -->|Control / Status| API
    UI_Auth -->|OAuth Flow| API
    UI_Search -->|Search Query| API

    API -->|Init| Auth
    API -->|Use| Client
    API -->|Start Mode| Choreographer

    Auth -->|Tokens| Client
    Auth <-->|OAuth Web Flow| AM_Auth

    Client <-->|Search / Playback Control| AM_API

    Choreographer -->|Poll Playback State| Client
    Choreographer -->|Request Audio| YTDL
    YTDL <-->|Download Audio| YouTube
    Choreographer -->|Analyze Audio| Librosa
    
    Choreographer -->|Send Moves| Mixer
    Mixer -->|Robot Commands| Mini
```

### Implementation Strategy

1.  **New Package**: Create `reachy_dance_suite/amazon/` to house all Amazon-specific logic.
2.  **`client.py`**: Implement `AmazonAuth` (handling "Login with Amazon") and `AmazonClient` (API wrapper).
3.  **`choreographer.py`**: Implement `AmazonChoreographer` (likely "Mode M"). This will use the same hybrid approach as Spotify:
    *   Search/Play via Amazon API.
    *   Download audio from YouTube for local beat analysis.
    *   Sync robot moves to local analysis while polling Amazon for playback progress.
4.  **Configuration**: Add `AMAZON_CONFIG` to `config.py` for Client ID/Secret and Redirect URIs.
5.  **Frontend**: Duplicate the Spotify tab structure in `index.html` to provide Amazon login and search capabilities.

---

## Part 2: Spotify Restructuring Recap

This section summarizes the refactoring work completed to organize the Spotify integration.

### Objectives Achieved
*   **Folder Restructuring**: Consolidate all Spotify-related files into a dedicated package.
*   **Port Update**: Change the application default port from `8200` to `9000`.
*   **Import Cleanup**: Ensure all imports point to the new locations and unnecessary code is removed.

### Changes Summary

1.  **File System Organization**:
    *   Created new directory: `reachy_dance_suite/spotify/`
    *   Moved `reachy_dance_suite/core/spotify_client.py` -> `reachy_dance_suite/spotify/client.py`
    *   Moved `reachy_dance_suite/behaviors/spotify_choreographer.py` -> `reachy_dance_suite/spotify/choreographer.py`
    *   Created `reachy_dance_suite/spotify/__init__.py` to expose classes (`SpotifyChoreographer`, `SpotifyAuth`, `SpotifyClient`) for cleaner top-level imports.

2.  **Configuration Updates (`config.py`)**:
    *   Updated `APP_CONFIG["port"]` to **9000**.
    *   Updated `SPOTIFY_CONFIG["redirect_uri"]` to `http://localhost:9000/api/spotify/callback`.

3.  **Application Entry Point (`app.py`)**:
    *   Updated imports to use the new `reachy_dance_suite.spotify` package.
    *   Cleaned up redundant imports.
    *   Verified `lifecycle` and `start_mode` handlers interact correctly with the moved classes.

4.  **Frontend (`static/index.html`)**:
    *   Updated the UI display for the Redirect URI to show port **9000**.

### Current State
 The application is now structured with a modular `spotify` package, running on port 9000, and fully prepared for the addition of similar modules (like the proposed `amazon` package) in the future.
