# Spotify Branding Guidelines - Developer Reference

*Comprehensive design system specifications for building Spotify-branded interfaces*

---

## Table of Contents
1. [Color System](#color-system)
2. [Typography](#typography)
3. [Logo Usage](#logo-usage)
4. [Spacing & Layout](#spacing--layout)
5. [UI Components](#ui-components)
6. [Album Artwork](#album-artwork)
7. [Dark Mode Interface](#dark-mode-interface)
8. [Accessibility](#accessibility)
9. [Design Principles](#design-principles)

---

## Color System

### Primary Colors

#### Spotify Green (Primary Brand Color)
- **Hex**: `#1DB954`
- **RGB**: `rgb(29, 185, 84)`
- **CMYK**: `80, 0, 80, 0`
- **HSL**: `hsl(141, 73%, 42%)`
- **Pantone**: PMS 7479 C

**Usage**: Primary logo color, key interactive elements, CTAs. Only use on black, white, or non-duotoned photography.

#### Spotify Green (Light - Logo Only)
- **Hex**: `#1ED760`
- **RGB**: `rgb(30, 215, 96)`

**Usage**: Official Spotify logo only. Not for general UI use.

#### Black (Background/UI)
- **Hex**: `#191414`
- **RGB**: `rgb(25, 20, 20)`

**Usage**: Primary dark background, app background

#### Pure Black
- **Hex**: `#000000`
- **RGB**: `rgb(0, 0, 0)`

**Usage**: True black for OLED optimization, deep shadows

#### White
- **Hex**: `#FFFFFF`
- **RGB**: `rgb(255, 255, 255)`

**Usage**: Light backgrounds, primary text on dark

### Grayscale Palette

#### Dark Grays
- **#282828** - Elevated surfaces
- **#181818** - Card backgrounds
- **#121212** - Deep backgrounds
- **#1A1A1A** - Surface variants

#### Mid Grays
- **#535353** - Disabled states
- **#404040** - Borders, dividers
- **#3E3E3E** - Secondary surfaces

#### Light Grays
- **#B3B3B3** - Secondary text
- **#A7A7A7** - Tertiary text
- **#6A6A6A** - Muted text

### Color Usage Rules

**DO:**
- Use Spotify Green for CTAs and primary actions
- Maintain high contrast (minimum 4.5:1 for normal text, 3:1 for large text)
- Use grayscale for hierarchy and depth
- Extract colors from album artwork for dynamic backgrounds

**DON'T:**
- Use Spotify Green with colored backgrounds (only black/white/non-duotoned photos)
- Introduce colors outside the brand palette
- Use over-saturated colors for CMYK printing
- Mix Spotify Green with duotoned images

---

## Typography

### Primary Typeface: Spotify Circular (LL Circular)

**Font Family**: LL Circular by Laurenz Brunner
**License**: Commercial (Lineto Type Foundry)
**Style**: Geometric sans-serif with rounded forms

### Font Weights

1. **Circular Light** (300)
   - Usage: Large display text, hero sections
   - Min size: 24px

2. **Circular Book** (400) - Regular
   - Usage: Body text, descriptions
   - Min size: 14px

3. **Circular Medium** (500)
   - Usage: Subheadings, emphasized text
   - Min size: 14px

4. **Circular Bold** (700)
   - Usage: Headlines, buttons, navigation
   - Min size: 12px

5. **Circular Black** (900)
   - Usage: Hero text, major headings
   - Min size: 32px

### Fallback Fonts

When Circular is unavailable, use this cascade:
```css
font-family: "Circular", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
```

### Typography Scale

```css
/* Display */
--font-size-xxxl: 96px;  /* Hero display */
--font-size-xxl: 72px;   /* Section headers */
--font-size-xl: 48px;    /* Page titles */
--font-size-lg: 32px;    /* Card titles */

/* Headings */
--font-size-h1: 28px;
--font-size-h2: 24px;
--font-size-h3: 20px;
--font-size-h4: 18px;

/* Body */
--font-size-base: 16px;    /* Base body text */
--font-size-sm: 14px;      /* Secondary text */
--font-size-xs: 12px;      /* Labels, metadata */
--font-size-xxs: 11px;     /* Fine print */
```

### Line Heights

```css
--line-height-tight: 1.2;    /* Headlines */
--line-height-normal: 1.5;   /* Body text */
--line-height-relaxed: 1.6;  /* Long-form content */
```

### Letter Spacing

```css
--letter-spacing-tight: -0.02em;  /* Large headings */
--letter-spacing-normal: 0;       /* Body text */
--letter-spacing-wide: 0.05em;    /* All caps labels */
```

### Usage Guidelines

- **Headlines**: Circular Bold/Black, tight line-height
- **Body text**: Circular Book, 16px minimum, 1.5 line-height
- **UI labels**: Circular Medium, 12-14px
- **Track names**: Circular Book/Medium, 14-16px, allow 23 characters
- **Artist names**: Circular Book, 14px, allow 18 characters
- **Playlist names**: Circular Medium, 14-16px, allow 25 characters

---

## Logo Usage

### Logo Variants

1. **Full Logo** (Icon + Wordmark)
   - Primary usage for all integrations
   - Use when space allows

2. **Icon Only**
   - Only when brand is established or space is limited
   - Never use wordmark without icon

### Logo Colors

#### Spotify Green Logo
- **Primary colorway**
- Only on: Black, White, or non-duotoned photography
- **File**: `spotify-full-logo-green.svg`

#### Black Logo
- On light colored backgrounds
- **File**: `spotify-full-logo-black.svg`

#### White Logo
- On dark colored backgrounds
- **File**: `spotify-full-logo-white.svg`

### Minimum Sizes

#### Full Logo
- **Digital**: 70px width minimum
- **Print**: 20mm width minimum

#### Icon
- **Digital**: 21px width minimum
- **Print**: 6mm width minimum

### Exclusion Zone

Clear space around logo = **0.5× the height of the icon**

```
    [×] = icon height
    Clear space = 0.5× on all sides
```

### Logo Misuse - NEVER DO THIS

❌ Rotate the logo
❌ Fill the lines of the logo
❌ Stretch or alter proportions
❌ Use logo as letter in text
❌ Change colors outside approved palette
❌ Place logo in busy/low-contrast areas
❌ Add effects (shadows, glows, gradients)
❌ Combine logo with other brand logos
❌ Place logo on album artwork

---

## Spacing & Layout

### Spacing Scale (8px Base Grid)

```css
/* Encore Foundation Spacing Tokens */
--space-xxxs: 4px;   /* Micro spacing */
--space-xxs: 8px;    /* Minimum touch target padding */
--space-xs: 12px;    /* Compact spacing */
--space-sm: 16px;    /* Default padding */
--space-md: 24px;    /* Section spacing */
--space-lg: 32px;    /* Component margins */
--space-xl: 48px;    /* Major sections */
--space-xxl: 64px;   /* Hero sections */
--space-xxxl: 96px;  /* Page-level spacing */
```

### Grid System

- **Base unit**: 8px
- **Container max-width**: 1200px
- **Gutter width**: 24px
- **Columns**: 12-column grid

### Component Spacing

#### Cards
- **Padding**: 16px (internal)
- **Margin**: 16px (between cards)
- **Border radius**: 4px (small/medium devices), 8px (large devices)

#### Buttons
- **Padding**: 
  - Small: 8px 16px
  - Medium: 12px 32px
  - Large: 16px 48px
- **Border radius**: 500px (fully rounded)
- **Min height**: 48px (touch target)

#### Lists
- **Item height**: 56px minimum
- **Item padding**: 12px 16px
- **Divider**: 1px, #282828

---

## UI Components

### Buttons

#### Primary Button (CTA)
```css
.button-primary {
  background: #1DB954;
  color: #FFFFFF;
  border: none;
  border-radius: 500px;
  padding: 14px 32px;
  font-family: Circular, sans-serif;
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  min-height: 48px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.button-primary:hover {
  background: #1ED760;
  transform: scale(1.04);
}

.button-primary:active {
  transform: scale(0.96);
}
```

#### Secondary Button
```css
.button-secondary {
  background: transparent;
  color: #FFFFFF;
  border: 2px solid #FFFFFF;
  border-radius: 500px;
  padding: 14px 32px;
  font-family: Circular, sans-serif;
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.button-secondary:hover {
  border-color: #1DB954;
  color: #1DB954;
}
```

#### Button Sizing
- **Small**: 32px height, 12px 24px padding
- **Medium**: 48px height, 14px 32px padding
- **Large**: 56px height, 16px 48px padding

### Cards

#### Album/Playlist Card
```css
.card {
  background: #181818;
  border-radius: 8px;
  padding: 16px;
  transition: background 0.2s ease;
}

.card:hover {
  background: #282828;
}

.card-artwork {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 4px;
  margin-bottom: 16px;
}

.card-title {
  font-family: Circular, sans-serif;
  font-weight: 700;
  font-size: 16px;
  color: #FFFFFF;
  margin-bottom: 4px;
  /* Clamp to 2 lines */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-subtitle {
  font-family: Circular, sans-serif;
  font-weight: 400;
  font-size: 14px;
  color: #B3B3B3;
  /* Clamp to 2 lines */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

### Navigation

#### Top Navigation Bar
```css
.navbar {
  height: 64px;
  background: #000000;
  padding: 0 32px;
  display: flex;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-link {
  color: #B3B3B3;
  font-family: Circular, sans-serif;
  font-weight: 700;
  font-size: 14px;
  margin: 0 16px;
  transition: color 0.2s ease;
}

.nav-link:hover,
.nav-link.active {
  color: #FFFFFF;
}
```

#### Sidebar Navigation
```css
.sidebar {
  width: 240px;
  background: #000000;
  padding: 24px 12px;
  overflow-y: auto;
}

.sidebar-item {
  height: 40px;
  padding: 0 16px;
  display: flex;
  align-items: center;
  border-radius: 4px;
  color: #B3B3B3;
  font-size: 14px;
  font-weight: 700;
  transition: color 0.2s ease;
}

.sidebar-item:hover,
.sidebar-item.active {
  color: #FFFFFF;
}
```

### Forms

#### Input Fields
```css
.input {
  width: 100%;
  height: 48px;
  background: #121212;
  border: 1px solid #727272;
  border-radius: 4px;
  padding: 0 16px;
  color: #FFFFFF;
  font-family: Circular, sans-serif;
  font-size: 16px;
}

.input:focus {
  border-color: #FFFFFF;
  outline: 2px solid #FFFFFF;
}

.input::placeholder {
  color: #6A6A6A;
}
```

### Icons

#### Like/Save Icon
- **State: Default**: Plus icon (+) in circle, #B3B3B3
- **State: Active**: Checkmark in filled circle, #1DB954
- **Size**: 24px × 24px
- **Usage**: Liking songs, saving albums/playlists

#### Play Button
- **Default**: Circle with play triangle, #1DB954 or white
- **Size**: 
  - Small: 32px
  - Medium: 48px
  - Large: 56px
- **Hover**: Scale 1.04

---

## Album Artwork

### Display Requirements

#### Sizing
- Maintain original aspect ratio (always square)
- Never crop, distort, or stretch
- Provide multiple sizes for responsive layouts

#### Border Radius
- **Small/Medium devices**: 4px corner radius
- **Large devices**: 8px corner radius
- **Lists**: 4px corner radius

#### Usage Rules

**DO:**
- Keep artwork in original form
- Round corners per specifications
- Extract dominant colors for backgrounds
- Use artwork provided by Spotify

**DON'T:**
- Crop or modify artwork
- Apply overlays, filters, or blur
- Place text, logos, or controls on artwork
- Animate or distort artwork
- Place your brand logo on artwork

### Dynamic Color Extraction

Extract dominant colors from album artwork for:
- Background gradients
- UI theming
- Accent colors

**Fallback**: Use #191414 if color extraction unavailable

---

## Dark Mode Interface

### Background Hierarchy

```css
/* Spotify's dark mode uses elevated surfaces */
--bg-base: #121212;         /* Base level */
--bg-elevated: #181818;     /* Cards, panels */
--bg-elevated-2: #282828;   /* Hover states */
--bg-elevated-3: #3E3E3E;   /* Active elements */
```

### Shadow System

```css
/* Subtle shadows for depth */
--shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.5);
--shadow-md: 0 4px 8px rgba(0, 0, 0, 0.6);
--shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.7);
--shadow-xl: 0 16px 32px rgba(0, 0, 0, 0.8);
```

### Text Hierarchy

```css
--text-primary: #FFFFFF;      /* Headlines, primary text */
--text-secondary: #B3B3B3;    /* Body text, descriptions */
--text-tertiary: #6A6A6A;     /* Metadata, timestamps */
--text-disabled: #535353;     /* Disabled states */
```

---

## Accessibility

### Color Contrast

**WCAG AA Compliance (Minimum)**
- Normal text (< 18pt): 4.5:1 contrast ratio
- Large text (≥ 18pt): 3:1 contrast ratio
- UI components: 3:1 contrast ratio

**Spotify Green on Backgrounds:**
- #1DB954 on #000000: ✅ 5.4:1 (Pass)
- #1DB954 on #FFFFFF: ✅ 3.3:1 (Large text only)
- #1DB954 on #191414: ✅ 5.1:1 (Pass)

### Focus States

All interactive elements must have visible focus indicators:
```css
.interactive-element:focus {
  outline: 2px solid #FFFFFF;
  outline-offset: 2px;
}

/* For dark backgrounds */
.interactive-element:focus-visible {
  outline: 3px solid #1DB954;
  outline-offset: 2px;
}
```

### Touch Targets

- **Minimum size**: 44px × 44px (WCAG AAA)
- **Recommended**: 48px × 48px
- **Padding**: Ensure 8px minimum space between targets

### Text Sizing

- **Minimum body text**: 14px
- **Minimum UI text**: 12px
- Allow user text scaling up to 200%

---

## Design Principles

### 1. Relevant
Make content feel personal and timely. Use dynamic data, contextual recommendations, and user preferences.

### 2. Human
Maintain warmth and personality. Use conversational copy, show creator identities, celebrate human artistry.

### 3. Unified
Create coherent experiences across platforms. Reuse patterns, maintain consistency, leverage the design system (Encore).

### Core Visual Principles

#### Simplicity
- Clean interfaces, minimal chrome
- Focus on content (music, podcasts, audiobooks)
- Remove unnecessary elements

#### Dark-First Design
- Optimized for low-light usage
- Reduces eye strain
- Emphasizes content and album artwork

#### Content-Forward
- Album artwork as hero element
- Bold typography
- Generous whitespace

#### Playful + Professional
- Vibrant accent colors in marketing
- Spotify Green for energy and growth
- Sophisticated in execution

---

## Encore Design System Structure

### Foundation Layer
Core design tokens used across all platforms:
- Color palette
- Typography scales
- Spacing units
- Motion timing
- Accessibility guidelines

### Platform Layers

#### Encore Web
Components for web applications:
- Buttons, forms, dialogs
- Navigation patterns
- Data visualization
- Layout grids

#### Encore Mobile
Components for native mobile (iOS/Android):
- Touch-optimized controls
- Platform-specific patterns
- Gesture interactions

### Local Systems
Product-specific components:
- Spotify for Artists
- Podcast creator tools
- Internal tools

---

## Component Patterns

### Player Interface

#### Mini Player
```css
.mini-player {
  height: 90px;
  background: #181818;
  border-top: 1px solid #282828;
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  padding: 16px;
  gap: 16px;
}

.track-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.track-artwork {
  width: 56px;
  height: 56px;
  border-radius: 4px;
}

.playback-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
}

.play-button {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #FFFFFF;
  color: #000000;
}
```

### Content Rows

```css
.content-row {
  margin-bottom: 40px;
}

.row-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding: 0 16px;
}

.row-title {
  font-size: 24px;
  font-weight: 700;
  color: #FFFFFF;
}

.row-link {
  font-size: 12px;
  font-weight: 700;
  color: #B3B3B3;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.row-content {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 24px;
  padding: 0 16px;
}
```

---

## Developer Notes

### Performance

- Use CSS transforms for animations (hardware accelerated)
- Lazy load album artwork
- Implement virtualized lists for long scrolling content
- Debounce search inputs (300ms)

### Responsive Breakpoints

```css
/* Encore responsive breakpoints */
--breakpoint-xs: 480px;   /* Mobile */
--breakpoint-sm: 768px;   /* Tablet */
--breakpoint-md: 1024px;  /* Desktop */
--breakpoint-lg: 1280px;  /* Large desktop */
--breakpoint-xl: 1920px;  /* Extra large */
```

### Animation Timing

```css
--duration-instant: 100ms;   /* Micro-interactions */
--duration-fast: 200ms;      /* Hover, focus states */
--duration-normal: 300ms;    /* Standard transitions */
--duration-slow: 500ms;      /* Page transitions */
--easing-standard: cubic-bezier(0.4, 0.0, 0.2, 1);
--easing-decelerate: cubic-bezier(0.0, 0.0, 0.2, 1);
--easing-accelerate: cubic-bezier(0.4, 0.0, 1, 1);
```

---

## Quick Reference

### Essential CSS Variables

```css
:root {
  /* Colors */
  --spotify-green: #1DB954;
  --spotify-black: #191414;
  --spotify-white: #FFFFFF;
  --bg-base: #121212;
  --bg-elevated: #181818;
  --text-primary: #FFFFFF;
  --text-secondary: #B3B3B3;
  
  /* Typography */
  --font-family: "Circular", -apple-system, sans-serif;
  --font-size-base: 16px;
  --line-height-base: 1.5;
  
  /* Spacing */
  --space-xs: 8px;
  --space-sm: 16px;
  --space-md: 24px;
  --space-lg: 32px;
  
  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-full: 500px;
  
  /* Transitions */
  --transition-fast: 200ms ease;
}
```

---

## Legal & Attribution

### Attribution Requirements

When using Spotify content, you MUST:
1. Display Spotify logo or icon
2. Link back to Spotify app/service
3. Keep metadata intact (track, artist, album names)
4. Maintain album artwork in original form

### Restricted Actions

**DO NOT:**
- Modify or crop album artwork
- Use Spotify branding with other services
- Include "Spotify" in your app name
- Imitate Spotify logo design
- Use Spotify Green outside specified contexts

### Links to Spotify

- If app installed: "OPEN SPOTIFY" / "PLAY ON SPOTIFY"
- If app not installed: "GET SPOTIFY FREE"

---

## Resources

### Official Links
- Developer Documentation: https://developer.spotify.com/documentation/design
- Spotify Design Blog: https://spotify.design
- Design Principles: https://spotify.design/article/introducing-spotifys-new-design-principles
- Encore Design System: https://spotify.design/stories/design/design-systems

### Downloads
- Logo assets: https://developer.spotify.com/documentation/design
- Brand guidelines PDF: Contact brandapproval@spotify.com

---

*This document compiled from official Spotify design resources and developer documentation. Last updated: December 2024.*

**For questions or approval of specific implementations, contact:** brandapproval@spotify.com
