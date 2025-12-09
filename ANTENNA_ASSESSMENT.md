# Antenna System Assessment: Scaling, Gain, and Motor Limits

## 1. The Core Issue: Physical Motor Overdrive
The observed "jittering" or "stuck at 1.0" behavior is consistent with **Motor Stall Protection**.

*   **Physical Limit**: The physical range of the antenna motors is `0` to `Pi` (3.14159 radians).
*   **The Error**: Our logic was generating commands up to **3.25 radians** (186 degrees).
*   **The Result**: When the hardware receives a command > 3.14, it rejects it or hits a hard mechanical stop. This causes the motor to "freeze" or bounce at a safe default position (approx ~1.0 rad) rather than completing the motion. This explains the "smearing grout" / stuck behavior.

## 2. Signal Path & Logic Fixes

To achieve the desired "Raw Power" without hitting this wall, we made three key changes to the signal path:

### A. Pre-Amp Gain (5x)
Raw RMS values from the audio are typically very low (`0.05` - `0.2`). Driving the motors directly with this results in barely visible movement.
*   **Fix**: Added a **5x Linear Gain** at the input stage.
*   *Effect*: A raw input of `0.2` becomes a usable signal of `1.0`.

### B. Boost Mixing Logic
The previous logic (`RMS * 0.4 + ...`) theoretically penalized sustained notes by attenuating them to 40%.
*   **Fix**: Switched to **Boost Logic**: `Signal = NormRMS * (1.0 + NormOnset)`
*   *Effect*:
    *   **Sustain (Bass/Synth)**: Signal = 100% of Volume.
    *   **Beat (Kick/Snare)**: Signal = Up to 200% of Volume (Multiplier Boost).
    *   This ensures the antennas never drop below the volume level, fixing the "5 degree" movement issue during loud sustained sections.

## 3. Recommended Configuration

To prevent the recurrence of the motor stall while maximizing visual impact, the system must be clamped **just below** the physical limit.

| Parameter | Setting | Reason |
| :--- | :--- | :--- |
| **Antenna Amplitude** | **3.0** ~ **3.05** | *Critical*. Must be slightly less than Pi (3.14) to avoid stalling. **3.15 is unsafe.** |
| **Sensitivity** | **1.0** - **3.0** | User adjustable gain. |
| **SafetyMixer Limit** | **±4.0** | Keep backend safety high so it doesn't interfere; rely on the Config limit (3.05) to protect motors. |

## 4. Summary of Logic
```python
# 1. Gain
norm_rms = raw_rms * 5.0

# 2. Boost Mix
combined = norm_rms * (1.0 + raw_onset)

# 3. User Gain
scaled = combined * sensitivity

# 4. Physical Mapping (The Danger Zone)
# If max_travel > 3.14, MOTORS WILL STALL on full power hits.
splay = min(scaled, 1.0) * max_travel 
```
