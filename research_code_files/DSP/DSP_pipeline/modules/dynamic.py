import numpy as np
import config
from scipy.ndimage import uniform_filter1d


def _rms_db_envelope(channel: np.ndarray, sample_rate: int) -> np.ndarray:
    """Short-term (20ms) RMS loudness envelope of `channel`, expressed in dB."""
    window_size = int(sample_rate * 0.02)
    squared = channel ** 2
    mean_squared = uniform_filter1d(squared, size=window_size, mode='reflect')
    rms_envelope = np.sqrt(np.maximum(mean_squared, 0))
    # 1e-9 offset prevents mathematical log(0) errors on pure silence
    return 20 * np.log10(rms_envelope + 1e-9)


def apply_dynamic_compression(
    audio_stem: np.ndarray,
    sample_rate: int = 48000,
    max_variance_db: float = 10.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    gate_offset_db: float = 45.0,
    max_boost_db: float = 18.0,
    max_passes: int = 5,
) -> np.ndarray:
    """
    Deterministic Dynamic Range Compression.
    Forces the RMS envelope variance to stay within max_variance_db using a
    bidirectional Hard-Knee mapping algorithm, closed-loop verified against
    the actual measured result rather than assumed from a single pass.

    BUGFIXES vs the original implementation (root-caused against
    therapy_compliance_report.csv, where compressed tracks still measured
    13-15.7dB despite targeting 12dB):

    1. GATING (root cause of most of the overshoot): percentiles/bounds
       were computed over EVERY sample, including near-silent decay tails
       and gaps between notes (down to -100/-150dB). The lower-bound
       "boost" step tried to lift those near-silent samples by 50-100+dB
       to reach the target floor. That's not physically meaningful audio,
       it's noise-floor amplification -- and because the required gain
       was so extreme, the attack/release smoothing (which is time-based,
       not target-based) could never fully catch up before the next note
       began. Those never-fully-corrected samples then dragged the p5
       percentile down, inflating the *measured* variance right back up
       even though compression had "run". Fix: gate out samples more than
       `gate_offset_db` below the track's peak before computing
       percentiles/bounds/gain -- true silence is left untouched, and only
       audible program material is shaped.
    2. BOOST CAP: even for legitimately quiet (but audible) passages, the
       requested boost is now capped at `max_boost_db` so the compressor
       never tries to manufacture large gain out of near-nothing.
    3. CLOSED-LOOP VERIFICATION: a single smoothing pass is a *model* of
       the result, not a guarantee -- attack/release lag means the actual
       achieved variance can still land above target. This function now
       re-measures the real output after each pass and, if still
       non-compliant, applies another pass with a tightened internal
       target, up to `max_passes` times. This mirrors the strict
       measure -> correct -> re-measure discipline used in smoothing.py's
       legato paths, instead of trusting the math in a single shot.
    """

    def _compress_pass(channel: np.ndarray, target_variance_db: float):
        rms_db = _rms_db_envelope(channel, sample_rate)
        peak = np.max(np.abs(channel))
        peak_db = 20 * np.log10(peak + 1e-9)

        # ── Gate: ignore true silence / decay-tails when measuring and
        # correcting. Only audio within `gate_offset_db` of the track's
        # own peak is considered "program material".
        gate_db = peak_db - gate_offset_db
        active_mask = rms_db > gate_db
        if np.count_nonzero(active_mask) < 10:
            return channel, 0.0

        active_db = rms_db[active_mask]
        p95 = np.percentile(active_db, 95)
        p5 = np.percentile(active_db, 5)
        variance = p95 - p5

        if variance <= target_variance_db:
            # Already compliant -- nothing to do this pass
            return channel, variance

        # ── Hard-Knee Boundaries around the (gated) median ──
        median_db = np.median(active_db)
        half_range = target_variance_db / 2.0
        upper_bound = median_db + half_range
        lower_bound = median_db - half_range

        # ── Generate Gain Map (only over active/gated samples) ──
        gain_db = np.zeros_like(rms_db)
        over = active_mask & (rms_db > upper_bound)
        under = active_mask & (rms_db < lower_bound)
        gain_db[over] = upper_bound - rms_db[over]
        # Cap the boost so we never try to amplify near-silence into range
        gain_db[under] = np.minimum(lower_bound - rms_db[under], max_boost_db)

        # ── Smooth Gain Changes (Attack / Release) ──
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))

        smoothed_gain_db = np.zeros_like(gain_db)
        for i in range(1, len(gain_db)):
            coeff = attack_coeff if gain_db[i] < smoothed_gain_db[i - 1] else release_coeff
            smoothed_gain_db[i] = coeff * smoothed_gain_db[i - 1] + (1 - coeff) * gain_db[i]

        gain_linear = 10 ** (smoothed_gain_db / 20.0)
        compressed = channel * gain_linear
        return compressed, variance

    def _measure_gated_variance(channel: np.ndarray) -> float:
        rms_db = _rms_db_envelope(channel, sample_rate)
        peak_db = 20 * np.log10(np.max(np.abs(channel)) + 1e-9)
        active = rms_db[rms_db > peak_db - gate_offset_db]
        if len(active) < 10:
            return 0.0
        return float(np.percentile(active, 95) - np.percentile(active, 5))

    def _compress(channel):
        original_peak = np.max(np.abs(channel))
        current = channel.astype(np.float64)
        target = max_variance_db

        for _pass in range(max_passes):
            current, _ = _compress_pass(current, target)
            achieved_variance = _measure_gated_variance(current)

            if achieved_variance <= max_variance_db:
                break

            # Smoothing lag left residual overshoot -- tighten the internal
            # target and run another corrective pass on the ACTUAL result
            # (not the theoretical one), rather than assuming success.
            target = max(2.0, target - 2.0)

        # Restore the original peak amplitude (a uniform linear scale,
        # which shifts every dB value by the same constant and therefore
        # does not change the achieved variance/spread).
        compressed_peak = np.max(np.abs(current))
        if compressed_peak > 0:
            current = current * (original_peak / compressed_peak)

        return current.astype(audio_stem.dtype)

    # Dispatch to handle both mono (1D) and stereo (2D) arrays natively
    if audio_stem.ndim == 1:
        return _compress(audio_stem)
    else:
        return np.stack([_compress(channel) for channel in audio_stem])


def apply_lfo_modulation(audio_stem: np.ndarray, sample_rate=48000, rate_hz=0.08,
                          depth=0.05, phase_deg=0.0):
    """
    Apply a Low Frequency Oscillator (LFO) to create a slow, gentle
    therapeutic pulsing effect on the audio volume.

    LFO is a sine wave oscillating between 0.05Hz and 0.1Hz — so slow
    that it is not perceived as a separate sound, but rather as a subtle
    breathing-like rhythm in the music. This mimics natural biological
    rhythms (breathing ~0.25Hz, heartbeat ~1Hz) at an even slower pace,
    promoting a calming, predictable sensation for individuals with ASD.

    Example at rate_hz=0.08:
        1 full cycle = 1/0.08 = 12.5 seconds
        → volume gently rises and falls every 12.5 seconds
        → too slow to hear as vibration, felt as "breathing" in the music

    Args:
        audio_stem  : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        sample_rate : int        — sample rate of the audio (default 48000)
        rate_hz     : float      — LFO frequency in Hz, must be 0.05–0.1 for ASD therapy (default 0.08)
        depth       : float      — how much the volume fluctuates (default 0.05 = ±5% volume change)
                                   0.0 = no modulation, 1.0 = volume drops to complete silence
        phase_deg   : float      — starting point of the sine wave in degrees (default 0.0)
                                   0.0   = start at middle, rising
                                   90.0  = start at peak volume
                                   270.0 = start at lowest volume

    Returns:
        np.ndarray — modulated audio, same shape as input
    """
    # ── 1. Validate LFO rate is within ASD therapy range ──
    # Rate outside 0.05–0.1Hz risks being perceived as tremolo (too fast)
    # or having no therapeutic effect (too slow)
    if not (0.05 <= rate_hz <= 0.1):
        raise ValueError(
            f"rate_hz ({rate_hz}Hz) must be between 0.05Hz and 0.1Hz for ASD therapy. "
            f"Below 0.05Hz has no perceptible effect. Above 0.1Hz sounds like tremolo."
        )

    # ── 2. Validate depth is within safe range ──
    # depth=1.0 would cause complete silence at the trough of the sine wave
    # which is jarring and counter-productive for ASD therapy
    if not (0.0 <= depth <= 0.3):
        raise ValueError(
            f"depth ({depth}) must be between 0.0 and 0.3. "
            f"Above 0.3 creates noticeable volume dips that may startle."
        )

    # ── 3. Generate Time Array ──
    # Create an array of time values, one per audio sample
    # Ex: sample_rate=48000, 1 second of audio → t = [0, 1/48000, 2/48000, ..., 1.0]
    num_samples = audio_stem.shape[-1]  # works for both (samples,) and (channels, samples)
    t = np.arange(num_samples) / sample_rate

    # ── 4. Convert Phase from Degrees to Radians ──
    # np.sin() expects radians, not degrees
    # 0°   → 0 rad    (sine starts at 0, rising)
    # 90°  → π/2 rad  (sine starts at peak = 1.0)
    # 180° → π rad    (sine starts at 0, falling)
    # 270° → 3π/2 rad (sine starts at trough = -1.0)
    phase_rad = np.deg2rad(phase_deg)

    # ── 5. Generate LFO Sine Wave ──
    # Standard sine wave formula: sin(2π × frequency × time + phase)
    # Output range: -1.0 to +1.0
    # At rate_hz=0.08: completes 1 full cycle every 12.5 seconds
    sine_wave = np.sin(2 * np.pi * rate_hz * t + phase_rad)

    # ── 6. Scale Sine Wave to Volume Multiplier ──
    # Raw sine oscillates -1.0 to +1.0 → would make audio go silent or flip phase
    # We want subtle volume flutter, not silence
    #
    # Formula: 1.0 + (depth × sine)
    # depth=0.05:
    #   sine at  1.0 → lfo = 1.0 + 0.05 = 1.05  (5% louder at peak)
    #   sine at  0.0 → lfo = 1.0 + 0.00 = 1.00  (unchanged at midpoint)
    #   sine at -1.0 → lfo = 1.0 - 0.05 = 0.95  (5% quieter at trough)
    #
    # Result: volume gently breathes between 0.95× and 1.05× original
    lfo = 1.0 + (depth * sine_wave)

    # ── SAFEGUARD: Dimension Alignment ──
    # If audio is stereo (2, samples), reshape LFO to (2, 1) to ensure 
    # it broadcasts correctly across both channels independently.
    if audio_stem.ndim > 1:
        # Reshape LFO from (samples,) to (1, samples) so it matches (channels, samples)
        lfo = lfo[np.newaxis, :]

    # ── 7. Apply LFO to Audio ──
    # Multiply every audio sample by its corresponding LFO value
    # Mono:   (samples,)   × (samples,)  → (samples,)
    # Stereo: (2, samples) × (samples,)  → (2, samples)  numpy broadcasts automatically
    modulated = audio_stem * lfo

    return modulated.astype(audio_stem.dtype)