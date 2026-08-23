import numpy as np
import librosa
from scipy.signal import butter, sosfilt, sosfiltfilt

def apply_low_pass_filter(audio_stem: np.ndarray, cutoff_freq=5000.0, sample_rate=48000, filter_order=4):
    """
    Apply a Butterworth low-pass filter to an audio signal to cut off audio > 5kHz.

    Args:
        audio_stem  : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        cutoff_freq : float      — cutoff frequency in Hz (default 5000Hz)
        sample_rate : int        — sample rate of the audio (default 48000)
        filter_order: int        — order of the Butterworth filter (default 4)

    Returns:
        np.ndarray — filtered audio, same shape as input
    """
    # Normalise cutoff to Nyquist (scipy expects a value in [0, 1] where 1 = Nyquist)
    # We use nyquist not the original sr because soundwave is longitudinal wave,
    # when draw on a graph it will have positvie and negative side (like sine graph) so we only take 1 side
    nyquist = sample_rate / 2.0
    normalised_cutoff = cutoff_freq / nyquist  # 5000/24000

    if not (0 < normalised_cutoff < 1):
        raise ValueError(
            f"cutoff_freq ({cutoff_freq} Hz) must be between 0 and Nyquist ({nyquist} Hz)."
        )

    # Design Butterworth filter — use second-order sections (sos) for numerical stability
    # filter_order: how steep will the cut will occur
    # normalise_cutoff: percentage where we start cut off
    # btype: direction for filtering, "low" mean only let low frequency through
    # output = "sos": method of filtering, second order sections filter through many smaller layer -> break down task for more accuracy
    sos = butter(filter_order, normalised_cutoff, btype="low", analog=False, output="sos")

    # Handle mono (1-D) and multi-channel (2-D: channels × samples)
    # sosfilt: apply sos filter
    # .astype: return array same type as audio_stem
    if audio_stem.ndim == 1:
        return sosfilt(sos, audio_stem).astype(audio_stem.dtype)
    else:
        return np.stack(
            [sosfilt(sos, channel).astype(audio_stem.dtype) for channel in audio_stem]
        )

def apply_high_pass_filter(audio_stem: np.ndarray, sample_rate=48000, cutoff_freq=80, filter_order=4):
    """
    Remove the sub-bass frequency range to eliminate feelings of heaviness and pressure.

    Args:
        audio_stem  : np.ndarray — Input audio signal.
        sample_rate : int        — Sampling rate (default 48000).
        cutoff_freq : float      — Cutoff frequency (default 80Hz).
        filter_order: int        — Filter order (cutoff slope, 4 is sufficiently deep).
    """
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff_freq / nyquist

    # Using high-pass filter to cleanup frequency below 80Hz
    sos = butter(filter_order, normal_cutoff, btype='highpass', analog=False, output='sos')
    
    def _apply(channel):
        # Use sosfiltfilt to filter out zero-phase, avoid phase distortion of the original signal
        return sosfiltfilt(sos, channel).astype(audio_stem.dtype)

    if audio_stem.ndim == 1:
        return _apply(audio_stem)
    else:
        return np.stack([_apply(channel) for channel in audio_stem])


def apply_parametric_eq(audio_stem: np.ndarray, sample_rate=48000, center_freq=250, gain_db=1.5, bandwidth=300, filter_order=2):
    """
    Apply a parametric EQ peak filter to boost the 100Hz-400Hz range.
    100Hz-400Hz are deep, familiar sounds which are ideal for therapy music.

    Args:
        audio_stem  : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        sample_rate : int        — sample rate of the audio (default 48000)
        center_freq : float      — center frequency of the boost in Hz (default 250Hz, midpoint of 100-400Hz)
        gain_db     : float      — amount of boost in decibels (default 3dB)
        bandwidth   : float      — width of the frequency band to boost in Hz (default 300Hz, covers 100-400Hz)
        filter_order: int        — order of the Butterworth band-pass filter (default 2)

    Returns:
        np.ndarray — EQ-applied audio, same shape as input
    """
    nyquist = sample_rate / 2.0

    low  = (center_freq - bandwidth / 2) / nyquist
    high = (center_freq + bandwidth / 2) / nyquist

    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError(
            f"Band edges must be between 0 and Nyquist ({nyquist}Hz). "
            f"Got low={low * nyquist:.1f}Hz, high={high * nyquist:.1f}Hz."
        )
    if low >= high:
        raise ValueError(
            f"low edge ({low * nyquist:.1f}Hz) must be less than high edge ({high * nyquist:.1f}Hz)."
        )

    sos = butter(filter_order, [low, high], btype="bandpass", analog=False, output="sos")

    # ── Adaptive gain: reduce boost if original is already warm ──────
    # Use bandpass RMS instead of STFT — much faster, sufficient accuracy
    effective_gain_db = gain_db
    try:
        audio_mono = audio_stem[0] if audio_stem.ndim > 1 else audio_stem

        # Measure energy in target band (100-400Hz) vs reference band (400-2000Hz)
        # using bandpass filters — O(N) instead of O(N log N) for STFT
        sos_target = butter(2, [100/nyquist, 400/nyquist],   btype="band", output="sos") # Low range
        sos_ref    = butter(2, [400/nyquist, 2000/nyquist],  btype="band", output="sos") # Mid range

        target_rms    = np.sqrt(np.mean(sosfilt(sos_target, audio_mono) ** 2))
        ref_rms       = np.sqrt(np.mean(sosfilt(sos_ref,    audio_mono) ** 2))
        orig_boost_db = 20 * np.log10((target_rms + 1e-9) / (ref_rms + 1e-9))

        # If original already has strong low-mid energy (> +5dB above reference/ mid range),
        # scale down the boost to avoid making the audio too muddy/heavy
        if orig_boost_db > 5.0:
            effective_gain_db = max(0.0, gain_db * (1.0 - (orig_boost_db - 5.0) / 5.0))
    except Exception:
        pass  # fallback to original gain_db if measurement fails

    gain_linear = 10 ** (effective_gain_db / 20)

    def _apply(channel):
        # sosfiltfilt: zero-phase filtering (forward + backward pass)
        # eliminates phase shift so boosted band aligns perfectly with original
        # sosfilt (single pass) introduces phase delay → subtle comb filtering when added back
        band_zero_phase = sosfiltfilt(sos, channel)
        boosted = channel + (gain_linear - 1.0) * band_zero_phase
        return boosted.astype(audio_stem.dtype)

    if audio_stem.ndim == 1:
        return _apply(audio_stem)
    else:
        return np.stack([_apply(channel) for channel in audio_stem])
    
def normalize_amplitude(audio_stem: np.ndarray, peak_db=-1.0):
    """
    Normalize the final mixed audio to a target peak level in dB.
    Prevents digital clipping while maximizing loudness.

    -1.0dB (not 0dB) is used as the ceiling to leave a small headroom
    buffer, which is standard practice in audio mastering to avoid
    clipping during file encoding/conversion.

    Args:
        audio_stem : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        peak_db    : float      — target peak level in dB (default -1.0dB)

    Returns:
        np.ndarray — normalized audio, same shape as input
    """
    # Convert target dB to linear scale
    # -1.0dB → 10^(-1/20) ≈ 0.891 (89.1% of maximum possible amplitude)
    target_peak_linear = 10 ** (peak_db / 20)

    # Find the current loudest sample across all channels
    current_peak = np.max(np.abs(audio_stem))

    # Guard against completely silent audio (avoid division by zero)
    if current_peak == 0:
        return audio_stem

    # Scale entire audio so the loudest sample hits exactly target_peak_linear
    return (audio_stem * (target_peak_linear / current_peak)).astype(audio_stem.dtype)
