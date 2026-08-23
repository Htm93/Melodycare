import librosa
import numpy as np


# =====================================================================
# MODULE 1: FEATURE ANALYSIS & COMPLIANCE
# =====================================================================

def analyze_audio_features(audio_stem: np.ndarray, original_audio: np.ndarray, sample_rate):
    """
    Analyze key musical properties of an audio stem to determine
    whether it already meets ASD therapy requirements.

    Args:
        audio_stem  : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        sample_rate : int        — sample rate of the audio

    Returns:
        dict with keys:
            "bpm"           : float — detected tempo in beats per minute
            "max_pitch_jump": float — largest pitch interval found in the audio (semitones)
            "is_bpm_valid"  : bool  — True if BPM is within ASD therapy range (60–80)
    """
    # Use mono channel for analysis — take left channel if stereo
    audio_mono = audio_stem[0] if audio_stem.ndim > 1 else audio_stem
    original_audio_mono = original_audio[0] if original_audio.ndim > 1 else original_audio

    # Detect BPM using beat tracking
    # beat_track returns (bpm, beat_frames) — we only need bpm
    bpm, _ = librosa.beat.beat_track(y=original_audio_mono, sr=sample_rate)
    bpm = float(np.squeeze(bpm))

    # Bug fix: use librosa.pyin to detect pitch for max_pitch_jump
    # Previous code used time_stretch which returns audio, not pitch intervals
    f0, voiced_flag, _ = librosa.pyin(
        y=audio_mono,
        fmin=65.4,    # C2
        fmax=2093.0,  # C7
        sr=sample_rate
    )

    # Clean NaN values from unvoiced frames
    f0 = np.nan_to_num(f0)

    # Convert Hz to MIDI semitones for interval calculation
    semitones = np.zeros_like(f0)
    valid_idx = f0 > 0
    if np.any(valid_idx):
        semitones[valid_idx] = librosa.hz_to_midi(f0[valid_idx])

    # Calculate pitch jumps between consecutive frames
    jumps = np.diff(semitones)

    # Only consider jumps where both frames have actual pitch (voiced)
    voiced_jumps = np.abs(jumps[voiced_flag[:-1] & voiced_flag[1:]])
    max_pitch_jump = float(np.max(voiced_jumps)) if len(voiced_jumps) > 0 else 0.0

    return {
        "bpm"           : bpm,
        "max_pitch_jump": max_pitch_jump,
        "is_bpm_valid"  : (60.0 <= bpm <= 80.0)
    }


def enforce_bpm(audio_stem: np.ndarray, current_bpm, target_bpm=60):
    """
    Time-stretch the audio to bring its tempo to the ASD therapy target BPM.
    If the current BPM is already within the 60–80 range, the audio is returned unchanged.

    Args:
        audio_stem  : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        current_bpm : float      — detected BPM of the input audio
        target_bpm  : float      — desired output BPM (default 60, lower bound of therapy range)

    Returns:
        np.ndarray — time-stretched audio at target BPM, or original if already valid
    """
    if 60 > round(current_bpm, 1) or  round(current_bpm, 1) > 80:
        stretch_ratio = target_bpm / current_bpm

        # Handle mono and stereo separately — time_stretch requires 1D input
        if audio_stem.ndim == 1:
            return librosa.effects.time_stretch(audio_stem, rate=stretch_ratio)
        else:
            return np.stack([
                librosa.effects.time_stretch(channel, rate=stretch_ratio)
                for channel in audio_stem
            ])

    return audio_stem