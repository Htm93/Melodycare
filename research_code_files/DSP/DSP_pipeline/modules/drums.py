import numpy as np
import librosa

def reduce_transient_attack(audio_stem: np.ndarray, sample_rate=48000, attack_ms=30.0, ratio=6.0):
    """
    Soften the sharp initial impact (transient) of each drum hit to remove
    harsh, startling sounds that are particularly overstimulating for
    individuals with ASD, while preserving the underlying rhythmic body
    of the drum sound (the thump after the crack).

    How it works:
        A drum hit has 2 phases:
            Transient: the sharp initial crack/click (first 5–30ms)  ← we reduce this
            Body:      the resonant thump that follows               ← we keep this

        By detecting each drum hit onset and applying a short gain ramp
        at the very start of each hit, the crack is softened into a
        gentle swell rather than a sudden impact.

    Args:
        audio_stem : np.ndarray — audio samples, shape (samples,) or (channels, samples)
        sample_rate: int        — sample rate of the audio (default 48000)
        attack_ms  : float      — duration of the transient softening ramp in ms (default 30ms)
                                  shorter = only the very initial crack is softened
                                  longer  = more of the hit body is also softened
        ratio      : float      — how aggressively the transient is reduced (default 6.0)
                                  1.0 = no reduction
                                  6.0 = transient reduced to ~1/6 of original level
                                  higher values make drums sound more like brushed/padded hits

    Returns:
        np.ndarray — transient-softened audio, same shape as input
    """
    # Bug fix: _reduce moved inside reduce_transient_attack so it can access
    # sample_rate, attack_ms, ratio, audio_stem.dtype from the enclosing scope
    # Previously _reduce was at module level with no access to these variables
    def _reduce(channel):
        # ── 1. Detect Drum Hit Onsets ──
        # librosa.onset.onset_detect finds the exact sample where each drum hit begins
        # units='samples' returns raw sample indices instead of frame indices
        onsets = librosa.onset.onset_detect(y=channel, sr=sample_rate, units='samples')

        # ── 2. Calculate Transient Window Size ──
        # Convert attack_ms to number of samples
        # Ex: 30ms at 48000Hz = 30 * 48000 / 1000 = 1440 samples
        attack_samples = int(sample_rate * attack_ms / 1000.0)

        # Make a copy so we don't modify the original array in place
        output = channel.copy()

        # ── 3. Process Each Drum Hit ──
        for onset in onsets:
            # Calculate the end of this transient window
            # Clamp to array length in case the last hit is near the end of the file
            window_end = min(onset + attack_samples, len(channel))

            # ── 4. Generate Transient Suppression Envelope ──
            # Create a gain curve that starts very low (1/ratio) and rises to 1.0
            # over the length of the attack window
            #
            # Ex: ratio=6, attack_samples=1440
            #   start_gain = 1/6 ≈ 0.167  (transient reduced to 16.7% of original)
            #   end_gain   = 1.0           (body of hit fully restored)
            #   curve = [0.167, 0.168, ..., 0.999, 1.0]  (linear ramp up)
            #
            # This makes each drum hit "swell in" instead of "snap in"
            window_length = window_end - onset
            envelope = np.linspace(1.0 / ratio, 1.0, window_length)

            # ── 5. Apply Envelope to Transient Region ──
            # Multiply only the first attack_samples of each drum hit by the ramp
            # Everything after the ramp window plays at full volume (body is preserved)
            output[onset:window_end] *= envelope

        return output.astype(audio_stem.dtype)

    # ── Handle Mono and Stereo ──
    if audio_stem.ndim == 1:
        return _reduce(audio_stem)
    else:
        return np.stack([_reduce(channel) for channel in audio_stem])
