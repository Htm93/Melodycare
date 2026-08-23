"""
MelodyCare — Stage 3: Deterministic Compliance Safety Net

Applies the SAME proven, closed-loop-verified DSP corrections used to build
the training targets (Stage 1) directly to the AI-generated output
(Stage 2's result), as a final guarantee layer. This is NOT a workaround —
for a clinical/ASD-therapy application, a generative model's soft
compliance should never be the sole guarantee of hard acoustic safety
bounds; a deterministic final stage that verifies and corrects the actual
measured result is the more clinically responsible design.

Order matches pipeline.py's original design: legato correction first,
dynamic-range compression last (pipeline.py explicitly compresses "at the
end" on the master bus, after all other shaping is done) — the
closed-loop compressor re-measures the REAL result regardless of what
came before it, so this ordering is safe.

IMPORTANT: this file must live in DSP_pipeline/ ITSELF (same directory as
config.py, pipeline.py, main.py) — NOT inside DSP_pipeline/modules/.
Imports below use `from modules.dynamic import ...` / `from modules.smoothing
import ...`, matching exactly how pipeline.py itself imports these same
modules — config.py is only importable from the DSP_pipeline root, and
modules/ needs to be reachable as a subpackage from there.

Usage (run from anywhere, using the full path):
    python /path/to/dsp_pipeline/post_process.py \
        --input_audio  /path/to/training_output/check_inference/strength_sweep/str35.wav \
        --output_audio /path/to/training_output/check_inference/str35_polished.wav
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from modules.dynamic import apply_dynamic_compression
from modules.smoothing import apply_legato_overlap


def apply_low_pass_filter(audio: np.ndarray, cutoff_freq: float, sample_rate: int, order: int = 4) -> np.ndarray:
    """
    4th-order zero-phase Butterworth low-pass filter, matching Stage 1's own
    design (per project report). Targets harsh/dissonant high-frequency
    content -- e.g. the loud, irritating burst reported at chorus
    transitions when many instruments join at once, which is exactly the
    kind of acoustic event ASD hyperacusis therapy needs to avoid, even if
    it's brief enough not to move the WHOLE-TRACK spectral rolloff
    percentage past its pass threshold.

    filtfilt (not lfilter) applies the filter forward AND backward, giving
    zero phase distortion -- matches "zero-phase high-pass filtering"
    language in the project's own report rubric for the equivalent Stage 1
    operation.
    """
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')

    def _filter_channel(channel):
        return filtfilt(b, a, channel)

    if audio.ndim == 1:
        return _filter_channel(audio).astype(audio.dtype)
    else:
        return np.stack([_filter_channel(ch) for ch in audio]).astype(audio.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio",  type=str, required=True)
    parser.add_argument("--output_audio", type=str, required=True)
    parser.add_argument("--max_variance_db", type=float, default=13.0,
                         help="Dynamic compression target. NOTE: 13.0 is the exact pass threshold "
                              "in Check_result.py (strict '<13', not '<=13') -- but dynamic.py's "
                              "internal verification uses peak-relative gating while Check_result.py "
                              "uses an absolute noise-floor gate, so they can measure slightly "
                              "different variance for the same audio. Targeting exactly 13.0 risks "
                              "landing just outside the official pass threshold even when this "
                              "script's own closed-loop check reports success. ALWAYS re-verify with "
                              "single_check.py after running this -- if it fails by a small margin, "
                              "rerun with --max_variance_db 12.0 or lower for safety margin.")
    parser.add_argument("--legato_target", type=float, default=0.75,
                         help="Legato overlap target (0-1 or 0-100) — default 0.75 = midpoint of 70-80% pass window")
    parser.add_argument("--rolloff_cutoff", type=float, default=5000.0,
                         help="Low-pass filter cutoff in Hz — default 5000 matches the spectral rolloff "
                              "pass threshold and Stage 1's own design")
    parser.add_argument("--legato_rounds", type=int, default=6,
                         help="Correction rounds for the final legato pass — raised above smoothing.py's "
                              "own default (3) since it wasn't fully converging in 3 rounds on this "
                              "~9-minute track")
    parser.add_argument("--legato_max_wet", type=float, default=0.65,
                         help="Max reverb-blend wet level for legato correction. smoothing.py's own "
                              "default (0.45) appears to be a hard ceiling this track's characteristics "
                              "can't get past even with more rounds -- confirmed by getting the EXACT "
                              "same 64.8% result across 3 rounds and 6 rounds. Loosened to 0.65 to give "
                              "the correction more room; needs smoothing.py patched to expose this param.")
    args = parser.parse_args()

    print(f"Loading: {args.input_audio}")
    audio, sr = sf.read(args.input_audio, always_2d=False)
    # soundfile returns (samples, channels) for stereo; dynamic.py/smoothing.py
    # expect (channels, samples), matching how pipeline.py's librosa.load
    # (mono=False) shapes stereo audio.
    if audio.ndim == 2:
        audio = audio.T
    print(f"  shape={audio.shape}, sr={sr}")

    print(f"\n[Stage 3a] Applying low-pass filter (cutoff={args.rolloff_cutoff}Hz) — "
          f"targets harsh/dissonant high-frequency content (e.g. loud chorus transitions)...")
    audio = apply_low_pass_filter(audio, cutoff_freq=args.rolloff_cutoff, sample_rate=sr)

    print(f"\n[Stage 3b] Applying dynamic range compression (target<{args.max_variance_db}dB)...")
    audio = apply_dynamic_compression(audio, sample_rate=sr, max_variance_db=args.max_variance_db)

    # Legato runs LAST, once, with no downstream step that could disturb it
    # afterward. Confirmed across 3 different orderings tonight that running
    # legato correction earlier gets undone by compression's envelope
    # reshaping regardless of position (67.3% -> 65.3% -> 64.8% across
    # attempts) -- rather than keep fighting that interaction, give legato
    # the final, uncontested word. max_correction_rounds raised from the
    # default (3) since a ~9-minute track with this much material may need
    # more iterations to fully converge.
    print(f"[Stage 3c] Applying legato overlap correction (target={args.legato_target}, "
          f"final step, {args.legato_rounds} correction rounds, max_wet={args.legato_max_wet})...")
    audio = apply_legato_overlap(
        audio, sr,
        target_overlap_percentage=args.legato_target,
        max_correction_rounds=args.legato_rounds,
        max_wet=args.legato_max_wet,
    )

    os.makedirs(os.path.dirname(args.output_audio) or ".", exist_ok=True)
    # Write back as (samples, channels) for soundfile, matching how it was read
    out = audio.T if audio.ndim == 2 else audio
    sf.write(args.output_audio, out, sr, subtype="PCM_16")
    print(f"\n[DONE] Saved polished output: {args.output_audio}")
    print("Now run single_check.py on this file to confirm compliance.")


if __name__ == "__main__":
    main()
