"""
MelodyCare — Compare AI-generated output against its DSP ground-truth pair.

For a given source song, this compares:
    /path/to/dsp_target.wav   (the DSP-processed therapeutic target — ground truth)
    /path/to/ai_output.wav    (your model's generated output for the same song)

Produces:
    1. A side-by-side table of the actual Check_result.py numbers for both
       files (not just pass/fail — the real values), so you can see how
       close the AI got to the ground truth on each dimension.
    2. Log-mel spectrogram distance (L1) — lower is more similar.
    3. Cosine similarity of mean log-mel spectra — a coarse "same overall
       tonal/spectral character" score, 1.0 = identical, 0 = unrelated.

Usage:
    python /path/to/training_output/check_inference/compare_dsp_vs_ai.py \
        --dsp_target /path/to/dataset/pass/buong_bui_anh_tuan_lyric_video_d_therapeutic.wav \
        --ai_output   /path/to/training_output/check_inference/test_str65.wav
"""
import argparse

import numpy as np
import librosa

from Check_result import check_file


def log_mel_spectrogram(path, sr=48000, n_mels=128):
    y, _ = librosa.load(path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def compare_spectra(dsp_path, ai_path, sr=48000):
    mel_dsp = log_mel_spectrogram(dsp_path, sr=sr)
    mel_ai  = log_mel_spectrogram(ai_path, sr=sr)

    # Align to the shorter of the two along the time axis
    T = min(mel_dsp.shape[1], mel_ai.shape[1])
    mel_dsp = mel_dsp[:, :T]
    mel_ai  = mel_ai[:, :T]

    l1_distance = float(np.mean(np.abs(mel_dsp - mel_ai)))

    mean_dsp = mel_dsp.mean(axis=1)
    mean_ai  = mel_ai.mean(axis=1)
    cos_sim = float(
        np.dot(mean_dsp, mean_ai) /
        (np.linalg.norm(mean_dsp) * np.linalg.norm(mean_ai) + 1e-9)
    )

    return l1_distance, cos_sim


def print_comparison_table(dsp_result, ai_result):
    checks = {
        "BPM"              : ("bpm", "value"),
        "Spectral Rolloff" : ("spectral_rolloff", "value"),
        "Dynamic Range dB" : ("dynamic_range", "value"),
        "Freq Boost dB"    : ("frequency_boost", "value"),
        "LFO Freq Hz"      : ("lfo_modulation", "value"),
        "Max Pitch Jump"   : ("pitch_intervals", "value"),
        "Legato Overlap %" : ("legato_overlap", "value"),
    }
    print(f"\n{'Metric':<20} {'DSP Target':>15} {'AI Output':>15} {'Diff':>15}")
    print("-" * 68)
    for label, (key, subkey) in checks.items():
        dsp_val = dsp_result.get(key, {}).get(subkey, "?")
        ai_val  = ai_result.get(key, {}).get(subkey, "?")
        try:
            diff = f"{abs(dsp_val - ai_val):.2f}"
        except TypeError:
            diff = "?"
        print(f"{label:<20} {str(dsp_val):>15} {str(ai_val):>15} {diff:>15}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsp_target", type=str, required=True)
    parser.add_argument("--ai_output",  type=str, required=True)
    args = parser.parse_args()

    print(f"Checking DSP target: {args.dsp_target}")
    dsp_result = check_file(args.dsp_target)

    print(f"Checking AI output:  {args.ai_output}")
    ai_result = check_file(args.ai_output)

    print_comparison_table(dsp_result, ai_result)

    print("\nSpectral similarity (log-mel)...")
    l1_distance, cos_sim = compare_spectra(args.dsp_target, args.ai_output)
    print(f"  L1 distance (lower = more similar): {l1_distance:.3f}")
    print(f"  Cosine similarity of mean spectra (1.0 = identical): {cos_sim:.4f}")


if __name__ == "__main__":
    main()