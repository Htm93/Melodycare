"""
MelodyCare — Musical Content Comparison (chroma + MFCC via DTW)

The coarse "cosine similarity of mean spectra" in compare_dsp_vs_ai.py
averages the ENTIRE track into a single vector before comparing — this can
give a misleadingly high similarity score for two pieces that have a
similar overall frequency balance (e.g. both bass-heavy) while sounding
completely different in melody, instrumentation, and structure, since
averaging over time throws away almost everything that makes two pieces of
music actually sound alike or different.

This script instead compares:
    - Chroma features (which notes/harmonies are present, over time) —
      melody/harmony content, roughly instrument-timbre-invariant.
    - MFCCs (timbral/instrument character, over time).
Both compared via DTW (dynamic time warping), which finds the best
time-alignment between the two sequences and measures how well the actual
musical content lines up — sensitive to real content differences, not just
an averaged EQ shape.

Usage:
    python compare_music_content.py \
        --dsp_target /path/to/dataset/pass/<song>_therapeutic.wav \
        --ai_output  /path/to/training_output/check_inference/test_str65.wav
"""
import argparse

import numpy as np
import librosa


def load_mono(path, sr=48000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def path_irregularity(wp, len1, len2):
    """
    Measures how far the DTW warping path deviates from a straight diagonal
    (the "natural", no-distortion alignment). High deviation means DTW is
    doing a lot of stretching/compressing/reordering to find matches — the
    kind of timing distortion that sounds wrong to a real-time listener
    even when it doesn't hurt frame-level similarity scores (since DTW gets
    to freely re-align, but a listener doesn't).

    Returns mean absolute deviation from the diagonal, normalized to
    [0, 1] roughly (0 = perfectly natural/diagonal alignment, higher =
    more distortion/warping happening).
    """
    wp = np.array(wp[::-1])  # chronological order
    # Expected diagonal position for each step, normalized to [0,1] on both axes
    expected_j = wp[:, 0] * (len2 / len1)
    deviation = np.abs(wp[:, 1] - expected_j) / max(len1, len2)
    return deviation.mean()


# DTW's cost matrix is O(N*M) in memory — on a ~9-minute track at librosa's
# default hop_length=512, that's a ~50,000 x 50,000 matrix, ~19GB for a
# SINGLE comparison. This caused real crashes (severe enough to trigger the
# Linux OOM-killer, taking down far more than just this script — see
# conversation history). Fixed two ways:
#   1. Much larger hop_length (4096, ~8x fewer frames -> ~64x less memory)
#   2. Sakoe-Chiba banding (global_constraints=True) restricts DTW's search
#      to a band around the diagonal — musically sensible for comparing two
#      versions of the same song (no reason to allow wild non-local
#      alignment), and cuts memory/time further on top of the hop_length fix.
DTW_HOP_LENGTH = 4096
DTW_BAND_RAD = 0.25  # fraction of sequence length allowed to deviate from diagonal


def chroma_dtw_similarity(y1, y2, sr):
    """
    Compare melodic/harmonic content over time via chroma + DTW.
    Returns (normalized_dtw_cost, per_frame_similarity_stats, path_irregularity).
    Lower normalized_dtw_cost = more similar melodic/harmonic content.
    """
    chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=DTW_HOP_LENGTH)
    chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=DTW_HOP_LENGTH)
    print(f"  Chroma frames: {chroma1.shape[1]} x {chroma2.shape[1]} "
          f"(~{chroma1.shape[1]*chroma2.shape[1]*8/1e6:.1f}MB DTW matrix)")

    D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2, metric="cosine",
                                  global_constraints=True, band_rad=DTW_BAND_RAD)
    normalized_cost = D[-1, -1] / len(wp)
    irregularity = path_irregularity(wp, chroma1.shape[1], chroma2.shape[1])

    wp_fwd = wp[::-1]
    sims = []
    for i, j in wp_fwd:
        v1, v2 = chroma1[:, i], chroma2[:, j]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        sims.append(sim)
    sims = np.array(sims)

    return normalized_cost, sims, irregularity


def mfcc_dtw_similarity(y1, y2, sr, n_mfcc=20):
    """
    Compare timbral/instrumentation content over time via MFCC + DTW.
    Returns (normalized_dtw_cost, per_frame_similarity_stats).
    """
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=n_mfcc, hop_length=DTW_HOP_LENGTH)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=n_mfcc, hop_length=DTW_HOP_LENGTH)
    print(f"  MFCC frames: {mfcc1.shape[1]} x {mfcc2.shape[1]} "
          f"(~{mfcc1.shape[1]*mfcc2.shape[1]*8/1e6:.1f}MB DTW matrix)")

    D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric="cosine",
                                  global_constraints=True, band_rad=DTW_BAND_RAD)
    normalized_cost = D[-1, -1] / len(wp)

    wp = wp[::-1]
    sims = []
    for i, j in wp:
        v1, v2 = mfcc1[:, i], mfcc2[:, j]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        sims.append(sim)
    sims = np.array(sims)

    return normalized_cost, sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsp_target", type=str, required=True)
    parser.add_argument("--ai_output", type=str, required=True)
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    print(f"Loading: {args.dsp_target}")
    y1 = load_mono(args.dsp_target, sr=args.sr)
    print(f"Loading: {args.ai_output}")
    y2 = load_mono(args.ai_output, sr=args.sr)

    print("\nComputing chroma (melody/harmony content) DTW comparison...")
    chroma_cost, chroma_sims, chroma_irregularity = chroma_dtw_similarity(y1, y2, args.sr)

    print("Computing MFCC (timbre/instrumentation content) DTW comparison...")
    mfcc_cost, mfcc_sims = mfcc_dtw_similarity(y1, y2, args.sr)

    print("\n" + "=" * 60)
    print("MUSICAL CONTENT COMPARISON (time-aware, not averaged)")
    print("=" * 60)
    print(f"\nChroma (melody/harmony) — lower cost = more similar musical content:")
    print(f"  Normalized DTW cost: {chroma_cost:.4f}")
    print(f"  Per-aligned-frame similarity — mean: {chroma_sims.mean():.4f}, "
          f"median: {np.median(chroma_sims):.4f}, "
          f"% frames >0.7 similar: {(chroma_sims > 0.7).mean()*100:.1f}%")
    print(f"  Warping path irregularity: {chroma_irregularity:.4f} "
          f"(near 0 = natural alignment; higher = DTW is stretching/reordering a lot")
    print(f"    to find matches — this can mean the high similarity above is misleading,")
    print(f"    since a real listener doesn't get DTW's free re-alignment)")

    print(f"\nMFCC (timbre/instrumentation) — lower cost = more similar instrument character:")
    print(f"  Normalized DTW cost: {mfcc_cost:.4f}")
    print(f"  Per-aligned-frame similarity — mean: {mfcc_sims.mean():.4f}, "
          f"median: {np.median(mfcc_sims):.4f}, "
          f"% frames >0.7 similar: {(mfcc_sims > 0.7).mean()*100:.1f}%")

    print("\n" + "-" * 60)
    print("Interpretation:")
    print("  These measure actual musical content over time (melody/harmony")
    print("  and instrument timbre), not an averaged frequency-balance shape.")
    print("  If these numbers look poor (low mean similarity, low % >0.7)")
    print("  despite a high 'cosine similarity of mean spectra' from")
    print("  compare_dsp_vs_ai.py, that CONFIRMS the earlier metric was")
    print("  misleading you — the two tracks share a rough EQ balance but")
    print("  not the actual musical content, matching what you're hearing.")
    print("-" * 60)


if __name__ == "__main__":
    main()