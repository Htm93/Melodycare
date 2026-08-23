"""
MelodyCare — Strength Sweep Analysis (generation-free)

Takes audio files you've ALREADY generated manually (one per --strength
value, via direct inference.py calls) and automates everything after that:
compliance checking (single_check.py logic) + musical content similarity
(compare_music_content.py logic) + one summary table.

WHY SPLIT FROM GENERATION: the original strength_sweep.py's subprocess
calls to inference.py failed silently and repeatedly (see conversation
history — likely leftover/leaked GPU memory from earlier crashed processes
this session), while the SAME command run directly, standalone, worked
every time. The compliance/content checks below never touch subprocess or
GPU at all (pure CPU/librosa), so they don't have that failure mode —
automating just this part is reliable, automating generation wasn't.

Usage — auto-discover files named str<NN>.wav in a directory, where NN is
strength*100 (e.g. str20.wav = strength 0.20, str65.wav = strength 0.65):
    python /path/to/training_output/result_data_check/analyze_strength_result.py \
        --results_dir /path/to/training_output/check_inference/strength_sweep \
        --dsp_target  /path/to/dataset/pass/buong_bui_anh_tuan_lyric_video_d_therapeutic.wav

Or specify files explicitly:
    python analyze_strength_result.py \
        --dsp_target /path/to/dataset/pass/buong_bui_anh_tuan_lyric_video_d_therapeutic.wav \
        --files 0.5=/path/str50.wav 0.65=/path/str65.wav
"""
import argparse
import glob
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np

from Check_result import check_file
from compare_music_content import chroma_dtw_similarity, mfcc_dtw_similarity, load_mono


def discover_files(results_dir):
    """
    Find files matching str<NN>.wav (no underscore, e.g. str20.wav, str65.wav)
    and parse the strength value. NN encodes strength*100 as an integer
    (str20.wav -> strength 0.20, str65.wav -> strength 0.65), matching the
    naming convention actually used for these files.
    """
    pattern = os.path.join(results_dir, "str*.wav")
    found = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"str(\d+)\.wav$", os.path.basename(path))
        if m:
            strength = int(m.group(1)) / 100.0
            found[strength] = path
    return found


def parse_explicit_files(file_args):
    """Parse --files 0.5=/path/x.wav 0.65=/path/y.wav ..."""
    found = {}
    for item in file_args:
        strength_str, path = item.split("=", 1)
        found[float(strength_str)] = path
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsp_target", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None,
                         help="Directory to auto-discover str<NN>.wav files in")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                         help="Explicit strength=path pairs, e.g. 0.5=/path/a.wav 0.65=/path/b.wav")
    args = parser.parse_args()

    if args.files:
        file_map = parse_explicit_files(args.files)
    elif args.results_dir:
        file_map = discover_files(args.results_dir)
    else:
        print("ERROR: provide either --results_dir or --files")
        sys.exit(1)

    if not file_map:
        print("No matching files found.")
        sys.exit(1)

    print(f"Found {len(file_map)} file(s): {sorted(file_map.keys())}\n")

    y1 = load_mono(args.dsp_target)

    results = []
    for strength in sorted(file_map.keys()):
        path = file_map[strength]
        if not os.path.exists(path):
            print(f"[SKIP] strength={strength}: file not found at {path}")
            continue

        print(f"Analyzing strength={strength} ({os.path.basename(path)}) ...")
        compliance = check_file(path)

        y2 = load_mono(path)
        _, chroma_sims, chroma_irregularity = chroma_dtw_similarity(y1, y2, 48000)
        _, mfcc_sims = mfcc_dtw_similarity(y1, y2, 48000)

        results.append({
            "strength": strength,
            "overall_pass": compliance["overall_pass"],
            "dynamic_range_db": compliance["dynamic_range"]["value"],
            "legato_pct": compliance["legato_overlap"]["value"],
            "bpm": compliance["bpm"]["value"],
            "chroma_mean_sim": chroma_sims.mean(),
            "chroma_irregularity": chroma_irregularity,
            "mfcc_mean_sim": mfcc_sims.mean(),
        })

    print("\n" + "=" * 100)
    print(f"{'Strength':>9} {'Pass?':>6} {'DynRange':>9} {'Legato%':>8} {'BPM':>6} "
          f"{'ChromaSim':>10} {'ChromaIrreg':>12} {'MFCCSim':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['strength']:>9.2f} {str(r['overall_pass']):>6} {r['dynamic_range_db']:>9.2f} "
              f"{r['legato_pct']:>8.1f} {r['bpm']:>6.1f} {r['chroma_mean_sim']:>10.4f} "
              f"{r['chroma_irregularity']:>12.4f} {r['mfcc_mean_sim']:>8.4f}")
    print("=" * 100)
    print("\nLooking for: DynRange getting closer to <13, Legato% closer to 70-80,")
    print("while ChromaSim/MFCCSim don't drop too much (structural/content fidelity lost).")


if __name__ == "__main__":
    main()