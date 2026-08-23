"""
MelodyCare — Strength Sweep

Runs inference.py at several --strength values on the SAME input, checks
each output's compliance (single_check.py logic) and musical content
similarity to the DSP target (compare_music_content.py logic), and prints
one summary table — so you can see the strength-vs-quality tradeoff
directly instead of manually running and comparing each value one at a time.

Usage:
    python /path/to/training_output/result_data_check/strength_sweep.py \
        --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-7410 \
        --input_audio    /path/to/dataset/Karaoke_HD_Bung_Full_Beat_Gc_Bi_Anh_Tun_Newtitan.wav \
        --dsp_target     /path/to/dataset/pass/buong_bui_anh_tuan_lyric_video_d_therapeutic.wav \
        --duration_ratio 2.344 \
        --strengths      0.5 0.65 0.75 0.85
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import librosa

from Check_result import check_file
from compare_music_content import chroma_dtw_similarity, mfcc_dtw_similarity, load_mono


def run_inference(python_exe, checkpoint_dir, input_audio, output_audio, duration_ratio, strength, num_steps=50):
    cmd = [
        python_exe, "/path/to/training_output/check_inference/inference.py",
        "--checkpoint_dir", checkpoint_dir,
        "--input_audio", input_audio,
        "--output_audio", output_audio,
        "--num_steps", str(num_steps),
        "--strength", str(strength),
        "--duration_ratio", str(duration_ratio),
    ]
    print(f"\n{'='*70}\nRunning strength={strength} ...\n{'='*70}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAILED] strength={strength}")
        print(result.stderr[-2000:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--input_audio",    type=str, required=True)
    parser.add_argument("--dsp_target",     type=str, required=True)
    parser.add_argument("--duration_ratio", type=float, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.65, 0.75, 0.85])
    parser.add_argument("--output_dir", type=str,
                         default="/path/to/training_output/check_inference/strength_sweep")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for strength in args.strengths:
        out_path = os.path.join(args.output_dir, f"strength_{strength:.2f}.wav")
        ok = run_inference(
            args.python_exe, args.checkpoint_dir, args.input_audio, out_path,
            args.duration_ratio, strength, args.num_steps,
        )
        if not ok:
            continue

        print(f"Checking compliance + content similarity for strength={strength} ...")
        compliance = check_file(out_path)

        y1 = load_mono(args.dsp_target)
        y2 = load_mono(out_path)
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