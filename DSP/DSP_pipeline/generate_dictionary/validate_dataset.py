"""
Step 3: Validate the generated CSV and all referenced files.

Checks:
    - All file paths in CSV exist on disk
    - Audio files are readable and have expected sample rate
    - Style vector is loadable and has correct shape
    - Text prompts are non-empty and within token length budget

Usage:
    python 3_validate_dataset.py
"""

import os
import csv
import sys
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from config import CSV_OUTPUT_PATH, CLAP_SR

MAX_PROMPT_CHARS  = 2000    # approximate token budget for Stable Audio 3
EXPECTED_VECTOR_DIM = 512   # CLAP embedding dimension


def validate_audio(path: str, expected_sr: int) -> tuple[bool, str]:
    """Check audio file is readable and has correct sample rate."""
    try:
        info = sf.info(path)
        if info.samplerate != expected_sr:
            return False, f"SR={info.samplerate} (expected {expected_sr})"
        if info.duration < 1.0:
            return False, f"Duration too short: {info.duration:.1f}s"
        return True, f"OK ({info.duration:.1f}s, {info.channels}ch)"
    except Exception as e:
        return False, str(e)


def validate_style_vector(path: str) -> tuple[bool, str]:
    """Check style vector is loadable and has correct shape."""
    try:
        vec = torch.load(path, map_location="cpu")
        if vec.shape != (EXPECTED_VECTOR_DIM,):
            return False, f"Shape {vec.shape} (expected ({EXPECTED_VECTOR_DIM},))"
        if torch.isnan(vec).any():
            return False, "Contains NaN values"
        if torch.isinf(vec).any():
            return False, "Contains Inf values"
        return True, f"OK (dim={vec.shape[0]}, norm={vec.norm():.3f})"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("MelodyCare — Dataset Validation")
    print("=" * 60)

    if not os.path.exists(CSV_OUTPUT_PATH):
        print(f"[ERROR] CSV not found: {CSV_OUTPUT_PATH}")
        print(f"        Run 2_generate_csv.py first")
        sys.exit(1)

    with open(CSV_OUTPUT_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n[INFO] Validating {len(rows)} training samples\n")

    total_errors   = 0
    total_warnings = 0

    for i, row in enumerate(tqdm(rows, desc="Validating")):
        song = row["song_name"]
        errors   = []
        warnings = []

        # ── Audio file checks ─────────────────────────────────────────
        audio_fields = [
            "input_melody", "input_bass", "input_drums",
            "target_melody", "target_bass",
        ]
        for field in audio_fields:
            path = row[field]
            if not os.path.exists(path):
                errors.append(f"{field}: file not found")
            else:
                ok, msg = validate_audio(path, CLAP_SR)
                if not ok:
                    errors.append(f"{field}: {msg}")

        # ── Style vector checks ───────────────────────────────────────
        sv_path = row["style_vector_pt"]
        if not os.path.exists(sv_path):
            errors.append(f"style_vector_pt: not found")
        else:
            ok, msg = validate_style_vector(sv_path)
            if not ok:
                errors.append(f"style_vector_pt: {msg}")

        # ── Prompt length check ───────────────────────────────────────
        prompt_len = len(row["text_prompt"])
        if prompt_len == 0:
            errors.append("text_prompt: empty")
        elif prompt_len > MAX_PROMPT_CHARS:
            warnings.append(
                f"text_prompt: {prompt_len} chars (>{MAX_PROMPT_CHARS} may truncate)"
            )

        # ── Mix weights sanity check ──────────────────────────────────
        try:
            w_melody = float(row["mix_weight_melody"])
            w_bass   = float(row["mix_weight_bass"])
            w_drum   = float(row["mix_weight_drum"])
            if not (0 <= w_melody <= 1 and 0 <= w_bass <= 1 and 0 <= w_drum <= 1):
                warnings.append(f"mix_weights outside [0,1]: {w_melody},{w_bass},{w_drum}")
        except ValueError:
            errors.append("mix_weights: non-numeric values")

        if errors:
            total_errors += len(errors)
            tqdm.write(f"\n  [FAIL] {song}:")
            for e in errors:
                tqdm.write(f"         ✗ {e}")
        if warnings:
            total_warnings += len(warnings)
            for w in warnings:
                tqdm.write(f"  [WARN] {song}: {w}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Validation complete: {len(rows)} samples")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors == 0:
        print(f"\n  ✓ Dataset is ready for Stable Audio 3 fine-tuning")
    else:
        print(f"\n  ✗ Fix {total_errors} errors before training")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()