"""
Step 2: Generate master CSV training manifest.

v2 UPDATE: Strict 1-to-1 mapping between pre-mixed input and target files.
           No stem loading or mixing logic.

Expected file naming convention:
    INPUT_MIXED_DIR/   song_name.wav  (original commercial audio, mixed)
    TARGET_MIXED_DIR/  song_name.wav  (DSP-processed therapeutic audio, mixed)

Both directories must contain identically named files.
"""
import os
import csv
import sys
import config


def build_therapy_prompt() -> str:
    """
    Build ASD therapy text prompt using constants from config.py.
    All numeric values reference config directly — no hardcoding.
    """
    return (
        f"A slow, calm instrumental therapeutic music track at exactly "
        f"{config.THERAPY_BPM_TARGET} BPM, strictly within the "
        f"{config.THERAPY_BPM_MIN} to {config.THERAPY_BPM_MAX} BPM therapeutic range. "
        f"The rhythm is steady, predictable, and metronomic. "
        f"No tempo variations, rubato, or rhythmic surprises. "

        f"The overall dynamic range is extremely compressed, "
        f"with a peak-to-floor variance of less than {config.THERAPY_DYNAMIC_DB:.0f} dB. "
        f"Volume is consistent with no sudden loud transients or abrupt silences. "

        f"Notes transition smoothly with legato style, overlapping between "
        f"{config.THERAPY_LEGATO_MIN}% and {config.THERAPY_LEGATO_MAX}% of their duration. "
        f"No staccato articulations. Each note sustains gently into the next. "

        f"A subtle breathing quality is present with amplitude modulation oscillating "
        f"between {config.THERAPY_LFO_MIN_HZ} Hz and {config.THERAPY_LFO_MAX_HZ} Hz. "

        f"The frequency spectrum rolls off sharply above {config.THERAPY_ROLLOFF_KHZ} kHz. "
        f"No harsh high-frequency components above {config.THERAPY_ROLLOFF_KHZ} kHz. "
        f"Sub-bass frequencies below {config.THERAPY_SUB_CUTOFF_HZ} Hz are absent. "

        f"Compared to the baseline therapeutic music reference, there is enhanced warmth "
        f"in the {config.THERAPY_BOOST_LOW_HZ} Hz to {config.THERAPY_BOOST_HIGH_HZ} Hz band. "
        f"This low-mid emphasis creates a grounded, warm tonal character without muddiness. "
        f"The boost is adaptive: reduced if input already has strong low-mid energy "
        f"exceeding the mid-range reference by more than 5 dB. "

        f"Minimalist, sparse note density with long durations and generous musical space. "
        f"Deeply calming, safe, and predictable — designed for sensory regulation "
        f"in individuals with Autism Spectrum Disorder. "
        f"No sudden changes in any acoustic parameter at any point in the track."
    )


def build_negative_prompt() -> str:
    return (
        "drums, percussion, cymbals, hi-hats, kick drum, snare, "
        "bright high frequencies, harsh treble, sibilance, "
        "fast tempo, energetic, upbeat, loud, sudden volume changes, "
        "staccato notes, sharp attack, distortion, noise, "
        "complex harmony, dense arrangement, multiple instruments, "
        "sub-bass, heavy bass, muddy low end, "
        "rubato, tempo fluctuation, syncopation, irregular rhythm, "
        "vocals, singing, speech, lyrics"
    )


def scan_paired_files() -> list[dict]:
    """
    Strict 1-to-1 mapping between INPUT_MIXED_DIR and TARGET_MIXED_DIR.

    Rules:
        1. Scan INPUT_MIXED_DIR for all .wav files → build set of names
        2. For each input name, check TARGET_MIXED_DIR for exact match
        3. Skip if target not found (log warning)
        4. Report unmatched target files (target exists but no input)

    Returns:
        List of dicts with keys: song_name, input_audio, target_audio
    """
    if not os.path.exists(config.INPUT_MIXED_DIR):
        print(f"[ERROR] INPUT_MIXED_DIR not found: {config.INPUT_MIXED_DIR}")
        sys.exit(1)

    if not os.path.exists(config.TARGET_MIXED_DIR):
        print(f"[ERROR] TARGET_MIXED_DIR not found: {config.TARGET_MIXED_DIR}")
        sys.exit(1)

    # Build sets of filenames (without directory)
    input_files  = {
        f for f in os.listdir(config.INPUT_MIXED_DIR)
        if f.lower().endswith(".wav")
    }
    target_files = {
        f for f in os.listdir(config.TARGET_MIXED_DIR)
        if f.lower().endswith(".wav")
    }

    # Report unmatched files
    only_in_input  = input_files  - target_files
    only_in_target = target_files - input_files

    if only_in_input:
        print(f"\n[WARN] {len(only_in_input)} input files have no matching target:")
        for f in sorted(only_in_input):
            print(f"       ✗ {f}")

    if only_in_target:
        print(f"\n[WARN] {len(only_in_target)} target files have no matching input:")
        for f in sorted(only_in_target):
            print(f"       ✗ {f}")

    # Build matched pairs (intersection)
    matched = sorted(input_files & target_files)

    if not matched:
        print("[ERROR] No matched pairs found between input and target directories")
        sys.exit(1)

    records = []
    for filename in matched:
        song_name  = os.path.splitext(filename)[0]   # strip .wav extension
        input_path = os.path.join(config.INPUT_MIXED_DIR,  filename)
        target_path= os.path.join(config.TARGET_MIXED_DIR, filename)

        records.append({
            "song_name"   : song_name,
            "input_audio" : input_path,
            "target_audio": target_path,
        })

    return records


def generate_csv(records: list[dict]) -> None:
    """
    Write master CSV with simplified schema (v2).

    CSV columns (v2 — no stem/mix columns):
        song_name       : identifier
        input_audio     : path to pre-mixed input .wav
        target_audio    : path to pre-mixed target .wav
        style_vector_pt : path to CLAP style vector (.pt)
        style_vector_npy: path to CLAP style vector (.npy)
        text_prompt     : ASD therapy text prompt
        negative_prompt : negative guidance prompt
    """
    style_pt  = os.path.join(config.STYLE_VECTOR_DIR, f"{config.STYLE_VECTOR_STEM}.pt")
    style_npy = os.path.join(config.STYLE_VECTOR_DIR, f"{config.STYLE_VECTOR_STEM}.npy")

    if not os.path.exists(style_pt):
        print(f"[WARN] Style vector not found: {style_pt}")
        print(f"       Run 1_extract_style_vector.py first")

    text_prompt     = build_therapy_prompt()
    negative_prompt = build_negative_prompt()

    # v2 simplified fieldnames — no stem or mix weight columns
    fieldnames = [
        "song_name",
        "input_audio",
        "target_audio",
        "style_vector_pt",
        "style_vector_npy",
        "text_prompt",
        "negative_prompt",
    ]

    os.makedirs(os.path.dirname(config.CSV_OUTPUT_PATH) or ".", exist_ok=True)

    with open(config.CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "song_name"       : rec["song_name"],
                "input_audio"     : rec["input_audio"],
                "target_audio"    : rec["target_audio"],
                "style_vector_pt" : style_pt,
                "style_vector_npy": style_npy,
                "text_prompt"     : text_prompt,
                "negative_prompt" : negative_prompt,
            })

    print(f"[SAVE] {config.CSV_OUTPUT_PATH} ({len(records)} rows)")


def main():
    print("=" * 60)
    print("MelodyCare — CSV Generation (v2: pre-mixed files)")
    print(f"  Input dir:  {config.INPUT_MIXED_DIR}")
    print(f"  Target dir: {config.TARGET_MIXED_DIR}")
    print(f"  Output CSV: {config.CSV_OUTPUT_PATH}")
    print("=" * 60)

    records = scan_paired_files()

    print(f"\n[INFO] {len(records)} matched pairs found")
    print(f"[PREVIEW] First pair:")
    print(f"    input : {records[0]['input_audio']}")
    print(f"    target: {records[0]['target_audio']}")

    generate_csv(records)
    print("\n[DONE]")


if __name__ == "__main__":
    main()