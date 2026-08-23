import os
import soundfile as sf
import config

def get_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate

def main():
    if not os.path.exists(config.INPUT_MIXED_DIR) or not os.path.exists(config.TARGET_MIXED_DIR):
        print(f"[ERROR] Directories do not exist:")
        print(f"  INPUT_MIXED_DIR:  {config.INPUT_MIXED_DIR}")
        print(f"  TARGET_MIXED_DIR: {config.TARGET_MIXED_DIR}")
        return

    # Map normalized stem names -> actual file names on disk
    input_map = {
        f.lower().removesuffix("_therapeutic.wav").removesuffix(".wav"): f
        for f in os.listdir(config.INPUT_MIXED_DIR) if f.lower().endswith(".wav")
    }
    
    target_map = {
        f.lower().removesuffix("_therapeutic.wav").removesuffix(".wav"): f
        for f in os.listdir(config.TARGET_MIXED_DIR) if f.lower().endswith(".wav")
    }

    print(f"Files found in INPUT_MIXED_DIR:  {len(input_map)}")
    print(f"Files found in TARGET_MIXED_DIR: {len(target_map)}")

    matched_keys = sorted(set(input_map.keys()) & set(target_map.keys()))
    print(f"Matched base pairs: {len(matched_keys)}\n")

    mismatched = []
    matched_ok = []

    for base in matched_keys:
        in_file = input_map[base]
        tgt_file = target_map[base]
        
        in_path = os.path.join(config.INPUT_MIXED_DIR, in_file)
        tgt_path = os.path.join(config.TARGET_MIXED_DIR, tgt_file)

        in_dur = get_duration(in_path)
        tgt_dur = get_duration(tgt_path)
        
        ratio = tgt_dur / in_dur if in_dur > 0 else 1.0
        if abs(ratio - 1.0) > 0.02:  # >2% duration difference = stretched
            mismatched.append((base, in_dur, tgt_dur, ratio))
        else:
            matched_ok.append(base)

    print(f"Total matched pairs: {len(matched_keys)}")
    print(f"  Duration-matched (no stretch): {len(matched_ok)}")
    print(f"  Duration-MISMATCHED (stretched, chunk alignment needed): {len(mismatched)}")
    print()

    if mismatched:
        print("Sample of mismatched pairs (base_name, input_s, target_s, ratio):")
        for base, i, t, r in mismatched[:10]:
            print(f"  {base}: {i:.1f}s -> {t:.1f}s (ratio {r:.3f})")

if __name__ == "__main__":
    main()