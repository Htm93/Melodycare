"""
MelodyCare - Sample-Accurate Audio Chunking (v2)

Purpose:
Slices matched commercial input audio and DSP-processed therapeutic audio 
into perfectly aligned 30-second chunks for Stable Audio 3 fine-tuning.

Input:
    INPUT_DIR/  song_name.wav
    TARGET_DIR/ song_name_therapeutic.wav

Output:
    CHUNK_INPUT_DIR/  song_name_chunk_001.wav
    CHUNK_TARGET_DIR/ song_name_chunk_001.wav
"""

import os
import sys
import numpy as np
import soundfile as sf


# =====================================================================
# PAIR DISCOVERY
# =====================================================================

def scan_paired_files(input_dir: str, target_dir: str) -> list[tuple[str, str, str]]:
    """
    Map raw input files (song_name.wav) to target therapeutic files (song_name_therapeutic.wav).
    Returns a list of tuples: (song_name, input_filename, target_filename)
    """
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    if not os.path.exists(target_dir):
        print(f"[ERROR] Target directory not found: {target_dir}")
        sys.exit(1)

    input_files  = {f for f in os.listdir(input_dir)  if f.lower().endswith(".wav")}
    target_files = {f for f in os.listdir(target_dir) if f.lower().endswith(".wav")}

    # Map target filenames back to base song names: "song_name_therapeutic.wav" -> "song_name"
    target_map = {
        f.removesuffix("_therapeutic.wav"): f
        for f in target_files
        if f.endswith("_therapeutic.wav")
    }

    input_map = {
        os.path.splitext(f)[0]: f
        for f in input_files
    }

    matched_keys = sorted(set(input_map.keys()) & set(target_map.keys()))
    
    only_in_input  = set(input_map.keys()) - set(target_map.keys())
    only_in_target = set(target_map.keys()) - set(input_map.keys())

    if only_in_input:
        print(f"\n[WARN] {len(only_in_input)} input file(s) have no matching target:")
        for k in sorted(only_in_input):
            print(f"       ✗ {input_map[k]}")

    if only_in_target:
        print(f"\n[WARN] {len(only_in_target)} target file(s) have no matching input:")
        for k in sorted(only_in_target):
            print(f"       ✗ {target_map[k]}")

    if not matched_keys:
        print("[ERROR] No matched pairs found between input and target directories")
        sys.exit(1)

    pairs = []
    for song_name in matched_keys:
        pairs.append((
            song_name,
            input_map[song_name],
            target_map[song_name]
        ))

    return pairs


# =====================================================================
# SAMPLE-ACCURATE CHUNKING
# =====================================================================

def _load_mono_or_stereo(path: str, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    if sr != sample_rate:
        import librosa
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        else:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
    return audio

def _num_frames(audio: np.ndarray) -> int:
    return audio.shape[0]

def _slice(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    return audio[start:end]

def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    current = audio.shape[0]
    if current >= length:
        return audio[:length]
    pad_shape = (length - current,) + audio.shape[1:]
    pad = np.zeros(pad_shape, dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)

def chunk_pair(
    song_name: str,
    input_path: str,
    target_path: str,
    input_out_dir: str,
    target_out_dir: str,
    sample_rate: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
    pad_last: bool,
    subtype: str,
) -> int:
    """
    Slice one matched (input, target) pair into aligned chunks.
    """
    input_audio  = _load_mono_or_stereo(input_path,  sample_rate)
    target_audio = _load_mono_or_stereo(target_path, sample_rate)

    n_frames = min(_num_frames(input_audio), _num_frames(target_audio))
    if _num_frames(input_audio) != _num_frames(target_audio):
        print(
            f"[WARN] {song_name}: input/target length mismatch "
            f"({_num_frames(input_audio)} vs {_num_frames(target_audio)} samples) "
            f"— chunking up to the shorter length."
        )

    chunk_frames = int(round(chunk_seconds * sample_rate))
    if chunk_frames <= 0:
        raise ValueError("chunk_seconds must be > 0")

    written = 0
    start = 0
    idx = 1
    
    while start < n_frames:
        end = min(start + chunk_frames, n_frames)
        length = end - start

        if length < chunk_frames:
            if pad_last:
                pass  
            elif length >= min_chunk_seconds * sample_rate:
                pass  
            else:
                break 

        in_chunk  = _slice(input_audio,  start, end)
        tgt_chunk = _slice(target_audio, start, end)

        if pad_last and length < chunk_frames:
            in_chunk  = _pad_to_length(in_chunk,  chunk_frames)
            tgt_chunk = _pad_to_length(tgt_chunk, chunk_frames)

        chunk_name = f"{song_name}_chunk_{idx:03d}.wav"
        sf.write(os.path.join(input_out_dir,  chunk_name), in_chunk,  sample_rate, subtype=subtype)
        sf.write(os.path.join(target_out_dir, chunk_name), tgt_chunk, sample_rate, subtype=subtype)

        written += 1
        idx += 1
        start += chunk_frames

    return written


# =====================================================================
# MAIN
# =====================================================================

def main():
    # --- HARD-CODE settings ---
    input_dir  = r"/path/to/dataset/raw/Tier_1_2"
    target_dir = r"/path/to/dataset/pass"
    input_out  = r"/path/to/dataset/raw_chunks"
    target_out = r"/path/to/dataset/therapy_chunks"
    
    sample_rate       = 48000
    chunk_seconds     = 30.0
    min_chunk_seconds = 30.0
    pad_last          = False
    subtype           = "PCM_24"
    # ------------------------------------

    os.makedirs(input_out, exist_ok=True)
    os.makedirs(target_out, exist_ok=True)

    print("=" * 60)
    print("MelodyCare — Audio Chunking (30s aligned pairs)")
    print(f"  Input dir:    {input_dir}")
    print(f"  Target dir:   {target_dir}")
    print(f"  Chunk input:  {input_out}")
    print(f"  Chunk target: {target_out}")
    print(f"  Chunk length: {chunk_seconds}s  (sample rate {sample_rate}Hz)")
    print(f"  Pad last:     {pad_last}  |  min kept partial: {min_chunk_seconds}s")
    print("=" * 60)

    matched_pairs = scan_paired_files(input_dir, target_dir)
    print(f"\n[INFO] {len(matched_pairs)} matched song pair(s) found\n")

    total_chunks = 0
    failed = []

    for song_name, input_filename, target_filename in matched_pairs:
        input_path  = os.path.join(input_dir,  input_filename)
        target_path = os.path.join(target_dir, target_filename)

        try:
            n = chunk_pair(
                song_name=song_name,
                input_path=input_path,
                target_path=target_path,
                input_out_dir=input_out,
                target_out_dir=target_out,
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                min_chunk_seconds=min_chunk_seconds,
                pad_last=pad_last,
                subtype=subtype,
            )
            print(f"[OK]   {song_name}: {n} chunk(s)")
            total_chunks += n
        except Exception as e:
            print(f"[FAIL] {song_name}: {e}")
            failed.append((song_name, str(e)))

    print(f"\n{'='*60}")
    print(f"Done: {len(matched_pairs) - len(failed)}/{len(matched_pairs)} songs processed, "
          f"{total_chunks} total chunk pairs written")
    
    if failed:
        print("\nFailed songs:")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()