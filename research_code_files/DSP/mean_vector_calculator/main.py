import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Import từ các module đã tách
from audio_utils import load_and_preprocess
from model_utils import (
    get_device, 
    load_clap_model, 
    extract_embedding_laion, 
    extract_embedding_transformers, 
    BACKEND
)

INPUT_DIR      = r"Dataset\Music\Real_therapy\chunks"
OUTPUT_DIR     = r"Dataset\Music\processed\training_data\mean_vector"
OUTPUT_STEM    = "therapeutic_style_vector"
TARGET_SR      = 48000
MAX_DURATION_S = 30
CLAP_SR        = 48000

def extract_style_vector(input_dir: str, output_dir: str, output_stem: str, target_sr: int, max_duration_s: int) -> torch.Tensor:
    device = get_device()
    model, processor = load_clap_model(device)
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".wav")])

    if not wav_files:
        print(f"[ERROR] No .wav files found in: {input_dir}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(wav_files)} .wav files in {input_dir}")
    print(f"[INFO] Processing up to {max_duration_s}s per file at {target_sr}Hz\n")

    embeddings = []
    failed_files = []

    for filepath in tqdm(wav_files, desc="Extracting embeddings"):
        try:
            audio = load_and_preprocess(filepath, target_sr, max_duration_s)

            if BACKEND == "laion":
                emb = extract_embedding_laion(model, audio, device)
            else:
                emb = extract_embedding_transformers(model, processor, audio, device, CLAP_SR)

            emb = emb.squeeze(0).cpu()
            embeddings.append(emb)

        except Exception as e:
            tqdm.write(f"  [SKIP] {os.path.basename(filepath)}: {e}")
            failed_files.append((filepath, str(e)))

    if failed_files:
        print(f"\n[WARN] {len(failed_files)} files failed:")
        for path, err in failed_files:
            print(f"  ✗ {os.path.basename(path)}: {err}")

    if not embeddings:
        print("[ERROR] No embeddings extracted — check input files")
        sys.exit(1)

    stacked = torch.stack(embeddings, dim=0)
    style_vector = stacked.mean(dim=0)

    pt_path = os.path.join(output_dir, f"{output_stem}.pt")
    npy_path = os.path.join(output_dir, f"{output_stem}.npy")
    meta_path = os.path.join(output_dir, f"{output_stem}_meta.txt")

    torch.save(style_vector, pt_path)
    np.save(npy_path, style_vector.numpy())

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("MelodyCare Therapeutic Style Vector\n")
        f.write("="*40 + "\n")
        f.write(f"CLAP backend:       {BACKEND}\n")
        f.write(f"Files processed:    {len(embeddings)}/{len(wav_files)}\n")
        f.write(f"Audio sample rate:  {target_sr} Hz\n")
        f.write(f"Vector norm:        {style_vector.norm().item():.6f}\n")

    print(f"[SAVE] PyTorch tensor → {pt_path}")
    print(f"[SAVE] NumPy array    → {npy_path}")
    print(f"[SAVE] Metadata       → {meta_path}")

    return style_vector

def verify_style_vector(pt_path: str) -> None:
    print(f"\n[VERIFY] Loading: {pt_path}")
    vec = torch.load(pt_path, map_location="cpu", weights_only=True)
    print(f"  Shape : {vec.shape}")
    print(f"  Norm  : {vec.norm().item():.4f}")
    print(f"  [OK] Style vector verified ✓")

if __name__ == "__main__":
    style_vector = extract_style_vector(INPUT_DIR, OUTPUT_DIR, OUTPUT_STEM, CLAP_SR, MAX_DURATION_S)
    pt_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_STEM}.pt")
    verify_style_vector(pt_path)