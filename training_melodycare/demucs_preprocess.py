"""
MelodyCare — Vocal Removal Preprocessing for Inference

Runs the SAME Demucs separation used in 3_source_separation.py (htdemucs,
4-stem: drums/bass/other→melody/vocals) on a raw input song, then discards
vocals AND drums — matching pipeline.py's DSP target-construction
convention exactly (that pipeline only ever processes melody+bass; drums
are explicitly excluded there too) — and mixes melody+bass at the same
1.0/0.1 weights pipeline.py uses.

WHY THIS EXISTS (Option B from conversation history):
    The fine-tuned model was trained on full commercial mixes (vocals
    included) as input, mapped to vocals-free melody+bass DSP targets. This
    makes "remove vocals" an implicit, harder part of what the model has to
    learn — and it currently handles that part poorly (garbled/bad-sounding
    audio specifically where vocals were present), even though the
    instrumental sections come out sounding like the right song. Rather
    than wait for more training to fix this internally, this script removes
    vocals BEFORE the audio ever reaches the model — sidestepping the
    model's weak spot entirely. The model still does 100% of the actual
    therapeutic transformation (tempo, dynamics, legato, spectral shaping);
    this just hands it a cleaner starting point, the same way a human sound
    engineer would remove an unwanted track before further processing.

Usage:
    python demucs_preprocess.py \
        --input_audio  /path/to/dataset/raw/Tier_2_Acceptable_81-100_BPM_95/buong_bui_anh_tuan_lyric_video_d.wav \
        --output_audio /path/to/training_output/check_inference/novocals_input.wav

Then feed --output_audio into inference.py's --input_audio instead of the
original vocal-containing file.
"""
import argparse
import gc
import os

import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def separate_and_mix(input_path: str, model_name: str = "htdemucs", melody_weight: float = 1.0,
                      bass_weight: float = 0.1) -> tuple:
    """
    Run Demucs separation, then mix melody(other)+bass only — same weights
    and stem selection as pipeline.py's DSP target construction, so the
    input the model now sees has a similar "what's present" character to
    what it needs to produce.

    Returns (mixed_audio_tensor, sample_rate).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using hardware: {device.upper()}")

    print(f"Loading '{model_name}' model...")
    model = get_model(model_name)
    model.to(device)
    model.eval()

    print(f"Loading input audio: {input_path}")
    wav, sr = torchaudio.load(input_path)
    wav = wav.to(device)
    wav_batched = wav.unsqueeze(0)  # (1, channels, time)

    print("Running Demucs separation (this may take a moment)...")
    with torch.no_grad():
        sources = apply_model(model, wav_batched, split=True, device=device)[0]

    # Stem order matches 3_source_separation.py: [drums, bass, other, vocals]
    stem_names = ["drums", "bass", "other", "vocals"]
    stems = {stem_names[i]: sources[i] for i in range(len(stem_names))}

    melody = stems["other"]   # renamed "melody" elsewhere in the project
    bass   = stems["bass"]
    # drums and vocals intentionally discarded — matches pipeline.py, and
    # is the entire point of this preprocessing step for vocals specifically.

    print(f"Mixing melody(other) + bass at weights ({melody_weight}, {bass_weight}) — "
          f"drums and vocals discarded.")
    mixed = melody * melody_weight + bass * bass_weight
    mixed = mixed.cpu()

    del sources, stems, melody, bass, wav, wav_batched
    torch.cuda.empty_cache()
    gc.collect()

    return mixed, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio",  type=str, required=True)
    parser.add_argument("--output_audio", type=str, required=True)
    parser.add_argument("--model_name",   type=str, default="htdemucs")
    parser.add_argument("--melody_weight", type=float, default=1.0)
    parser.add_argument("--bass_weight",   type=float, default=0.1)
    args = parser.parse_args()

    mixed, sr = separate_and_mix(
        args.input_audio,
        model_name=args.model_name,
        melody_weight=args.melody_weight,
        bass_weight=args.bass_weight,
    )

    os.makedirs(os.path.dirname(args.output_audio) or ".", exist_ok=True)
    torchaudio.save(args.output_audio, mixed, sr)
    print(f"[DONE] Saved vocal-free instrumental mix: {args.output_audio}")
    print(f"       Feed this into inference.py's --input_audio instead of the original.")


if __name__ == "__main__":
    main()
