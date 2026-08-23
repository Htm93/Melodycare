"""
Diagnostic: run the BASE pretrained stable-audio-open-1.0 (no fine-tuned
checkpoint weights, no style/input-audio conditioning) through the same
sampler used in inference.py, on a plain text prompt.

Purpose: isolate whether "sounds like noise" is a sampler/pipeline bug
(this test would also produce noise) or a fine-tuning problem (this test
should produce coherent, if generic, music).

Usage:
    python /path/to/training_output/check_inference/sanity_base_model.py --output_audio /path/to/training_output/check_inference/base_model_test.wav
"""
import argparse
import os

import sys
from pathlib import Path

# Resolve project root (/path/to/training_dir) from nested script location
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.loader import load_stable_audio_model

import torch
import soundfile as sf

from accelerate.state import PartialState
PartialState()

import config
from models.loader import load_stable_audio_model

# Reuse the exact same sampler used in inference.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import sample_v_objective, get_alphas_sigmas  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_audio", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--prompt", type=str,
                         default="A calm, slow, minimal piano melody, peaceful and warm.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BASE pretrained model (no fine-tuned weights)...")
    model, style_proj, input_audio_proj, model_config = load_stable_audio_model(
        model_id=config.PRETRAINED_MODEL_ID, device=device, gradient_checkpointing=False,
    )
    dit = model.model
    dit.eval()

    # Text + seconds conditioning ONLY — no style / input-audio tokens
    cond_input = [{
        "prompt": args.prompt,
        "seconds_start": 0,
        "seconds_total": config.CHUNK_DURATION_S,
    }]
    with torch.no_grad():
        conditioning_tensors = model.conditioner(cond_input, device=device)
        conditioning = model.get_conditioning_inputs(conditioning_tensors)
        if "cross_attn_mask" not in conditioning and "cross_attn_cond_mask" in conditioning:
            conditioning["cross_attn_mask"] = conditioning["cross_attn_cond_mask"]

    latent_channels = 64
    downsampling_ratio = model_config.get("model", {}).get("pretransform", {}).get(
        "config", {}).get("downsampling_ratio", 2048
    )
    latent_T = (config.SAMPLE_RATE * config.CHUNK_DURATION_S) // downsampling_ratio
    shape = (1, latent_channels, latent_T)

    print(f"Sampling ({args.num_steps} steps) with shape {shape}...")
    output_latents = sample_v_objective(dit, shape, conditioning, device, num_steps=args.num_steps)

    print("Decoding...")
    with torch.no_grad():
        output_audio = model.pretransform.decode(output_latents)
    output_audio = output_audio.squeeze(0).clamp(-1.0, 1.0).cpu()

    os.makedirs(os.path.dirname(args.output_audio) or ".", exist_ok=True)
    sf.write(args.output_audio, output_audio.T.numpy(), config.SAMPLE_RATE)
    print(f"[DONE] Saved: {args.output_audio}")
    print("\nListen to this file. If it sounds like coherent (if generic) music,")
    print("the sampler/pipeline is correct and the noise issue is from fine-tuning.")
    print("If THIS also sounds like noise, there's a bug in the sampler/conditioning code.")


if __name__ == "__main__":
    main()