"""
Quick diagnostic: how far has input_audio_proj grown from its zero-init
starting point? Compares the norm of its output tokens against the norm of
the (non-zero-init) text embeddings, as a rough scale reference.

Usage:
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint_before_extend/checkpoint-2450
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-5000
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-5500
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-6000
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-6500
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-7000
    python /path/to/training_output/check_inference/check_module_growth.py --checkpoint_dir /path/to/training_output/checkpoint/checkpoint-7410
"""
import argparse
import torch
from accelerate.state import PartialState
PartialState()

# =============================================================================
# Dynamic System Path Resolution
# =============================================================================
# Resolves project root (/path/to/training_dir) from nested script:
# training_result/check_inference/inference.py (2 levels up)
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from models.loader import load_stable_audio_model
from inference import load_checkpoint_weights, build_therapy_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, style_proj, input_audio_proj, model_config = load_stable_audio_model(
    model_id=config.PRETRAINED_MODEL_ID, device=device, gradient_checkpointing=False,
)
load_checkpoint_weights(args.checkpoint_dir, model.model, style_proj, input_audio_proj)

# Dummy input latents just to get a representative token norm
dummy_latents = torch.randn(1, 64, 700, device=device)
with torch.no_grad():
    input_tokens = input_audio_proj(dummy_latents)
    style_tokens = style_proj(torch.randn(1, config.STYLE_VECTOR_DIM, device=device))

    cond_input = [{"prompt": build_therapy_prompt(), "seconds_start": 0, "seconds_total": config.CHUNK_DURATION_S}]
    conditioning_tensors = model.conditioner(cond_input, device=device)
    base_cond = model.get_conditioning_inputs(conditioning_tensors)
    text_tokens = base_cond["cross_attn_cond"]

print(f"input_audio_proj output norm (mean per-token): {input_tokens.norm(dim=-1).mean().item():.4f}")
print(f"style_proj output norm (mean per-token):        {style_tokens.norm(dim=-1).mean().item():.4f}")
print(f"text embedding norm (mean per-token, reference): {text_tokens.norm(dim=-1).mean().item():.4f}")
print("\nIf input_audio_proj's norm is near 0 (<< text norm), it genuinely hasn't grown much from zero-init yet.")
print("If it's a meaningful fraction of the text norm but output still barely changes, the issue is more likely")
print("the DiT's cross-attention not yet learning to USE these tokens (points toward raising DIT_LR_SCALE).")