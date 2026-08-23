import torch
import numpy as np
import os
from diffusers import AudioLDM2Pipeline
from peft import PeftModel

# 1. Configuration & Paths
model_id = "cvssp/audioldm2-music"
lora_path = "thera_lora_weight"  # Folder name for your LoRA
test_input_path = r"Dataset\Training\flow_npy\[_thuy_tien_]_giac_mo_tuyet_trang_lyric_chunk_001.npy"  # Update this to your file

# 2. Patch function for GPT2 compatibility
def patch_language_model(pipe):
    if not hasattr(pipe.language_model, '_update_model_kwargs_for_generation'):
        def _dummy_update(outputs, model_kwargs):
            return model_kwargs
        pipe.language_model._update_model_kwargs_for_generation = _dummy_update
    return pipe

# 3. Load Pipeline
print("Loading model...")
pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = patch_language_model(pipe)

# 4. Memory Optimization
# This keeps the model off the GPU until the exact moment of computation
pipe.enable_model_cpu_offload()

# 5. Inject LoRA Weights
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

# 6. Prepare Input
print(f"Loading input: {test_input_path}")
orig_spec = np.load(test_input_path)
# Ensure input is moved to the device correctly within the offloaded pipeline
input_tensor = torch.from_numpy(orig_spec).unsqueeze(0).unsqueeze(0).to("cuda").half()

# 7. Translation (Single-step)
print("Running translation...")
with torch.no_grad():
    # Encode
    latents = pipe.vae.encode(input_tensor).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    
    # Prompt Embedding (FIXED: num_waveforms_per_prompt)
    prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        prompt="therapy music", 
        device="cuda", 
        do_classifier_free_guidance=False, 
        num_waveforms_per_prompt=1
    )
    
    # Single-step translation (timestep=0)
    timesteps = torch.zeros((1,), device="cuda", dtype=torch.long)
    
    # U-Net Forward Pass
    output_latents = pipe.unet(
        latents.half(),
        timestep=timesteps,
        encoder_hidden_states=generated_prompt_embeds.half(),
        encoder_hidden_states_1=prompt_embeds.half()
    ).sample
    
    # Decode
    decoded_audio = pipe.vae.decode(output_latents / pipe.vae.config.scaling_factor).sample

# 8. Save
np.save("translated_therapy.npy", decoded_audio.cpu().numpy())
print("Inference complete. Saved as translated_therapy.npy")