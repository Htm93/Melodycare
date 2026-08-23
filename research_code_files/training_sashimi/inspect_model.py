from stable_audio_tools import get_pretrained_model

with open("full_dump.txt", "w") as f:
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    f.write(f"input_concat_ids: {model.input_concat_ids}\n")
    f.write(f"prepend_cond_ids: {model.prepend_cond_ids}\n")
    f.write(f"local_add_cond_ids: {model.local_add_cond_ids}\n")
    f.write(f"modular_local_cond_ids: {model.modular_local_cond_ids}\n")
    f.write(f"io_channels: {model.io_channels}\n")
    # Check the DiT's underlying network for an input-concat-aware projection layer
    dit_inner = model.model.model  # DiTWrapper.model -> the actual transformer
    f.write(f"\ntype(model.model.model): {type(dit_inner)}\n")
    f.write(f"dir (filtered for 'concat' or 'proj' or 'embed'):\n")
    f.write(str([a for a in dir(dit_inner) if any(k in a.lower() for k in ['concat','proj','embed','channel'])]))
print("wrote /tmp/concat_dump.txt")