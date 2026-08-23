import sys
import torch
import numpy as np

try:
    import laion_clap
    BACKEND = "laion"
    print("[INFO] Using LAION-CLAP backend")
except ImportError:
    try:
        from transformers import ClapModel, ClapProcessor
        BACKEND = "transformers"
        print("[INFO] Using HuggingFace Transformers CLAP backend")
    except ImportError:
        print("[ERROR] No CLAP backend found. Install laion-clap or transformers")
        sys.exit(1)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[CPU] CUDA not available — using CPU (slower)")
    return device

def load_clap_model(device: torch.device):
    if BACKEND == "laion":
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        model.load_ckpt()
        model = model.to(device)
        model.eval()
        return model, None
    else:
        model_id = "laion/larger_clap_music"
        model = ClapModel.from_pretrained(model_id).to(device)
        processor = ClapProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor

def extract_embedding_laion(model, audio: np.ndarray, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        embedding = model.get_audio_embedding_from_data(x=[audio], use_tensor=False)
        embedding = torch.from_numpy(embedding).float().to(device)
    return embedding

def extract_embedding_transformers(model, processor, audio: np.ndarray, device: torch.device, clap_sr: int) -> torch.Tensor:
    inputs = processor(audios=audio, sampling_rate=clap_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model.get_audio_features(**inputs)
    return embedding.float()