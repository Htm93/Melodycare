import torch
import numpy as np

# ── Device setup (chỉ chạy 1 lần trong main process) ──
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU] GPU not available")
    return device

# Audio constants
SAMPLE_RATE     = 48000
TARGET_BPM      = 60
MAX_DYNAMIC     = 13.0
MAX_SEMITONES   = 12
LFO_RATE_HZ     = 0.08
LFO_DEPTH       = 0.12
LEGATO_OVERLAP  = 0.70
LEGATO_DECAY = -4.0