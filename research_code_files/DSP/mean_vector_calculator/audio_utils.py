import numpy as np
import soundfile as sf
import librosa

def load_and_preprocess(filepath: str, target_sr: int, max_duration_s: int) -> np.ndarray:
    """
    Load a .wav file, resample to target_sr, convert to mono,
    and clip/pad to a fixed length.
    """
    audio, sr = sf.read(filepath, always_2d=True)
    audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    target_len = max_duration_s * target_sr

    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')

    return audio.astype(np.float32)