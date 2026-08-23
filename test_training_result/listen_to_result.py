import torch
import numpy as np
import scipy.io.wavfile as wavfile
import torch.nn.functional as F
from diffusers import AudioLDM2Pipeline

file_path = r"test_training_result\translated_therapy.npy"

print("Loading Vocoder...")
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16).to("cuda")

# 1. Load your successful spectrogram
spectrogram = np.load(file_path)

# 2. Squeeze out empty dimensions to get (128, 2584) which is (Freq, Time)
spec_2d = torch.from_numpy(spectrogram).squeeze()

# 3. Transpose to (Time, Freq) -> (2584, 128)
spec_transposed = spec_2d.T 

# 4. Downsample the 128 Freq bins to 64 Freq bins to satisfy the Vocoder
# We reshape to (Batch=Time, Channels=1, Length=Freq) for the interpolation
spec_for_resize = spec_transposed.unsqueeze(1).contiguous().float() 
spec_resized = F.interpolate(spec_for_resize, size=64, mode='linear', align_corners=False)

# 5. Reshape back and add the final batch dimension for the vocoder -> (1, 2584, 64)
spectrogram_tensor = spec_resized.squeeze(1).unsqueeze(0).to("cuda").half()

print(f"Fixed Tensor Shape: {spectrogram_tensor.shape}")
print("Converting Spectrogram to Audio Waveform...")

with torch.no_grad():
    # Pass the correctly shaped tensor through the vocoder
    audio_wave = pipe.vocoder(spectrogram_tensor)

# Convert to numpy and save
audio_wave_np = audio_wave.cpu().numpy().squeeze().astype(np.float32)
wavfile.write("translated_therapy_audio.wav", 16000, audio_wave_np)

print("Success! You can now listen to 'translated_therapy_audio.wav'")