import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

# 1. Load the distorted audio
file_path = "translated_therapy_audio.wav"
sample_rate, audio_data = wavfile.read(file_path)

# 2. Design a High-Pass Filter 
# This cuts off frequencies below 250 Hz (where the loud bass rumble is located)
cutoff_freq = 250.0
nyquist = 0.5 * sample_rate
normalized_cutoff = cutoff_freq / nyquist

# Create a Butterworth filter (Order=4)
sos = butter(4, normalized_cutoff, btype='high', analog=False, output='sos')

# 3. Apply the filter to the audio array
print("Applying High-Pass Filter to remove bass...")
filtered_audio = sosfilt(sos, audio_data)

# 4. Normalize the audio to prevent clipping
max_amp = np.max(np.abs(filtered_audio))
if max_amp > 0:
    filtered_audio = (filtered_audio / max_amp) * 0.9  # Scale to 90% volume

# 5. Save the corrected audio
output_path = "translated_therapy_audio_fixed.wav"
wavfile.write(output_path, sample_rate, filtered_audio.astype(np.float32))

print(f"Success! Listen to '{output_path}'")