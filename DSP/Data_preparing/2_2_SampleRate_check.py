import os
import torchaudio
import librosa

def get_song_sample_rate(song_folder):
    """
    Chỉ kiểm tra file 'other.wav' (melody) để lấy sample rate đại diện cho cả 4 file.
    """
    representative_file = os.path.join(song_folder, "other.wav")
    
    if os.path.exists(representative_file):
        metadata = torchaudio.info(representative_file)
        return metadata.sample_rate
    return None

input_dir = r"Dataset\Music\processed\Full"
raw_dir = r"Dataset\Raw\Music1"
all_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
all_raw = [f for f in os.listdir(raw_dir) if f.endswith(".wav")]
correct_count = 0

for index, song_name in enumerate(all_raw):
    print(f"[{index+1}/{len(all_raw)}] PROCESSING")
    folder_path = os.path.join(raw_dir, song_name)
    sr = torchaudio.info(folder_path).sample_rate
    if sr == 48000:
        correct_count += 1
    else:
        print(f"ERROR: {song_name}\n SR: {sr}")

print(f"Correct song: {correct_count}/{len(all_raw)}")
