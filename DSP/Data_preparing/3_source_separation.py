#Tensor are just array but it is effective when working with gpu because it store data in a straight line

import gc #Garbage collector
import os
import torch
import torchaudio
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model

input_dir = r"Dataset\Raw\Tier_2_Acceptable (81-100 BPM)(95)"
output_dir = r"Dataset\Music\processed\separated_sources"
model_name = "htdemucs"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Force to run on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Initialization ---")
print(f"Using hardware: {device.upper()}")

# Load the AI model demucs
print(f"Loading '{model_name}' model...")
model = get_model(model_name)
model.to(device)
model.eval() # Set to evaluation mode to disable dropout/training behavior -> maximize efficiency
print("Model loaded successfully.\n")


# Process song
valid_extensions = ('.wav', '.mp3', '.flac')
songs = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)] #append valid extension file into song array

print(f"Found {len(songs)} songs to process. Starting batch job...\n")

for i, song_filename in enumerate(songs): #list out the file in song array
    song_path = os.path.join(input_dir, song_filename) #take the full path of the song file
    song_name = os.path.splitext(song_filename)[0] #để lấy tên bài hát và loại bỏ extension

    print(f"[{i+1}/{len(songs)}] Processing: {song_filename}...")

    # Create a specific folder for this song's stems
    song_output = os.path.join(output_dir, song_name)
    os.makedirs(song_output, exist_ok=True)

    # Initialize wav and sources in case it cannot loaded
    wav = None
    sources = None
    
    try:
        # Trim out the silence part
        # Load audio
        wav, sr = torchaudio.load(song_path)

        # Convert Pytorch tensor to Numpy array for librosa trimming
        wav_np = wav.numpy()

        # Trimming (top_db = 60 is the standard threshold for digital silence )
        wav_trimmed_np, index = librosa.effects.trim(wav_np, top_db = 60)
        
        # Confirm if cut process happen
        cut_samples = wav_np.shape[1] - wav_trimmed_np.shape[1]
        print(f"-> Trimmed {cut_samples / sr:.2f} seconds of dead air.")

        # Convert back to Pytorch tensor and move to GPU
        wav = torch.from_numpy(wav_trimmed_np).to(device)
        
        # Add the batch dimension (Channels, Time) -> (Batch, Channels, Time)
        wav = wav.unsqueeze(0)
        
        # Apply Demucs Separation
        with torch.no_grad():       #Tắt tính năng lưu dữ liệu để học của torch -> giảm lượng GPU sử dụng

            #Tách nhạc
            # split=True -> tách file nhạc dài ra các chunk nhỏ để tách từng file 1 tránh tràn bộ nhớ
            sources = apply_model(model, wav, split=True, device=device)[0]    
        
        # Define stem names: [drums, bass, other, vocals]
        stem_names = ["drums", "bass", "other", "vocals"]
        # required_stems = ["other", "vocals"]
        
        #Create stem_file.wave with corresponding index from the sources
        for stem_idx, stem_tensor in enumerate(sources):
            stem_name = "melody" if stem_names[stem_idx] == "other" else stem_names[stem_idx]
            output_file = os.path.join(song_output, f"{stem_name}.wav")
            
            # Save the stem (Move tensor back to CPU before saving)
            torchaudio.save(output_file, stem_tensor.cpu(), sr)
            
        print(f"    -> Success! Stems saved to {song_output}/")
        
    except Exception as e:
        print(f"    -> ERROR processing {song_filename}: {e}")
        print("    -> Skipping to next song...")
    
    # Make sure sources and wav are not empty/load correctly before clean up
    finally:
        if sources is not None:
            del sources # Giải phóng biến
        if wav is not None:
            del wav

    torch.cuda.empty_cache() # Ép sạch cache GPU
    gc.collect() # Ép sạch rác trong RAM hệ thống

print("\n=== BATCH PROCESSING COMPLETE ===")