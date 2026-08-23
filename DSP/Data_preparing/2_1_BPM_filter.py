import os
import csv
import librosa
import warnings

# Disable librosa warnings to keep the console clean
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
raw_dir = r"Dataset\Raw\Tier_4_Archive (Over 120 BPM)"
csv_path = r"Dataset\Raw\bpm_metadata4_sorted_log.csv"

# Valid audio formats
valid_extensions = ('.wav', '.mp3', '.flac')
songs = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_extensions)]

print(f"Starting BPM analysis for {len(songs)} songs...")
print(f"Results will be saved to: {csv_path}\n")

# Open CSV file for writing
with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # Create column headers
    writer.writerow(["Filename", "BPM"])
    
    for i, song_filename in enumerate(songs):
        song_path = os.path.join(raw_dir, song_filename)
        
        try:
            print(f"[{i+1}/{len(songs)}] Scanning: {song_filename}...", end=" ")
            
            # OPTIMIZATION: Only load the first 60 seconds with a low Sample Rate 
            # for ultra-fast BPM scanning.
            y, sr = librosa.load(song_path, sr=22050, duration=60.0)
            
            # Analyze BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Compatibility handling (newer librosa versions return a 1D array, 
            # older versions return a float)
            bpm_value = float(tempo[0]) if isinstance(tempo, (list, tuple)) or hasattr(tempo, '__iter__') else float(tempo)
            
            # Round BPM to 1 decimal place (e.g., 120.0, 75.5)
            bpm_rounded = round(bpm_value, 1)
            
            print(f"BPM: {bpm_rounded}")

            # Write the data row to the CSV immediately 
            # (prevents data loss if a crash occurs mid-process)
            writer.writerow([song_filename, bpm_rounded])
            
        except Exception as e:
            print(f"ERROR! Skipping. Details: {e}")
            writer.writerow([song_filename, "ERROR"])

print(f"\n=== COMPLETE! Results saved to {os.path.basename(csv_path)} ===")