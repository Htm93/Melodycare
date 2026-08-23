import os
import pandas as pd
import shutil

# --- 1. CONFIGURATION ---
# Point this to one of your 5 CSV files (you can run this script 5 times, once for each CSV)
csv_source = r"Dataset\Raw\bpm_metadata6.csv" 
raw_directory = r"Dataset\Raw\Music6" # Where the files are currently located
target_directory = r"Dataset\Music\BPM_Sorted"

# --- 2. DEFINE BPM RANGES ---
def get_bpm_folder_name(bpm):
    """Categorizes a BPM value into a specific folder name."""
    try:
        bpm_value = float(bpm)
        if bpm_value <= 80:
            return "Tier_1_Golden (60-80 BPM)"
        elif bpm_value <= 100:
            return "Tier_2_Acceptable (81-100 BPM)"
        elif bpm_value <= 120:
            return "Tier_3_Heavy_Stretch (101-120 BPM)"
        else:
            return "Tier_4_Archive (Over 120 BPM)"
    except (ValueError, TypeError):
        # Catches strings like "ERROR" if librosa failed to read the BPM previously
        return "Tier_5_Unknown_Errors"

# --- 3. MAIN LOGIC ---
def sort_files_by_bpm():
    print(f"Reading {csv_source}...")
    data_table = pd.read_csv(csv_source)
    
    # Add a column to track where the file was moved
    if "New_Location" not in data_table.columns:
        data_table["New_Location"] = ""

    moved_count = 0

    for row_idx, row in data_table.iterrows():
        filename = row["Filename"]
        bpm = row["BPM"]
        
        # Determine the correct folder based on BPM
        folder_name = get_bpm_folder_name(bpm)
        
        # Define paths
        source_path = os.path.join(raw_directory, filename)
        destination_folder = os.path.join(target_directory, folder_name)
        destination_path = os.path.join(destination_folder, filename)
        
        # Create the BPM folder if it doesn't exist yet
        os.makedirs(destination_folder, exist_ok=True)
        
        # Move the file
        if os.path.exists(source_path):
            if not os.path.exists(destination_path):
                shutil.move(source_path, destination_path)
                data_table.at[row_idx, "New_Location"] = destination_folder
                moved_count += 1
            else:
                print(f"Skip: {filename} is already in the destination folder.")
        else:
            print(f"Missing: Cannot find {filename} in {raw_directory}")

    # Save the updated CSV
    log_name = os.path.basename(csv_source).replace(".csv", "_sorted_log.csv")
    log_path = os.path.join(target_directory, log_name)
    data_table.to_csv(log_path, index=False)
    
    print(f"=== Sorting Complete ===")
    print(f"Successfully moved {moved_count} files into BPM folders.")
    print(f"Log saved to: {log_path}")

# Run the function
if __name__ == "__main__":
    sort_files_by_bpm()