import os

training_path = r"Dataset\Music\processed\training_data"
original_path = r"Dataset\Raw\Tier_1+2(60-100BPM)(131)"

training_files = [f.replace("_therapeutic","") for f in os.listdir(training_path) if f.endswith(".wav")]
original_files = [f for f in os.listdir(original_path) if f.endswith(".wav")]

set_train = set(training_files)
set_orig = set(original_files)

mismatch = set_orig - set_train

if mismatch:
    print(f"Number of missing files: {len(mismatch)}")
    for orig in mismatch:
        print(f"Missing: {orig}")

else:
    print("No mismatch found!")