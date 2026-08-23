import unicodedata
import re
import os

def clean_file_name(name):
    # Separate file name and file extension
    base_name, extension = os.path.splitext(name)
    base_name = str(base_name)
    extension = str(extension)

    # Convert Vietnamese letter into raw Latin characters
    base_name = base_name.replace('đ', 'd').replace('Đ', 'D')

    # Convert Vietnamese accents into raw Latin characters
    # Separate Vietnamese accent and letter
    normalize_form_kd = unicodedata.normalize('NFKD', base_name)

    # Remove all the separate accent, only keep letter
    only_ascii = normalize_form_kd.encode('ASCII', 'ignore').decode('utf-8')

    # Replace space and special characters with underscore '_', clean up duplicates
    # Find any character that is NOT a lowercase letter, uppercase letter, or number and replace with '_'
    clean = re.sub(r'[^a-zA-Z0-9]', '_', only_ascii)

    # Remove any '_' that consecutively sitting next to each other
    clean = re.sub(r'_+', '_', clean).strip('_')

    extension = str(extension)
    
    return clean.lower() + extension.lower()

source_dir = r"Dataset\Music\Real_therapy"
all_file = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

for idx, file_name in enumerate(all_file):
    original_dir = os.path.join(source_dir, file_name)
    clean_name = clean_file_name(file_name)
    clean_dir = os.path.join(source_dir, clean_name)

    # Don't rename if the name is already clean
    if original_dir != clean_dir:
        if not os.path.exists(clean_dir):
            os.rename(original_dir, clean_dir)
            print(f"Renamed: {file_name} -> {clean_name}")

        else:
            print(f"Collision Warning: Cannot rename '{file_name}'. '{clean_name}' already exists.")