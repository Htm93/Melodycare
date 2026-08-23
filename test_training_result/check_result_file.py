import numpy as np
import matplotlib.pyplot as plt

# 1. Load the generated npy file
file_path = "translated_therapy.npy"

try:
    data = np.load(file_path)
    print("--- File Information ---")
    print(f"Successfully loaded: {file_path}")
    print(f"Array Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")
    print(f"Value Range: Min = {data.min():.4f}, Max = {data.max():.4f}")
    
    # 2. Reshape for visualization if necessary
    # AudioLDM2 outputs are typically (Batch, Channels, Freq, Time) -> e.g., (1, 1, 8, 1024)
    # We need a 2D matrix (Freq, Time) to plot it.
    squeezed_data = np.squeeze(data)
    
    print(f"Squeezed Shape for plotting: {squeezed_data.shape}")
    
    # If it's a 2D matrix, plot it
    if len(squeezed_data.shape) == 2:
        plt.figure(figsize=(10, 4))
        plt.imshow(squeezed_data, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Translated Therapy Spectrogram Latent")
        plt.xlabel("Time Bin")
        plt.ylabel("Frequency/Latent Bin")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        
        # Save the plot as an image so you can look at it
        plt.savefig("translated_spectrogram.png")
        print("Visualization saved as 'translated_spectrogram.png'")
        plt.show()
    else:
        print("Warning: Data has unexpected dimensions and cannot be plotted directly as a 2D image.")

except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Make sure the script runs in the correct folder.")
except Exception as e:
    print(f"An error occurred: {e}")