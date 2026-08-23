import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def plot_spectrogram_comparison(
    input_wav: str, 
    target_wav: str, 
    output_wav: str, 
    save_path: str, 
    sr: int = 48000, 
    n_fft: int = 2048, 
    hop_length: int = 512,
    y_axis_max: int = 8000
) -> None:
    """
    Generates a 3-way vertical spectrogram comparison for MelodyCare validation.
    
    Args:
        input_wav  : Path to raw commercial audio (Input Condition).
        target_wav : Path to DSP-processed audio (Therapy Target).
        output_wav : Path to Stable Audio generated audio (AI Output).
        save_path  : Destination path to save the .png figure.
        sr         : Sample rate (default 48000Hz).
        n_fft      : FFT window size.
        hop_length : Number of samples between successive frames.
        y_axis_max : Maximum frequency to display (Hz) to highlight the 5kHz roll-off.
    """
    files = {
        "Input (Raw Commercial)": input_wav,
        "Target (DSP NMT Pipeline)": target_wav,
        "Output (AI Generated)": output_wav
    }

    # Verify all files exist before processing
    for name, path in files.items():
        if not os.path.exists(path):
            logging.error(f"Missing file for {name}: {path}")
            return

    logging.info("Loading audio files and computing STFT...")
    
    spectrograms = {}
    for name, path in files.items():
        # Load audio (mono conversion for structural visualization)
        y, _ = librosa.load(path, sr=sr, mono=True)
        
        # Compute Short-Time Fourier Transform
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Convert amplitude to Decibels (dB)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        spectrograms[name] = S_db

    # Initialize Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)
    fig.suptitle("MelodyCare: Generative AI Spectral Validation", fontsize=16, fontweight='bold')

    # Global min/max for consistent colorbar scaling across all plots
    vmin = min(S.min() for S in spectrograms.values())
    vmax = max(S.max() for S in spectrograms.values())

    for idx, (title, S_db) in enumerate(spectrograms.items()):
        ax = axes[idx]
        img = librosa.display.specshow(
            S_db, 
            sr=sr, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='linear', 
            ax=ax, 
            cmap='magma',
            vmin=vmin, 
            vmax=vmax
        )
        
        ax.set_title(title, fontsize=12)
        ax.set_ylim([0, y_axis_max])
        ax.set_ylabel("Frequency (Hz)")
        
        # Draw a dashed line at 5kHz to highlight the clinical threshold
        ax.axhline(y=5000, color='cyan', linestyle='--', alpha=0.8, label="5kHz Cutoff")
        if idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (Seconds)")
    
    # Add a unified colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")

    # Save and cleanup
    plt.subplots_adjust(right=0.9)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Spectrogram comparison successfully saved to: {save_path}")

if __name__ == "__main__":
    # Define Linux-compatible paths
    INPUT_PATH = "/path/to/dataset/Karaoke_HD_Bung_Full_Beat_Gc_Bi_Anh_Tun_Newtitan.wav"
    TARGET_PATH = "/path/to/dataset/pass/buong_bui_anh_tuan_lyric_video_d_therapeutic.wav"
    OUTPUT_PATH = "/path/to/training_output/check_inference/test_str65.wav"
    SAVE_DEST = "/path/to/training_output/check_inference/spectrogram_comparison.png"

    plot_spectrogram_comparison(
        input_wav=INPUT_PATH,
        target_wav=TARGET_PATH,
        output_wav=OUTPUT_PATH,
        save_path=SAVE_DEST
    )