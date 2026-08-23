import librosa
import DSP_pipeline.config as config
import numpy as np
import os

def analyze_reference_from_memory(filepath, start_sec, sample_rate= config.SAMPLE_RATE, duration_sec=60.0):
    print(f"Đang tải và phân tích {duration_sec} giây đầu của: {os.path.basename(filepath)}...")
    
    # 1. Đọc metadata để lấy tổng thời lượng (không load dữ liệu vào RAM)
    total_duration = librosa.get_duration(path=filepath)

    # Đảm bảo duration không vượt quá thời lượng thực tế của bài hát
    actual_duration = min(duration_sec, total_duration)

    # 3. Load đoạn âm thanh chỉ định
    audio, sr = librosa.load(
        filepath, 
        sr=sample_rate, 
        offset=start_sec, 
        duration=actual_duration, 
        mono=True
    )
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # 2. Phân tích trực tiếp trên RAM
    # -- 1. BPM (60-80) --
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo_candidates = librosa.beat.tempo(
        onset_envelope=onset_env, sr=sample_rate, start_bpm=65, aggregate=None
    )
    bpm = float(np.median(tempo_candidates)) if len(tempo_candidates) > 0 else 0.0

    # Trích xuất STFT và tần số dùng chung cho Spectral Balance và Frequency Profile
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # -- 2. Spectral Balance: Roll off at 5kHz --
    freq_mask_5k = freqs <= 5000
    energy_below_5k = np.sum(stft[freq_mask_5k, :] ** 2)
    total_energy_spec = np.sum(stft ** 2) + 1e-9
    ratio_5k = float(energy_below_5k / total_energy_spec)
    
    cumulative_energy = np.cumsum(np.sum(stft ** 2, axis=1))
    total_cum = cumulative_energy[-1] if cumulative_energy[-1] > 0 else 1.0
    rolloff_idx = np.searchsorted(cumulative_energy, total_cum * 0.85)
    actual_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # -- 3. Dynamic Range: Low (< 6dB variance) --
    frame_length = int(sample_rate * 0.1)
    hop_length = frame_length // 2
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    active_rms = rms[rms > 10**(-50 / 20)]
    if len(active_rms) > 0:
        rms_db = 20 * np.log10(active_rms)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    else:
        dynamic_range = 0.0

    # -- 4. Frequency Profile: Boost 100Hz – 400Hz --
    target_mask = (freqs >= 100) & (freqs <= 400)
    ref_mask = (freqs > 400) & (freqs <= 2000)
    target_energy = np.mean(stft[target_mask, :] ** 2)
    ref_energy = np.mean(stft[ref_mask, :] ** 2)
    boost_db = 10 * np.log10((target_energy + 1e-9) / (ref_energy + 1e-9))

    # -- 5. Modulation: 0.05Hz – 0.1Hz (LFO trên biên độ RMS) --
    rms_mod = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    if len(rms_mod) > sample_rate / 512:
        f_rms = np.fft.rfft(rms_mod - np.mean(rms_mod))
        freqs_rms = np.fft.rfftfreq(len(rms_mod), d=512 / sample_rate)
        target_lfo_mask = (freqs_rms >= 0.05) & (freqs_rms <= 0.1)
        if np.any(target_lfo_mask):
            peak_idx = np.argmax(np.abs(f_rms[target_lfo_mask]))
            dominant_lfo = float(freqs_rms[target_lfo_mask][peak_idx])
        else:
            dominant_lfo = 0.0
    else:
        dominant_lfo = 0.0

    # -- 6. Pitch Interval Limit: 12 semitones/beat --
    f0, voiced_flag, _ = librosa.pyin(
        y=audio, fmin=65.4, fmax=2093.0, sr=sr, hop_length=1024
    )
    valid_f0 = f0[voiced_flag & (f0 > 0)]
    if len(valid_f0) >= 2:
        semitones = 12 * np.log2(valid_f0[1:] / (valid_f0[:-1] + 1e-9))
        max_pitch_jump = float(np.max(np.abs(semitones)))
    else:
        max_pitch_jump = 0.0

    # -- 7. Note Overlap (Legato): 70% – 80% --
    rms_short = librosa.feature.rms(y=audio, frame_length=512, hop_length=512)[0]
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames', backtrack=True)
    overlap_ratios = []
    
    for i in range(len(onset_frames) - 1):
        curr_f = int(onset_frames[i])
        next_f = int(onset_frames[i + 1])
        if next_f >= len(rms_short) or curr_f >= len(rms_short):
            continue
        peak_end = min(curr_f + 10, next_f)
        if peak_end <= curr_f:
            continue
        peak_e = np.max(rms_short[curr_f:peak_end])
        if peak_e < 1e-6:
            continue
        ratio = min(1.0, max(0.0, float(rms_short[next_f] / peak_e)))
        overlap_ratios.append(ratio)
    
    median_overlap = float(np.median(overlap_ratios) * 100) if overlap_ratios else 0.0

    # In kết quả báo cáo đầy đủ
    print("=" * 60)
    print("  KẾT QUẢ PHÂN TÍCH MẪU CHUẨN (60s ĐẦU TRÊN RAM)")
    print("=" * 60)
    print(f"  • BPM (60-80)                : {bpm:.1f} BPM {'(Đạt)' if 60 <= bpm <= 80 else '(Chưa đạt)'}")
    print(f"  • Spectral Balance (≤5kHz)   : {actual_rolloff:.1f} Hz (Tỷ lệ dưới 5kHz: {ratio_5k*100:.1f}%)")
    print(f"  • Dynamic Range (<6dB)       : {dynamic_range:.2f} dB variance")
    print(f"  • Frequency Profile (Boost)  : {boost_db:+.2f} dB (100Hz - 400Hz)")
    print(f"  • Modulation (0.05-0.1Hz)    : {dominant_lfo:.3f} Hz")
    print(f"  • Pitch Interval (≤12 semit) : {max_pitch_jump:.1f} semitones")
    print(f"  • Note Overlap (70-80%)      : {median_overlap:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    reference_file = r"Dataset\Raw\Tier_1+2(60-100BPM)(131)\anh_danh_roi_nguoi_yeu_nay_andiez_ft_amee_ost_ttvkobe.wav"
    analyze_reference_from_memory(reference_file, start_sec=60.0, duration_sec=60.0)