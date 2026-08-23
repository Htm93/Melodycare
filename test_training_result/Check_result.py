import os
import librosa
import config
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf
from tqdm import tqdm

# =====================================================================
# THERAPY COMPLIANCE CHECKER
# Verifies processed audio files meet all 8 ASD therapy requirements
# =====================================================================

def check_bpm(audio, sample_rate):
    """
    Check if BPM is within ASD therapy range (60-80, target 60).
    Uses multiple strategies and picks the most reliable result.
    """
    # onset_strength: used by check_bpm (2 strategies both need it)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)

    # Strategy 1: Standard beat tracking với hint range 50-90
    # start_bpm tells librosa to prioritize this tempo range
    tempo_standard, _ = librosa.beat.beat_track(
        onset_envelope = onset_env,
        sr=sample_rate,
        start_bpm=65,       # hint: expect around 60-80 BPM
        tightness=80        # higher = stricter tempo tracking
    )
    tempo_standard = float(np.squeeze(tempo_standard))

    # Strategy 2: Tempogram-based detection (more robust after time-stretch)
    # Tempogram analyzes periodicity directly from onset strength
    

   # Get tempo candidates from tempogram
    tempo_tempogram = librosa.feature.rhythm.tempo(
        onset_envelope=onset_env,
        sr=sample_rate,
        start_bpm=65,
        aggregate=np.median   # use median instead of mean — more robust
    )
    tempo_tempogram = float(np.squeeze(tempo_tempogram))

    # Strategy 3: Check if either value is a harmonic of the other
    # e.g. 120 BPM when true is 60 BPM → divide by 2
    def normalize_to_therapy_range(bpm):
        """Fold BPM into 60-80 range by halving or doubling."""
        while bpm > 80:
            bpm /= 2
        while bpm < 50:
            bpm *= 2
        return bpm

    normalized_standard   = normalize_to_therapy_range(tempo_standard)
    normalized_tempogram  = normalize_to_therapy_range(tempo_tempogram)

    # Pick the estimate closest to 60-80 range after normalization
    # If both agree after normalization → high confidence
    if abs(normalized_standard - normalized_tempogram) < 5:
        # Both agree → use average
        final_bpm = (normalized_standard + normalized_tempogram) / 2
    else:
        # Disagree → prefer tempogram (more robust to time-stretch artifacts)
        final_bpm = normalized_tempogram

    final_bpm = round(final_bpm, 1)

    return {
        "value"              : final_bpm,
        "pass"               : 59.0 <= final_bpm <= 81.0,
        "ideal"              : 60.0,
        "distance"           : round(abs(final_bpm - 60.0), 1),
        "raw_standard"       : round(tempo_standard, 1),
        "raw_tempogram"      : round(tempo_tempogram, 1),
    }


def check_spectral_rolloff(stft, freqs, target_cutoff=5000):
    """
    Check if spectral energy rolls off at 5kHz.
    Measures what percentage of spectral energy is below the cutoff.

    Returns:
        dict: % energy below cutoff, pass/fail
    """

    # Calculate total energy and energy below cutoff
    total_energy  = np.sum(stft ** 2)
    below_cutoff  = np.sum(stft[freqs <= target_cutoff, :] ** 2)

    pct_below = (below_cutoff / (total_energy + 1e-9)) * 100

    # Pass if >= 85% of energy is below 5kHz (rolloff achieved)
    return {
        "value"         : round(pct_below, 1),
        "unit"          : "% energy below 5kHz",
        "pass"          : pct_below >= 85.0,
        "threshold"     : "≥ 85%"
    }


def check_dynamic_range(audio_mono, sample_rate, max_variance_db=config.MAX_DYNAMIC):
    """
    Check if dynamic range (RMS variance) is below 6dB.
    Only measures segments where audio is actually present (non-silent),
    avoiding false inflation from silence gaps between notes.

    True dynamic range of digital audio:
        16-bit: max ~96dB
        24-bit: max ~144dB
    If result exceeds ~50dB something is wrong with the measurement.
    """
    # ── 1. Calculate RMS envelope ──────────────────────────────────────
    # 100ms windows — long enough to capture note energy, short enough for detail
    frame_length = int(sample_rate * 0.1)
    hop_length   = frame_length // 2

    rms = librosa.feature.rms(
        y=audio_mono,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # ── 2. Filter out silent frames BEFORE converting to dB ───────────
    # Silent frames (near 0 linear) cause extreme negative dB values
    # which inflate the variance enormously
    #
    # Noise floor threshold: -50dB = 10^(-50/20) ≈ 0.00316 linear
    # Any frame below this is considered silence, not music
    noise_floor_linear = 10 ** (-50 / 20)   # -50dB noise floor
    active_frames = rms[rms > noise_floor_linear]

    if len(active_frames) < 10:
        return {
            "value"     : 0.0,
            "unit"      : "dB variance",
            "pass"      : True,
            "threshold" : f"< {max_variance_db} dB",
            "note"      : "Audio is mostly silent — skipping"
        }

    # ── 3. Convert only active frames to dB ───────────────────────────
    # Now safe to convert because all values are well above 0
    rms_db = 20 * np.log10(active_frames)

    # ── 4. Use percentile range instead of max - min ───────────────────
    # max - min is too sensitive to outliers (a single loud transient)
    # P95 - P5 captures the effective dynamic range, ignoring extremes
    #
    # Example:
    #   99 frames at -20dB, 1 frame at -5dB (transient)
    #   max - min = 15dB  ← misleading
    #   P95 - P5  ≈  2dB  ← reflects actual consistent range
    p5  = float(np.percentile(rms_db, 5))
    p95 = float(np.percentile(rms_db, 95))
    variance_db = p95 - p5

    return {
        "value"         : round(variance_db, 2),
        "unit"          : "dB variance (P95-P5 of active frames)",
        "pass"          : variance_db < max_variance_db,
        "threshold"     : f"< {max_variance_db} dB",
        "active_frames" : len(active_frames),
        "p5_db"         : round(p5, 1),
        "p95_db"        : round(p95, 1),
    }


def check_frequency_boost(stft, freqs, low=100, high=400):
    """
    Check if 100Hz-400Hz band is boosted relative to neighboring bands.
    Compares energy in target band vs bands immediately above and below.

    Returns:
        dict: relative boost in dB, pass/fail
    """

    # Energy in target band (100-400Hz)
    target_mask = (freqs >= low) & (freqs <= high)
    target_energy = np.mean(stft[target_mask, :] ** 2)

    # Energy in reference band (400-2000Hz) for comparison
    # If target > reference → boost exists
    ref_mask = (freqs > high) & (freqs <= 2000)
    ref_energy = np.mean(stft[ref_mask, :] ** 2)

    # Convert ratio to dB
    boost_db = 10 * np.log10((target_energy + 1e-9) / (ref_energy + 1e-9))

    return {
        "value"     : round(boost_db, 2),
        "unit"      : "dB (target band vs 400-2kHz reference)",
        "pass"      : boost_db > 0,      # target band louder than reference = boost exists
        "threshold" : "> 0 dB (100-400Hz louder than 400-2kHz)"
    }


def check_lfo_modulation(audio_mono, sample_rate, low_hz=0.05, high_hz=0.1):
    """
    Check if LFO modulation (volume breathing) is present in 0.05-0.1Hz range.

    Strategy: Instead of finding the dominant frequency (which gets confused
    by natural phrase dynamics), we band-pass filter the RMS envelope into
    exactly the LFO range and measure how much energy exists there.

    If apply_lfo_modulation worked correctly:
    → The filtered signal should have measurable, consistent amplitude
    → We compare LFO-band energy vs total envelope energy
    → High ratio = LFO is present and dominant in that band
    """
    # ── 1. Extract RMS envelope ───────────────────────────────────────
    hop_length   = 512
    rms_envelope = librosa.feature.rms(
        y=audio_mono,
        frame_length=2048,
        hop_length=hop_length
    )[0].astype(np.float64)

    # Envelope sample rate (how many RMS values per second)
    envelope_sr = sample_rate / hop_length   # ≈ 93.75 fps

    # ── 2. Remove DC offset and very slow trend ───────────────────────
    # Subtract moving average to remove overall loudness trend
    # (crescendo/diminuendo) which would otherwise dominate
    from scipy.signal import butter, sosfilt, detrend
    rms_detrended = detrend(rms_envelope, type='linear')   # remove linear trend

    # ── 3. Band-pass filter envelope into LFO range (0.05-0.1 Hz) ────
    # This isolates ONLY the frequency range our LFO operates in
    nyquist_env = envelope_sr / 2.0
    low_norm    = low_hz  / nyquist_env
    high_norm   = high_hz / nyquist_env

    # Need order=2 because the band is very narrow relative to Nyquist
    sos = butter(2, [low_norm, high_norm], btype='band', output='sos')
    lfo_filtered = sosfilt(sos, rms_detrended)

    # ── 4. Measure energy ratio ───────────────────────────────────────
    # How much of the total envelope energy is in the LFO band?
    total_energy = np.var(rms_detrended) + 1e-12
    lfo_energy   = np.var(lfo_filtered)  + 1e-12
    energy_ratio = float(lfo_energy / total_energy)

    # ── 5. Measure LFO amplitude (depth) ─────────────────────────────
    # How strong is the modulation in absolute terms?
    # apply_lfo_modulation uses depth=0.05 → ±5% amplitude change
    # In RMS terms this should be measurable
    lfo_amplitude_pct = float(np.std(lfo_filtered) / (np.mean(rms_envelope) + 1e-9)) * 100

    # ── 6. Also check dominant frequency for reporting ────────────────
    # Use autocorrelation instead of FFT — more robust for periodic signals
    # buried in noise
    from numpy.fft import rfft, rfftfreq

    # Zero-pad to increase frequency resolution
    n_fft     = max(len(rms_detrended) * 4, 65536)   # 4x zero-padding
    fft_mag   = np.abs(rfft(rms_detrended, n=n_fft))
    fft_freqs = rfftfreq(n_fft, d=1.0 / envelope_sr)

    # Find peak within LFO range
    lfo_mask = (fft_freqs >= low_hz * 0.5) & (fft_freqs <= high_hz * 1.5)
    if np.any(lfo_mask):
        peak_idx      = np.argmax(fft_mag[lfo_mask])
        dominant_freq = float(fft_freqs[lfo_mask][peak_idx])
    else:
        dominant_freq = 0.0

    # ── 7. Pass criteria ──────────────────────────────────────────────
    # Pass if:
    #   (a) dominant frequency is in range, OR
    #   (b) sufficient energy exists in LFO band (energy_ratio > 1%)
    #       AND amplitude is detectable (> 0.5%)
    freq_in_range    = low_hz <= dominant_freq <= high_hz
    energy_present   = energy_ratio > 0.01 and lfo_amplitude_pct > 0.5
    passed           = freq_in_range or energy_present

    return {
        "value"             : round(dominant_freq, 4),
        "unit"              : "Hz (dominant LFO frequency)",
        "pass"              : passed,
        "threshold"         : f"{low_hz}-{high_hz} Hz",
        "energy_ratio_pct"  : round(energy_ratio * 100, 2),
        "lfo_amplitude_pct" : round(lfo_amplitude_pct, 3),
        "freq_in_range"     : freq_in_range,
        "energy_present"    : energy_present,
        "note"              : (
            "Pass via frequency detection" if freq_in_range else
            "Pass via energy detection"    if energy_present else
            f"LFO band energy {energy_ratio*100:.2f}% — too weak or absent"
        )
    }


def check_pitch_intervals(f0, voiced_flag, max_semitones=12):
    """
    Check if pitch intervals stay within 12 semitones (1 octave) per beat.

    Returns:
        dict: max pitch jump detected, pass/fail, number of violations
    """

    f0 = np.nan_to_num(f0)

    semitones = np.zeros_like(f0)
    valid_idx = f0 > 0
    if np.any(valid_idx):
        semitones[valid_idx] = librosa.hz_to_midi(f0[valid_idx])

    jumps = np.diff(semitones)

    # Only count jumps where both frames have actual pitch
    voiced_jumps = np.abs(jumps[voiced_flag[:-1] & voiced_flag[1:]])

    if len(voiced_jumps) == 0:
        return {"value": 0.0, "pass": True, "violations": 0, "note": "No pitched content detected"}

    max_jump   = float(np.max(voiced_jumps))
    violations = int(np.sum(voiced_jumps > max_semitones))

    return {
        "value"      : round(max_jump, 2),
        "unit"       : "semitones (max jump)",
        "pass"       : max_jump <= max_semitones,
        "violations" : violations,
        "threshold"  : f"≤ {max_semitones} semitones"
    }


def check_legato_overlap(audio_mono, sample_rate, onset_frames, min_pct=0.70, max_pct=0.82):
    """
    Estimate note overlap by measuring how much energy remains
    at the next onset relative to the current onset peak.

    Core insight:
        No overlap  → energy drops to ~0 before next note starts
        70% overlap → energy still at ~70% of peak when next note starts
        100% overlap → energy never drops, notes fully blend

    This measures the ACTUAL residual energy at onset boundaries,
    which directly reflects how much the previous note is still
    sounding when the next note begins — i.e. legato overlap.
    """
    # ── 1. Get onset times ────────────────────────────────────────────
    if len(onset_frames) < 4:
        return {
            "value" : 0.0,
            "pass"  : False,
            "note"  : "Too few onsets detected (< 4)",
            "p25"   : 0.0,
            "p75"   : 0.0,
            "iqr"   : 0.0,
        }

    # ── 2. Compute energy envelope ────────────────────────────────────
    # ĐỒNG BỘ: Sử dụng hop_length=512 để khớp tuyệt đối với tham số mặc định 
    # của librosa.onset.onset_detect đã tạo ra onset_frames.
    hop_length = 512
    energy = librosa.feature.rms(
        y=audio_mono,
        frame_length=512,
        hop_length=hop_length
    )[0]

    # ── 3. For each pair of consecutive onsets, measure residual energy ──
    overlap_ratios = []

    for i in range(len(onset_frames) - 1):
        # Vì hop_length đã đồng bộ, chỉ số frame của onset và energy ánh xạ 1:1
        current_energy_frame = int(onset_frames[i])
        next_energy_frame    = int(onset_frames[i + 1])

        # Safety bounds
        if next_energy_frame >= len(energy) or current_energy_frame >= len(energy):
            continue

        # Peak energy in the window just after current onset
        peak_window_end = min(current_energy_frame + 10, next_energy_frame)
        if peak_window_end <= current_energy_frame:
            continue

        peak_energy = np.max(energy[current_energy_frame:peak_window_end])

        if peak_energy < 1e-6:   # skip silent segments
            continue

        # Energy at the moment the next note begins
        residual_energy = energy[next_energy_frame]

        # Overlap ratio = how much of the previous note is still present
        ratio = float(residual_energy / peak_energy)
        ratio = min(1.0, max(0.0, ratio))   # clamp to [0, 1]
        overlap_ratios.append(ratio)

    if len(overlap_ratios) < 3:
        return {
            "value" : 0.0,
            "pass"  : False,
            "note"  : "Could not compute enough valid overlap pairs",
            "p25"   : 0.0,
            "p75"   : 0.0,
            "iqr"   : 0.0,
        }

    # ── 4. Use median to avoid outliers ──────────────────────────────
    median_overlap = float(np.median(overlap_ratios)) * 100
    median_overlap = round(median_overlap, 1)

    # ── 5. Distribution info for debugging ───────────────────────────
    p25 = round(float(np.percentile(overlap_ratios, 25)) * 100, 1)
    p75 = round(float(np.percentile(overlap_ratios, 75)) * 100, 1)

    # Calculate Interquartile Range to check stability
    iqr = round(p75 - p25, 1)

    return {
        "value"          : median_overlap,
        "unit"           : "% residual energy at next onset (median)",
        "pass"           : (min_pct * 100 <= median_overlap <= max_pct * 100) and (iqr <= 30.0),
        "threshold"      : f"{min_pct*100:.0f}%-{max_pct*100:.0f}%",
        "p25"            : p25,
        "p75"            : p75,
        "iqr"            : iqr,
        "n_pairs"        : len(overlap_ratios),
    }


def check_file(filepath, sample_rate=48000):
    """
    Run all ASD therapy compliance checks on a single audio file.
    Shared computations (STFT, onset_strength, pyin) are computed ONCE
    and reused across all checks to avoid redundant expensive operations.
    """
    audio, sr = librosa.load(filepath, sr=sample_rate, mono=True)

    # ── Shared computations — computed ONCE, reused by multiple checks ──

    # STFT: used by check_spectral_rolloff AND check_frequency_boost
    stft  = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)

    # pyin: used by check_pitch_intervals
    # hop_length=1024 is sufficient for pitch interval detection
    f0, voiced_flag, _ = librosa.pyin(
        y=audio, fmin=65.4, fmax=2093.0, sr=sr, hop_length=1024
    )

    # onset frames: used by check_legato_overlap
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, units='frames', backtrack=True
    )

    # ── Run all checks, passing shared data ──
    # legato_result computed once and reused for p25/p75 below -- it was
    # previously called 3 separate times here (same deterministic inputs,
    # same expensive RMS/onset work redone twice for nothing).
    legato_result = check_legato_overlap(audio, sr, onset_frames)

    results = {
        "file"             : os.path.basename(filepath),
        "bpm"              : check_bpm(audio, sr),
        "spectral_rolloff" : check_spectral_rolloff(stft, freqs),
        "dynamic_range"    : check_dynamic_range(audio, sr),
        "frequency_boost"  : check_frequency_boost(stft, freqs),
        "lfo_modulation"   : check_lfo_modulation(audio, sr),
        "pitch_intervals"  : check_pitch_intervals(f0, voiced_flag),
        "legato_overlap"   : legato_result,
        "p25"              : legato_result.get("p25", 0.0),
        "p75"              : legato_result.get("p75", 0.0)
    }

    check_keys = ["bpm", "spectral_rolloff", "dynamic_range",
                  "frequency_boost", "lfo_modulation", "pitch_intervals", "legato_overlap"]
    results["overall_pass"] = all(results[k]["pass"] for k in check_keys)
    return results


def print_report(results):
    """
    Print a formatted compliance report for a single file.
    """
    status = "✓ PASS" if results["overall_pass"] else "✗ FAIL"
    print(f"\n{'='*60}")
    print(f"  {results['file']}")
    print(f"  Overall: {status}")
    print(f"{'='*60}")

    checks = {
        "BPM (60-80)"           : results["bpm"],
        "Spectral Rolloff 5kHz" : results["spectral_rolloff"],
        "Dynamic Range <13dB"    : results["dynamic_range"],
        "Freq Boost 100-400Hz"  : results["frequency_boost"],
        "LFO Modulation"        : results["lfo_modulation"],
        "Pitch Intervals ≤12st" : results["pitch_intervals"],
        "Legato Overlap 70-80%" : results["legato_overlap"],
    }

    for name, result in checks.items():
        icon  = "✓" if result["pass"] else "✗"
        value = result.get("value", "?")
        unit  = result.get("unit", "")
        print(f"  {icon} {name:<28} {value} {unit}")

    print()

import csv
import os

def export_results_to_csv(all_results, output_filename="therapy_compliance_report.csv"):
    """
    Xuất danh sách kết quả kiểm tra ra tệp CSV báo cáo tổng hợp.
    Dữ liệu được chia cột rõ ràng, định dạng dễ đọc và bổ sung p25, p75.
    """
    # Định nghĩa các tiêu đề cột chuẩn hóa để dễ đọc trên Excel
    fieldnames = [
        "File Name",
        "Overall Status",
        "BPM Value", "BPM Pass",
        "Rolloff 5kHz (%)", "Rolloff Pass",
        "Dynamic Range (dB)", "Dyn Range Pass",
        "Freq Boost (dB)", "Boost Pass",
        "LFO Freq (Hz)", "LFO Pass",
        "Max Pitch Jump (st)", "Pitch Pass",
        "Legato Overlap (%)", "Legato p25 (%)", "Legato p75 (%)", "Legato IQR", "Legato Pass"
    ]

    try:
        with open(output_filename, mode="w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Ghi tiêu đề cột
            writer.writeheader()
            
            for res in all_results:
                # Trích xuất dữ liệu Legato an toàn (xử lý trường hợp legato_overlap bị ghi đè thành float)
                legato_data = res.get("legato_overlap", {})
                if isinstance(legato_data, dict):
                    l_val  = legato_data.get("value", "")
                    l_pass = legato_data.get("pass", False)
                    l_iqr  = legato_data.get("iqr", "")
                else:
                    l_val  = legato_data
                    l_pass = res.get("overall_pass", False)
                    l_iqr  = ""

                # Định dạng dữ liệu từng hàng
                row = {
                    "File Name": res.get("file", ""),
                    "Overall Status": "PASS" if res.get("overall_pass") else "FAIL",
                    
                    "BPM Value": res.get("bpm", {}).get("value", "") if isinstance(res.get("bpm"), dict) else "",
                    "BPM Pass": "PASS" if isinstance(res.get("bpm"), dict) and res.get("bpm", {}).get("pass") else "FAIL",
                    
                    "Rolloff 5kHz (%)": res.get("spectral_rolloff", {}).get("value", "") if isinstance(res.get("spectral_rolloff"), dict) else "",
                    "Rolloff Pass": "PASS" if isinstance(res.get("spectral_rolloff"), dict) and res.get("spectral_rolloff", {}).get("pass") else "FAIL",
                    
                    "Dynamic Range (dB)": res.get("dynamic_range", {}).get("value", "") if isinstance(res.get("dynamic_range"), dict) else "",
                    "Dyn Range Pass": "PASS" if isinstance(res.get("dynamic_range"), dict) and res.get("dynamic_range", {}).get("pass") else "FAIL",
                    
                    "Freq Boost (dB)": res.get("frequency_boost", {}).get("value", "") if isinstance(res.get("frequency_boost"), dict) else "",
                    "Boost Pass": "PASS" if isinstance(res.get("frequency_boost"), dict) and res.get("frequency_boost", {}).get("pass") else "FAIL",
                    
                    "LFO Freq (Hz)": res.get("lfo_modulation", {}).get("value", "") if isinstance(res.get("lfo_modulation"), dict) else "",
                    "LFO Pass": "PASS" if isinstance(res.get("lfo_modulation"), dict) and res.get("lfo_modulation", {}).get("pass") else "FAIL",
                    
                    "Max Pitch Jump (st)": res.get("pitch_intervals", {}).get("value", "") if isinstance(res.get("pitch_intervals"), dict) else "",
                    "Pitch Pass": "PASS" if isinstance(res.get("pitch_intervals"), dict) and res.get("pitch_intervals", {}).get("pass") else "FAIL",
                    
                    "Legato Overlap (%)": l_val,
                    # Lấy p25 và p75 từ khóa gốc theo như cấu trúc dict hiện tại
                    "Legato p25 (%)": res.get("p25", ""),
                    "Legato p75 (%)": res.get("p75", ""),
                    "Legato IQR": l_iqr,
                    "Legato Pass": "PASS" if l_pass else "FAIL",
                }
                writer.writerow(row)
                
        print(f"\n[+] Đã xuất báo cáo thành công ra tệp: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"\n[✗] Lỗi khi ghi tệp CSV: {str(e)}")

from concurrent.futures import ProcessPoolExecutor, as_completed

def _check_one(args):
    """Top-level wrapper for parallel execution."""
    filepath, sample_rate = args
    try:
        return check_file(filepath, sample_rate), None
    except Exception as e:
        return None, (filepath, str(e))

def batch_check(output_dir, sample_rate=48000, max_workers=4):
    """
    Run compliance checks on all processed files in the output directory.
    Prints per-file reports and a final summary.

    Args:
        output_dir  : str — directory containing processed _therapeutic.wav files
        sample_rate : int — sample rate of the files (default 48000)
    """
        
    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("_therapeutic.wav")
    ]

    if not files:
        print(f"No _therapeutic.wav files found in {output_dir}")
        return

    print(f"Found {len(files)} files to check\n")

    args_list    = [(f, sample_rate) for f in files]
    all_results  = []
    failed_files = []

    # Tự động điều chỉnh số lượng tiến trình dựa trên luồng CPU trống
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_one, a): a[0] for a in args_list}
        with tqdm(total=len(futures), desc="Checking files") as pbar:
            for future in as_completed(futures):
                result, error = future.result()
                if error:
                    filepath, msg = error
                    # Sử dụng tqdm.write để không làm vỡ thanh tiến trình
                    tqdm.write(f"  ✗ ERROR: {os.path.basename(filepath)}: {msg}")
                    failed_files.append(os.path.basename(filepath))
                else:
                    all_results.append(result)
                    if not result["overall_pass"]:
                        failed_files.append(result["file"])
                pbar.update(1)

    # Chỉ in báo cáo chi tiết SAU KHI tiến trình batch hoàn tất
    for result in all_results:
        print_report(result)

    pass_count = len(all_results) - len(failed_files)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {pass_count}/{len(files)} files passed all checks")
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  ✗ {f}")
    print(f"{'='*60}\n")

    if all_results:
        export_results_to_csv(all_results)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    output_dir = r"Dataset\Music\processed\training_data\unsorted"
    batch_check(output_dir)