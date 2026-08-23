import config
import librosa
import numpy as np
from scipy.signal import fftconvolve

# =====================================================================
# SHARED UTILITIES (CACHED)
# =====================================================================

def _measure_overlap(audio_mono: np.ndarray, sample_rate: int, precomputed_onsets=None, precomputed_energy=None) -> dict:
    """
    Measure current legato overlap of audio, with support for precomputed onsets/energy.
    """
    hop_length = 256
    
    if precomputed_onsets is not None:
        onset_frames = precomputed_onsets
    else:
        onset_frames = librosa.onset.onset_detect(
            y=audio_mono, sr=sample_rate, units='frames', backtrack=True
        )
        
    if precomputed_energy is not None:
        energy = precomputed_energy
    else:
        energy = librosa.feature.rms(
            y=audio_mono, frame_length=512, hop_length=hop_length
        )[0]

    if len(onset_frames) < 4:
        return {"median": 0.0, "p25": 0.0, "p75": 0.0, "iqr": 0.0}

    ratios = []
    for i in range(len(onset_frames) - 1):
        cur = int(onset_frames[i])
        nxt = int(onset_frames[i + 1])
        if nxt >= len(energy) or cur >= len(energy):
            continue
        peak_end = min(cur + 10, nxt)
        if peak_end <= cur:
            continue
        peak = np.max(energy[cur:peak_end])
        if peak < 1e-6:
            continue
        residual = energy[nxt]
        ratio    = float(np.clip(residual / peak, 0.0, 1.0))
        ratios.append(ratio)

    if len(ratios) < 3:
        return {"median": 0.0, "p25": 0.0, "p75": 0.0, "iqr": 0.0}

    p25 = float(np.percentile(ratios, 25)) * 100
    p75 = float(np.percentile(ratios, 75)) * 100

    return {
        "median" : float(np.median(ratios)) * 100,
        "p25"    : p25,
        "p75"    : p75,
        "iqr"    : p75 - p25,
    }


def _is_valid(stats: dict, lo: float, hi: float, max_iqr: float = 35.0) -> bool:
    median_ok = lo <= stats["median"] <= hi
    iqr_ok    = stats["iqr"] < max_iqr
    return median_ok and iqr_ok


def _normalize_to_peak(audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_peak = np.max(np.abs(reference))
    out_peak = np.max(np.abs(audio))
    if out_peak > 0 and ref_peak > 0:
        return audio * (ref_peak / out_peak)
    return audio


def _build_reverb_ir(sr: int, decay_time_sec: float, wet_character: float = 0.4) -> np.ndarray:
    ir_length = int(sr * decay_time_sec)
    t         = np.linspace(0, 1, ir_length)
    decay_env = np.exp(-4.0 * t)
    noise     = np.random.randn(ir_length) * 0.15 * wet_character
    ir        = decay_env * (1.0 + noise)
    ir       /= (np.sum(np.abs(ir)) + 1e-9)
    return ir.astype(np.float32)


def _get_onset_samples(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    onsets = librosa.onset.onset_detect(
        y=audio_mono, sr=sr, units='samples', backtrack=True
    )
    return np.append(onsets, len(audio_mono))


def _get_per_note_energy(
    audio_mono   : np.ndarray,
    onset_samples: np.ndarray,
    sr           : int,
    hop_length   : int = 256,
    precomputed_energy = None,
) -> np.ndarray:
    energy = precomputed_energy if precomputed_energy is not None else librosa.feature.rms(
        y=audio_mono, frame_length=512, hop_length=hop_length
    )[0]
    note_energies = []
    for i in range(len(onset_samples) - 1):
        start_frame = int(onset_samples[i])   // hop_length
        end_frame   = int(onset_samples[i+1]) // hop_length
        end_frame   = min(end_frame, len(energy) - 1)
        if end_frame > start_frame:
            note_energies.append(float(np.mean(energy[start_frame:end_frame])))
        else:
            note_energies.append(0.0)
    return np.array(note_energies, dtype=np.float32)


# =====================================================================
# PATH A — ADDITIVE (OPTIMIZED)
# =====================================================================

def _increase_overlap(
    audio_stem    : np.ndarray,
    sr            : int,
    current_stats : dict,
    target_pct    : float,
    lo            : float,
    hi            : float,
    max_iqr       : float,
    min_wet       : float = 0.05,
    max_wet       : float = 0.45,
    max_iterations: int   = 5,
    cached_onsets = None,
    cached_onset_frames = None,
    cached_energy = None,
) -> np.ndarray:
    gap        = max(0.0, target_pct - current_stats["median"])
    decay_time = np.clip(0.1 + (gap / 40.0) * 0.7, 0.1, 0.8)
    ir         = _build_reverb_ir(sr, decay_time_sec=decay_time)

    onset_samples  = cached_onsets if cached_onsets is not None else _get_onset_samples(audio_stem, sr)
    # note_energies is derived from the ORIGINAL (unmodified) audio_stem, which
    # never changes in this function, so reusing cached_energy here is safe.
    note_energies  = _get_per_note_energy(audio_stem, onset_samples, sr, precomputed_energy=cached_energy)

    reverb_full = fftconvolve(audio_stem, ir, mode='full')[:len(audio_stem)]

    e_min = np.min(note_energies)
    e_max = np.max(note_energies)
    e_range = e_max - e_min + 1e-9
    wet_weights = 1.0 - (note_energies - e_min) / e_range

    def _render(global_wet: float) -> np.ndarray:
        wet_mask = np.zeros(len(audio_stem), dtype=np.float32)
        for i in range(len(onset_samples) - 1):
            start = int(onset_samples[i])
            end   = int(onset_samples[i + 1])
            wet_mask[start:end] = global_wet * wet_weights[i]
        dry_mask = 1.0 - wet_mask
        mixed    = dry_mask * audio_stem + wet_mask * reverb_full
        return _normalize_to_peak(mixed, audio_stem)

    def _score_at(global_wet: float):
        candidate = _render(global_wet)
        # NEVER reuse cached_energy here: the candidate has just been remixed
        # with reverb, so its RMS energy differs from the original's. Only
        # onset *positions* are stable across the correction -- energy must
        # be recomputed fresh on every candidate or the feedback loop is
        # measuring the wrong signal.
        stats = _measure_overlap(candidate, sr, precomputed_onsets=cached_onset_frames)
        return candidate, stats

    best_result = audio_stem.copy()
    best_median = current_stats["median"]
    best_iqr    = current_stats["iqr"]
    best_score  = abs(best_median - target_pct) + best_iqr

    # Overlap increases monotonically with wet level, so bisection converges
    # directly on the target instead of a fixed 5-point grid that can
    # straddle (and miss) the target window entirely.
    lo_wet, hi_wet = min_wet, max_wet

    for iteration in range(max_iterations):
        mid_wet = (lo_wet + hi_wet) / 2.0
        candidate, stats = _score_at(mid_wet)
        out_median = stats["median"]
        out_iqr    = stats["iqr"]

        iqr_acceptable = (out_iqr < max_iqr) or (out_iqr <= current_stats["iqr"])
        score          = abs(out_median - target_pct) + out_iqr

        if iqr_acceptable and score < best_score:
            best_result = candidate.copy()
            best_median = out_median
            best_iqr    = out_iqr
            best_score  = score

        if _is_valid(stats, lo, hi, max_iqr):
            best_result = candidate.copy()
            break

        if out_median < target_pct:
            lo_wet = mid_wet
        else:
            hi_wet = mid_wet

    return best_result.astype(audio_stem.dtype)


# =====================================================================
# PATH B — SUBTRACTIVE (OPTIMIZED)
# =====================================================================

def _decrease_overlap(
    audio_stem    : np.ndarray,
    sr            : int,
    current_stats : dict,
    target_pct    : float,
    lo            : float,
    hi            : float,
    max_iqr       : float,
    max_iterations: int = 5,
    cached_onsets = None,
    cached_onset_frames = None,
    cached_energy = None,
) -> np.ndarray:
    onset_samples  = cached_onsets if cached_onsets is not None else _get_onset_samples(audio_stem, sr)
    hop_length     = 256
    energy         = cached_energy if cached_energy is not None else librosa.feature.rms(
        y=audio_stem, frame_length=512, hop_length=hop_length
    )[0]

    excess         = current_stats["median"] - target_pct
    initial_rate   = np.clip(3.0 + (excess / 40.0) * 12.0, 3.0, 15.0)
    if current_stats["p75"] > 90.0:
        initial_rate = max(initial_rate, 10.0)

    best_result = audio_stem.copy()
    best_median = current_stats["median"]
    best_iqr    = current_stats["iqr"]
    best_score  = abs(best_median - target_pct) + best_iqr

    def _render(release_rate: float) -> np.ndarray:
        gain_envelope = np.ones(len(audio_stem), dtype=np.float32)

        for i in range(len(onset_samples) - 1):
            note_start = int(onset_samples[i])
            note_end   = int(onset_samples[i + 1])
            if note_end - note_start < sr * 0.02:
                continue

            start_frame = note_start // hop_length
            end_frame   = min(note_end // hop_length, len(energy) - 1)
            if start_frame >= end_frame:
                continue

            local_energy = energy[start_frame:end_frame]
            peak_frame   = start_frame + int(np.argmax(local_energy))
            peak_sample  = peak_frame * hop_length
            attack_end   = min(peak_sample + int(sr * 0.02), note_end)

            if attack_end >= note_end:
                continue

            release_len = note_end - attack_end
            t           = np.linspace(0, 1, release_len)
            floor       = float(np.clip(1.0 - (release_rate / 10.0), 0.0, 0.7))
            release_curve = floor + (1.0 - floor) * np.exp(-release_rate * t)

            gain_envelope[attack_end:note_end] = np.minimum(
                gain_envelope[attack_end:note_end],
                release_curve.astype(np.float32)
            )

        shaped = audio_stem * gain_envelope
        return _normalize_to_peak(shaped, audio_stem)

    def _score_at(release_rate: float):
        candidate = _render(release_rate)
        # Energy must be measured fresh on the shaped candidate -- the gain
        # envelope changes RMS energy directly, so cached_energy (computed
        # on the original, unshaped audio) would silently score the wrong
        # signal and the loop would never actually converge.
        stats = _measure_overlap(candidate, sr, precomputed_onsets=cached_onset_frames)
        return candidate, stats

    # Higher release_rate -> faster decay -> lower overlap (monotonic),
    # so bisection replaces the fixed grid scan that could straddle the
    # target window without ever landing a sample inside it.
    lo_rate, hi_rate = initial_rate, initial_rate * 5.0

    for iteration in range(max_iterations):
        mid_rate = (lo_rate + hi_rate) / 2.0
        candidate, stats = _score_at(mid_rate)
        out_median = stats["median"]
        out_iqr    = stats["iqr"]

        iqr_acceptable = (out_iqr < max_iqr) or (out_iqr <= current_stats["iqr"])
        score  = abs(out_median - target_pct) + out_iqr

        if iqr_acceptable and score < best_score:
            best_result = candidate.copy()
            best_median = out_median
            best_iqr    = out_iqr
            best_score  = score

        if _is_valid(stats, lo, hi, max_iqr):
            best_result = candidate.copy()
            break

        if out_median > target_pct:
            lo_rate = mid_rate
        else:
            hi_rate = mid_rate

    return best_result.astype(audio_stem.dtype)


# =====================================================================
# PATH C — HYBRID (OPTIMIZED)
# =====================================================================

def _compress_spread(
    audio_stem    : np.ndarray,
    sr            : int,
    current_stats : dict,
    target_pct    : float,
    lo            : float,
    hi            : float,
    max_iqr       : float,
    max_iterations: int = 5,
    cached_onsets = None,
    cached_onset_frames = None,
    cached_energy = None,
) -> np.ndarray:
    onset_samples = cached_onsets if cached_onsets is not None else _get_onset_samples(audio_stem, sr)
    hop_length    = 256
    energy        = cached_energy if cached_energy is not None else librosa.feature.rms(
        y=audio_stem, frame_length=512, hop_length=hop_length
    )[0]
    note_energies = _get_per_note_energy(audio_stem, onset_samples, sr, precomputed_energy=energy)

    ir_light = _build_reverb_ir(sr, decay_time_sec=0.15, wet_character=0.2)

    e_p25 = float(np.percentile(note_energies, 25))
    e_p75 = float(np.percentile(note_energies, 75))

    best_result = audio_stem.copy()
    best_score  = current_stats["iqr"]

    def _render(strength: float) -> np.ndarray:
        result = audio_stem.copy()
        release_rate  = 5.0 + strength * 10.0
        gain_envelope = np.ones(len(result), dtype=np.float32)

        for i in range(len(onset_samples) - 1):
            if note_energies[i] < e_p75:
                continue

            note_start = int(onset_samples[i])
            note_end   = int(onset_samples[i + 1])
            if note_end - note_start < sr * 0.02:
                continue

            start_frame  = note_start // hop_length
            end_frame    = min(note_end // hop_length, len(energy) - 1)
            if start_frame >= end_frame:
                continue

            peak_frame   = start_frame + int(np.argmax(energy[start_frame:end_frame]))
            peak_sample  = peak_frame * hop_length
            attack_end   = min(peak_sample + int(sr * 0.02), note_end)
            if attack_end >= note_end:
                continue

            release_len   = note_end - attack_end
            t             = np.linspace(0, 1, release_len)
            floor         = max(0.0, 0.4 - strength * 0.5)
            release_curve = floor + (1.0 - floor) * np.exp(-release_rate * t)

            gain_envelope[attack_end:note_end] = np.minimum(
                gain_envelope[attack_end:note_end],
                release_curve.astype(np.float32)
            )

        result = result * gain_envelope

        reverb_full = fftconvolve(audio_stem, ir_light, mode='full')[:len(audio_stem)]
        wet_mask    = np.zeros(len(result), dtype=np.float32)
        wet_level   = strength * 0.25

        for i in range(len(onset_samples) - 1):
            if note_energies[i] > e_p25:
                continue
            start = int(onset_samples[i])
            end   = int(onset_samples[i + 1])
            wet_mask[start:end] = wet_level

        dry_mask = 1.0 - wet_mask
        result   = dry_mask * result + wet_mask * reverb_full
        return _normalize_to_peak(result, audio_stem)

    def _score_at(strength: float):
        candidate = _render(strength)
        # Fresh energy measurement is required: both the release envelope and
        # the reverb blend change RMS energy relative to the original, so
        # cached_energy would score a signal that no longer exists.
        stats = _measure_overlap(candidate, sr, precomputed_onsets=cached_onset_frames)
        return candidate, stats

    # Higher strength compresses loud-note releases and adds wet signal to
    # quiet notes -- both push IQR down monotonically -- so bisection
    # converges on the smallest strength that satisfies max_iqr instead of
    # sampling 5 fixed points that might all overshoot or undershoot it.
    lo_strength, hi_strength = 0.1, 0.5

    for iteration in range(max_iterations):
        mid_strength = (lo_strength + hi_strength) / 2.0
        candidate, stats = _score_at(mid_strength)
        out_median = stats["median"]
        out_iqr    = stats["iqr"]

        median_ok  = lo <= out_median <= hi
        iqr_better = out_iqr < best_score

        if median_ok and iqr_better:
            best_result = candidate.copy()
            best_score  = out_iqr

        if _is_valid(stats, lo, hi, max_iqr):
            best_result = candidate.copy()
            break

        if out_iqr > max_iqr:
            lo_strength = mid_strength
        else:
            hi_strength = mid_strength

    return best_result.astype(audio_stem.dtype)


# =====================================================================
# MAIN ENTRY POINT (CACHING ONSET & RMS ONCE PER CHANNEL)
# =====================================================================

def apply_legato_overlap(
    audio_stem: np.ndarray,
    sr: int,
    target_overlap_percentage: float = 0.75,
    lo: float = 70.0,
    hi: float = 80.0,
    max_iqr: float = 35.0,
    max_correction_rounds: int = 3,
    max_wet: float = 0.45,
) -> np.ndarray:
    target_pct = (
        target_overlap_percentage * 100.0
        if target_overlap_percentage <= 1.0
        else target_overlap_percentage
    )

    # ── Defensive clamp: target_pct MUST live inside (lo, hi) ──────────
    # ROOT CAUSE of the CSV failures (medians landing at 53-64%, well below
    # the 70-80% pass window): pipeline.py calls this with
    # target_overlap_percentage=0.65, i.e. target_pct=65, which is BELOW
    # lo=70. Every correction path (_increase_overlap, _decrease_overlap,
    # _compress_spread) steers strictly toward target_pct, not toward the
    # [lo, hi] window -- their bisection direction (`if out_median <
    # target_pct: ...`) and best-candidate scoring (`abs(out_median -
    # target_pct)`) are both anchored on target_pct alone. So an
    # out-of-range target makes the loop actively converge outside the
    # pass window (too low), or stall partway through correction for
    # tracks that started above `hi` (too high, e.g. the 82-84% failures),
    # since the "best" candidate by that score is never one that's
    # actually inside [lo, hi].
    #
    # Snapping to the exact edge (e.g. clamping 65 -> 70) is still risky:
    # the measured median is a statistic with noise, so aiming exactly at
    # a boundary means roughly half of runs will land just outside it.
    # Instead, if the caller's target isn't safely inside the window
    # (with a margin away from both edges), retarget the midpoint of
    # [lo, hi] so convergence has buffer room on both sides.
    margin = min(2.0, (hi - lo) / 4.0)
    if not (lo + margin <= target_pct <= hi - margin):
        target_pct = (lo + hi) / 2.0

    def _correct_channel(channel: np.ndarray) -> np.ndarray:
        current = channel

        # Onset positions are structurally stable across corrections -- the
        # reverb/envelope edits shift *energy*, not where notes start -- so
        # these are safe to compute once and reuse across every round.
        hop_length = 256
        cached_onset_frames = librosa.onset.onset_detect(
            y=current, sr=sr, units='frames', backtrack=True
        )
        cached_onsets = np.append(librosa.onset.onset_detect(
            y=current, sr=sr, units='samples', backtrack=True
        ), len(current))

        for _round in range(max_correction_rounds):
            # RMS energy MUST be recomputed on `current` every round: after
            # round 1, `current` is no longer the original channel, so an
            # energy array cached before the loop would silently measure a
            # signal that no longer exists -- this was the root cause of the
            # median getting stuck instead of converging.
            fresh_energy = librosa.feature.rms(
                y=current, frame_length=512, hop_length=hop_length
            )[0]

            stats = _measure_overlap(current, sr, precomputed_onsets=cached_onset_frames, precomputed_energy=fresh_energy)

            if _is_valid(stats, lo, hi, max_iqr):
                break

            if stats["median"] < lo:
                current = _increase_overlap(
                    current, sr, stats, target_pct, lo, hi, max_iqr, 
                    max_wet=max_wet,
                    cached_onsets=cached_onsets, cached_onset_frames=cached_onset_frames, cached_energy=fresh_energy
                )
            elif stats["median"] > hi:
                current = _decrease_overlap(
                    current, sr, stats, target_pct, lo, hi, max_iqr, 
                    cached_onsets=cached_onsets, cached_onset_frames=cached_onset_frames, cached_energy=fresh_energy
                )
            else:
                current = _compress_spread(
                    current, sr, stats, target_pct, lo, hi, max_iqr, 
                    cached_onsets=cached_onsets, cached_onset_frames=cached_onset_frames, cached_energy=fresh_energy
                )

        return current

    is_stereo = audio_stem.ndim > 1
    if is_stereo:
        corrected = np.stack([
            _correct_channel(audio_stem[c].astype(np.float64))
            for c in range(audio_stem.shape[0])
        ])
    else:
        corrected = _correct_channel(audio_stem.astype(np.float64))

    return corrected.astype(audio_stem.dtype)


def enforce_pitch_limit(audio_stem: np.ndarray, sample_rate, stem_type="melody", max_semitones=12):
    hop_length = 1024
    is_stereo = audio_stem.ndim > 1
    audio_for_modification = audio_stem[0].copy() if is_stereo else audio_stem.copy()

    if stem_type == "bass":
        fmin, fmax = 30.0, 800.0
    else:
        fmin = 65.4
        fmax = 2093.0

    f0, voiced_flag, _ = librosa.pyin(
        y=audio_for_modification, fmin=fmin, fmax=fmax, sr=sample_rate, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0)

    semitones = np.zeros_like(f0)
    valid_idx = f0 > 0
    semitones[valid_idx] = librosa.hz_to_midi(f0[valid_idx])
    jumps = np.diff(semitones)

    for i, jump in enumerate(jumps):
        if abs(jump) > max_semitones and voiced_flag[i] and voiced_flag[i + 1]:
            note_end_frame = i + 1
            while note_end_frame < len(voiced_flag) and voiced_flag[note_end_frame]:
                note_end_frame += 1

            start_sample = (i + 1) * hop_length
            end_sample = note_end_frame * hop_length

            direction = -1 if jump > 0 else 1
            shift_amount = 0
            current_jump = abs(jump)

            while current_jump > max_semitones:
                shift_amount += 12
                current_jump -= 12

            shift_amount = shift_amount * direction
            segment = audio_for_modification[start_sample:end_sample]

            if len(segment) > 0:
                shifted_segment = librosa.effects.pitch_shift(
                    y=segment, sr=sample_rate, n_steps=shift_amount
                )
                target_len = end_sample - start_sample
                shifted_segment = librosa.util.fix_length(shifted_segment, size=target_len)
                audio_for_modification[start_sample:end_sample] = shifted_segment

    if is_stereo:
        audio_stem[0] = audio_for_modification
        audio_stem[1] = librosa.util.fix_length(audio_for_modification, size=audio_stem.shape[1])
        return audio_stem
    else:
        return audio_for_modification