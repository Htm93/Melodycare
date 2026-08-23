"""
Shared configuration for MelodyCare data preparation pipeline.
"""
import os

# ── Directory paths ───────────────────────────────────────────────────
BASE_DIR              = r"Dataset"

# Input: original unedited stems (from Demucs separation)
INPUT_STEMS_DIR       = os.path.join(BASE_DIR, "Music", "processed", "separated_sources")

# Target: DSP-processed therapeutic output stems
TARGET_STEMS_DIR      = os.path.join(BASE_DIR, "Music", "processed", "training_data")

# Benchmark clinical tracks for CLAP style vector extraction
BENCHMARK_TRACKS_DIR  = os.path.join(BASE_DIR, "BenchmarkTracks")

# Style vectors output directory
STYLE_VECTOR_DIR      = os.path.join(BASE_DIR, "StyleVectors")

# Master CSV output
CSV_OUTPUT_PATH       = os.path.join(BASE_DIR, "melodycare_training_manifest.csv")

# ── CLAP config ───────────────────────────────────────────────────────
CLAP_SR               = 48000
CLAP_MAX_DURATION_S   = 30
STYLE_VECTOR_STEM     = "therapeutic_style_vector"

# ── DSP therapy target constraints (for prompt engineering) ──────────
THERAPY_BPM_MIN       = 60
THERAPY_BPM_MAX       = 80
THERAPY_BPM_TARGET    = 60
THERAPY_DYNAMIC_DB    = 13.0
THERAPY_LEGATO_MIN    = 70
THERAPY_LEGATO_MAX    = 80
THERAPY_LFO_MIN_HZ    = 0.05
THERAPY_LFO_MAX_HZ    = 0.10
THERAPY_ROLLOFF_KHZ   = 5.0
THERAPY_BOOST_LOW_HZ  = 100
THERAPY_BOOST_HIGH_HZ = 400
THERAPY_SUB_CUTOFF_HZ = 80

# ── Mix weights (melody : bass : drum) ───────────────────────────────
MIX_WEIGHT_MELODY     = 1.0
MIX_WEIGHT_BASS       = 0.1
MIX_WEIGHT_DRUM       = 0.0    # drums excluded from therapeutic output

# ── Stem filename conventions ─────────────────────────────────────────
STEM_MELODY           = "melody.wav"
STEM_BASS             = "bass.wav"
STEM_DRUMS            = "drums.wav"
STEM_MELODY_DSP       = "_melody_therapeutic.wav"
STEM_BASS_DSP         = "_bass_therapeutic.wav"