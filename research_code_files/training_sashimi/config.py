"""
MelodyCare — Shared Project Configuration
Single source of truth for all paths, constants, and DSP parameters.

UPDATE (v2): Stem-level paths and mix weights removed.
             Dataset now uses pre-mixed input/target audio files.
"""
import os

# ── Directory paths ───────────────────────────────────────────────────
BASE_DIR              = "/path/to/dataset"

# v2: Pre-mixed files — no stem separation needed during training
INPUT_MIXED_DIR       = r"/path/to/dataset/raw/Tier_1_2"
TARGET_MIXED_DIR      = r"/path/to/dataset/pass"

# Benchmark clinical tracks for CLAP style vector extraction
BENCHMARK_TRACKS_DIR  = os.path.join(BASE_DIR, "BenchmarkTracks")

# Style vectors output directory
STYLE_VECTOR_DIR      = r"/path/to/dataset/StyleVectors"

# Master CSV output
CSV_OUTPUT_PATH       = os.path.join(BASE_DIR, "melodycare_training_manifest.csv")

# ── CLAP ──────────────────────────────────────────────────────────────
CLAP_SR               = 48000
CLAP_MAX_DURATION_S   = 30
STYLE_VECTOR_STEM     = "therapeutic_style_vector"
STYLE_VECTOR_DIM      = 512

# ── Audio ─────────────────────────────────────────────────────────────
SAMPLE_RATE           = 48000
CHUNK_DURATION_S      = 30
NUM_AUDIO_CHANNELS    = 2      # stereo output for Stable Audio

# ── Model ─────────────────────────────────────────────────────────────
PRETRAINED_MODEL_ID   = "stabilityai/stable-audio-open-1.0"
MODEL_DIM             = 768

# ── Training ──────────────────────────────────────────────────────────
NUM_EPOCHS            = 50
BATCH_SIZE_PER_GPU    = 2
GRADIENT_ACCUM_STEPS  = 4
LEARNING_RATE         = 1e-5
# Applied only to the pretrained DiT's parameters — style_proj and
# input_audio_proj (newly-initialized, need to learn from scratch) train at
# the full LEARNING_RATE. Fine-tuning the whole DiT at the same aggressive
# LR as brand-new modules, on a small dataset, risks catastrophic forgetting
# (confirmed as the likely cause of noise-like generations — see loop.py).
DIT_LR_SCALE          = 0.1
# Applied only to the pretrained DiT's parameters — style_proj and
# input_audio_proj (newly-initialized, need to learn from scratch) train at
# the full LEARNING_RATE. Fine-tuning the whole DiT at the same aggressive
# LR as brand-new modules, on a small dataset, risks catastrophic forgetting
# (confirmed as the likely cause of noise-like generations — see loop.py).
DIT_LR_SCALE          = 0.1
LR_WARMUP_STEPS       = 500
WEIGHT_DECAY          = 1e-2
MAX_GRAD_NORM         = 1.0
SEED                  = 42
MIXED_PRECISION       = "bf16"
GRADIENT_CHECKPOINTING= True
SAVE_EVERY_N_STEPS    = 500
LOG_EVERY_N_STEPS     = 50
MAX_CHECKPOINTS       = 3

# ── Diffusion ─────────────────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS   = 1000
NOISE_OFFSET          = 0.05

# ── DSP therapy constraints (for prompt engineering only) ─────────────
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

# ── Checkpoints & logs ────────────────────────────────────────────────
OUTPUT_DIR            = r"/path/to/training_output/checkpoint"
LOGGING_DIR           = r"/path/to/training_output/logs"