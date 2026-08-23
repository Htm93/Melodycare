"""
MelodyCare TrainConfig Dataclass (v2)
Wraps config.py constants into a structured dataclass for training.
"""
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class TrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────
    csv_path             : str           = config.CSV_OUTPUT_PATH
    output_dir           : str           = config.OUTPUT_DIR
    logging_dir          : str           = config.LOGGING_DIR
    pretrained_model_id  : str           = config.PRETRAINED_MODEL_ID
    resume_from          : Optional[str] = None
    reset_scheduler      : bool          = False

    # ── Audio ─────────────────────────────────────────────────────────
    sample_rate          : int           = config.SAMPLE_RATE
    chunk_duration_s     : int           = config.CHUNK_DURATION_S
    num_audio_channels   : int           = config.NUM_AUDIO_CHANNELS

    # ── Training ──────────────────────────────────────────────────────
    num_epochs           : int           = config.NUM_EPOCHS
    batch_size_per_gpu   : int           = config.BATCH_SIZE_PER_GPU
    gradient_accum_steps : int           = config.GRADIENT_ACCUM_STEPS
    learning_rate        : float         = config.LEARNING_RATE
    dit_lr_scale         : float         = config.DIT_LR_SCALE
    lr_warmup_steps      : int           = config.LR_WARMUP_STEPS
    weight_decay         : float         = config.WEIGHT_DECAY
    max_grad_norm        : float         = config.MAX_GRAD_NORM
    seed                 : int           = config.SEED
    mixed_precision      : str           = config.MIXED_PRECISION
    gradient_checkpointing: bool         = config.GRADIENT_CHECKPOINTING
    save_every_n_steps   : int           = config.SAVE_EVERY_N_STEPS
    log_every_n_steps    : int           = config.LOG_EVERY_N_STEPS
    max_checkpoints      : int           = config.MAX_CHECKPOINTS

    # ── Model ─────────────────────────────────────────────────────────
    style_vector_dim     : int           = config.STYLE_VECTOR_DIM
    model_dim            : int           = config.MODEL_DIM

    # ── Diffusion ─────────────────────────────────────────────────────
    num_train_timesteps  : int           = config.NUM_TRAIN_TIMESTEPS
    noise_offset         : float         = config.NOISE_OFFSET

    @property
    def chunk_len(self) -> int:
        return self.sample_rate * self.chunk_duration_s

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.gradient_accum_steps