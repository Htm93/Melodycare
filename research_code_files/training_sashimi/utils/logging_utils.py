"""
MelodyCare Logging Utilities
Centralized logger setup and metric tracking.
"""
import logging
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger as accelerate_get_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger that only prints from the main accelerate process.
    Prevents duplicate log messages in multi-GPU runs.
    """
    return accelerate_get_logger(name, log_level="INFO")


def setup_logging(accelerator: Accelerator) -> None:
    """
    Configure root logging format.
    Called once at the start of training.
    """
    logging.basicConfig(
        format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level   = logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )


def log_training_summary(
    accelerator       : Accelerator,
    config,
    total_train_steps : int,
    dataset_size      : int,
) -> None:
    """
    Print a formatted training configuration summary.
    Only prints from main process.
    """
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("MelodyCare — Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  GPUs:                   {accelerator.num_processes}")
    logger.info(f"  Batch per GPU:          {config.batch_size_per_gpu}")
    logger.info(f"  Gradient accum steps:   {config.gradient_accum_steps}")
    logger.info(f"  Effective batch size:   "
                f"{config.batch_size_per_gpu * config.gradient_accum_steps * accelerator.num_processes}")
    logger.info(f"  Total training steps:   {total_train_steps}")
    logger.info(f"  Mixed precision:        {config.mixed_precision}")
    logger.info(f"  Learning rate:          {config.learning_rate}")
    logger.info(f"  LR warmup steps:        {config.lr_warmup_steps}")
    logger.info(f"  Training samples:       {dataset_size}")
    logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    logger.info(f"  Noise offset:           {config.noise_offset}")
    logger.info(f"  Resume from:            {config.resume_from or 'None'}")
    logger.info("=" * 60)


class MetricTracker:
    """
    Tracks running average of training metrics across gradient
    accumulation steps and logs to accelerate tracker (TensorBoard).
    """

    def __init__(self):
        self._sum   : dict[str, float] = {}
        self._count : dict[str, int]   = {}

    def update(self, key: str, value: float) -> None:
        """Add a new value to the running average."""
        self._sum[key]   = self._sum.get(key, 0.0)   + value
        self._count[key] = self._count.get(key, 0)   + 1

    def average(self, key: str) -> float:
        """Get current average for a metric."""
        count = self._count.get(key, 0)
        return self._sum[key] / count if count > 0 else 0.0

    def reset(self, key: str) -> None:
        """Reset a metric's accumulator."""
        self._sum[key]   = 0.0
        self._count[key] = 0

    def reset_all(self) -> None:
        """Reset all metric accumulators."""
        self._sum.clear()
        self._count.clear()