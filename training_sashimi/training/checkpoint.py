"""
MelodyCare Checkpoint Management
Save, load, and cleanup training checkpoints.

UPDATED: dropped the standalone `model` argument. accelerator.save_state()/
load_state() already track whatever was passed to accelerator.prepare() —
which is now model.model (the DiT), not the whole ConditionedDiffusionModel
Wrapper — so there's nothing extra to do for it here. input_audio_proj is
now saved/restored the same way style_proj already was.
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from accelerate import Accelerator
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    accelerator     : Accelerator,
    style_proj      : nn.Module,
    input_audio_proj: nn.Module,
    optimizer       : torch.optim.Optimizer,
    scheduler       : object,
    global_step     : int,
    epoch           : int,
    output_dir      : str,
    max_checkpoints : int = 3,
) -> str:
    """
    Save a full training checkpoint.

    Saves:
        - Full accelerate state (DiT weights, optimizer, scheduler, RNG —
          whatever was passed to accelerator.prepare() in loop.py)
        - Style projection weights separately (redundant safety net)
        - Input-audio projection weights separately (same reason)
        - Training metadata (step, epoch, LR)

    Cleans up oldest checkpoints to stay within max_checkpoints limit.

    Args:
        accelerator      : Accelerator instance (handles DDP unwrapping)
        style_proj        : style projection module
        input_audio_proj  : input-audio projection module
        optimizer         : AdamW optimizer
        scheduler         : LR scheduler
        global_step       : current training step
        epoch             : current epoch
        output_dir        : base checkpoint directory
        max_checkpoints   : maximum number of checkpoints to retain

    Returns:
        path to the saved checkpoint directory
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # accelerator.save_state handles DDP unwrapping automatically
    # saves DiT weights, optimizer state, scheduler state, RNG state
    accelerator.save_state(checkpoint_dir)

    # Style + input-audio projections are separate small modules — save
    # independently as well (accelerator.save_state already covers them
    # since they were passed to accelerator.prepare(), this is just a
    # cheap, easy-to-restore-standalone redundant copy).
    unwrapped_style_proj = accelerator.unwrap_model(style_proj)
    torch.save(
        unwrapped_style_proj.state_dict(),
        os.path.join(checkpoint_dir, "style_projection.pt")
    )

    unwrapped_input_audio_proj = accelerator.unwrap_model(input_audio_proj)
    torch.save(
        unwrapped_input_audio_proj.state_dict(),
        os.path.join(checkpoint_dir, "input_audio_projection.pt")
    )

    # Save training metadata for resume and monitoring
    metadata = {
        "global_step"  : global_step,
        "epoch"        : epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_dir}")

    # Cleanup old checkpoints
    _cleanup_old_checkpoints(output_dir, max_checkpoints)

    return checkpoint_dir


def load_checkpoint(
    accelerator      : Accelerator,
    dit              : nn.Module,
    style_proj       : nn.Module,
    input_audio_proj : nn.Module,
    optimizer        : torch.optim.Optimizer,
    output_dir       : str,
    resume_from      : Optional[str] = None,
    reset_scheduler  : bool = False,
) -> tuple[int, int]:
    """
    Load the latest (or specified) checkpoint to resume training.

    Args:
        accelerator      : Accelerator instance
        dit               : DDP-wrapped DiT (model.model) — needed directly
                            when reset_scheduler=True, since we bypass
                            accelerator.load_state() and restore it manually
        style_proj        : style projection module (restored separately)
        input_audio_proj  : input-audio projection module (restored separately)
        optimizer         : optimizer — needed directly for the same reason as dit
        output_dir        : base checkpoint directory
        resume_from       : specific checkpoint path; if None, uses latest
        reset_scheduler   : if True, restore DiT/style_proj/input_audio_proj/
                            optimizer weights but deliberately do NOT restore
                            scheduler/RNG/sampler state. Use this when
                            extending a run with a larger NUM_EPOCHS — the
                            old checkpoint's scheduler is already fully
                            annealed (sized for the SHORTER original run),
                            so resuming it as-is leaves LR near eta_min for
                            the entire extension. With this flag, loop.py
                            builds a FRESH scheduler sized for the new,
                            larger total_train_steps, left untouched here.
                            If False (default): normal resume, restores
                            everything via accelerator.load_state() — use
                            this for resuming an interrupted run with no
                            change to NUM_EPOCHS.

    Returns:
        (global_step, starting_epoch) tuple
    """
    checkpoint_dir = _resolve_checkpoint_dir(output_dir, resume_from)

    if checkpoint_dir is None:
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0

    logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

    if reset_scheduler:
        logger.info(
            "reset_scheduler=True: restoring model/optimizer weights only — "
            "scheduler/RNG/sampler state will NOT be restored (a fresh "
            "scheduler sized for the new total_train_steps is used instead)."
        )
        accelerator.unwrap_model(dit).load_state_dict(
            load_safetensors(os.path.join(checkpoint_dir, "model.safetensors"))
        )
        accelerator.unwrap_model(style_proj).load_state_dict(
            load_safetensors(os.path.join(checkpoint_dir, "model_1.safetensors"))
        )
        accelerator.unwrap_model(input_audio_proj).load_state_dict(
            load_safetensors(os.path.join(checkpoint_dir, "model_2.safetensors"))
        )
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
        if os.path.exists(optimizer_path):
            try:
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                logger.info("Optimizer state restored (Adam momentum/variance preserved)")
            except Exception as e:
                logger.warning(
                    f"Could not restore optimizer state ({e}) — continuing with a "
                    f"freshly-initialized optimizer. Model weights are still restored "
                    f"correctly; only Adam's momentum/variance history is lost, which "
                    f"causes a few noisy early steps but is not otherwise harmful."
                )
        logger.info("Model + optimizer weights restored (scheduler reset)")
    else:
        # Normal resume — restore everything, including scheduler/RNG/sampler
        accelerator.load_state(checkpoint_dir)

        style_proj_path = os.path.join(checkpoint_dir, "style_projection.pt")
        if os.path.exists(style_proj_path):
            accelerator.unwrap_model(style_proj).load_state_dict(
                torch.load(style_proj_path, map_location="cpu")
            )
            logger.info("Style projection weights restored")
        else:
            logger.warning(f"style_projection.pt not found in {checkpoint_dir}")

        input_audio_proj_path = os.path.join(checkpoint_dir, "input_audio_projection.pt")
        if os.path.exists(input_audio_proj_path):
            accelerator.unwrap_model(input_audio_proj).load_state_dict(
                torch.load(input_audio_proj_path, map_location="cpu")
            )
            logger.info("Input-audio projection weights restored")
        else:
            logger.warning(f"input_audio_projection.pt not found in {checkpoint_dir}")

    # Read step/epoch from metadata. In reset_scheduler mode we deliberately
    # still read the ORIGINAL global_step/epoch — loop.py uses these to know
    # where to resume the epoch loop from; only the LR schedule restarts fresh.
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        global_step    = meta.get("global_step", 0)
        starting_epoch = meta.get("epoch", 0)
        logger.info(f"Resuming at step={global_step}, epoch={starting_epoch}")
        return global_step, starting_epoch

    return 0, 0


def _resolve_checkpoint_dir(
    output_dir  : str,
    resume_from : Optional[str],
) -> Optional[str]:
    """Find the checkpoint directory to resume from."""
    if resume_from is not None:
        if os.path.exists(resume_from):
            return resume_from
        logger.warning(f"Specified checkpoint not found: {resume_from}")
        return None

    if not os.path.exists(output_dir):
        return None

    checkpoints = _list_checkpoints(output_dir)
    return str(checkpoints[-1]) if checkpoints else None


def _list_checkpoints(output_dir: str) -> list[Path]:
    """List checkpoints sorted by step number (ascending)."""
    return sorted(
        [
            d for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda x: int(x.name.split("-")[1])
    )


def _cleanup_old_checkpoints(output_dir: str, max_checkpoints: int) -> None:
    """Remove oldest checkpoints beyond max_checkpoints limit."""
    checkpoints = _list_checkpoints(output_dir)

    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest)
        logger.info(f"Removed old checkpoint: {oldest.name}")
