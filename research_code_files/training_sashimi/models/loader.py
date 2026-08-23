"""
MelodyCare Model Loading
"""
import torch
import torch.nn as nn

from stable_audio_tools import get_pretrained_model
from models.style_projection import StyleVectorProjection
from models.input_audio_projection import InputAudioProjection
from utils.logging_utils import get_logger
import config

logger = get_logger(__name__)


def load_stable_audio_model(
    model_id              : str  = config.PRETRAINED_MODEL_ID,
    device                : torch.device = torch.device("cpu"),
    gradient_checkpointing: bool = config.GRADIENT_CHECKPOINTING,
    model_dim              : int  = config.MODEL_DIM,
    style_vector_dim      : int  = config.STYLE_VECTOR_DIM,
    input_audio_channels  : int  = 64,   # stable-audio-open-1.0 VAE latent channels
    input_audio_downsample: int  = 8,    # see models/input_audio_projection.py
) -> tuple[nn.Module, nn.Module, nn.Module, dict]:
    """
    Load Stable Audio Open and configure for audio-to-audio fine-tuning.

    Freeze : VAE (pretransform) + text/number conditioners (prompt, seconds_start,
             seconds_total).
    Train  : DiT (model.model) + StyleVectorProjection + InputAudioProjection.

    NOTE: model.model.io_channels == 64 and model.input_concat_ids /
    prepend_cond_ids / local_add_cond_ids are all empty for this checkpoint —
    there is no built-in path for conditioning on a full audio clip. Source
    audio is injected as extra cross-attention tokens via InputAudioProjection
    (see that module's docstring for why).

    Returns:
        model          : ConditionedDiffusionModelWrapper (frozen conditioner/
                          pretransform; model.model is the trainable DiT —
                          this is what should be passed to accelerator.prepare(),
                          NOT `model` itself, since DDP-wrapping the whole
                          wrapper breaks the ability to splice in extra
                          cross-attention tokens — see training/loop.py)
        style_proj     : trainable StyleVectorProjection
        input_audio_proj: trainable InputAudioProjection
        model_config   : dict, raw model config from stable_audio_tools
    """
    logger.info(f"Loading Stable Audio: {model_id}")
    model, model_config = get_pretrained_model(model_id)
    model = model.to(device)

    # Freeze VAE
    if hasattr(model, 'pretransform'):
        for param in model.pretransform.parameters():
            param.requires_grad = False
        logger.info("  VAE (pretransform): FROZEN")

    # Freeze text/number conditioners (prompt, seconds_start, seconds_total)
    if hasattr(model, 'conditioner'):
        for name, module in model.conditioner.named_children():
            for param in module.parameters():
                param.requires_grad = False
        logger.info("  Conditioner (prompt/seconds_start/seconds_total): FROZEN")

    # Gradient checkpointing on the DiT
    if gradient_checkpointing:
        if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing: ENABLED")

    # Style projection (new trainable layer) — projects style_vector into a
    # single cross-attention token
    style_proj = StyleVectorProjection(
        style_dim = style_vector_dim,
        model_dim = model_config.get("model_dim", model_dim),
    ).to(device)

    # Input-audio projection (new trainable layer) — projects encoded source
    # audio into a short sequence of cross-attention tokens (see module
    # docstring for why this is needed instead of input_concat_cond).
    input_audio_proj = InputAudioProjection(
        latent_channels    = input_audio_channels,
        cond_token_dim      = model_config.get("model_dim", model_dim),
        downsample_factor   = input_audio_downsample,
    ).to(device)

    _log_parameter_summary(model, style_proj, input_audio_proj)
    return model, style_proj, input_audio_proj, model_config


def _log_parameter_summary(model: nn.Module, style_proj: nn.Module, input_audio_proj: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj      = sum(p.numel() for p in style_proj.parameters())
    input_proj = sum(p.numel() for p in input_audio_proj.parameters())
    grand_total     = total + proj + input_proj
    grand_trainable = trainable + proj + input_proj
    logger.info(f"  Trainable: {grand_trainable:,} / {grand_total:,} "
                f"({100*grand_trainable/grand_total:.1f}%)")