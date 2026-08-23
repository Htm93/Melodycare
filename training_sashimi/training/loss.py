"""
MelodyCare Diffusion Training Loss

Rewritten to match the ACTUAL stable-audio-open-1.0 API (confirmed via
runtime inspection of the installed stable_audio_tools package — see
training/loop.py header comment for the inspection trail). The model:

    - diffusion_objective == "v"  (NOT discrete DDPM — no alphas_cumprod,
      no integer num_timesteps, no q_sample/get_v methods on the model)
    - Continuous timestep t in [0, 1), noise schedule:
          alphas, sigmas = cos(t * pi/2), sin(t * pi/2)   (get_alphas_sigmas)
    - v-target:  target = noise * alphas - clean * sigmas
    - t is sampled via a scrambled Sobol low-discrepancy sequence, matching
      stable_audio_tools.training.diffusion.DiffusionCondTrainingWrapper's
      own trainer (self.rng = torch.quasirandom.SobolEngine(1, scramble=True))
    - The DiT's forward() (model.model, a DiTWrapper) takes conditioning as
      individual kwargs — critically `cross_attn_mask`, NOT
      `cross_attn_cond_mask` — see loop.py for where `conditioning` is built.
"""
import torch
import torch.nn as nn
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_diffusion_loss(
    dit            : nn.Module,
    target_latents : torch.Tensor,
    conditioning   : dict,
    cfg_dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    Compute the v-objective diffusion training loss.

    NOTE: unlike the original version of this function, `input_latents` is
    NOT passed here or channel-concatenated. Source-audio conditioning is
    now injected upstream (in loop.py) as extra cross-attention tokens via
    InputAudioProjection, and arrives already merged into `conditioning`.
    See models/input_audio_projection.py for why (this checkpoint's DiT has
    io_channels=64 with no input_concat_ids configured — channel-concat
    would crash or corrupt the pretrained input projection).

    Args:
        dit             : the DDP-wrapped DiTWrapper (model.model), called
                          directly so gradients sync correctly across GPUs.
                          NOT the top-level ConditionedDiffusionModelWrapper —
                          that class's forward() recomputes conditioning
                          internally from raw conditioner output and gives no
                          hook to splice in extra tokens.
        target_latents  : (B, C, T) — encoded target (therapeutic) audio,
                          this is what gets noised and is the denoising target
        conditioning    : dict with keys matching DiTWrapper.forward's kwargs:
                          cross_attn_cond, cross_attn_mask, global_cond,
                          (optionally prepend_cond, prepend_cond_mask,
                          input_concat_cond — unused here, always None)
        cfg_dropout_prob: classifier-free-guidance conditioning dropout
                          probability during training (matches stable-audio-
                          tools' own default of 0.1)

    Returns:
        scalar loss tensor (gradient-attached)
    """
    B      = target_latents.shape[0]
    device = target_latents.device

    # ── 1. Sample continuous timesteps via scrambled Sobol sequence ──────
    # Matches stable_audio_tools' own DiffusionCondTrainingWrapper exactly
    # (self.rng = torch.quasirandom.SobolEngine(1, scramble=True)).
    rng = torch.quasirandom.SobolEngine(1, scramble=True)
    t   = rng.draw(B)[:, 0].to(device)

    # ── 2. v-objective noise schedule ─────────────────────────────────
    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas[:, None, None].to(device)
    sigmas = sigmas[:, None, None].to(device)

    # ── 3. Forward diffusion — noise the target latents ────────────────
    noise        = torch.randn_like(target_latents)
    noised_input = target_latents * alphas + noise * sigmas
    v_target     = noise * alphas - target_latents * sigmas

    # ── 4. DiT forward pass ─────────────────────────────────────────────
    # Calling `dit` directly (not model.model.model) — DiTWrapper.forward()
    # is the layer that translates our kwarg names into the inner
    # DiffusionTransformer's expected args (e.g. global_cond -> global_embed).
    model_pred = dit(
        noised_input,
        t,
        cross_attn_cond   = conditioning.get("cross_attn_cond"),
        cross_attn_mask   = conditioning.get("cross_attn_mask"),
        global_cond       = conditioning.get("global_cond"),
        prepend_cond      = conditioning.get("prepend_cond"),
        prepend_cond_mask = conditioning.get("prepend_cond_mask"),
        cfg_dropout_prob  = cfg_dropout_prob,
    )

    # ── 5. MSE loss against the v-target ────────────────────────────────
    loss = nn.functional.mse_loss(model_pred, v_target)

    return loss