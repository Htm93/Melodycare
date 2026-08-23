"""
MelodyCare Training Loop
Main training orchestration using HuggingFace Accelerate.

ARCHITECTURE NOTES (confirmed via runtime inspection of the installed
stable_audio_tools package on the training server — see chat history for the full
inspection trail):

  - model (ConditionedDiffusionModelWrapper) is NOT what gets DDP-wrapped.
    Its .forward() recomputes conditioning internally via
    get_conditioning_inputs(cond), which only routes conditioner ids the
    model was built with (prompt, seconds_start, seconds_total) — there is
    no hook to splice in our style/input-audio tokens through it.

    Instead we DDP-wrap model.model (a DiTWrapper) directly, and call it
    with a manually-assembled conditioning dict. model (the wrapper) stays
    unwrapped and is only used for its frozen .pretransform, .conditioner,
    and the .get_conditioning_inputs() utility method.

  - diffusion_objective == "v" (continuous-time, NOT discrete DDPM). See
    training/loss.py for the schedule/target math.

  - This checkpoint has input_concat_ids == prepend_cond_ids ==
    local_add_cond_ids == [] — no built-in path for conditioning on a full
    audio clip. Source audio is injected via InputAudioProjection as extra
    cross-attention tokens (same mechanism as the style vector), NOT via
    channel-concatenation (which would crash: DiT io_channels == 64, fixed).
"""
import os
import math
import torch
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from configs.train_config import TrainConfig
from data.dataset import MelodyCareDataset
from data.collate import build_dataloader
from models.loader import load_stable_audio_model
from training.loss import compute_diffusion_loss
from training.checkpoint import save_checkpoint, load_checkpoint
from utils.logging_utils import (
    get_logger, setup_logging,
    log_training_summary, MetricTracker
)

logger = get_logger(__name__)


def train(config: TrainConfig) -> None:
    """
    Main training loop for MelodyCare Stable Audio Open fine-tuning.
    """

    # ── 1. Accelerator ────────────────────────────────────────────────
    project_config = ProjectConfiguration(
        project_dir = config.output_dir,
        logging_dir = config.logging_dir,
    )

    accelerator = Accelerator(
        mixed_precision             = config.mixed_precision,
        gradient_accumulation_steps = config.gradient_accum_steps,
        log_with                    = "tensorboard",
        project_config              = project_config,
    )

    setup_logging(accelerator)
    set_seed(config.seed)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        accelerator.init_trackers("melodycare", config=vars(config))

    # ── 2. Dataset & DataLoader ───────────────────────────────────────
    logger.info("Initializing dataset...")

    dataset    = MelodyCareDataset(
        csv_path         = config.csv_path,
        sample_rate      = config.sample_rate,
        chunk_duration_s = config.chunk_duration_s,
        augment          = True,
    )

    dataloader = build_dataloader(
        dataset    = dataset,
        batch_size = config.batch_size_per_gpu,
        shuffle    = True,
        num_workers= 4,
        pin_memory = True,
    )

    # ── 3 & 4. Model + Style/Input-Audio Projections ────────────────────
    logger.info("Loading model...")

    # `model` is the ConditionedDiffusionModelWrapper — it stays unwrapped
    # and is used only for its frozen .pretransform / .conditioner /
    # .get_conditioning_inputs(). `model.model` (the trainable DiT) is what
    # actually gets DDP-wrapped below, along with style_proj and
    # input_audio_proj.
    model, style_proj, input_audio_proj, model_config = load_stable_audio_model(
        model_id               = config.pretrained_model_id,
        device                 = accelerator.device,
        gradient_checkpointing = config.gradient_checkpointing,
        model_dim               = config.model_dim,
        style_vector_dim       = config.style_vector_dim,
    )

    # ── 5. Optimizer & Schedulers ─────────────────────────────────────
    # Differential learning rates: the pretrained DiT trains at a much
    # lower rate (config.DIT_LR_SCALE) than the newly-initialized style_proj
    # / input_audio_proj, which need to learn from scratch. Combined with
    # zero-init on those two modules (see their _init_weights), this avoids
    # hitting the pretrained network with large, disruptive updates before
    # it's had a chance to gradually learn to use the new conditioning —
    # the likely cause of noise-like generations from full unfrozen
    # fine-tuning on a small dataset at a uniform LR.
    dit_params = [p for p in model.model.parameters() if p.requires_grad]
    new_module_params = list(style_proj.parameters()) + list(input_audio_proj.parameters())
    trainable_params = dit_params + new_module_params

    param_groups = [
        {"params": dit_params,        "lr": config.learning_rate * config.dit_lr_scale},
        {"params": new_module_params, "lr": config.learning_rate},
    ]

    if _HAS_BNB:
        # 8-bit optimizer state (bitsandbytes) — cuts Adam's exp_avg/
        # exp_avg_sq memory ~4x vs standard fp32 AdamW. Confirmed necessary:
        # this training run OOMs inside optimizer.step() with ~1.06B
        # trainable params (full DiT fine-tune), by only 15-70MB, REGARDLESS
        # of batch size (tested batch_size_per_gpu=1 and 2, same failure) —
        # meaning optimizer state itself, not activations, is the bottleneck.
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            weight_decay = config.weight_decay,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )
        logger.info("Using bitsandbytes 8-bit AdamW (reduced optimizer memory)")
    else:
        logger.warning(
            "bitsandbytes not installed — falling back to standard fp32 AdamW. "
            "This WILL likely OOM again given ~1.06B trainable params (confirmed "
            "in prior runs). Install with: pip install bitsandbytes --break-system-packages"
        )
        optimizer = AdamW(
            param_groups,
            weight_decay = config.weight_decay,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )

    # ── 6. Accelerate preparation (model/optimizer/dataloader FIRST) ────
    # NOTE: we prepare model.model (the DiT), NOT `model` itself. `model`
    # (frozen conditioner + pretransform) stays a plain object on
    # accelerator.device, already placed there by load_stable_audio_model().
    #
    # CRITICAL ORDERING: accelerator.prepare(dataloader) is what SHARDS the
    # dataset across GPUs (each process then only sees len(dataset)/num_gpus
    # samples per epoch). Step counts / LR schedule length MUST be computed
    # from the dataloader's length AFTER this sharding, not before — doing
    # it before (the original bug here) computes total_train_steps as if
    # running on a single GPU, which both mis-reports the tqdm ETA and cuts
    # the cosine LR schedule short relative to the real (fewer, since data
    # is now split across GPUs) number of steps that will actually run.
    dit, style_proj, input_audio_proj, optimizer, dataloader = accelerator.prepare(
        model.model, style_proj, input_audio_proj, optimizer, dataloader,
    )

    steps_per_epoch   = math.ceil(len(dataloader) / config.gradient_accum_steps)
    total_train_steps = config.num_epochs * steps_per_epoch

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = max(1, total_train_steps - config.lr_warmup_steps),
        eta_min = config.learning_rate * 0.1,
    )

    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, config.lr_warmup_steps))
    )

    cosine_scheduler, warmup_scheduler = accelerator.prepare(cosine_scheduler, warmup_scheduler)

    # ── 7. Resume from checkpoint ─────────────────────────────────────
    # reset_scheduler=True skips restoring the old (already fully-annealed)
    # scheduler state, leaving the freshly-built one above (sized for THIS
    # run's total_train_steps) intact — use this when extending training
    # with a larger num_epochs than the checkpoint was originally built for.
    global_step, starting_epoch = load_checkpoint(
        accelerator      = accelerator,
        dit               = dit,
        style_proj       = style_proj,
        input_audio_proj = input_audio_proj,
        optimizer        = optimizer,
        output_dir       = config.output_dir,
        resume_from      = config.resume_from,
        reset_scheduler  = config.reset_scheduler,
    )

    log_training_summary(accelerator, config, total_train_steps, len(dataset))

    # ── 8. Training loop ──────────────────────────────────────────────
    metrics = MetricTracker()
    dit.train()
    style_proj.train()
    input_audio_proj.train()

    progress_bar = tqdm(
        range(total_train_steps),
        initial = global_step,
        desc    = "Training",
        disable = not accelerator.is_local_main_process,
    )

    for epoch in range(starting_epoch, config.num_epochs):
        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(dit):

                # ── Unpack batch ──────────────────────────────────────
                input_audio  = batch["input_audio"]    # (B, 2, L)
                target_audio = batch["target_audio"]   # (B, 2, L)
                style_vector = batch["style_vector"]   # (B, 512)
                text_prompts = batch["text_prompt"]    # list[str]

                # ── Encode audio → latents (frozen VAE) ──────────────
                with torch.no_grad():
                    input_latents  = model.pretransform.encode(input_audio)
                    target_latents = model.pretransform.encode(target_audio)

                # ── Encode text/timing → conditioning (frozen conditioner) ──
                with torch.no_grad():
                    # cross_attn_cond_ids == ['prompt','seconds_start','seconds_total']
                    # global_cond_ids     == ['seconds_start','seconds_total']
                    # All three keys required or the conditioner raises
                    # ValueError: Conditioner key <key> not found in batch metadata.
                    cond_input = [
                        {
                            "prompt"        : p,
                            "seconds_start" : 0,
                            "seconds_total" : config.chunk_duration_s,
                        }
                        for p in text_prompts
                    ]
                    conditioning_tensors = model.conditioner(
                        cond_input, device=accelerator.device
                    )
                    # Routes prompt -> cross-attention, seconds_start/total ->
                    # both cross-attention AND global conditioning, per the
                    # ids above. Dict keys match DiTWrapper.forward's kwarg
                    # names directly (cross_attn_cond, cross_attn_mask,
                    # global_cond, ...) since ConditionedDiffusionModelWrapper
                    # .forward() unpacks this same dict as **kwargs into it.
                    base_cond = model.get_conditioning_inputs(conditioning_tensors)

                text_embeddings = base_cond["cross_attn_cond"]        # (B, S_text, 768)
                # Defensive lookup: DiTWrapper.forward's param is named
                # `cross_attn_mask`; fall back to the old assumed name in
                # case a different stable_audio_tools version differs.
                text_mask = base_cond.get("cross_attn_mask", base_cond.get("cross_attn_cond_mask"))

                # ── Trainable conditioning projections ─────────────────
                style_embeddings    = style_proj(style_vector)          # (B, 1, 768)
                input_audio_tokens  = input_audio_proj(input_latents)   # (B, T', 768)

                # ── Splice style + input-audio tokens onto the text tokens ──
                combined_cond = torch.cat(
                    [text_embeddings, style_embeddings, input_audio_tokens], dim=1
                )
                style_mask = torch.ones(
                    (style_embeddings.shape[0], 1),
                    device = accelerator.device,
                    dtype  = text_mask.dtype,
                )
                input_audio_mask = torch.ones(
                    input_audio_tokens.shape[:2],
                    device = accelerator.device,
                    dtype  = text_mask.dtype,
                )
                combined_mask = torch.cat([text_mask, style_mask, input_audio_mask], dim=1)

                conditioning = dict(base_cond)
                conditioning["cross_attn_cond"] = combined_cond
                conditioning["cross_attn_mask"] = combined_mask
                conditioning.pop("cross_attn_cond_mask", None)  # drop stale key if present

                # ── Diffusion loss ────────────────────────────────────
                # `dit` is the DDP-wrapped DiTWrapper — calling it directly
                # (not model.model.model) ensures gradients sync across GPUs.
                loss = compute_diffusion_loss(
                    dit            = dit,
                    target_latents = target_latents,
                    conditioning   = conditioning,
                )

                # ── Backward ──────────────────────────────────────────
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, config.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # ── LR schedule ───────────────────────────────────────
                if global_step < config.lr_warmup_steps:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

            # ── Metrics & logging ─────────────────────────────────────
            metrics.update("loss", loss.detach().item())

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss" : f"{metrics.average('loss'):.4f}",
                    "lr"   : f"{optimizer.param_groups[0]['lr']:.2e}",
                    "epoch": epoch,
                })

                if global_step % config.log_every_n_steps == 0:
                    accelerator.log(
                        {
                            "train/loss"   : metrics.average("loss"),
                            "train/lr"     : optimizer.param_groups[0]["lr"],
                            "train/epoch"  : epoch,
                        },
                        step=global_step,
                    )
                    metrics.reset_all()

                # ── Checkpoint ────────────────────────────────────────
                if global_step % config.save_every_n_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator      = accelerator,
                            style_proj       = style_proj,
                            input_audio_proj = input_audio_proj,
                            optimizer        = optimizer,
                            scheduler        = cosine_scheduler,
                            global_step      = global_step,
                            epoch            = epoch,
                            output_dir       = config.output_dir,
                            max_checkpoints  = config.max_checkpoints,
                        )

    # ── Final checkpoint ──────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator      = accelerator,
            style_proj       = style_proj,
            input_audio_proj = input_audio_proj,
            optimizer        = optimizer,
            scheduler        = cosine_scheduler,
            global_step      = global_step,
            epoch            = config.num_epochs - 1,
            output_dir       = config.output_dir,
            max_checkpoints  = config.max_checkpoints,
        )

    accelerator.end_training()
    logger.info(f"Training complete — {global_step} total steps")
